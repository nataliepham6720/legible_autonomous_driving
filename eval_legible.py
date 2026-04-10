# =============================================================
# Observer-Aware Legibility Evaluation
# =============================================================

import re
import argparse
import os
import json
import math
import torch
import numpy as np
from transformers import AutoModelForCausalLM
from safetensors.torch import load_file

from utils.model_utils import load_llama_tokenizer, load_model
from utils.training_utils import get_train_val_data
from train_legible import VectorRouteGuard, patch_vector_fields


def set_global_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

# =============================================================
# UTILITIES
# =============================================================

def to_tensor(x, device):
    if isinstance(x, list):
        x = torch.tensor(x)
    return x.to(device)

def get_obs(sample):
    return sample["observation"] if "observation" in sample else sample

def get_text_prompt(sample, tokenizer=None, device=None):
    if "input" in sample and isinstance(sample["input"], str):
        return sample["input"]

    if "input_ids" in sample and tokenizer is not None:
        return tokenizer.decode(sample["input_ids"], skip_special_tokens=True)

    raise KeyError(f"No usable text prompt found. Keys: {sample.keys()}")

def extract_dataset_answers(sample):
    answers = []
    if "response_content" in sample:
        rc = sample["response_content"]
        if isinstance(rc, str):
            try:
                parsed = json.loads(rc)
                if isinstance(parsed, list):
                    answers = parsed
                elif isinstance(parsed, dict):
                    answers = [parsed]
            except Exception:
                pass
    return answers

def find_dataset_answer(dataset_answers, question):
    if not isinstance(dataset_answers, list):
        return "N/A"

    q_clean = question.strip().lower()
    for qa in dataset_answers:
        if qa.get("question", "").strip().lower() == q_clean:
            return qa.get("answer", "N/A")
    return "N/A"

def select_pedestrian_samples(dataset, n=10, seed=42):
    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(len(dataset))

    selected = []
    for idx in shuffled:
        item = dataset[int(idx)]
        obs = get_obs(item)
        ped = obs["pedestrian_descriptors"]
        if isinstance(ped, list):
            ped = torch.tensor(ped)
        if (ped[:, 0] > 0).sum().item() > 0:
            selected.append(item)
        if len(selected) >= n:
            break
    return selected

# =============================================================
# MANUAL GENERATION (BASELINE ONLY)
# =============================================================

@torch.no_grad()
def greedy_generate_with_vectors(model, tokenizer, model_inputs, max_new_tokens=120):
    input_ids = model_inputs["input_ids"].clone()
    attention_mask = model_inputs["attention_mask"].clone()
    print(model)
    for _ in range(max_new_tokens):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            route_descriptors=model_inputs["route_descriptors"],
            vehicle_descriptors=model_inputs["vehicle_descriptors"],
            pedestrian_descriptors=model_inputs["pedestrian_descriptors"],
            ego_vehicle_descriptor=model_inputs["ego_vehicle_descriptor"],
        )

        logits, _ = outputs
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

        input_ids = torch.cat([input_ids, next_token], dim=1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones_like(next_token)], dim=1
        )

        if next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

# =============================================================
# BUILD MODEL INPUTS (FOR BASELINE)
# =============================================================

def build_model_inputs(model, tokenizer, sample, question):
    device = next(model.parameters()).device
    obs = get_obs(sample)

    prompt = get_text_prompt(sample, tokenizer, device) + "\n\nQ: " + question + "\nA:"
    enc = tokenizer(prompt, return_tensors="pt").to(device)

    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "route_descriptors": to_tensor(obs["route_descriptors"], device).unsqueeze(0).float(),
        "vehicle_descriptors": to_tensor(obs["vehicle_descriptors"], device).unsqueeze(0).float(),
        "pedestrian_descriptors": to_tensor(obs["pedestrian_descriptors"], device).unsqueeze(0).float(),
        "ego_vehicle_descriptor": to_tensor(obs["ego_vehicle_descriptor"], device).unsqueeze(0).float(),
    }

def run_model_on_sample(model, tokenizer, sample, question):
    model_inputs = build_model_inputs(model, tokenizer, sample, question)
    return greedy_generate_with_vectors(model, tokenizer, model_inputs)

# =============================================================
# OBSERVER-AWARE LEGIBILITY (RO-MAN 2022) + CROSSING
# =============================================================

CROSSING_BOOST = 1.5   # multiplier for crossing pedestrians
BETA = 0.5            # temperature for distance penalty
FOV_DEG = 120.0       # field of view

def get_ego_heading_vector(ego: torch.Tensor):
    yaw = ego[..., 2]  # adjust if your schema differs
    return torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)

def visibility_score(ego_pos, ego_dir, obj_pos, fov_deg=FOV_DEG):
    v = obj_pos - ego_pos
    v_norm = torch.norm(v) + 1e-6

    cos_theta = torch.sum(ego_dir * v, dim=-1) / v_norm
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    theta = torch.acos(cos_theta)

    half_fov = (fov_deg / 2.0) * (math.pi / 180.0)
    vis = 1.0 - (theta / half_fov)
    return torch.clamp(vis, min=0.0)

def distance_to_straight_ahead_goal(ego_pos, ego_dir, obj_pos):
    v = obj_pos - ego_pos
    proj = torch.sum(v * ego_dir, dim=-1, keepdim=True) * ego_dir
    perp = v - proj
    return torch.norm(perp, dim=-1)


@torch.no_grad()
def safe_forward(model, inputs):
    """
    Run TWO forwards:
    - LoRA path → logits (QA)
    - Base path → hidden (route)
    """

    # ===== 1. LORA forward (for logits) =====
    try:
        out_lora = model.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            return_dict=True,
        )
        logits = getattr(out_lora, "logits", None)
    except:
        logits = None

    # ===== 2. BASE forward (for hidden) =====
    try:
        out_base = model.model.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            return_dict=True,
        )
        hidden = out_base.last_hidden_state
    except:
        hidden = None

    return logits, hidden

@torch.no_grad()
def safe_forward_legible(model, inputs):
    """
    Run TWO forwards:
    - LoRA path → logits (QA)
    - Base path → hidden (route)
    """

    # ===== 1. LORA forward (for logits) =====
    try:
        out_lora = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            return_dict=True,
        )
        logits = getattr(out_lora, "logits", None)
    except:
        logits = None

    # ===== 2. BASE forward (for hidden) =====
    try:
        out_base = model.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            return_dict=True,
        )
        hidden = out_base.last_hidden_state
    except:
        hidden = None

    return logits, hidden

def compute_legibility_from_predicted_route(route_pred, vehicle, pedestrian):
    beta = 0.5
    crossing_boost = 1.5

    # ---- existence masks ----
    veh_exists = vehicle[:, :, 0] > 0
    ped_exists = pedestrian[:, :, 0] > 0

    # ---- positions ----
    veh_xy = vehicle[:, :, 3:5]
    ped_xy = pedestrian[:, :, 2:4]
    ego_xy = route_pred[:, 0]

    # ---- trajectory direction (Eq. 3) ----
    traj_vec = route_pred[:, -1] - route_pred[:, 0]
    utraj = traj_vec / (torch.norm(traj_vec, dim=-1, keepdim=True) + 1e-6)

    # ---- observer orientations ----
    veh_yaw = vehicle[:, :, 5]
    veh_dir = torch.stack([torch.cos(veh_yaw), torch.sin(veh_yaw)], dim=-1)

    # pedestrians face ego (proxy)
    ped_dir = ego_xy.unsqueeze(1) - ped_xy
    ped_dir = ped_dir / (torch.norm(ped_dir, dim=-1, keepdim=True) + 1e-6)

    def visibility(xo, ro, fov_deg):
        # Eq. (4)
        v = ego_xy.unsqueeze(1) - xo
        v_norm = torch.norm(v, dim=-1) + 1e-6

        cos_theta = (v * ro).sum(-1) / v_norm
        cos_theta = torch.clamp(cos_theta, -0.999, 0.999)

        theta = torch.acos(cos_theta)

        # Eq. (5)
        half_fov = (fov_deg / 2.0) * math.pi / 180.0
        return torch.clamp(1 - theta / half_fov, min=0.0)

    def perp_distance(xo):
        # Eq. (6)-(7)
        v = xo - ego_xy.unsqueeze(1)
        proj = (v * utraj.unsqueeze(1)).sum(-1, keepdim=True) * utraj.unsqueeze(1)
        perp = v - proj
        d = torch.norm(perp, dim=-1)

        # distance cap (IMPORTANT)
        return torch.minimum(d, torch.tensor(50.0, device=d.device))

    # =========================
    # VEHICLES (FOV = 300°)
    # =========================
    veh_vis = visibility(veh_xy, veh_dir, fov_deg=300.0)
    veh_d = perp_distance(veh_xy)

    veh_leg = veh_vis * torch.exp(-beta * veh_d)
    veh_leg *= veh_exists.float()

    # =========================
    # PEDESTRIANS (FOV = 120°)
    # =========================
    ped_vis = visibility(ped_xy, ped_dir, fov_deg=120.0)
    ped_d = perp_distance(ped_xy)

    ped_leg = ped_vis * torch.exp(-beta * ped_d)

    # crossing term: (1 + γ c_o)
    ped_cross = pedestrian[:, :, 8]
    ped_leg *= (1 + crossing_boost * ped_cross)

    ped_leg *= ped_exists.float()

    # =========================
    # SCENE AGGREGATION (Eq. 10)
    # =========================
    total = veh_leg.sum(1) + ped_leg.sum(1)
    num = veh_exists.sum(1) + ped_exists.sum(1)

    # handle empty scene
    scene_leg = total / (num + 1e-6)

    return scene_leg.mean().item(),veh_leg, ped_leg

@torch.no_grad()
def compute_per_object_legibility(sample, beta=BETA, fov_deg=FOV_DEG):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    obs = get_obs(sample)

    vehicle = to_tensor(obs["vehicle_descriptors"], device).unsqueeze(0).float()
    pedestrian = to_tensor(obs["pedestrian_descriptors"], device).unsqueeze(0).float()
    ego = to_tensor(obs["ego_vehicle_descriptor"], device).unsqueeze(0).float()

    veh_exists = vehicle[:, :, 0] > 0
    ped_exists = pedestrian[:, :, 0] > 0

    veh_xy = vehicle[:, :, 3:5]
    ped_xy = pedestrian[:, :, 2:4]
    ego_xy = ego[:, :2]
    ego_dir = get_ego_heading_vector(ego)[0]

    results = []

    # ---- Vehicles (no crossing concept) ----
    for i in range(veh_xy.shape[1]):
        if veh_exists[0, i]:
            obj_pos = veh_xy[0, i]
            dist = torch.norm(obj_pos - ego_xy[0]).item()

            vis = visibility_score(ego_xy[0], ego_dir, obj_pos)
            d_to_goal = distance_to_straight_ahead_goal(ego_xy[0], ego_dir, obj_pos)
            leg = (vis * torch.exp(-beta * d_to_goal)).item()

            results.append({
                "type": "vehicle",
                "index": i,
                "distance_m": round(dist, 2),
                "crossing": False,
                "legibility": round(float(leg), 6),
            })

    # ---- Pedestrians (CROSSING-AWARE) ----
    for i in range(ped_xy.shape[1]):
        if ped_exists[0, i]:
            obj_pos = ped_xy[0, i]
            dist = torch.norm(obj_pos - ego_xy[0]).item()

            vis = visibility_score(ego_xy[0], ego_dir, obj_pos)
            d_to_goal = distance_to_straight_ahead_goal(ego_xy[0], ego_dir, obj_pos)

            # === CROSSING FLAG (your schema: adjust index if needed) ===
            # Common pattern: pedestrian[..., 8] = crossing (0/1)
            crossing_flag = bool(int(pedestrian[0, i, 8].item()))

            cross_weight = CROSSING_BOOST if crossing_flag else 1.0

            leg = (vis * torch.exp(-beta * d_to_goal) * cross_weight).item()

            results.append({
                "type": "pedestrian",
                "index": i,
                "distance_m": round(dist, 2),
                "crossing": crossing_flag,
                "legibility": round(float(leg), 6),
            })

    return sorted(results, key=lambda x: x["distance_m"])

@torch.no_grad()
def get_predicted_route(model, sample):
    device = next(model.parameters()).device
    obs = get_obs(sample)

    inputs = {
        "input_ids": torch.zeros((1, 1), dtype=torch.long, device=device),
        "attention_mask": torch.ones((1, 1), device=device),
    }

    _, hidden = safe_forward_legible(model, inputs)

    if hidden is None:
        raise RuntimeError("Failed to extract hidden states")

    B = hidden.shape[0]

    route_queries = model.route_queries.unsqueeze(0).expand(B, -1, -1)
    global_feat = hidden.mean(dim=1, keepdim=True)
    route_tokens = global_feat + route_queries

    route_pred = model.route_head(route_tokens)

    return route_pred

@torch.no_grad()
def get_route_for_eval(model, sample):
    device = next(model.parameters()).device
    obs = get_obs(sample)

    route_gt = to_tensor(obs["route_descriptors"], device).unsqueeze(0).float()[..., :2]

    # =========================================================
    # Case 1: Finetuned model (has route_head)
    # =========================================================
    if hasattr(model, "route_head"):

        inputs = {
            "input_ids": torch.zeros((1, 1), dtype=torch.long, device=device),
            "attention_mask": torch.ones((1, 1), device=device),
            "route_descriptors": to_tensor(obs["route_descriptors"], device).unsqueeze(0).float(),
            "vehicle_descriptors": to_tensor(obs["vehicle_descriptors"], device).unsqueeze(0).float(),
            "pedestrian_descriptors": to_tensor(obs["pedestrian_descriptors"], device).unsqueeze(0).float(),
            "ego_vehicle_descriptor": to_tensor(obs["ego_vehicle_descriptor"], device).unsqueeze(0).float(),
        }

        outputs = model(**inputs)

        if isinstance(outputs, dict) and "route_pred" in outputs:
            return outputs["route_pred"], route_gt, "predicted"

    # =========================================================
    # Case 2: Baseline model → use GT route
    # =========================================================
    return route_gt, route_gt, "ground_truth"

def get_base_lm(model):
    """
    Robustly find the underlying model that supports `.generate()`
    """

    visited = set()
    current = model

    # walk through possible wrappers
    while current is not None and id(current) not in visited:
        visited.add(id(current))

        # ✅ FOUND correct model
        if hasattr(current, "generate"):
            return current

        # try common wrapper attributes
        if hasattr(current, "model"):
            current = current.model
            continue

        if hasattr(current, "base_model"):
            current = current.base_model
            continue

        if hasattr(current, "module"):  # DDP case
            current = current.module
            continue

        break

    raise RuntimeError("Could not find a model with `.generate()`")

@torch.no_grad()
def greedy_generate_with_vectors(model, tokenizer, model_inputs, max_new_tokens=120):

    base_lm = get_base_lm(model)

    input_ids = model_inputs["input_ids"].clone()
    attention_mask = model_inputs["attention_mask"].clone()

    for _ in range(max_new_tokens):

        outputs = base_lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        logits = outputs.logits

        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

        input_ids = torch.cat([input_ids, next_token], dim=1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones_like(next_token)], dim=1
        )

        if next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


def extract_number(text):
    """
    Extract first number from LLM output.
    Handles integers and floats.
    """
    matches = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    if len(matches) == 0:
        return None
    return float(matches[0])

def extract_dataset_answers(sample):
    """
    Parse multiple JSON lines from response_content.
    """
    answers = []

    if "response_content" not in sample:
        return answers

    rc = sample["response_content"]

    if not isinstance(rc, str):
        return answers

    # split by lines (each line is one JSON)
    lines = rc.strip().split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        try:
            qa = json.loads(line)
            if isinstance(qa, dict) and "question" in qa:
                answers.append(qa)
        except:
            continue

    return answers

# def compute_qa_accuracy(pred, gt):
#     pred = pred.lower().strip()
#     gt = gt.lower().strip()

#     if gt == "n/a" or gt == "":
#         return None

#     # ---- exact match ----
#     if gt in pred:
#         return 1

#     # ---- number matching (VERY IMPORTANT for this dataset) ----
#     import re
#     gt_nums = re.findall(r"\d+\.?\d*", gt)
#     pred_nums = re.findall(r"\d+\.?\d*", pred)

#     if len(gt_nums) > 0 and len(pred_nums) > 0:
#         if gt_nums[0] == pred_nums[0]:
#             return 1

#     # ---- keyword overlap (fallback) ----
#     gt_words = set(gt.split())
#     pred_words = set(pred.split())

#     overlap = len(gt_words & pred_words) / (len(gt_words) + 1e-6)

#     return 1 if overlap > 0.5 else 0

@torch.no_grad()
def evaluate_qa_sample(model, tokenizer, sample):

    # model.eval()

    # ✅ always use base LM for QA
    base_lm = get_base_lm(model)
    print(type(base_lm))
    base_lm.eval()

    device = next(base_lm.parameters()).device

    def generate_answer(prompt):
        enc = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)

        input_len = enc["input_ids"].shape[1]

        output_ids = base_lm.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_new_tokens=80,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

        # ✅ ONLY take generated tokens (remove prompt)
        generated_ids = output_ids[0][input_len:]

        return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # =========================================================
    # CASE 1: QA dataset
    # =========================================================
    if "response_content" in sample:

        dataset_answers = extract_dataset_answers(sample)
        results = []

        base_prompt = get_text_prompt(sample, tokenizer)

        for qa in dataset_answers:
            question = qa["question"]
            gt = qa["answer"]

            prompt = f"{base_prompt}\n\nQ: {question}\nA:"
            pred = generate_answer(prompt)

            obs = get_obs(sample)
            acc = compute_qa_accuracy(pred, gt, tokenizer, base_lm)

            print("\n================ FRAME CONTEXT ================")
            print("🚗 #Vehicles:", int((torch.tensor(obs["vehicle_descriptors"])[:,0] > 0).sum()))
            print("🚶 #Pedestrians:", int((torch.tensor(obs["pedestrian_descriptors"])[:,0] > 0).sum()))
            print("==============================================\n")

            print("\n" + "="*60)
            print(f"FRAME QA COMPARISON")
            print("="*60)

            print(f"\n❓ Question:\n{question}")

            print("\n🟡 Predicted Answer:")
            print(pred)

            print("\n🔵 Ground Truth Answer:")
            print(gt)

            print("\n📊 Score:", acc)
            print("="*60 + "\n")

            results.append(acc if acc is not None else 0)

        return results

    # =========================================================
    # CASE 2: instruction dataset
    # =========================================================
    elif "output" in sample:
        QUESTION_SET = [
            "What is the distance of the farthest pedestrian?",
            "What is your current speed?",
            "Are there any traffic light ahead?",
            "How many pedestrians are on the scene at the current point?",
            "How many pedestrians are on the scene at the current point?",
        ]
        total_acc = 0
        for question in QUESTION_SET:
            # question = qa["question"]
            # gt = qa["answer"]

            prompt = f"Q: {question}\nA:"
            # prompt = sample.get("input", "")
            print("Prompt:", prompt)
            gt = sample["output"]

            pred = generate_answer(prompt)
            print("\n🟡 Predicted Answer:")
            print(pred)

            obs = get_obs(sample)
            acc = compute_qa_accuracy(pred, gt, tokenizer, base_lm)
            total_acc += acc

        print("\n================ FRAME CONTEXT ================")
        print("🚗 #Vehicles:", int((torch.tensor(obs["vehicle_descriptors"])[:,0] > 0).sum()))
        print("🚶 #Pedestrians:", int((torch.tensor(obs["pedestrian_descriptors"])[:,0] > 0).sum()))
        print("==============================================\n")

        # print("\n" + "="*60)
        # print(f"FRAME QA COMPARISON")
        # print("="*60)

        # print(f"\n❓ Question:\n{question}")

        print("\n🟡 Predicted Answer:")
        print(pred)

        print("\n🔵 Ground Truth Answer:")
        print(gt)

        print("\n📊 Score:", acc)
        print("="*60 + "\n")

        # results.append(acc if acc is not None else 0)

        return [total_acc/len(QUESTION_SET)]

    return []


def compute_qa_accuracy(pred, gt, tokenizer=None, model=None):
    """
    Compare prediction and ground truth in embedding space.
    Uses token-level distributions (mean pooled).
    """

    if gt in ["", "n/a"]:
        return None

    print(gt)

    # -------------------------
    # tokenize
    # -------------------------
    device = next(model.parameters()).device

    pred_tokens = tokenizer(
        pred,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    ).to(device)

    gt_tokens = tokenizer(
        gt,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    ).to(device)

    # -------------------------
    # get embeddings (last hidden)
    # -------------------------
    with torch.no_grad():
        pred_hidden = model(
            input_ids=pred_tokens["input_ids"],
            attention_mask=pred_tokens["attention_mask"],
            output_hidden_states=True,
            return_dict=True,
        ).hidden_states[-1]

        gt_hidden = model(
            input_ids=gt_tokens["input_ids"],
            attention_mask=gt_tokens["attention_mask"],
            output_hidden_states=True,
            return_dict=True,
        ).hidden_states[-1]

    # -------------------------
    # mean pool (token distribution summary)
    # -------------------------
    pred_emb = pred_hidden.mean(dim=1)
    gt_emb = gt_hidden.mean(dim=1)

    # -------------------------
    # cosine similarity
    # -------------------------
    sim = torch.nn.functional.cosine_similarity(pred_emb, gt_emb).item()

    # -------------------------
    # threshold decision
    # -------------------------
    # tune this if needed
    return sim
# def compute_grounded_qa_accuracy(pred, question, obs):
#     """
#     Grounded QA evaluation based on scene semantics.
    
#     Args:
#         pred: model output (string)
#         question: question string
#         obs: observation dict (contains descriptors)

#     Returns:
#         1 / 0 / None
#     """

#     pred = pred.lower().strip()
#     question = question.lower()

#     # =========================
#     # 🚶 COUNT PEDESTRIANS
#     # =========================
#     if "how many pedestrians" in question:
#         ped = torch.tensor(obs["pedestrian_descriptors"])
#         gt = int((ped[:, 0] > 0).sum().item())

#         pred_num = extract_number(pred)
#         if pred_num is None:
#             return 0

#         return 1 if int(pred_num) == gt else 0

#     # =========================
#     # 🚗 COUNT VEHICLES
#     # =========================
#     if "how many vehicles" in question:
#         veh = torch.tensor(obs["vehicle_descriptors"])
#         gt = int((veh[:, 0] > 0).sum().item())

#         pred_num = extract_number(pred)
#         if pred_num is None:
#             return 0

#         return 1 if int(pred_num) == gt else 0

#     # =========================
#     # 📏 FARTHEST PEDESTRIAN
#     # =========================
#     if "farthest pedestrian" in question:
#         ped = torch.tensor(obs["pedestrian_descriptors"])

#         mask = ped[:, 0] > 0
#         if mask.sum() == 0:
#             return None

#         ped_xy = ped[mask][:, 2:4]
#         dist = torch.norm(ped_xy, dim=-1)
#         gt = dist.max().item()

#         pred_num = extract_number(pred)
#         if pred_num is None:
#             return 0

#         # allow tolerance (meters)
#         return 1 if abs(pred_num - gt) < 2.0 else 0

#     # =========================
#     # 🚗 SPEED (if available)
#     # =========================
#     if "speed" in question:
#         ego = torch.tensor(obs["ego_vehicle_descriptor"])

#         # adjust index if needed
#         gt = ego[0].item()

#         pred_num = extract_number(pred)
#         if pred_num is None:
#             return 0

#         return 1 if abs(pred_num - gt) < 2.0 else 0

#     # =========================
#     # ❓ OPEN-ENDED → skip
#     # =========================
#     return None

def extract_dataset_answers(sample):
    """
    Extract QA pairs from response_content.
    Handles multiple JSON lines.
    """

    answers = []

    if "response_content" not in sample:
        return answers

    rc = sample["response_content"]

    if not isinstance(rc, str):
        return answers

    # each line is one JSON object
    lines = rc.strip().split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        try:
            qa = json.loads(line)
            if isinstance(qa, dict) and "question" in qa:
                answers.append(qa)
        except:
            continue

    return answers

def compute_legibility_from_predicted_route(route_pred, vehicle, pedestrian):
    beta = 0.5
    fov_deg = 120.0
    crossing_boost = 1.5

    veh_exists = vehicle[:, :, 0] > 0
    ped_exists = pedestrian[:, :, 0] > 0

    veh_xy = vehicle[:, :, 3:5]
    ped_xy = pedestrian[:, :, 2:4]

    ego_xy = route_pred[:, 0]

    traj_dir = route_pred[:, -1] - route_pred[:, 0]
    traj_dir = traj_dir / (torch.norm(traj_dir, dim=-1, keepdim=True) + 1e-6)

    # ---- observer directions ----
    veh_yaw = vehicle[:, :, 5]
    veh_dir = torch.stack([torch.cos(veh_yaw), torch.sin(veh_yaw)], dim=-1)

    ped_dir = ego_xy.unsqueeze(1) - ped_xy
    ped_dir = ped_dir / (torch.norm(ped_dir, dim=-1, keepdim=True) + 1e-6)

    def visibility(obs_pos, obs_dir):
        v = ego_xy.unsqueeze(1) - obs_pos
        v_norm = torch.norm(v, dim=-1) + 1e-6

        cos_theta = (v * obs_dir).sum(-1) / v_norm
        cos_theta = torch.clamp(cos_theta, -0.999, 0.999)

        theta = torch.acos(cos_theta)
        half_fov = (fov_deg / 2) * math.pi / 180.0

        return torch.clamp(1 - theta / half_fov, min=0.0)

    def dist_to_traj(obs_pos):
        v = obs_pos - ego_xy.unsqueeze(1)
        proj = (v * traj_dir.unsqueeze(1)).sum(-1, keepdim=True) * traj_dir.unsqueeze(1)
        return torch.norm(v - proj, dim=-1)

    veh_leg = visibility(veh_xy, veh_dir) * torch.exp(-beta * dist_to_traj(veh_xy))
    veh_leg *= veh_exists.float()

    ped_cross = pedestrian[:, :, 8]
    ped_weight = 1 + (crossing_boost - 1) * ped_cross

    ped_leg = visibility(ped_xy, ped_dir) * torch.exp(-beta * dist_to_traj(ped_xy)) * ped_weight
    ped_leg *= ped_exists.float()

    total = veh_leg.sum(1) + ped_leg.sum(1)
    num = veh_exists.sum(1) + ped_exists.sum(1)

    return (total / (num + 1e-6)).mean().item()

# =============================================================
# MAIN EVALUATION LOOP
# =============================================================
def evaluate_legibility_behavior(model, tokenizer, val_data, n_samples=20, seed=42):

    QUESTION_SET = [
        "What is the distance of the farthest pedestrian?",
        "What is your current speed?",
        "Are there any traffic light ahead?",
        "How many pedestrians are on the scene at the current point?",
        "How many pedestrians are on the scene at the current point?",
    ]

    samples = select_pedestrian_samples(val_data, n=n_samples, seed=seed)

    total_acc = []
    total_leg = []
    total_l2 = []

    print("\n================ EVALUATION ================\n")

    for i, sample in enumerate(samples):

        print(f"\n================ SAMPLE {i+1} ================\n")

        dataset_answers = extract_dataset_answers(sample)
        obs = get_obs(sample)

        veh = to_tensor(obs["vehicle_descriptors"], "cuda").unsqueeze(0).float()
        ped = to_tensor(obs["pedestrian_descriptors"], "cuda").unsqueeze(0).float()
        route_gt = to_tensor(obs["route_descriptors"], "cuda").unsqueeze(0).float()[..., :2]

        # =========================================================
        # 🚗 ROUTE PREDICTION
        # =========================================================
        # route_pred = get_predicted_route(model, sample)
        route_pred, route_gt, route_type = get_route_for_eval(model, sample)

        print("🔹 Predicted Route (first 5 points):")
        print(route_pred[0, :5])

        print("\n🔹 Ground Truth Route (first 5 points):")
        print(route_gt[0, :5])

        # ---- L2 LOSS ----
        min_len = min(route_pred.shape[1], route_gt.shape[1])
        l2 = torch.norm(route_pred[:, :min_len] - route_gt[:, :min_len], dim=-1).mean().item()
        total_l2.append(l2/torch.norm(route_gt[:, :min_len], dim=-1).mean().item())

        print(f"\n📏 Route L2 Error: {l2:.4f}")

        # =========================================================
        # 👀 LEGIBILITY PER AGENT
        # =========================================================

        def per_agent_legibility(route_pred, vehicle, pedestrian):

            beta = 0.5
            fov_deg = 120.0
            crossing_boost = 1.5

            veh_exists = vehicle[:, :, 0] > 0
            ped_exists = pedestrian[:, :, 0] > 0

            veh_xy = vehicle[:, :, 3:5]
            ped_xy = pedestrian[:, :, 2:4]

            ego_xy = route_pred[:, 0]

            traj_dir = route_pred[:, -1] - route_pred[:, 0]
            traj_dir = traj_dir / (torch.norm(traj_dir, dim=-1, keepdim=True) + 1e-6)

            veh_yaw = vehicle[:, :, 5]
            veh_dir = torch.stack([torch.cos(veh_yaw), torch.sin(veh_yaw)], dim=-1)

            ped_dir = ego_xy.unsqueeze(1) - ped_xy
            ped_dir = ped_dir / (torch.norm(ped_dir, dim=-1, keepdim=True) + 1e-6)

            def visibility(obs_pos, obs_dir):
                v = ego_xy.unsqueeze(1) - obs_pos
                v_norm = torch.norm(v, dim=-1) + 1e-6

                cos_theta = (v * obs_dir).sum(-1) / v_norm
                cos_theta = torch.clamp(cos_theta, -0.999, 0.999)

                theta = torch.acos(cos_theta)
                half_fov = (fov_deg / 2) * math.pi / 180.0

                return torch.clamp(1 - theta / half_fov, min=0.0)

            def dist_to_traj(obs_pos):
                v = obs_pos - ego_xy.unsqueeze(1)
                proj = (v * traj_dir.unsqueeze(1)).sum(-1, keepdim=True) * traj_dir.unsqueeze(1)
                return torch.norm(v - proj, dim=-1)

            veh_leg = visibility(veh_xy, veh_dir) * torch.exp(-beta * dist_to_traj(veh_xy))
            ped_leg = visibility(ped_xy, ped_dir) * torch.exp(-beta * dist_to_traj(ped_xy))

            ped_cross = pedestrian[:, :, 8]
            ped_leg *= (1 + (crossing_boost - 1) * ped_cross)

            return veh_leg, ped_leg, veh_exists, ped_exists

        veh_leg, ped_leg, veh_exists, ped_exists = per_agent_legibility(route_pred, veh, ped)
        veh_leg_gt, ped_leg_gt, _, _ = per_agent_legibility(route_gt, veh, ped)
        print("\n👀 Legibility per agent:")

        for j in range(veh_leg.shape[1]):
            if veh_exists[0, j]:
                print(f"  🚗 Vehicle {j}: {veh_leg[0, j].item():.4f}")
                print(f"  🚗 Vehicle {j} GT route: {veh_leg_gt[0, j].item():.4f}")

        for j in range(ped_leg.shape[1]):
            if ped_exists[0, j]:
                cross = int(ped[0, j, 8].item())
                print(f"  🚶 Pedestrian {j} (cross={cross}): {ped_leg[0, j].item():.4f}")
                print(f"  🚶 Pedestrian {j} GT route (cross={cross}): {ped_leg_gt[0, j].item():.4f}")

        leg_score = compute_legibility_from_predicted_route(route_pred, veh, ped)
        total_leg.append(leg_score)
        # total_leg.append("--------------")
        leg_score_gt = compute_legibility_from_predicted_route(route_gt, veh, ped)
        total_leg.append(leg_score_gt)

        print(f"\n📊 Total Legibility Score on new route: {leg_score:.4f}")
        print(f"\n📊 Total Legibility Score on gt route: {leg_score_gt:.4f}")

        # =========================================================
        # 🧠 QA EVALUATION
        # =========================================================
        print(f"\n🧠 Evaluating QA / Instruction for sample {i+1}")

        accs = evaluate_qa_sample(model, tokenizer, sample)
        # acc = compute_qa_accuracy(pred, gt, tokenizer, base_lm)

        # keep only valid scores
        valid_accs = [a for a in accs if a is not None]

        if len(valid_accs) > 0:
            total_acc.extend(valid_accs)
        else:
            print("⚠️ No valid QA found for this sample")
    # =========================================================
    # FINAL METRICS
    # =========================================================

    avg_acc = np.mean(total_acc) if total_acc else 0.0
    avg_leg = np.mean(total_leg)
    avg_l2 = np.mean(total_l2)

    print("\n================ FINAL RESULTS ================\n")
    print(f"QA Accuracy:        {avg_acc:.4f}")
    print(f"Route Legibility:   {avg_leg:.4f}")
    print(f"Route relative L2 Error:     {avg_l2:.4f}")
    print("\n=============================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--model_type", default="legible")
    parser.add_argument("--data_path", default="data/vqa_test_1k.pkl")
    parser.add_argument("--n_samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_global_seed(args.seed)

    print(f"🔹 Loading checkpoint from: {args.model_dir}")

    base_model_token = "meta-llama/Llama-2-7b-hf"

    if args.model_type=="legible":
        base_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        dtype=torch.float16,
        device_map="auto",
        )

        model = VectorRouteGuard(base_model).cuda()

        safetensors_path = os.path.join(args.model_dir, "model.safetensors")
        state_dict = load_file(safetensors_path, device="cuda")
        model.load_state_dict(state_dict, strict=False)

        print("✅ Loaded trained VectorRouteGuard weights.")
    else:
        base_model = load_model(
            base_model=base_model_token,
            resume_from_checkpoint=args.model_dir,
            lora_r=16,
        ).cuda()

        model = VectorRouteGuard(base_model).cuda()

        print("✅ Loaded trained VectorRouteGuard + LoRA weights.")

    tokenizer = load_llama_tokenizer("meta-llama/Llama-2-7b-hf")

    _, val_data = get_train_val_data(
        args.data_path,
        tokenizer,
        val_set_size=0.9,
    )

    val_data = patch_vector_fields(val_data)

    evaluate_legibility_behavior(
        model,
        tokenizer,
        val_data,
        n_samples=args.n_samples,
        seed=args.seed,
    )
