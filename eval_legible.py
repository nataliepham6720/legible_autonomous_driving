# =============================================================
# eval_legible.py
# Observer-Aware Legibility Evaluation (RO-MAN 2022)
# Goal assumed to be straight ahead; crossing-aware pedestrians
# =============================================================

import argparse
import os
import json
import math
import torch
import numpy as np
from transformers import AutoModelForCausalLM
from safetensors.torch import load_file

from utils.model_utils import load_llama_tokenizer
from utils.training_utils import get_train_val_data
from train_legible import VectorRouteGuard, patch_vector_fields

# =============================================================
# REPRODUCIBILITY
# =============================================================

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

    for _ in range(max_new_tokens):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            route_descriptors=model_inputs["route_descriptors"],
            vehicle_descriptors=model_inputs["vehicle_descriptors"],
            pedestrian_descriptors=model_inputs["pedestrian_descriptors"],
            ego_vehicle_descriptor=model_inputs["ego_vehicle_descriptor"],
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

    _, hidden = safe_forward(model, inputs)

    if hidden is None:
        raise RuntimeError("Failed to extract hidden states")

    B = hidden.shape[0]

    route_queries = model.route_queries.unsqueeze(0).expand(B, -1, -1)
    global_feat = hidden.mean(dim=1, keepdim=True)
    route_tokens = global_feat + route_queries

    route_pred = model.route_head(route_tokens)

    return route_pred



@torch.no_grad()
def greedy_generate_with_vectors(model, tokenizer, model_inputs, max_new_tokens=120):
    input_ids = model_inputs["input_ids"].clone()
    attention_mask = model_inputs["attention_mask"].clone()

    for _ in range(max_new_tokens):

        logits, _ = safe_forward(model, {
            **model_inputs,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        })

        # ---- fallback if logits missing ----
        if logits is None:
            raise RuntimeError("Model did not return logits — cannot generate text")

        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

        input_ids = torch.cat([input_ids, next_token], dim=1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones_like(next_token)], dim=1
        )

        if next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

def compute_qa_accuracy(pred, gt):
    pred = pred.lower().strip()
    gt = gt.lower().strip()

    if gt == "n/a":
        return None

    return 1 if gt in pred else 0

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
        "If a pedestrian suddenly starts crossing the road in front of you, how will you drive and why?",
        "What is the distance of the farthest pedestrian?",
        "What is your current speed?",
        "How are you going to drive in this situation and why?",
        "What is the distance and the direction of a certain pedestrian?",
        "Can you describe the current driving scenario?",
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
        route_pred = get_predicted_route(model, sample)

        print("🔹 Predicted Route (first 5 points):")
        print(route_pred[0, :5])

        print("\n🔹 Ground Truth Route (first 5 points):")
        print(route_gt[0, :5])

        # ---- L2 LOSS ----
        min_len = min(route_pred.shape[1], route_gt.shape[1])
        l2 = torch.norm(route_pred[:, :min_len] - route_gt[:, :min_len], dim=-1).mean().item()
        total_l2.append(l2)

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

        print("\n👀 Legibility per agent:")

        for j in range(veh_leg.shape[1]):
            if veh_exists[0, j]:
                print(f"  🚗 Vehicle {j}: {veh_leg[0, j].item():.4f}")

        for j in range(ped_leg.shape[1]):
            if ped_exists[0, j]:
                cross = int(ped[0, j, 8].item())
                print(f"  🚶 Pedestrian {j} (cross={cross}): {ped_leg[0, j].item():.4f}")

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

        for q_idx, question in enumerate(QUESTION_SET):

            print(f"\n❓ Q{q_idx+1}: {question}")

            pred = run_model_on_sample(model, tokenizer, sample, question)
            gt = find_dataset_answer(dataset_answers, question)

            print("\n🟡 Model Answer:")
            print(pred.strip())

            print("\n🔵 Ground Truth:")
            print(gt.strip())

            acc = compute_qa_accuracy(pred, gt)
            if acc is not None:
                total_acc.append(acc)
                print(f"\n✅ Correct: {acc}")
            else:
                print("\n⚠️ Skipped (no GT)")

            print("-" * 80)

    # =========================================================
    # FINAL METRICS
    # =========================================================

    avg_acc = np.mean(total_acc) if total_acc else 0.0
    avg_leg = np.mean(total_leg)
    avg_l2 = np.mean(total_l2)

    print("\n================ FINAL RESULTS ================\n")
    print(f"QA Accuracy:        {avg_acc:.4f}")
    print(f"Route Legibility:   {avg_leg:.4f}")
    print(f"Route L2 Error:     {avg_l2:.4f}")
    print("\n=============================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--data_path", default="data/vqa_test_1k.pkl")
    parser.add_argument("--n_samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_global_seed(args.seed)

    print(f"🔹 Loading checkpoint from: {args.model_dir}")

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
