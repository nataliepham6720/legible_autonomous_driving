import re
import argparse
import os
import json
import math
import torch
import numpy as np
from transformers import AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
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
BETA = 0.5
CROSSING_BOOST = 1.5
FOV_DEG_VEH = 300
FOV_DEG_PED = 120

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

@torch.no_grad()
def safe_forward_legible(model, inputs):
    """
    Run TWO forwards:
    - LoRA path → logits (QA)
    - Base path → hidden (route)
    """
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
    # print(route_gt) #

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

# @torch.no_grad()
# def greedy_generate_with_vectors(model, tokenizer, model_inputs, max_new_tokens=120):

#     base_lm = get_base_lm(model)

#     input_ids = model_inputs["input_ids"].clone()
#     attention_mask = model_inputs["attention_mask"].clone()

#     for _ in range(max_new_tokens):

#         outputs = base_lm(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             return_dict=True,
#         )

#         logits = outputs.logits

#         next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

#         input_ids = torch.cat([input_ids, next_token], dim=1)
#         attention_mask = torch.cat(
#             [attention_mask, torch.ones_like(next_token)], dim=1
#         )

#         if next_token.item() == tokenizer.eos_token_id:
#             break

#     return tokenizer.decode(input_ids[0], skip_special_tokens=True)

#     # =========================================================
#     # CASE 1: QA dataset
#     # =========================================================
#     if "response_content" in sample:

#         dataset_answers = extract_dataset_answers(sample)
#         results = []

#         base_prompt = get_text_prompt(sample, tokenizer)

#         for qa in dataset_answers:
#             question = qa["question"]
#             gt = qa["answer"]

#             prompt = f"{base_prompt}\n\nQ: {question}\nA:"
#             pred = generate_answer(prompt)

#             obs = get_obs(sample)
#             acc = compute_qa_accuracy(pred, gt, tokenizer, base_lm)

#             print("\n================ FRAME CONTEXT ================")
#             print("🚗 #Vehicles:", int((torch.tensor(obs["vehicle_descriptors"])[:,0] > 0).sum()))
#             print("🚶 #Pedestrians:", int((torch.tensor(obs["pedestrian_descriptors"])[:,0] > 0).sum()))
#             print("==============================================\n")

#             print("\n" + "="*60)
#             print(f"FRAME QA COMPARISON")
#             print("="*60)

#             print(f"\n❓ Question:\n{question}")

#             print("\n🟡 Predicted Answer:")
#             print(pred)

#             print("\n🔵 Ground Truth Answer:")
#             print(gt)

#             print("\n📊 Score:", acc)
#             print("="*60 + "\n")

#             results.append(acc if acc is not None else 0)

#         return results

@torch.no_grad()
def generate_answer(prompt, route, veh, ped, base_lm):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        enc = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)

        input_len = enc["input_ids"].shape[1]

        output_ids = base_lm.model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            route_descriptors=route,
            vehicle_descriptors=veh,
            pedestrian_descriptors=ped,
            max_new_tokens=80,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

        gen_ids = output_ids[0][input_len:]
        return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

@torch.no_grad()
def evaluate_qa_sample(model, tokenizer, sample, embed_model):

    base_lm = get_base_lm(model)
    base_lm.eval()
    device = next(base_lm.parameters()).device

    # =========================
    # QUESTIONS
    # =========================
    QUESTION_SET = [
        "What is the distance of the farthest pedestrian?",
        "What is your current speed?",
        "Are there any traffic light ahead?",
        "How many pedestrians are on the scene at the current point?",
        "How many pedestrians are on the scene at the current point?",
    ]
    
    # =========================
    # EMBEDDING COSINE SIM
    # =========================
    def cosine_sim(a, b):
        emb = embed_model.encode([a, b], convert_to_tensor=True)
        return torch.nn.functional.cosine_similarity(emb[0], emb[1], dim=0).item()

    # =========================
    # SCENE GROUND TRUTH
    # =========================
    def compute_scene_answers(veh, ped, ego, route):
        ped_exists = ped[:, 0] > 0
        veh_exists = veh[:, 0] > 0

        ped_xy = ped[:, 2:4]
        ego_xy = route[0,:2]

        # ---- 1. farthest pedestrian distance ----
        if ped_exists.any():
            dists = torch.norm(ped_xy[ped_exists] - ego_xy, dim=-1)
            farthest = dists.max().item()
        else:
            farthest = 0.0

        # ---- 2. ego speed ----
        print(route[0])
        speed = route[0,6].item() if route.shape[0] > 0 else 0.0

        # ---- 3. traffic light (simple proxy) ----
        # adjust if you have explicit signal
        has_light = route[0,10].item() # placeholder unless dataset provides

        # ---- 4. pedestrian count ----
        ped_count = int(ped_exists.sum().item())

        return {
            QUESTION_SET[0]: farthest,
            QUESTION_SET[1]: speed,
            QUESTION_SET[2]: has_light,
            QUESTION_SET[3]: ped_count,
            QUESTION_SET[4]: ped_count,
        }

    # =========================
    # PARSE LLM OUTPUT
    # =========================
    def parse_number(text):
        import re
        match = re.search(r"[-+]?\d*\.?\d+", text)
        return float(match.group()) if match else None

    def parse_yes_no(text):
        text = text.lower()
        if "yes" in text:
            return "yes"
        if "no" in text:
            return "no"
        return None

    # =========================
    # MAIN LOOP
    # =========================
    obs = get_obs(sample)
    veh = torch.tensor(obs["vehicle_descriptors"])
    ped = torch.tensor(obs["pedestrian_descriptors"])
    ego = torch.tensor(obs["ego_vehicle_descriptor"])
    route = torch.tensor(obs["route_descriptors"])
    
    scene_gt = compute_scene_answers(veh, ped, ego, route)

    cosine_scores = []
    scene_accs = []

    for question in QUESTION_SET:

        prompt = f"Q: {question}\nA:"
        pred = generate_answer(prompt, route, veh, ped, base_lm)

        # ---- cosine similarity vs GT text ----
        gt_text = str(scene_gt[question])
        cos = cosine_sim(pred, gt_text)
        cosine_scores.append(cos)

        # ---- scene accuracy ----
        gt_val = scene_gt[question]

        if isinstance(gt_val, float):  # distance / speed
            pred_val = parse_number(pred)
            if pred_val is None:
                acc = 0
            else:
                acc = float(abs(pred_val - gt_val) < 2.0)  # tolerance

        elif isinstance(gt_val, int):  # count
            pred_val = parse_number(pred)
            acc = float(pred_val == gt_val) if pred_val is not None else 0

        elif isinstance(gt_val, str):  # yes/no
            pred_val = parse_yes_no(pred)
            acc = float(pred_val == gt_val)

        else:
            acc = 0

        scene_accs.append(acc)

        # ---- debug print ----
        print("\n" + "="*50)
        print("Q:", question)
        print("Pred:", pred)
        print("GT:", gt_val)
        print("Cosine:", round(cos, 4))
        print("Scene Acc:", acc)

    # =========================
    # FINAL METRICS
    # =========================
    avg_cos = sum(cosine_scores) / len(cosine_scores)
    avg_acc = sum(scene_accs) / len(scene_accs)

    print("\n================ QA METRICS ================")
    print(f"Cosine Similarity: {avg_cos:.4f}")
    print(f"Scene Accuracy:    {avg_acc:.4f}")
    print("===========================================\n")

    return {
        "cosine_similarity": avg_cos,
        "scene_accuracy": avg_acc,
        "per_question_cosine": cosine_scores,
        "per_question_accuracy": scene_accs,
    }

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

def per_agent_legibility(route_pred, vehicle, pedestrian):
      beta = BETA
      crossing_boost = CROSSING_BOOST

      veh_exists = vehicle[:, :, 0] > 0
      ped_exists = pedestrian[:, :, 0] > 0

      veh_xy = vehicle[:, :, 3:5]
      ped_xy = pedestrian[:, :, 2:4]

      ego_xy = route_pred[:, 0]

      # ---- Eq. (3): trajectory direction ----
      traj_vec = route_pred[:, -1] - route_pred[:, 0]
      traj_dir = traj_vec / (torch.norm(traj_vec, dim=-1, keepdim=True) + 1e-6)

      # ---- observer directions ----
      veh_yaw = vehicle[:, :, 5]
      veh_dir = torch.stack([torch.cos(veh_yaw), torch.sin(veh_yaw)], dim=-1)

      ped_dir = ego_xy.unsqueeze(1) - ped_xy
      ped_dir = ped_dir / (torch.norm(ped_dir, dim=-1, keepdim=True) + 1e-6)

      # =========================
      # VISIBILITY (Eq. 4–5)
      # =========================
      def visibility(obs_pos, obs_dir, fov_deg):
          v = ego_xy.unsqueeze(1) - obs_pos
          v_norm = torch.norm(v, dim=-1) + 1e-6

          cos_theta = (v * obs_dir).sum(-1) / v_norm
          cos_theta = torch.clamp(cos_theta, -0.999, 0.999)

          theta = torch.acos(cos_theta)
          half_fov = (fov_deg / 2.0) * math.pi / 180.0

          return torch.clamp(1 - theta / half_fov, min=0.0)

      # =========================
      # DISTANCE (Eq. 6–7)
      # =========================
      def dist_to_traj(obs_pos):
          v = obs_pos - ego_xy.unsqueeze(1)
          proj = (v * traj_dir.unsqueeze(1)).sum(-1, keepdim=True) * traj_dir.unsqueeze(1)
          perp = v - proj
          d = torch.norm(perp, dim=-1)

          return torch.minimum(d, torch.tensor(50.0, device=d.device))

      # =========================
      # VEHICLES
      # =========================
      veh_vis = visibility(veh_xy, veh_dir, FOV_DEG_VEH)
      veh_d = dist_to_traj(veh_xy)

      veh_leg = veh_vis * torch.exp(-beta * veh_d)
      veh_leg *= veh_exists.float()

      # =========================
      # PEDESTRIANS
      # =========================
      ped_vis = visibility(ped_xy, ped_dir, FOV_DEG_PED)
      ped_d = dist_to_traj(ped_xy)

      ped_cross = pedestrian[:, :, 8]
      ped_weight = 1 + crossing_boost * ped_cross

      ped_leg = ped_vis * torch.exp(-beta * ped_d) * ped_weight
      ped_leg *= ped_exists.float()

      return veh_leg, ped_leg, veh_exists, ped_exists
    
# =============================================================
# MAIN EVALUATION LOOP
# =============================================================
def evaluate_legibility_behavior(model, tokenizer, val_data, model_type="vanilla", n_samples=20, seed=42):

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
        veh_leg, ped_leg, veh_exists, ped_exists = per_agent_legibility(route_pred, veh, ped)
        veh_leg_gt, ped_leg_gt, veh_exists_gt, ped_exists_gt = per_agent_legibility(route_gt, veh, ped)
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

        total = veh_leg.sum(1) + ped_leg.sum(1)
        num = veh_exists.sum(1) + ped_exists.sum(1)

        total_leg = total / (num + 1e-6)
        
        total_gt = veh_leg_gt.sum(1) + ped_leg_gt.sum(1)
        num_gt = veh_exists_gt.sum(1) + ped_exists_gt.sum(1)

        total_leg_gt = total_gt / (num_gt + 1e-6)
        # print(total_leg_gt)
        # leg_score, _, _ = compute_legibility_from_predicted_route(route_pred, veh, ped)
        # total_leg.append(leg_score)
        # # total_leg.append("--------------")
        # leg_score_gt, _, _ = compute_legibility_from_predicted_route(route_gt, veh, ped)
        # total_leg.append(leg_score_gt)

        print(f"\n📊 Total Legibility Score on new route: {total_leg.item():.4f}")
        print(f"\n📊 Total Legibility Score on gt route: {total_leg_gt.item():.4f}")

        # =========================================================
        # 🧠 QA EVALUATION
        # =========================================================
        print(f"\n🧠 Evaluating QA / Instruction for sample {i+1}")
        embed_model = SentenceTransformer("all-mpnet-base-v2")
        accs = evaluate_qa_sample(model, tokenizer, sample, embed_model)
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
        model_type=args.model_type,
        n_samples=args.n_samples,
        seed=args.seed,
    )
