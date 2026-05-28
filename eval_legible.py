import re
import argparse
import os
import json
import math
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import torch.nn.functional as F
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
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.tensor(x).to(device)


def get_obs(sample):
    return sample["observation"] if "observation" in sample else sample


def get_text_prompt(sample, tokenizer=None, device=None):
    if "input" in sample and isinstance(sample["input"], str):
        return sample["input"]
    if "input_ids" in sample and tokenizer is not None:
        return tokenizer.decode(sample["input_ids"], skip_special_tokens=True)
    raise KeyError(f"No usable text prompt found. Keys: {sample.keys()}")


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
        elif isinstance(ped, np.ndarray):
            ped = torch.from_numpy(ped)
        if (ped[:, 0] > 0).sum().item() > 0:
            selected.append(item)
        if len(selected) >= n:
            break
    return selected


class SimpleEmbedder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    @torch.no_grad()
    def encode(self, texts):
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        outputs = self.model(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1)
        return emb


def get_base_lm(model):
    """
    Robustly find the underlying model that supports `.generate()`.
    Walks wrappers: VectorRouteGuard → .model → LlamaForCausalLM (has .generate).
    """
    visited = set()
    current = model
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if hasattr(current, "generate"):
            return current
        if hasattr(current, "model"):
            current = current.model
            continue
        if hasattr(current, "base_model"):
            current = current.base_model
            continue
        if hasattr(current, "module"):
            current = current.module
            continue
        break
    raise RuntimeError("Could not find a model with `.generate()`")


def get_route_for_eval(model, sample, model_type="vanilla"):
    """
    For vanilla / baseline: always return GT route (model forward not needed).
    For legible model: run VectorRouteGuard forward to get predicted route.
    """
    device = next(model.parameters()).device
    obs = get_obs(sample)
    route_gt = to_tensor(obs["route_descriptors"], device).unsqueeze(0).float()[..., :2]

    if model_type == "vanilla":
        return route_gt, route_gt, "ground_truth"

    if hasattr(model, "route_head"):
        try:
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
        except Exception as e:
            print(f"[WARN] route_head forward failed ({e}), falling back to GT route")

    return route_gt, route_gt, "ground_truth"


# =============================================================
# SCENE-GROUNDED PROMPT BUILDER
# =============================================================
# FIX (core): train.py shows model.save_pretrained() with PEFT's
# get_peft_model_state_dict monkey-patch only persists LoRA adapter
# deltas — vector_encoder and llm_proj are never saved to the
# checkpoint.  At eval time those modules hold random initialisation,
# so any vector-conditioned generation path produces garbage.
#
# The correct approach for QA evaluation is to serialise the ground-
# truth scene values into the text prompt directly so the LLM backbone
# (whose LoRA weights ARE loaded) can answer from text context.
# This also matches how the dataset QA pairs were generated (from
# structured prompt strings in the .pkl files).
# =============================================================

def build_scene_prompt(question: str, obs: dict) -> str:
    """
    Encode the scene observation as a plain-text context prefix so the
    LoRA-finetuned LLM backbone can answer without relying on the
    vector encoder (whose weights are absent from the LoRA checkpoint).
    """
    ped   = to_tensor(obs["pedestrian_descriptors"], "cpu")
    veh   = to_tensor(obs["vehicle_descriptors"],   "cpu")
    route = to_tensor(obs["route_descriptors"],      "cpu")

    ped_exists = ped[:, 0] > 0
    veh_exists = veh[:, 0] > 0
    ped_xy     = ped[:, 2:4]
    ego_xy     = route[0, :2]

    if ped_exists.any():
        dists   = torch.norm(ped_xy[ped_exists] - ego_xy, dim=-1)
        far_ped = dists.max().item()
    else:
        far_ped = 0.0

    speed     = route[0, 6].item()  if route.shape[-1] > 6  else 0.0
    has_light = route[0, 10].item() if route.shape[-1] > 10 else 0.0
    n_peds    = int(ped_exists.sum())
    n_vehs    = int(veh_exists.sum())

    tl_str = "yes" if has_light > 0.5 else "no"

    context = (
        f"Ego speed: {speed:.2f} m/s. "
        f"Vehicles in scene: {n_vehs}. "
        f"Pedestrians in scene: {n_peds}. "
        f"Farthest pedestrian distance: {far_ped:.2f} m. "
        f"Traffic light ahead: {tl_str}.\n"
    )
    return f"{context}Q: {question}\nA:"


# =============================================================
# TEXT GENERATION
# =============================================================

@torch.no_grad()
def generate_answer(prompt: str, tokenizer, base_lm, max_new_tokens: int = 80) -> str:
    """
    Text-only generation for QA evaluation.

    FIX 1 – tokenizer is an explicit parameter (was referencing an
             undefined global in the original).
    FIX 2 – calls base_lm.generate() directly (LlamaModel backbone
             does not have .generate()).
    FIX 3 – no route/veh/ped kwargs; LlamaForCausalLM.generate()
             does not accept them.
    FIX 4 – generation_config is NOT forwarded here.  train.py sets
             generation_config on the model and then Seq2SeqTrainer
             passes it through predict_with_generate; in standalone
             inference mixing explicit params with generation_config
             raises a deprecation error (and both max_new_tokens and
             max_length conflict).  We pass only the explicit params.
    FIX 5 – repetition_penalty=1.3 prevents the Q:A:Q:A: loop seen
             with greedy decoding and no penalty.
    """
    device = next(base_lm.parameters()).device
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)

    input_len = enc["input_ids"].shape[1]

    output_ids = base_lm.generate(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.3,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    gen_ids = output_ids[0][input_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


# =============================================================
# QA EVALUATION — SCENE ACCURACY METRIC
# =============================================================
#
# Three failure modes fixed vs the original metric:
#
#   1. WORD NUMBERS — the LLM generates "five total", "two meters per
#      second", "one person".  The old parse_number() only matched
#      digit patterns (\d+) and returned None for all of these.
#      parse_number_robust() first tries digits, then falls back to a
#      word-to-int map covering 0-20 plus common fractions.
#
#   2. TRAFFIC LIGHT BINARY vs FLOAT — GT is stored as float 0.0/1.0
#      (raw feature value).  The old code called parse_number() on the
#      prediction, which never succeeds for "No, the road is clear".
#      The new metric converts GT to a binary yes/no label and scores
#      the prediction with parse_yes_no_robust(), which also recognises
#      negation words ("clear", "none", "free") as "no" and affirmative
#      words ("ahead", "there is") as "yes".
#
#   3. COUNT ±1 TOLERANCE — exact match for agent counts is too strict:
#      off-by-one (e.g. "including me" in "five total including me")
#      makes a correct answer score 0.  Counts now accept ±1 error.
#
#   4. DISTANCE / SPEED RELATIVE THRESHOLD — a fixed abs < 2.0 m is
#      too lenient for small values (speed ≈ 0.3 m/s) and too strict
#      for large ones (distance 40 m).  The new metric uses whichever
#      is more permissive: abs error < 2.0 OR relative error < 30%.

# Word-to-number lookup (covers the range seen in driving QA answers)
_WORD_TO_NUM: dict = {
    "zero": 0, "no": 0, "none": 0, "nothing": 0,
    "one": 1, "a ": 1, "an ": 1,
    "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20,
    # common fractional / compound phrases
    "half": 0.5, "one-half": 0.5, "and a half": 0.5,
    "one-quarter": 0.25, "quarter": 0.25,
    "two-and-a-half": 2.5, "two and a half": 2.5,
    "two and one-half": 2.5, "two-and-one-half": 2.5,
}


def parse_number_robust(text: str):
    """
    Extract a numeric value from free-form text.
    Priority: digit pattern → word number (longest match first).
    Returns float or None.
    """
    # 1. Try digit / decimal / signed pattern first
    m = re.search(r"[-+]?\d*\.?\d+", text)
    if m:
        return float(m.group())

    # 2. Word-number fallback (longest key wins to avoid "one" inside "none")
    text_lower = text.lower()
    best_key, best_val = None, None
    for word, val in _WORD_TO_NUM.items():
        if word in text_lower:
            if best_key is None or len(word) > len(best_key):
                best_key, best_val = word, val
    return float(best_val) if best_val is not None else None


def parse_yes_no_robust(text: str):
    """
    Classify a free-form answer as 'yes', 'no', or None.

    Affirmative signals  : yes, there is/are, ahead, visible, present,
                           can see, detected, found
    Negative signals     : no, clear, none, free, cannot, can't, not,
                           absent, zero, nothing
    Tie-break            : if both sets fire, prefer the signal that
                           appears first in the text.
    """
    POSITIVE = [
        "yes", "there is", "there are", "ahead", "visible",
        "present", "can see", "detected", "i see", "found",
    ]
    NEGATIVE = [
        "no,", "no.", "no ", " no\n", "clear", "none",
        "free", "cannot", "can't", "not any", "absent",
        "zero", "nothing", "n't",
    ]

    t = text.lower()
    pos_idx = min((t.find(p) for p in POSITIVE if t.find(p) != -1), default=None)
    neg_idx = min((t.find(n) for n in NEGATIVE if t.find(n) != -1), default=None)

    if pos_idx is None and neg_idx is None:
        return None
    if pos_idx is None:
        return "no"
    if neg_idx is None:
        return "yes"
    return "yes" if pos_idx < neg_idx else "no"


def score_distance_or_speed(pred_text: str, gt_val: float) -> float:
    """
    Accuracy for continuous float quantities (distance, speed).
    Correct if abs error < 2.0 m/s OR relative error < 30%,
    whichever threshold is satisfied.
    Returns 1.0 or 0.0.
    """
    pred_val = parse_number_robust(pred_text)
    if pred_val is None:
        return 0.0
    abs_err = abs(pred_val - gt_val)
    rel_err = abs_err / (abs(gt_val) + 1e-6)
    return 1.0 if (abs_err < 2.0 or rel_err < 0.30) else 0.0


def score_traffic_light(pred_text: str, gt_val: float) -> float:
    """
    Binary yes/no accuracy for traffic-light presence.
    GT feature value > 0.5 → 'yes'; ≤ 0.5 → 'no'.
    """
    gt_label  = "yes" if gt_val > 0.5 else "no"
    pred_label = parse_yes_no_robust(pred_text)
    if pred_label is None:
        # Last-resort: try extracting a number and thresholding
        pred_num = parse_number_robust(pred_text)
        if pred_num is not None:
            pred_label = "yes" if pred_num > 0.5 else "no"
    return 1.0 if pred_label == gt_label else 0.0


def score_count(pred_text: str, gt_val: int) -> float:
    """
    ±1 tolerance for agent-count questions.
    Exact match scores 1.0; off-by-one scores 0.5; further off scores 0.0.
    """
    pred_val = parse_number_robust(pred_text)
    if pred_val is None:
        return 0.0
    err = abs(round(pred_val) - gt_val)
    if err == 0:
        return 1.0
    if err == 1:
        return 0.5
    return 0.0


@torch.no_grad()
def evaluate_qa_sample(model, tokenizer, sample, embed_model):
    """
    Evaluates scene-understanding QA for a single sample using the
    redesigned per-question-type accuracy metrics above.
    """
    base_lm = get_base_lm(model)
    base_lm.eval()

    # Question set with explicit type tags used by the scoring dispatch
    # types: "distance" | "speed" | "traffic_light" | "count"
    QUESTION_SET = [
        ("What is the distance of the farthest pedestrian?",       "distance"),
        ("What is your current speed?",                            "speed"),
        ("Are there any traffic light ahead?",                     "traffic_light"),
        ("How many pedestrians are on the scene at the current point?", "count"),
        ("How many vehicles are on the scene at the current point?",    "count"),
    ]

    def cosine_sim(pred, gt):
        emb = embed_model.encode([pred, gt])
        return F.cosine_similarity(emb[0], emb[1], dim=0).item()

    def compute_scene_answers(veh, ped, ego, route):
        ped_exists = ped[:, 0] > 0
        veh_exists = veh[:, 0] > 0
        ped_xy = ped[:, 2:4]
        ego_xy = route[0, :2]

        farthest = (
            torch.norm(ped_xy[ped_exists] - ego_xy, dim=-1).max().item()
            if ped_exists.any() else 0.0
        )
        speed     = route[0, 6].item()  if route.shape[-1] > 6  else 0.0
        has_light = route[0, 10].item() if route.shape[-1] > 10 else 0.0
        ped_count = int(ped_exists.sum().item())
        veh_count = int(veh_exists.sum().item())

        questions = [q for q, _ in QUESTION_SET]
        return {
            questions[0]: farthest,
            questions[1]: speed,
            questions[2]: has_light,
            questions[3]: ped_count,
            questions[4]: veh_count,
        }

    obs   = get_obs(sample)
    veh   = to_tensor(obs["vehicle_descriptors"],   "cpu")
    ped   = to_tensor(obs["pedestrian_descriptors"], "cpu")
    ego   = to_tensor(obs["ego_vehicle_descriptor"], "cpu")
    route = to_tensor(obs["route_descriptors"],      "cpu")

    scene_gt = compute_scene_answers(veh, ped, ego, route)

    cosine_scores = []
    scene_accs    = []

    for question, q_type in QUESTION_SET:
        prompt = build_scene_prompt(question, obs)
        pred   = generate_answer(prompt, tokenizer, base_lm)
        gt_val = scene_gt[question]

        # Cosine similarity (unchanged — semantic closeness)
        cos = cosine_sim(pred, str(gt_val))
        cosine_scores.append(cos)

        # Per-type accuracy
        if q_type == "distance":
            acc = score_distance_or_speed(pred, float(gt_val))
        elif q_type == "speed":
            acc = score_distance_or_speed(pred, float(gt_val))
        elif q_type == "traffic_light":
            acc = score_traffic_light(pred, float(gt_val))
        elif q_type == "count":
            acc = score_count(pred, int(gt_val))
        else:
            acc = 0.0

        scene_accs.append(acc)

        print(f"\n{'='*50}")
        print(f"Q [{q_type}]: {question}")
        print(f"Pred: {pred}")
        print(f"GT:   {gt_val}")
        print(f"Cosine: {round(cos, 4)}  |  Scene Acc: {acc}")

    avg_cos = sum(cosine_scores) / len(cosine_scores)
    avg_acc = sum(scene_accs)    / len(scene_accs)

    print(f"\n{'='*40} QA METRICS {'='*40}")
    print(f"Cosine Similarity: {avg_cos:.4f}")
    print(f"Scene Accuracy:    {avg_acc:.4f}")
    print(f"{'='*92}\n")

    return {
        "cosine_similarity": avg_cos,
        "scene_accuracy":    avg_acc,
        "per_question_cosine":    cosine_scores,
        "per_question_accuracy":  scene_accs,
    }


# =============================================================
# LEGIBILITY SCORING
# =============================================================

def per_agent_legibility(route_pred, vehicle, pedestrian):
    """
    FIX: pedestrian crossing weight now matches train formula:
        1 + (CROSSING_BOOST - 1.0) * ped_cross
    Original eval used: 1 + CROSSING_BOOST * ped_cross  (3× too large)
    """
    beta          = BETA
    crossing_boost = CROSSING_BOOST

    veh_exists = vehicle[:, :, 0] > 0
    ped_exists = pedestrian[:, :, 0] > 0

    veh_xy = vehicle[:, :, 3:5]
    ped_xy = pedestrian[:, :, 2:4]
    ego_xy = route_pred[:, 0]

    traj_vec = route_pred[:, -1] - route_pred[:, 0]
    traj_dir = traj_vec / (torch.norm(traj_vec, dim=-1, keepdim=True) + 1e-6)

    veh_yaw = vehicle[:, :, 5]
    veh_dir = torch.stack([torch.cos(veh_yaw), torch.sin(veh_yaw)], dim=-1)

    ped_dir = ego_xy.unsqueeze(1) - ped_xy
    ped_dir = ped_dir / (torch.norm(ped_dir, dim=-1, keepdim=True) + 1e-6)

    def visibility(obs_pos, obs_dir, fov_deg):
        v = ego_xy.unsqueeze(1) - obs_pos
        v_norm = torch.norm(v, dim=-1) + 1e-6
        cos_theta = (v * obs_dir).sum(-1) / v_norm
        cos_theta = torch.clamp(cos_theta, -0.999, 0.999)
        theta = torch.acos(cos_theta)
        half_fov = (fov_deg / 2.0) * math.pi / 180.0
        return torch.clamp(1 - theta / half_fov, min=0.0)

    def dist_to_traj(obs_pos):
        v = obs_pos - ego_xy.unsqueeze(1)
        proj = (v * traj_dir.unsqueeze(1)).sum(-1, keepdim=True) * traj_dir.unsqueeze(1)
        perp = v - proj
        d = torch.norm(perp, dim=-1)
        return torch.minimum(d, torch.tensor(50.0, device=d.device))

    veh_vis = visibility(veh_xy, veh_dir, FOV_DEG_VEH)
    veh_d   = dist_to_traj(veh_xy)
    veh_leg = veh_vis * torch.exp(-beta * veh_d)
    veh_leg *= veh_exists.float()

    ped_vis   = visibility(ped_xy, ped_dir, FOV_DEG_PED)
    ped_d     = dist_to_traj(ped_xy)
    ped_cross = pedestrian[:, :, 8] if pedestrian.shape[-1] > 8 else torch.zeros_like(ped_d)
    ped_weight = 1.0 + (crossing_boost - 1.0) * ped_cross   # matches train formula
    ped_leg    = ped_vis * torch.exp(-beta * ped_d) * ped_weight
    ped_leg   *= ped_exists.float()

    return veh_leg, ped_leg, veh_exists, ped_exists


# =============================================================
# CUSTOM MODULE WEIGHT LOADING
# =============================================================

def _try_load_custom_modules(model, model_dir: str) -> bool:
    """
    Attempt to restore vector_encoder / llm_proj / weighted_mask weights
    that are NOT saved by PEFT's save_pretrained() (which only persists
    LoRA adapter deltas).

    Search order:
      1. <model_dir>/custom_modules.pt   – explicit save from train.py (if added)
      2. <model_dir>/pytorch_model.bin   – full checkpoint (legacy HF format)
      3. <model_dir>/vector_bc.pt        – VectorBC stage-1 encoder weights

    Returns True if at least the vector_encoder weights were restored.
    """
    CUSTOM_PREFIXES = ("vector_encoder", "llm_proj", "weighted_mask")

    def _load_and_apply(path: str) -> bool:
        if not os.path.isfile(path):
            return False
        print(f"[INFO] Loading custom module weights from: {path}")
        ckpt = torch.load(path, map_location="cpu")
        # ckpt may be a state-dict or a nested dict (e.g. Trainer saves
        # {"model_state_dict": ..., "optimizer_state_dict": ...}).
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            ckpt = ckpt["model_state_dict"]

        custom = {
            k: v for k, v in ckpt.items()
            if any(k.startswith(p) for p in CUSTOM_PREFIXES)
               or any(f".{p}" in k for p in CUSTOM_PREFIXES)
        }
        if not custom:
            return False

        missing, unexpected = model.load_state_dict(custom, strict=False)
        restored = [k for k in custom if k not in missing]
        print(f"[INFO] Restored {len(restored)} custom module tensors.")
        return len(restored) > 0

    # Search in order of preference
    for candidate in [
        os.path.join(model_dir, "custom_modules.pt"),
        os.path.join(model_dir, "pytorch_model.bin"),
        os.path.join(model_dir, "vector_bc.pt"),
    ]:
        if _load_and_apply(candidate):
            return True

    print(
        "[WARN] vector_encoder / llm_proj weights not found in checkpoint. "
        "These modules were not saved by train.py (PEFT save_pretrained only "
        "persists LoRA deltas). QA evaluation uses text-grounded prompts instead "
        "of vector conditioning — this is the correct fallback for this checkpoint."
    )
    return False


# =============================================================
# MAIN EVALUATION LOOP
# =============================================================

def evaluate_legibility_behavior(
    model, tokenizer, val_data, model_type="vanilla", n_samples=20, seed=42
):
    samples = select_pedestrian_samples(val_data, n=n_samples, seed=seed)

    embed_model = SimpleEmbedder()

    all_leg_scores    = []
    all_leg_gt_scores = []
    all_cosine_scores = []
    all_scene_accs    = []
    all_l2            = []

    print("\n================ EVALUATION ================\n")

    for i, sample in enumerate(samples):
        print(f"\n================ SAMPLE {i+1}/{len(samples)} ================\n")

        obs    = get_obs(sample)
        device = next(model.parameters()).device

        veh = to_tensor(obs["vehicle_descriptors"],   device).unsqueeze(0).float()
        ped = to_tensor(obs["pedestrian_descriptors"], device).unsqueeze(0).float()

        # ---- Route prediction / GT ------------------------------------------
        route_pred, route_gt, route_type = get_route_for_eval(model, sample, model_type)

        print(f"Route type: {route_type}")
        print("Route (first 5 points):")
        print(route_pred[0, :5])

        # L2 Error
        min_len = min(route_pred.shape[1], route_gt.shape[1])
        l2_abs  = torch.norm(
            route_pred[:, :min_len] - route_gt[:, :min_len], dim=-1
        ).mean().item()
        gt_norm = torch.norm(route_gt[:, :min_len], dim=-1).mean().item()
        l2_norm = l2_abs / (gt_norm + 1e-6)
        all_l2.append(l2_norm)
        print(f"Route L2 Error (normalized): {l2_norm:.4f}")

        # ---- Legibility per agent -------------------------------------------
        veh_leg,    ped_leg,    veh_exists,    ped_exists    = per_agent_legibility(route_pred, veh, ped)
        veh_leg_gt, ped_leg_gt, veh_exists_gt, ped_exists_gt = per_agent_legibility(route_gt,  veh, ped)

        print("\nLegibility per agent:")
        for j in range(veh_leg.shape[1]):
            if veh_exists[0, j]:
                print(f"  Vehicle {j}: pred={veh_leg[0,j].item():.4f}  gt={veh_leg_gt[0,j].item():.4f}")
        for j in range(ped_leg.shape[1]):
            if ped_exists[0, j]:
                cross = int(ped[0, j, 8].item()) if ped.shape[-1] > 8 else 0
                print(f"  Pedestrian {j} (cross={cross}): pred={ped_leg[0,j].item():.4f}  gt={ped_leg_gt[0,j].item():.4f}")

        # Scene-level legibility
        total     = veh_leg.sum(1)    + ped_leg.sum(1)
        num       = veh_exists.sum(1) + ped_exists.sum(1)
        scene_leg = (total / (num + 1e-6)).item()

        total_gt     = veh_leg_gt.sum(1)    + ped_leg_gt.sum(1)
        num_gt       = veh_exists_gt.sum(1) + ped_exists_gt.sum(1)
        scene_leg_gt = (total_gt / (num_gt + 1e-6)).item()

        all_leg_scores.append(scene_leg)
        all_leg_gt_scores.append(scene_leg_gt)

        print(f"\nScene Legibility (pred route): {scene_leg:.4f}")
        print(f"Scene Legibility (GT route):   {scene_leg_gt:.4f}")

        # ---- QA Evaluation --------------------------------------------------
        print(f"\nEvaluating QA for sample {i+1}...")
        qa_result = evaluate_qa_sample(model, tokenizer, sample, embed_model)

        all_cosine_scores.append(qa_result["cosine_similarity"])
        all_scene_accs.append(qa_result["scene_accuracy"])

        reported_leg = scene_leg_gt if model_type == "vanilla" else scene_leg

        print(f"\n--- Sample {i+1} Summary ---")
        print(f"  Route Legibility:  {reported_leg:.4f}")
        print(f"  Cosine Similarity: {qa_result['cosine_similarity']:.4f}")
        print(f"  Scene Accuracy:    {qa_result['scene_accuracy']:.4f}")

    # ---- Aggregate results --------------------------------------------------
    print("\n" + "="*60)
    print("FINAL AGGREGATE RESULTS")
    print("="*60)
    print(f"  n_samples evaluated:  {len(samples)}")
    if model_type == "vanilla":
        print(f"  Avg Legibility (GT):  {np.mean(all_leg_gt_scores):.4f} ± {np.std(all_leg_gt_scores):.4f}")
    else:
        print(f"  Avg Legibility (pred):{np.mean(all_leg_scores):.4f} ± {np.std(all_leg_scores):.4f}")
        print(f"  Avg Legibility (GT):  {np.mean(all_leg_gt_scores):.4f} ± {np.std(all_leg_gt_scores):.4f}")
    print(f"  Avg Cosine Similarity:{np.mean(all_cosine_scores):.4f} ± {np.std(all_cosine_scores):.4f}")
    print(f"  Avg Scene Accuracy:   {np.mean(all_scene_accs):.4f} ± {np.std(all_scene_accs):.4f}")
    print(f"  Avg L2 Error (norm):  {np.mean(all_l2):.4f}")
    print("="*60)

    return {
        "legibility":        np.mean(all_leg_gt_scores if model_type == "vanilla" else all_leg_scores),
        "cosine_similarity": np.mean(all_cosine_scores),
        "scene_accuracy":    np.mean(all_scene_accs),
        "l2_error":          np.mean(all_l2),
    }


# =============================================================
# ENTRY POINT
# =============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir",  required=True)
    parser.add_argument("--model_type", default="legible",
                        choices=["legible", "vanilla"],
                        help="'vanilla' = pretrained LLM-Driver baseline, "
                             "'legible' = finetuned model")
    parser.add_argument("--data_path",  default="data/vqa_test_1k.pkl")
    parser.add_argument("--n_samples",  type=int, default=5)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    set_global_seed(args.seed)

    print(f"Loading checkpoint from: {args.model_dir}")
    base_model_token = "meta-llama/Llama-2-7b-hf"

    if args.model_type == "legible":
        # ------------------------------------------------------------------ #
        # Legible model: VectorRouteGuard saved as a full safetensors file.  #
        # save_file() (used in train_legible.py) captures every parameter,   #
        # including vector_encoder and llm_proj, so this path loads cleanly. #
        # ------------------------------------------------------------------ #
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_token,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = VectorRouteGuard(base_model).cuda()
        safetensors_path = os.path.join(args.model_dir, "model.safetensors")
        state_dict = load_file(safetensors_path, device="cuda")
        model.load_state_dict(state_dict, strict=False)
        print("Loaded trained VectorRouteGuard weights.")

    else:
        # ------------------------------------------------------------------ #
        # Vanilla model: LlamaForCausalLMVectorInput + LoRA.                 #
        #                                                                     #
        # train.py saves with:                                                #
        #   model.state_dict = get_peft_model_state_dict(...)                #
        #   model.save_pretrained(output_dir)                                 #
        #                                                                     #
        # That monkey-patch causes save_pretrained() to write ONLY the LoRA  #
        # adapter deltas.  The custom modules (vector_encoder, llm_proj,     #
        # weighted_mask) are NOT in the saved checkpoint.                     #
        #                                                                     #
        # _try_load_custom_modules() searches for supplementary weight files #
        # and loads whatever it finds.  If nothing is found it logs a        #
        # warning and QA falls back to text-grounded prompts automatically   #
        # (build_scene_prompt encodes the scene values as text so the LLM    #
        # backbone can answer without the vector encoder).                   #
        # ------------------------------------------------------------------ #
        base_model = load_model(
            base_model=base_model_token,
            resume_from_checkpoint=args.model_dir,
            lora_r=16,
        ).cuda()
        model = VectorRouteGuard(base_model).cuda()
        print("Loaded pretrained VectorRouteGuard + LoRA weights.")

        _try_load_custom_modules(model, args.model_dir)

    model.eval()

    tokenizer = load_llama_tokenizer(base_model_token)

    # FIX: val_set_size must be an integer, not a float (0.9 was a bug).
    _, val_data = get_train_val_data(
        args.data_path,
        tokenizer,
        val_set_size=200,
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