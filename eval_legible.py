# =============================================================
# eval_legible.py
# Qualitative evaluation of legibility-aware vs baseline behavior
# =============================================================

import argparse
import os
import torch
import numpy as np

from transformers import AutoModelForCausalLM
from safetensors.torch import load_file

from utils.model_utils import load_llama_tokenizer
from utils.training_utils import get_train_val_data
from train_legible import (
    compute_dragan_legibility_loss,
    VectorRouteGuard,
    patch_vector_fields,
)

# =============================================================
# UTILITIES
# =============================================================

def to_tensor(x, device):
    """Convert list → tensor if needed, and move to device."""
    if isinstance(x, list):
        x = torch.tensor(x)
    return x.to(device)


def get_obs(sample):
    """Handle both nested and flat dataset formats."""
    return sample["observation"] if "observation" in sample else sample


def get_text_prompt(sample, tokenizer=None, device=None):
    """
    Robust way to recover the text prompt.
    Priority:
    1) sample["input"]
    2) decode from input_ids
    """
    if "input" in sample:
        return sample["input"]

    if "input_ids" in sample and tokenizer is not None:
        return tokenizer.decode(sample["input_ids"], skip_special_tokens=True)

    raise KeyError(
        f"No usable text prompt found. Keys available: {sample.keys()}"
    )


def select_pedestrian_samples(dataset, n=10):
    """Select samples that actually contain pedestrians."""
    selected = []

    for item in dataset:
        obs = get_obs(item)
        ped = obs["pedestrian_descriptors"]

        if isinstance(ped, list):
            ped = torch.tensor(ped)

        if (ped[:, 0] > 0).sum().item() > 0:
            selected.append(item)

        if len(selected) >= n:
            break

    return selected


def make_inputs_for_model(model, tokenizer, sample, forced_route=None):
    """Format one sample exactly as in training."""
    device = next(model.parameters()).device
    obs = get_obs(sample)

    text = get_text_prompt(sample, tokenizer, device)
    enc = tokenizer(text, return_tensors="pt").to(device)

    routes = to_tensor(obs["route_descriptors"], device).unsqueeze(0).float()

    if forced_route is not None:
        routes = forced_route

    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "route_descriptors": routes,
        "vehicle_descriptors": to_tensor(obs["vehicle_descriptors"], device).unsqueeze(0).float(),
        "pedestrian_descriptors": to_tensor(obs["pedestrian_descriptors"], device).unsqueeze(0).float(),
        "ego_vehicle_descriptor": to_tensor(obs["ego_vehicle_descriptor"], device).unsqueeze(0).float(),
    }


# =============================================================
# CUSTOM GENERATION (KEY FIX)
# =============================================================

@torch.no_grad()
def greedy_generate_with_vectors(model, tokenizer, model_inputs, max_new_tokens=120):
    """
    Manual greedy decoding loop that CALLS VectorRouteGuard directly.
    This is REQUIRED because base LLaMA cannot accept vector inputs in .generate().
    """

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

        logits = outputs.logits  # [1, seq_len, vocab]
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

        input_ids = torch.cat([input_ids, next_token], dim=1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones_like(next_token)], dim=1
        )

        if next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


# =============================================================
# BASELINE & LEGIBILITY FUNCTIONS
# =============================================================

def run_model_on_sample(model, tokenizer, sample, question):
    """Baseline generation (no legibility)."""
    device = next(model.parameters()).device
    obs = get_obs(sample)

    prompt = get_text_prompt(sample, tokenizer, device) + "\n\nQ: " + question + "\nA:"
    enc = tokenizer(prompt, return_tensors="pt").to(device)

    model_inputs = {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "route_descriptors": to_tensor(obs["route_descriptors"], device).unsqueeze(0),
        "vehicle_descriptors": to_tensor(obs["vehicle_descriptors"], device).unsqueeze(0),
        "pedestrian_descriptors": to_tensor(obs["pedestrian_descriptors"], device).unsqueeze(0),
        "ego_vehicle_descriptor": to_tensor(obs["ego_vehicle_descriptor"], device).unsqueeze(0),
    }

    return greedy_generate_with_vectors(model, tokenizer, model_inputs)


def pick_most_legible_route(model, tokenizer, sample):
    """Re-rank K routes by Dragan-style legibility score."""
    device = next(model.parameters()).device
    obs = get_obs(sample)

    routes = to_tensor(obs["route_descriptors"], device).unsqueeze(0)
    K = routes.shape[1]
    scores = []

    for g in range(K):
        goal_route = routes.clone()
        goal_route.zero_()
        goal_route[:, 0, :] = routes[:, g, :]

        text = get_text_prompt(sample, tokenizer, device)
        enc = tokenizer(text, return_tensors="pt").to(device)

        leg_inputs = {
            "labels": torch.zeros(1, 1, device=device, dtype=torch.long),  # <-- KEY FIX
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "route_descriptors": goal_route,
            "vehicle_descriptors": to_tensor(obs["vehicle_descriptors"], device).unsqueeze(0),
            "pedestrian_descriptors": to_tensor(obs["pedestrian_descriptors"], device).unsqueeze(0),
            "ego_vehicle_descriptor": to_tensor(obs["ego_vehicle_descriptor"], device).unsqueeze(0),
        }
        with torch.no_grad():
            leg = compute_dragan_legibility_loss(model, leg_inputs)

        # leg = compute_dragan_legibility_loss(model, leg_inputs)
        scores.append(float(leg.item()))

    best_g = int(np.argmax(scores))
    return best_g, scores


def run_legible_generation(model, tokenizer, sample, best_g, question):
    """Generate answer using the most legible route."""
    device = next(model.parameters()).device
    obs = get_obs(sample)

    routes = to_tensor(obs["route_descriptors"], device).unsqueeze(0)
    leg_route = routes.clone()
    leg_route.zero_()
    leg_route[:, 0, :] = routes[:, best_g, :]

    prompt = get_text_prompt(sample, tokenizer, device) + "\n\nQ: " + question + "\nA:"
    enc = tokenizer(prompt, return_tensors="pt").to(device)

    model_inputs = {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "route_descriptors": leg_route,
        "vehicle_descriptors": to_tensor(obs["vehicle_descriptors"], device).unsqueeze(0),
        "pedestrian_descriptors": to_tensor(obs["pedestrian_descriptors"], device).unsqueeze(0),
        "ego_vehicle_descriptor": to_tensor(obs["ego_vehicle_descriptor"], device).unsqueeze(0),
    }

    return greedy_generate_with_vectors(model, tokenizer, model_inputs)


# =============================================================
# MAIN EVALUATION LOOP
# =============================================================

def evaluate_legibility_behavior(model, tokenizer, val_data, n_samples=5):
    question = (
        "If a pedestrian suddenly starts crossing the road in front of you, "
        "how will you drive and why?"
    )

    samples = select_pedestrian_samples(val_data, n=n_samples)

    print("\n================ LEGIBILITY EVALUATION ================\n")

    for i, sample in enumerate(samples):
        print(f"\n🔹 SAMPLE {i+1}")
        print("--------------------------------------------------")

        base_answer = run_model_on_sample(model, tokenizer, sample, question)

        best_g, scores = pick_most_legible_route(model, tokenizer, sample)
        leg_answer = run_legible_generation(model, tokenizer, sample, best_g, question)
        
        print("\n🟡 BASELINE ANSWER (No Legibility):")
        print(base_answer)

        print("\n🟢 LEGIBILITY-AWARE ANSWER:")
        print(leg_answer)

        print("\n🔹 Top-5 route legibility scores:", scores[:5])
        print("Chosen legible route index:", best_g)
        print("\n=====================================================\n")


# =============================================================
# CLI
# =============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        required=True,
        help="Path to trained checkpoint directory (contains model.safetensors)",
    )
    parser.add_argument("--data_path", default="data/vqa_test_1k.pkl")
    parser.add_argument("--n_samples", type=int, default=5)
    args = parser.parse_args()

    ckpt_dir = args.model_dir
    print(f"🔹 Loading checkpoint from: {ckpt_dir}")

    base_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        torch_dtype=torch.float16,
        device_map="auto",
    )

    model = VectorRouteGuard(base_model).cuda()

    safetensors_path = os.path.join(ckpt_dir, "model.safetensors")
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
    )
