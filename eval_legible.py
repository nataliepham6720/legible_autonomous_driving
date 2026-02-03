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

def get_obs(sample):
    """Handle both nested and flat dataset formats."""
    return sample["observation"] if "observation" in sample else sample


def get_text_prompt(sample, tokenizer=None, device=None):
    """
    Robust way to recover the text prompt.
    Priority:
    1) sample["input"]
    2) decode from input_ids if needed
    """
    if "input" in sample:
        return sample["input"]

    if "input_ids" in sample and tokenizer is not None:
        return tokenizer.decode(sample["input_ids"], skip_special_tokens=True)

    raise KeyError(
        f"No usable text prompt found. Keys available: {sample.keys()}"
    )


def get_llm(model: VectorRouteGuard):
    """
    Return the underlying causal LM used for generation.
    Works regardless of how you named the attribute.
    """
    for attr in ["base_model", "model", "llm", "backbone"]:
        if hasattr(model, attr):
            return getattr(model, attr)
    raise AttributeError(
        "Cannot find underlying LLM inside VectorRouteGuard. "
        "Expected one of: base_model, model, llm, backbone."
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


def make_inputs_for_legibility(model, tokenizer, sample, forced_route=None):
    """Format one sample exactly as in training."""
    device = next(model.parameters()).device
    obs = get_obs(sample)

    text = get_text_prompt(sample, tokenizer, device)
    enc = tokenizer(text, return_tensors="pt").to(device)

    routes = obs["route_descriptors"].unsqueeze(0).to(device).float()
    if forced_route is not None:
        routes = forced_route

    return {
        "labels": torch.zeros(1, 1, device=device),
        "route_descriptors": routes,
        "vehicle_descriptors": obs["vehicle_descriptors"].unsqueeze(0).to(device).float(),
        "pedestrian_descriptors": obs["pedestrian_descriptors"].unsqueeze(0).to(device).float(),
        "ego_vehicle_descriptor": obs["ego_vehicle_descriptor"].unsqueeze(0).to(device).float(),
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
    }


def run_model_on_sample(model, tokenizer, sample, question):
    """
    Baseline generation (no legibility).
    Calls .generate() on the underlying LLM, NOT the wrapper.
    """
    obs = get_obs(sample)
    llm = get_llm(model)

    prompt = get_text_prompt(sample, tokenizer, model.device) + "\n\nQ: " + question + "\nA:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = llm.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        route_descriptors=obs["route_descriptors"].unsqueeze(0).to(model.device),
        vehicle_descriptors=obs["vehicle_descriptors"].unsqueeze(0).to(model.device),
        pedestrian_descriptors=obs["pedestrian_descriptors"].unsqueeze(0).to(model.device),
        ego_vehicle_descriptor=obs["ego_vehicle_descriptor"].unsqueeze(0).to(model.device),
        max_new_tokens=120,
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def pick_most_legible_route(model, tokenizer, sample):
    """Re-rank K routes by Dragan-style legibility score."""
    obs = get_obs(sample)
    routes = obs["route_descriptors"].unsqueeze(0).to(model.device)

    K = routes.shape[1]
    scores = []

    for g in range(K):
        goal_route = routes.clone()
        goal_route.zero_()
        goal_route[:, 0, :] = routes[:, g, :]

        inputs = make_inputs_for_legibility(
            model, tokenizer, sample, forced_route=goal_route
        )

        leg = compute_dragan_legibility_loss(model, inputs)
        scores.append(float(leg.item()))

    best_g = int(np.argmax(scores))
    return best_g, scores


def run_legible_generation(model, tokenizer, sample, best_g, question):
    """Generate answer using the most legible route."""
    obs = get_obs(sample)
    llm = get_llm(model)

    routes = obs["route_descriptors"].unsqueeze(0).to(model.device)
    leg_route = routes.clone()
    leg_route.zero_()
    leg_route[:, 0, :] = routes[:, best_g, :]

    prompt = get_text_prompt(sample, tokenizer, model.device) + "\n\nQ: " + question + "\nA:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = llm.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        route_descriptors=leg_route,
        vehicle_descriptors=obs["vehicle_descriptors"].unsqueeze(0).to(model.device),
        pedestrian_descriptors=obs["pedestrian_descriptors"].unsqueeze(0).to(model.device),
        ego_vehicle_descriptor=obs["ego_vehicle_descriptor"].unsqueeze(0).to(model.device),
        max_new_tokens=120,
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


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

    # 1) Load base LLaMA backbone
    base_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # 2) Wrap with VectorRouteGuard
    model = VectorRouteGuard(base_model).cuda()

    # 3) Load trained weights from safetensors
    safetensors_path = os.path.join(ckpt_dir, "model.safetensors")
    assert os.path.exists(safetensors_path), f"Missing {safetensors_path}"

    state_dict = load_file(safetensors_path, device="cuda")
    model.load_state_dict(state_dict, strict=False)

    print("✅ Loaded trained VectorRouteGuard weights from model.safetensors.")

    # 4) Load tokenizer
    tokenizer = load_llama_tokenizer("meta-llama/Llama-2-7b-hf")

    # ------------------------------
    # LOAD DATA
    # ------------------------------
    _, val_data = get_train_val_data(
        args.data_path,
        tokenizer,
        val_set_size=0.9,  # fraction avoids split error
    )

    val_data = patch_vector_fields(val_data)

    # ------------------------------
    # RUN EVALUATION
    # ------------------------------
    evaluate_legibility_behavior(
        model,
        tokenizer,
        val_data,
        n_samples=args.n_samples,
    )
