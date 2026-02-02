# eval_legibility.py
# -------------------------------------------------------------
# Qualitative evaluation of legibility-aware vs baseline behavior
# -------------------------------------------------------------

import argparse
import torch
import numpy as np
from transformers import AutoTokenizer
from utils.model_utils import load_llama_tokenizer, load_model
from utils.training_utils import get_train_val_data
from train_legible import compute_dragan_legibility_loss, VectorRouteGuard, patch_vector_fields


# ============================================================
# UTILITIES
# ============================================================

def select_pedestrian_samples(dataset, n=10):
    """
    Select n samples that contain at least one real pedestrian.
    """
    selected = []
    for item in dataset:
        ped = item["observation"]["pedestrian_descriptors"]
        # existence flag is first column
        if (ped[:, 0] > 0).sum() > 0:
            selected.append(item)
        if len(selected) >= n:
            break
    return selected


def run_model_on_sample(model, tokenizer, sample, question):
    """
    Generate answer from the model using its default route.
    """
    obs = sample["observation"]

    prompt = sample["input_prompt"] + "\n\nQ: " + question + "\nA:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
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
    """
    Re-rank the K candidate routes using your Dragan-style legibility score.
    """
    obs = sample["observation"]
    routes = obs["route_descriptors"].unsqueeze(0).to(model.device)
    vehicle = obs["vehicle_descriptors"].unsqueeze(0).to(model.device)
    pedestrian = obs["pedestrian_descriptors"].unsqueeze(0).to(model.device)
    ego = obs["ego_vehicle_descriptor"].unsqueeze(0).to(model.device)

    # Dummy labels just to satisfy the loss interface
    dummy_labels = torch.zeros(1, 1, device=model.device)

    # Tokenize prompt (needed for compute_dragan_legibility_loss)
    enc = tokenizer(sample["input_prompt"], return_tensors="pt").to(model.device)

    K = routes.shape[1]
    scores = []

    for g in range(K):
        goal_route = routes.clone()
        goal_route.zero_()
        goal_route[:, 0, :] = routes[:, g, :]

        leg = compute_dragan_legibility_loss(
            model,
            {
                "labels": dummy_labels,
                "route_descriptors": goal_route,
                "vehicle_descriptors": vehicle,
                "pedestrian_descriptors": pedestrian,
                "ego_vehicle_descriptor": ego,
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
            },
        )
        scores.append(leg.item())

    best_g = int(np.argmax(scores))
    return best_g, scores


def run_legible_generation(model, tokenizer, sample, best_g, question):
    """
    Generate answer after forcing the most legible route.
    """
    obs = sample["observation"]

    # Override route to the most legible one
    routes = obs["route_descriptors"].unsqueeze(0).to(model.device)
    leg_route = routes.clone()
    leg_route.zero_()
    leg_route[:, 0, :] = routes[:, best_g, :]

    prompt = sample["input_prompt"] + "\n\nQ: " + question + "\nA:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        route_descriptors=leg_route,
        vehicle_descriptors=obs["vehicle_descriptors"].unsqueeze(0).to(model.device),
        pedestrian_descriptors=obs["pedestrian_descriptors"].unsqueeze(0).to(model.device),
        ego_vehicle_descriptor=obs["ego_vehicle_descriptor"].unsqueeze(0).to(model.device),
        max_new_tokens=120,
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# ============================================================
# MAIN EVALUATION LOOP
# ============================================================

def evaluate_legibility_behavior(
    model,
    tokenizer,
    val_data,
    n_samples=5,
):
    question = (
        "If a pedestrian suddenly starts crossing the road in front of you, "
        "how will you drive and why?"
    )

    samples = select_pedestrian_samples(val_data, n=n_samples)

    print("\n================ LEGIBILITY EVALUATION ================\n")

    for i, sample in enumerate(samples):
        print(f"\n🔹 SAMPLE {i+1}")
        print("--------------------------------------------------")

        # ---- Baseline (no legibility) ----
        base_answer = run_model_on_sample(model, tokenizer, sample, question)

        # ---- Legibility-aware ----
        best_g, scores = pick_most_legible_route(model, tokenizer, sample)
        leg_answer = run_legible_generation(model, tokenizer, sample, best_g, question)

        print("\n🟡 BASELINE ANSWER (No Legibility):")
        print(base_answer)

        print("\n🟢 LEGIBILITY-AWARE ANSWER:")
        print(leg_answer)

        print("\n🔹 Route legibility scores (first 5 shown):", scores[:5])
        print("Chosen legible route index:", best_g)
        print("\n=====================================================\n")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="Path to trained checkpoint")
    parser.add_argument("--data_path", default="data/vqa_test_1k.pkl")
    parser.add_argument("--n_samples", type=int, default=5)
    args = parser.parse_args()

    # Load base model + wrapper
    base_model = load_model(
        base_model="meta-llama/Llama-2-7b-hf",
        resume_from_checkpoint=args.model_dir,
        lora_r=16,
    ).cuda()

    model = VectorRouteGuard(base_model).cuda()
    tokenizer = load_llama_tokenizer("meta-llama/Llama-2-7b-hf")

    # Load validation data
    _, val_data = get_train_val_data(
        args.data_path,
        tokenizer,
        val_set_size=1000,
    )

    val_data = patch_vector_fields(val_data)

    # Run evaluation
    evaluate_legibility_behavior(model, tokenizer, val_data, n_samples=args.n_samples)
