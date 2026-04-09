# =============================================================
# eval.py
# Unified Evaluation Script (Batch + Modular + WandB)
# =============================================================

import argparse
import os
import math
import torch
import numpy as np
import wandb
from transformers import AutoModelForCausalLM
from safetensors.torch import load_file

from utils.model_utils import load_llama_tokenizer
from utils.training_utils import get_train_val_data
from train_legible import VectorRouteGuard, patch_vector_fields

# =============================================================
# CONFIG
# =============================================================

BETA = 0.5
FOV_DEG = 120.0
CROSSING_BOOST = 1.5

# =============================================================
# UTILS
# =============================================================

def get_obs(sample):
    return sample["observation"] if "observation" in sample else sample

def collate_batch(samples, device):
    route, veh, ped, ego = [], [], [], []
    input_ids, labels = [], []

    for s in samples:
        obs = get_obs(s)

        route.append(obs["route_descriptors"])
        veh.append(obs["vehicle_descriptors"])
        ped.append(obs["pedestrian_descriptors"])
        ego.append(obs["ego_vehicle_descriptor"])

        if "input_ids" in s:
            input_ids.append(torch.tensor(s["input_ids"]))
            labels.append(torch.tensor(s["labels"]))

    route = torch.tensor(route, device=device).float()
    veh = torch.tensor(veh, device=device).float()
    ped = torch.tensor(ped, device=device).float()
    ego = torch.tensor(ego, device=device).float()

    if input_ids:
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=0
        ).to(device)
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        ).to(device)
    else:
        input_ids, labels = None, None

    return route, veh, ped, ego, input_ids, labels

# =============================================================
# LEGIBILITY (BATCH)
# =============================================================

@torch.no_grad()
def compute_legibility_batch(route, veh, ped, ego):

    ego_xy = ego[:, :2]
    ego_yaw = ego[:, 2]

    ego_dir = torch.stack(
        [torch.cos(ego_yaw), torch.sin(ego_yaw)], dim=-1
    )

    def compute(xy, exists, is_ped=False, crossing=None):
        v = xy - ego_xy.unsqueeze(1)
        v_norm = torch.norm(v, dim=-1) + 1e-6

        cos_theta = (v * ego_dir.unsqueeze(1)).sum(-1) / v_norm
        cos_theta = torch.clamp(cos_theta, -0.999, 0.999)
        theta = torch.acos(cos_theta)

        half_fov = (FOV_DEG / 2) * math.pi / 180.0
        vis = torch.clamp(1 - theta / half_fov, min=0.0)

        proj = (v * ego_dir.unsqueeze(1)).sum(-1, keepdim=True) * ego_dir.unsqueeze(1)
        dist = torch.norm(v - proj, dim=-1)

        leg = vis * torch.exp(-BETA * torch.clamp(dist, max=50.0))

        if is_ped:
            weight = 1.0 + (CROSSING_BOOST - 1.0) * crossing
            leg = leg * weight

        return leg * exists, vis

    veh_leg, veh_vis = compute(veh[:, :, 3:5], veh[:, :, 0] > 0)

    ped_leg, ped_vis = compute(
        ped[:, :, 2:4],
        ped[:, :, 0] > 0,
        is_ped=True,
        crossing=ped[:, :, 8],
    )

    return {
        "scene": (veh_leg.sum(1) + ped_leg.sum(1)),
        "veh": veh_leg.sum(1),
        "ped": ped_leg.sum(1),
        "cross": (ped_leg * ped[:, :, 8]).sum(1),
        "visible": (veh_leg * (veh_vis > 0)).sum(1)
                   + (ped_leg * (ped_vis > 0)).sum(1),
        "non_visible": (veh_leg * (veh_vis == 0)).sum(1)
                       + (ped_leg * (ped_vis == 0)).sum(1),
    }

# =============================================================
# LANGUAGE (PPL)
# =============================================================

@torch.no_grad()
def compute_ppl_batch(model, input_ids, labels, route, veh, ped, ego):
    if input_ids is None:
        return None

    outputs = model(
        input_ids=input_ids,
        labels=labels,
        route_descriptors=route,
        vehicle_descriptors=veh,
        pedestrian_descriptors=ped,
        ego_vehicle_descriptor=ego,
    )

    return torch.exp(outputs.loss).item()

# =============================================================
# MODES
# =============================================================

def eval_legibility(model, dataset, batch_size, max_samples):
    logs = {k: [] for k in ["scene","veh","ped","cross","visible","non_visible"]}

    device = next(model.parameters()).device

    for i in range(0, min(len(dataset), max_samples), batch_size):
        batch = dataset[i:i+batch_size]
        route, veh, ped, ego, _, _ = collate_batch(batch, device)

        out = compute_legibility_batch(route, veh, ped, ego)

        for k in logs:
            logs[k].extend(out[k].cpu().numpy())

    return {k: (np.mean(v), np.std(v)) for k,v in logs.items()}

def eval_language(model, dataset, batch_size, max_samples):
    ppls = []
    device = next(model.parameters()).device

    for i in range(0, min(len(dataset), max_samples), batch_size):
        batch = dataset[i:i+batch_size]
        route, veh, ped, ego, input_ids, labels = collate_batch(batch, device)

        ppl = compute_ppl_batch(model, input_ids, labels, route, veh, ped, ego)
        if ppl is not None:
            ppls.append(ppl)

    return {"ppl": (np.mean(ppls), np.std(ppls))}

def eval_full(model, dataset, batch_size, max_samples):
    res1 = eval_legibility(model, dataset, batch_size, max_samples)
    res2 = eval_language(model, dataset, batch_size, max_samples)
    return {**res1, **res2}

# =============================================================
# MAIN
# =============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--data_path", default="data/vqa_test_1k.pkl")
    parser.add_argument("--mode", default="full",
                        choices=["full","legibility","language"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="legible_eval")

    args = parser.parse_args()

    if args.use_wandb:
        wandb.init(project=args.wandb_project, name=f"eval_{args.mode}")

    print("🔹 Loading model...")

    base_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        torch_dtype=torch.float16,
        device_map="auto",
    )

    model = VectorRouteGuard(base_model).cuda()

    state_dict = load_file(
        os.path.join(args.model_dir, "model.safetensors"),
        device="cuda"
    )
    model.load_state_dict(state_dict, strict=False)

    tokenizer = load_llama_tokenizer("meta-llama/Llama-2-7b-hf")

    _, val_data = get_train_val_data(
        args.data_path,
        tokenizer,
        val_set_size=0.9,
    )

    val_data = patch_vector_fields(val_data)

    if args.mode == "legibility":
        results = eval_legibility(model, val_data, args.batch_size, args.max_samples)

    elif args.mode == "language":
        results = eval_language(model, val_data, args.batch_size, args.max_samples)

    else:
        results = eval_full(model, val_data, args.batch_size, args.max_samples)

    print("\n========= RESULTS =========\n")
    for k,(m,s) in results.items():
        print(f"{k:20s} | mean={m:.4f} | std={s:.4f}")
    print("\n===========================\n")

    if args.use_wandb:
        wandb.log({f"{k}_mean": v[0] for k,v in results.items()} |
                  {f"{k}_std": v[1] for k,v in results.items()})
        wandb.finish()

if __name__ == "__main__":
    main()