# pylint: skip-file
import datetime
import tempfile
from typing import Tuple

import os
import fire
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from peft import get_peft_model_state_dict
from transformers import logging

import wandb
from utils.model_utils import load_llama_tokenizer, load_model
from utils.training_utils import (
    decode_generation_seqeunces,
    eval_action,
    eval_tl,
    get_eval_distance_errors,
    get_train_val_data,
    log_txt_as_img,
)

# ============================================================
# TRUE DATASET DIMS
# ============================================================

ROUTE_RAW_DIM = 6
VEHICLE_RAW_DIM = 6
PEDESTRIAN_RAW_DIM = 6
EGO_RAW_DIM = 6

# ============================================================
# MODEL EXPECTED DIMS (Wayve VectorEncoder Contract)
# ============================================================

MODEL_ROUTE_POINTS = 30
ROUTE_MODEL_DIM = 17
VEHICLE_MODEL_DIM = 33
PEDESTRIAN_MODEL_DIM = 9
EGO_MODEL_DIM = 31

ROUTE_SHAPE = (MODEL_ROUTE_POINTS, ROUTE_RAW_DIM)
VEHICLE_SHAPE = (16, VEHICLE_RAW_DIM)
PEDESTRIAN_SHAPE = (16, PEDESTRIAN_RAW_DIM)
EGO_SHAPE = (EGO_RAW_DIM,)

# ============================================================
# AUTO VECTOR SHAPE TRACER (CATCHES FUTURE BUGS)
# ============================================================

DEBUG_VECTOR_TRACE = True
TRACE_RATE = 0.002  # ~0.2% batches print

def trace_vector(name, x):
    if x is None:
        print(f"[TRACE] {name}: None")
        return
    if torch.rand(1).item() < TRACE_RATE:
        print(f"[TRACE] {name}: shape={tuple(x.shape)} dtype={x.dtype}")

# ============================================================
# GENERIC VECTOR ADAPTER (LEARNED SAFE PROJECTION)
# ============================================================

class VectorAdapter(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.proj = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x):
        if x is None:
            return None

        # Ego case [B, D]
        if x.ndim == 2:
            if x.shape[-1] == self.out_dim:
                return x
            return self.proj(x)

        # Agent case [B, K, D]
        B, K, D = x.shape
        if D == self.out_dim:
            return x

        if D != self.in_dim:
            raise RuntimeError(f"Adapter expected dim {self.in_dim}, got {D}")

        x = self.proj(x.reshape(-1, D))
        return x.view(B, K, self.out_dim)

# ============================================================
# ROUTE ADAPTER (COUNT + FEATURE SAFE)
# ============================================================

class RouteAdapter(torch.nn.Module):
    def __init__(self, in_dim=ROUTE_RAW_DIM, out_dim=ROUTE_MODEL_DIM, max_routes=MODEL_ROUTE_POINTS):
        super().__init__()
        self.max_routes = max_routes
        self.proj = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x):
        B, K, D = x.shape

        # Fix route count
        if K > self.max_routes:
            x = x[:, :self.max_routes]
        elif K < self.max_routes:
            pad = torch.zeros(B, self.max_routes - K, D, device=x.device)
            x = torch.cat([x, pad], dim=1)

        # Project features
        if D == ROUTE_MODEL_DIM:
            return x

        x = self.proj(x.reshape(-1, D))
        return x.view(B, self.max_routes, ROUTE_MODEL_DIM)

# ============================================================
# INIT ADAPTERS
# ============================================================

route_adapter = RouteAdapter().cuda()
vehicle_adapter = VectorAdapter(VEHICLE_RAW_DIM, VEHICLE_MODEL_DIM).cuda()
pedestrian_adapter = VectorAdapter(PEDESTRIAN_RAW_DIM, PEDESTRIAN_MODEL_DIM).cuda()
ego_adapter = VectorAdapter(EGO_RAW_DIM, EGO_MODEL_DIM).cuda()

# ============================================================
# MODEL WRAPPER — ENFORCES SAFE VECTOR CONTRACT
# ============================================================

class VectorRouteGuard(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        route_descriptors=None,
        vehicle_descriptors=None,
        pedestrian_descriptors=None,
        ego_vehicle_descriptor=None,
        **kwargs
    ):

        # Adapt vectors
        if route_descriptors is not None:
            route_descriptors = route_adapter(route_descriptors.float())

        if vehicle_descriptors is not None:
            vehicle_descriptors = vehicle_adapter(vehicle_descriptors.float())

        if pedestrian_descriptors is not None:
            pedestrian_descriptors = pedestrian_adapter(pedestrian_descriptors.float())

        if ego_vehicle_descriptor is not None:
            ego_vehicle_descriptor = ego_adapter(ego_vehicle_descriptor.float())

        # Debug tracing
        if DEBUG_VECTOR_TRACE:
            trace_vector("route", route_descriptors)
            trace_vector("vehicle", vehicle_descriptors)
            trace_vector("pedestrian", pedestrian_descriptors)
            trace_vector("ego", ego_vehicle_descriptor)

        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            route_descriptors=route_descriptors,
            vehicle_descriptors=vehicle_descriptors,
            pedestrian_descriptors=pedestrian_descriptors,
            ego_vehicle_descriptor=ego_vehicle_descriptor,
            **kwargs
        )

# ============================================================
# DATA PATCHER — SHAPE SAFE LOADING
# ============================================================

def patch_vector_fields(dataset):
    print("🔍 Validating dataset vector fields...")

    for i in range(len(dataset)):
        d = dataset[i]
        obs = d.get("observation", d)

        def ensure(name, shape):
            x = obs.get(name)
            if x is None:
                return np.zeros(shape, dtype=np.float32)

            arr = np.array(x)
            fixed = np.zeros(shape, dtype=np.float32)

            slices = tuple(slice(0, min(a, b)) for a, b in zip(arr.shape, shape))
            fixed[slices] = arr[slices]
            return fixed

        obs["route_descriptors"] = ensure("route_descriptors", ROUTE_SHAPE)
        obs["vehicle_descriptors"] = ensure("vehicle_descriptors", VEHICLE_SHAPE)
        obs["pedestrian_descriptors"] = ensure("pedestrian_descriptors", PEDESTRIAN_SHAPE)
        obs["ego_vehicle_descriptor"] = ensure("ego_vehicle_descriptor", EGO_SHAPE)

    print("✅ Dataset sanitized")
    return dataset

# ============================================================
# LEGIBILITY LOSS (DRAGAN)
# ============================================================
def compute_dragan_legibility_loss(model, inputs, alpha: float = 1.0):
    """
    Multi-observer, distance-weighted legibility loss (Dragan-style),
    with EARLY EXIT when no real vehicles/pedestrians exist.

    - Observers = vehicles + pedestrians.
    - Closer observers get higher weight.
    - True goal is assumed to be route index 0.
    - Empty rows (flag=0) are ignored.
    """

    labels = inputs["labels"]
    routes = inputs["route_descriptors"]           # [B, 30, 17]
    vehicle = inputs["vehicle_descriptors"]       # [B, 30, 33]
    pedestrian = inputs["pedestrian_descriptors"] # [B, 20, 9]
    ego = inputs["ego_vehicle_descriptor"]        # [B, 31]

    B, K, _ = routes.shape  # K = 30 route candidates

    # --------------------------------------------------
    # 1) BUILD OBSERVER MASKS (CHEAP CHECK FIRST)
    # --------------------------------------------------

    # Existence flags (first column in both tensors)
    veh_exists = vehicle[:, :, 0] > 0        # [B, 30]
    ped_exists = pedestrian[:, :, 0] > 0     # [B, 20]

    # 👉 FAST EXIT: if there are NO real objects in the batch, skip legibility
    if (veh_exists.sum() + ped_exists.sum()) == 0:
        return torch.tensor(0.0, device=routes.device)

    # --------------------------------------------------
    # 2) BUILD OBSERVER POSITIONS (ONLY IF NEEDED)
    # --------------------------------------------------

    # Extract relative (x, y)
    veh_xy = vehicle[:, :, 3:5]   # [B, 30, 2]
    ped_xy = pedestrian[:, :, 2:4]# [B, 20, 2]

    # Mask out empty rows
    veh_xy = veh_xy * veh_exists.unsqueeze(-1)
    ped_xy = ped_xy * ped_exists.unsqueeze(-1)

    # Concatenate observers
    obs_xy = torch.cat([veh_xy, ped_xy], dim=1)      # [B, 50, 2]
    obs_exists = torch.cat([veh_exists, ped_exists], dim=1)  # [B, 50]

    # --------------------------------------------------
    # 3) DISTANCE-BASED OBSERVER WEIGHTS
    # --------------------------------------------------

    ego_xy = ego[:, :2]  # [B, 2]

    dists = torch.norm(obs_xy - ego_xy.unsqueeze(1), dim=-1)

    # Push empty rows far away
    dists = dists + (1 - obs_exists.float()) * 1e6

    obs_weights = torch.softmax(-alpha * dists, dim=1)  # [B, 50]

    # --------------------------------------------------
    # 4) LEGIBILITY (GOAL LIKELIHOODS)
    # --------------------------------------------------

    log_likelihoods = []

    for g in range(K):
        goal_route = routes.clone()
        goal_route.zero_()
        goal_route[:, 0, :] = routes[:, g, :]

        with torch.no_grad():
            out = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=labels,
                route_descriptors=goal_route,
                vehicle_descriptors=vehicle,
                pedestrian_descriptors=pedestrian,
                ego_vehicle_descriptor=ego,
            )

        log_likelihoods.append(-out["loss"])  # proxy for likelihood

    log_likelihoods = torch.stack(log_likelihoods, dim=0)  # [K]

    goal_posterior = F.log_softmax(log_likelihoods, dim=0)

    leg_loss = -goal_posterior[0]  # scalar

    # Weight by closest observer (cheapest multi-observer proxy)
    closest_weight = obs_weights.max(dim=1).values.mean()

    return closest_weight * leg_loss


# ============================================================
# TRAINER
# ============================================================

class TrainerWithGeneration(transformers.Seq2SeqTrainer):

    def __init__(self, *args, legibility_weight=0.3, **kwargs):
        self.legibility_weight = legibility_weight
        super().__init__(*args, **kwargs)
        self.tokenizer = kwargs["data_collator"].tokenizer

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        device = next(model.parameters()).device
        obs = inputs.get("observation", inputs)

        route = torch.tensor(obs["route_descriptors"], device=device).float()
        veh = torch.tensor(obs["vehicle_descriptors"], device=device).float()
        ped = torch.tensor(obs["pedestrian_descriptors"], device=device).float()
        ego = torch.tensor(obs["ego_vehicle_descriptor"], device=device).float()

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
            route_descriptors=route,
            vehicle_descriptors=veh,
            pedestrian_descriptors=ped,
            ego_vehicle_descriptor=ego,
        )
        print(outputs)
        lm_loss = outputs["loss"]

        leg_loss = compute_dragan_legibility_loss(
            model,
            {
                "labels": inputs["labels"],
                "route_descriptors": route_adapter(route),
                "vehicle_descriptors": veh,
                "pedestrian_descriptors": ped,
                "ego_vehicle_descriptor": ego,
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
            },
        )
        print("legibility loss", leg_loss)

        total_loss = lm_loss + self.legibility_weight * leg_loss
        return (total_loss, outputs) if return_outputs else total_loss


def train(
    base_model="meta-llama/Llama-2-7b-hf",
    data_path="data/vqa_test_600.pkl",
    batch_size=128,
    micro_batch_size=32,
    num_epochs=5,
    learning_rate=3e-4,
    val_set_size=200,
    eval_steps=10,
    lora_r=16,
    resume_from_checkpoint="models/weights/stage1_pretrained_model/",
    legibility_weight=0.5,
    wandb_project: str = "legible_driving",
    wandb_run_name: str = "legible_0.5",
):

    # ------------------------------
    # 1) SET UP WANDB
    # ------------------------------
    os.environ["WANDB_PROJECT"] = wandb_project
    if wandb_run_name:
        os.environ["WANDB_RUN_NAME"] = wandb_run_name

    wandb.init(
        project=wandb_project,
        name=wandb_run_name if wandb_run_name else None,
        config={
            "base_model": base_model,
            "batch_size": batch_size,
            "micro_batch_size": micro_batch_size,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "legibility_weight": legibility_weight,
        },
    )

    # ------------------------------
    # 2) MODEL & DATA
    # ------------------------------
    output_dir = tempfile.mkdtemp(
        prefix=f"legible_safe_{datetime.datetime.now():%Y%m%d_%H%M%S}_"
    )

    base_model_token = base_model
    base_model = load_model(
        base_model=base_model_token,
        resume_from_checkpoint=resume_from_checkpoint,
        lora_r=16,
    ).cuda()

    model = VectorRouteGuard(base_model).cuda()

    tokenizer = load_llama_tokenizer(base_model_token)

    train_data, val_data = get_train_val_data(
        data_path, tokenizer, val_set_size=val_set_size
    )

    train_data = patch_vector_fields(train_data)
    val_data = patch_vector_fields(val_data)

    # ------------------------------
    # 3) TRAINER WITH WANDB LOGGING
    # ------------------------------
    trainer = TrainerWithGeneration(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        legibility_weight=legibility_weight,
        args=transformers.Seq2SeqTrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=batch_size // micro_batch_size,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            eval_steps=eval_steps,
            save_steps=100,
            output_dir=output_dir,
            save_total_limit=3,
            predict_with_generate=True,
            generation_max_length=384,
            report_to="wandb",              # <-- KEY LINE
            run_name=wandb_run_name,        # <-- KEY LINE
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, padding=True
        ),
    )

    # ------------------------------
    # 4) PATCH TRAINER TO LOG LEGIBILITY LOSS
    # ------------------------------
    original_compute_loss = trainer.compute_loss

    def compute_loss_with_legibility_logging(model, inputs, *args, **kwargs):
        loss = original_compute_loss(model, inputs, *args, **kwargs)

        # If loss is a tuple (loss, outputs), extract loss
        if isinstance(loss, tuple):
            total_loss = loss[0]
        else:
            total_loss = loss

        # Try to access last computed legibility loss if stored in trainer
        if hasattr(trainer, "last_leg_loss"):
            wandb.log(
                {
                    "train/total_loss": total_loss.item(),
                    "train/legibility_loss": trainer.last_leg_loss,
                    "train/lm_loss": trainer.last_lm_loss,
                }
            )

        return loss

    trainer.compute_loss = compute_loss_with_legibility_logging

    # ------------------------------
    # 5) TRAIN
    # ------------------------------
    logging.set_verbosity_info()
    trainer.train()
    # model.save_pretrained(output_dir)
    output_dir = '/content/legible_autonomous_driving/results/legible_finetune'
    trainer.save_model(output_dir)
    torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
    tokenizer.save_pretrained(output_dir)

    print("Training complete →", output_dir)


if __name__ == "__main__":
    fire.Fire(train)
