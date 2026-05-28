# pylint: skip-file
"""
Legibility-aware fine-tuning for LLM-based autonomous driving.

This version drops the 6-dim adapter detour entirely and feeds the
Wayve-format descriptors (route [30,17], vehicle [30,33], pedestrian
[20,9], ego [31]) straight through to the base model's VectorEncoder,
the way the underlying Driving-with-LLMs model expects.
"""

import datetime
import math
import os
import tempfile

import fire
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import wandb
from transformers import logging

from utils.model_utils import load_llama_tokenizer, load_model
from utils.training_utils import get_train_val_data


# ============================================================
# Native Wayve descriptor shapes
# ------------------------------------------------------------
# These are what the base model's VectorEncoder already expects,
# so no learned projection is required.
# ============================================================
ROUTE_SHAPE      = (30, 17)   # [route_points, feat]
VEHICLE_SHAPE    = (30, 33)   # [num_vehicles, feat]
PEDESTRIAN_SHAPE = (20, 9)    # [num_pedestrians, feat]
EGO_SHAPE        = (31,)      # [feat]

# Per-descriptor column indices used by the legibility loss.
# Verify against your local utils/training_utils.py if the schema
# differs in your fork.
VEH_EXISTS_IDX = 0
VEH_XY_SLICE   = slice(3, 5)   # relative (x, y)
VEH_YAW_IDX    = 5             # relative yaw

PED_EXISTS_IDX = 0
PED_XY_SLICE   = slice(2, 4)   # relative (x, y)
PED_YAW_IDX    = 5             # orientation
PED_CROSS_IDX  = 8             # crossing intent

# Debug toggle for shape tracing.
DEBUG_VECTOR_TRACE = False
TRACE_RATE = 0.002


def _trace(name, x):
    if not DEBUG_VECTOR_TRACE or x is None:
        return
    if torch.rand(1).item() < TRACE_RATE:
        print(f"[TRACE] {name}: shape={tuple(x.shape)} dtype={x.dtype}")


# ============================================================
# Model wrapper
# ------------------------------------------------------------
# Forwards descriptors straight to the base causal LM (which owns
# the VectorEncoder), pulls the last hidden state, and emits a
# trajectory prediction used by the legibility loss.
# ============================================================
# ============================================================
# Model wrapper
# ------------------------------------------------------------
# The Wayve causal LM owns the vector fusion and returns only the
# LM loss (its forward computes cross-entropy internally over the
# fused sequence, handling the vector-token offset correctly).
# We therefore:
#   * pass `labels` through and use the model's own `loss`, and
#   * capture the final hidden state via a forward hook on the
#     transformer's last norm layer, to feed the route head.
# This avoids depending on the return dict exposing logits/hidden
# states (it doesn't) and avoids mis-aligning labels with the
# extra vector tokens.
# ============================================================
def _find_final_norm(model):
    """Locate the transformer's final norm (the module whose output
    is the last hidden state). It is named '...model.norm' and lives
    outside the decoder layers (unlike '...layers.N.input_layernorm')."""
    chosen = None
    for name, module in model.named_modules():
        if name.split(".")[-1] == "norm" and ".layers." not in name:
            chosen = (name, module)
    return chosen


class VectorRouteGuard(nn.Module):
    def __init__(self, model, num_route_pts: int = 30):
        super().__init__()
        self.model = model
        self.hidden_size = model.config.hidden_size
        self.num_route_pts = num_route_pts

        self.route_head = nn.Linear(self.hidden_size, 2)

        # Small init so the pooled hidden state actually influences
        # the prediction early in training.
        self.route_queries = nn.Parameter(
            torch.empty(num_route_pts, self.hidden_size)
        )
        nn.init.normal_(self.route_queries, std=0.02)

        # Hook the final norm to capture the last hidden state.
        found = _find_final_norm(model)
        if found is None:
            raise RuntimeError(
                "Could not locate the transformer's final norm to hook. "
                "Inspect model.named_modules() and adjust _find_final_norm."
            )
        self._norm_name, norm_module = found
        self._captured = {}

        def _capture_hook(_module, _inp, out):
            self._captured["hidden"] = out[0] if isinstance(out, tuple) else out

        norm_module.register_forward_hook(_capture_hook)

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
        **kwargs,
    ):
        # Match base model dtype to avoid fp16/fp32 mismatches under autocast.
        param_dtype = next(self.model.parameters()).dtype

        def _cast(x):
            return None if x is None else x.to(dtype=param_dtype)

        route_descriptors      = _cast(route_descriptors)
        vehicle_descriptors    = _cast(vehicle_descriptors)
        pedestrian_descriptors = _cast(pedestrian_descriptors)
        ego_vehicle_descriptor = _cast(ego_vehicle_descriptor)

        if DEBUG_VECTOR_TRACE:
            _trace("route",      route_descriptors)
            _trace("vehicle",    vehicle_descriptors)
            _trace("pedestrian", pedestrian_descriptors)
            _trace("ego",        ego_vehicle_descriptor)

        # Strip trainer-supplied kwargs the base model doesn't accept.
        kwargs.pop("num_items_in_batch", None)

        self._captured.clear()

        # Let the Wayve causal LM compute the LM loss itself (passing
        # labels). The forward hook fills self._captured["hidden"].
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            route_descriptors=route_descriptors,
            vehicle_descriptors=vehicle_descriptors,
            pedestrian_descriptors=pedestrian_descriptors,
            ego_vehicle_descriptor=ego_vehicle_descriptor,
            return_dict=True,
            **kwargs,
        )

        lm_loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss

        hidden = self._captured.get("hidden")              # [B, T, H]
        if hidden is None:
            raise RuntimeError(
                f"Forward hook on '{self._norm_name}' did not capture a hidden "
                f"state. The final-norm module may have a different name."
            )

        B = hidden.shape[0]
        global_feat   = hidden.mean(dim=1, keepdim=True)                   # [B, 1, H]
        route_queries = self.route_queries.unsqueeze(0).expand(B, -1, -1)  # [B, K, H]
        route_tokens  = global_feat + route_queries                        # [B, K, H]
        route_pred    = self.route_head(route_tokens)                      # [B, K, 2]

        return {
            "loss": lm_loss,
            "last_hidden_state": hidden,
            "route_pred": route_pred,
        }


# ============================================================
# Dataset patcher
# ------------------------------------------------------------
# Pads / truncates only the agent-count axis. The feature axis is
# preserved exactly — if a sample's feature dim disagrees with the
# expected schema we raise instead of silently corrupting it.
# ============================================================
DESCRIPTOR_SHAPES = {
    "route_descriptors":      ROUTE_SHAPE,
    "vehicle_descriptors":    VEHICLE_SHAPE,
    "pedestrian_descriptors": PEDESTRIAN_SHAPE,
    "ego_vehicle_descriptor": EGO_SHAPE,
}


def fit_to_shape(name, x, target_shape):
    """Pad / truncate `x` to `target_shape`, adjusting only the
    agent-count (leading) axis. The feature axis must already match —
    a mismatch raises rather than silently slicing columns."""
    if x is None:
        return np.zeros(target_shape, dtype=np.float32)

    arr = np.asarray(x, dtype=np.float32)

    if arr.ndim == 1:
        # Ego descriptor.
        out = np.zeros(target_shape, dtype=np.float32)
        n = min(arr.shape[0], target_shape[0])
        out[:n] = arr[:n]
        return out

    if arr.ndim == 2:
        N_target, F_target = target_shape
        if arr.shape[1] != F_target:
            raise ValueError(
                f"{name}: feature dim mismatch — got {arr.shape[1]}, "
                f"expected {F_target}. Update *_SHAPE constants or check "
                f"your dataset version."
            )
        out = np.zeros(target_shape, dtype=np.float32)
        n = min(arr.shape[0], N_target)
        out[:n] = arr[:n]
        return out

    raise ValueError(f"{name}: unexpected ndim={arr.ndim}")


def patch_vector_fields(dataset):
    """Best-effort in-place normalization. The collator re-applies
    fit_to_shape per batch, so this is mainly a fail-fast validation
    pass (and a no-op if the dataset returns copies rather than refs)."""
    print("Validating dataset vector fields...")
    for i in range(len(dataset)):
        sample = dataset[i]
        obs = sample.get("observation", sample)
        for name, target_shape in DESCRIPTOR_SHAPES.items():
            obs[name] = fit_to_shape(name, obs.get(name), target_shape)
    print("Dataset sanitized")
    return dataset


# ============================================================
# Custom collator
# ------------------------------------------------------------
# DataCollatorForSeq2Seq hands every field to tokenizer.pad, which
# cannot handle the structured descriptor arrays. We pad the text
# fields with the standard collator and stack the vector fields
# (already fixed-size via fit_to_shape) ourselves.
# ============================================================
class LegibleDataCollator:
    TEXT_KEYS = ("input_ids", "attention_mask", "labels")

    def __init__(self, tokenizer, pad_to_multiple_of=8):
        self.tokenizer = tokenizer
        self.text_collator = transformers.DataCollatorForSeq2Seq(
            tokenizer,
            padding=True,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors="pt",
        )

    def __call__(self, features):
        text_features = []
        vector_store = {k: [] for k in DESCRIPTOR_SHAPES}

        for f in features:
            obs = f.get("observation", f)
            for k, shape in DESCRIPTOR_SHAPES.items():
                vector_store[k].append(fit_to_shape(k, obs.get(k), shape))
            # Only forward paddable text fields to the tokenizer.
            text_features.append({k: f[k] for k in self.TEXT_KEYS if k in f})

        batch = self.text_collator(text_features)

        observation = {
            k: torch.from_numpy(np.stack(v, axis=0)).float()
            for k, v in vector_store.items()
        }
        batch["observation"] = observation
        return batch


# ============================================================
# Observer-aware legibility loss (fully differentiable)
# ============================================================
def compute_legibility_from_predicted_route(
    route_pred,                    # [B, K, 2]
    vehicle,                       # [B, Nv, F_v]
    pedestrian,                    # [B, Np, F_p]
    beta: float = 0.5,
    veh_fov_deg: float = 300.0,    # cars: wide effective FOV
    ped_fov_deg: float = 120.0,    # pedestrians: narrower FOV
    crossing_boost: float = 1.5,
    max_distance: float = 50.0,
):
    device = route_pred.device

    veh_exists = vehicle[:, :, VEH_EXISTS_IDX] > 0       # [B, Nv]
    ped_exists = pedestrian[:, :, PED_EXISTS_IDX] > 0    # [B, Np]

    if (veh_exists.sum() + ped_exists.sum()) == 0:
        return torch.tensor(0.0, device=device)

    veh_xy = vehicle[:, :, VEH_XY_SLICE]                  # [B, Nv, 2]
    ped_xy = pedestrian[:, :, PED_XY_SLICE]               # [B, Np, 2]
    ego_xy = route_pred[:, 0]                             # [B, 2]

    # Trajectory direction from predicted waypoints.
    traj_vec = route_pred[:, -1] - route_pred[:, 0]                # [B, 2]
    traj_dir = traj_vec / (traj_vec.norm(dim=-1, keepdim=True) + 1e-6)

    # Observer viewing directions from yaw.
    veh_yaw = vehicle[:, :, VEH_YAW_IDX]
    veh_dir = torch.stack([torch.cos(veh_yaw), torch.sin(veh_yaw)], dim=-1)

    if pedestrian.shape[-1] > PED_YAW_IDX:
        ped_yaw = pedestrian[:, :, PED_YAW_IDX]
        ped_dir = torch.stack([torch.cos(ped_yaw), torch.sin(ped_yaw)], dim=-1)
    else:
        v = ego_xy.unsqueeze(1) - ped_xy
        ped_dir = v / (v.norm(dim=-1, keepdim=True) + 1e-6)

    def visibility(obs_pos, obs_dir, fov_deg):
        v = ego_xy.unsqueeze(1) - obs_pos                          # [B, N, 2]
        v_norm = v.norm(dim=-1) + 1e-6
        cos_theta = (v * obs_dir).sum(dim=-1) / v_norm
        cos_theta = cos_theta.clamp(-0.999, 0.999)
        theta = torch.acos(cos_theta)
        half_fov = (fov_deg / 2.0) * (math.pi / 180.0)
        return (1.0 - theta / half_fov).clamp(min=0.0)

    def distance_to_traj(obs_pos):
        v = obs_pos - ego_xy.unsqueeze(1)                          # [B, N, 2]
        proj = (v * traj_dir.unsqueeze(1)).sum(dim=-1, keepdim=True) \
               * traj_dir.unsqueeze(1)
        return (v - proj).norm(dim=-1)

    # Vehicles
    veh_vis = visibility(veh_xy, veh_dir, veh_fov_deg)
    veh_d   = distance_to_traj(veh_xy)
    veh_leg = veh_vis * torch.exp(-beta * veh_d.clamp(max=max_distance))
    veh_leg = veh_leg * veh_exists.float()

    # Pedestrians (with crossing boost)
    ped_vis = visibility(ped_xy, ped_dir, ped_fov_deg)
    ped_d   = distance_to_traj(ped_xy)
    if pedestrian.shape[-1] > PED_CROSS_IDX:
        ped_cross = pedestrian[:, :, PED_CROSS_IDX]
    else:
        ped_cross = torch.zeros_like(ped_vis)
    ped_weight = 1.0 + (crossing_boost - 1.0) * ped_cross
    ped_leg = ped_vis * torch.exp(-beta * ped_d.clamp(max=max_distance)) * ped_weight
    ped_leg = ped_leg * ped_exists.float()

    total_leg = veh_leg.sum(dim=1) + ped_leg.sum(dim=1)              # [B]
    num_obs   = veh_exists.sum(dim=1) + ped_exists.sum(dim=1)        # [B]
    scene_leg = total_leg / (num_obs.float() + 1e-6)                 # [B]

    # Maximize legibility ⇒ minimize its negative.
    return -scene_leg.mean()


# ============================================================
# Route-validity losses
# ------------------------------------------------------------
# The legibility loss alone has a degenerate optimum (point the
# trajectory at nearby observers). These terms keep the suggested
# route VALID — anchored to the reference route (goal-reaching,
# on-road), starting at the ego position, and drivable — so the
# legibility term acts as a soft perturbation that makes a valid
# route *more legible* rather than inventing an arbitrary one.
#
# Assumes route_descriptors[:, :, :2] are the reference (x, y)
# waypoints in the same ego-centric frame as the observer
# positions used by the legibility loss.
# ============================================================
def compute_route_losses(route_pred, route_descriptors):
    """Returns (recon, start, smooth) scalar losses."""
    device = route_pred.device
    ref_xy = route_descriptors[..., :2]                     # [B, R, 2]

    K = route_pred.shape[1]
    R = ref_xy.shape[1]
    n = min(K, R)
    pred = route_pred[:, :n]                                # [B, n, 2]
    ref  = ref_xy[:, :n]                                    # [B, n, 2]

    # Valid route points = non-padded rows in the descriptor.
    valid = (route_descriptors[:, :n].abs().sum(-1) > 0).float()   # [B, n]
    valid_e = valid.unsqueeze(-1)                                  # [B, n, 1]
    denom = valid_e.sum().clamp(min=1.0)

    # Reconstruction: stay on the reference route (validity + goal).
    recon = (F.smooth_l1_loss(pred, ref, reduction="none") * valid_e).sum() / denom

    # Start anchor: y0 == current ego position (== first route point).
    start = F.smooth_l1_loss(route_pred[:, 0], ref_xy[:, 0], reduction="mean")

    # Smoothness: penalize curvature (second difference) for drivability.
    if n >= 3:
        second_diff = pred[:, 2:] - 2.0 * pred[:, 1:-1] + pred[:, :-2]   # [B, n-2, 2]
        sm_mask = valid[:, 2:n]                                          # [B, n-2]
        smooth = (second_diff.pow(2).sum(-1) * sm_mask).sum() / denom
    else:
        smooth = torch.tensor(0.0, device=device)

    return recon, start, smooth
class TrainerWithGeneration(transformers.Seq2SeqTrainer):
    def __init__(
        self,
        *args,
        legibility_weight: float = 0.05,
        route_weight: float = 1.0,
        start_weight: float = 0.1,
        smooth_weight: float = 0.01,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.legibility_weight = legibility_weight
        self.route_weight = route_weight
        self.start_weight = start_weight
        self.smooth_weight = smooth_weight
        self.last_lm_loss     = None
        self.last_leg_loss    = None
        self.last_recon_loss  = None
        self.last_start_loss  = None
        self.last_smooth_loss = None

    @staticmethod
    def _to_device(x, device):
        if torch.is_tensor(x):
            return x.to(device=device, dtype=torch.float32)
        return torch.as_tensor(np.asarray(x), device=device, dtype=torch.float32)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        device = next(model.parameters()).device
        obs = inputs.get("observation", inputs)

        route = self._to_device(obs["route_descriptors"],      device)
        veh   = self._to_device(obs["vehicle_descriptors"],    device)
        ped   = self._to_device(obs["pedestrian_descriptors"], device)
        ego   = self._to_device(obs["ego_vehicle_descriptor"], device)

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
            route_descriptors=route,
            vehicle_descriptors=veh,
            pedestrian_descriptors=ped,
            ego_vehicle_descriptor=ego,
        )

        # The wrapper returns the Wayve model's own LM loss (computed
        # internally over the fused sequence with correct label offset).
        lm_loss    = outputs["loss"]
        route_pred = outputs["route_pred"].float()

        # Validity anchors: keep the suggested route on the reference
        # route, starting at the ego, and drivable.
        recon, start, smooth = compute_route_losses(route_pred, route)

        # Legibility: nudge the (valid) route to express intent earlier.
        leg_loss = compute_legibility_from_predicted_route(route_pred, veh, ped)

        total_loss = (
            lm_loss
            + self.route_weight       * recon
            + self.start_weight       * start
            + self.smooth_weight      * smooth
            + self.legibility_weight  * leg_loss
        )

        # Cached for the logging callback (logged at the trainer's step).
        self.last_lm_loss     = lm_loss.detach().float().item()
        self.last_leg_loss    = leg_loss.detach().float().item()
        self.last_recon_loss  = recon.detach().float().item()
        self.last_start_loss  = start.detach().float().item()
        self.last_smooth_loss = smooth.detach().float().item()

        return (total_loss, outputs) if return_outputs else total_loss


class LegibilityLogger(transformers.TrainerCallback):
    """Attaches loss components to the trainer's own log step, so W&B
    sees them at the right global_step rather than once per microbatch."""

    def __init__(self, trainer):
        self._trainer = trainer

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        t = self._trainer
        for key, val in [
            ("train/lm_loss",         t.last_lm_loss),
            ("train/legibility_loss", t.last_leg_loss),
            ("train/route_recon",     t.last_recon_loss),
            ("train/route_start",     t.last_start_loss),
            ("train/route_smooth",    t.last_smooth_loss),
        ]:
            if val is not None:
                logs[key] = val


# ============================================================
# Train entrypoint
# ============================================================
def train(
    base_model: str = "meta-llama/Llama-2-7b-hf",
    data_path: str = "data/vqa_train_10k.pkl",
    batch_size: int = 128,
    micro_batch_size: int = 32,
    num_epochs: int = 1,
    learning_rate: float = 3e-4,
    val_set_size: int = 2000,
    eval_steps: int = 50,
    lora_r: int = 16,
    resume_from_checkpoint: str = "models/weights/stage1_pretrained_model/",
    legibility_weight: float = 0.05,
    route_weight: float = 1.0,
    start_weight: float = 0.1,
    smooth_weight: float = 0.01,
    output_dir: str = "results/legible_finetune",
    wandb_project: str = "legible_driving",
    wandb_run_name: str = "observer_legible_0.05",
):
    os.environ["WANDB_PROJECT"] = wandb_project
    if wandb_run_name:
        os.environ["WANDB_RUN_NAME"] = wandb_run_name

    wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        config={
            "base_model": base_model,
            "batch_size": batch_size,
            "micro_batch_size": micro_batch_size,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "legibility_weight": legibility_weight,
            "route_weight": route_weight,
            "start_weight": start_weight,
            "smooth_weight": smooth_weight,
            "lora_r": lora_r,
        },
    )

    tmp_dir = tempfile.mkdtemp(
        prefix=f"legible_{datetime.datetime.now():%Y%m%d_%H%M%S}_"
    )

    base = load_model(
        base_model=base_model,
        resume_from_checkpoint=resume_from_checkpoint,
        lora_r=lora_r,
    ).cuda()

    model = VectorRouteGuard(base).cuda()
    tokenizer = load_llama_tokenizer(base_model)

    train_data, val_data = get_train_val_data(
        data_path, tokenizer, val_set_size=val_set_size
    )
    train_data = patch_vector_fields(train_data)
    val_data   = patch_vector_fields(val_data)

    # NOTE on evaluation:
    #   predict_with_generate=False because VectorRouteGuard does
    #   not implement .generate(). Eval falls back to loss-only.
    #   To enable generation during eval, add a .generate() method
    #   that adapts descriptors and delegates to self.model.generate().
    #
    # NOTE on eval_strategy:
    #   This is the transformers >= 4.41 name. On older versions
    #   (<= 4.40) it was `evaluation_strategy` — rename if needed.
    training_args = transformers.Seq2SeqTrainingArguments(
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=max(1, batch_size // micro_batch_size),
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_steps=100,
        save_total_limit=3,
        output_dir=tmp_dir,
        predict_with_generate=False,
        report_to="wandb",
        run_name=wandb_run_name,
        remove_unused_columns=False,
    )

    trainer = TrainerWithGeneration(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        legibility_weight=legibility_weight,
        route_weight=route_weight,
        start_weight=start_weight,
        smooth_weight=smooth_weight,
        args=training_args,
        data_collator=LegibleDataCollator(
            tokenizer, pad_to_multiple_of=8
        ),
    )

    trainer.add_callback(LegibilityLogger(trainer))

    logging.set_verbosity_info()
    trainer.train()

    # ----- Save artifacts -------------------------------------------------
    final_dir = output_dir
    os.makedirs(final_dir, exist_ok=True)

    # PEFT adapter + tokenizer (small).
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    # Just the wrapper-specific heads — kilobytes, not gigabytes.
    extras = {
        "route_head":    model.route_head.state_dict(),
        "route_queries": model.route_queries.detach().cpu(),
    }
    torch.save(extras, os.path.join(final_dir, "wrapper_heads.pt"))

    print("Training complete →", final_dir)


if __name__ == "__main__":
    fire.Fire(train)