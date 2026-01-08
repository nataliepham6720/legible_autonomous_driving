# pylint: skip-file
import datetime
import os
import tempfile
from typing import List, Optional, Tuple

import fire
import numpy as np
import torch
import transformers
from peft import get_peft_model_state_dict
from transformers import logging

import wandb
from utils.model_utils import load_llama_tokenizer, load_model
from utils.training_utils import (
    DEFAULT_EVAL_ITEMS,
    decode_generation_seqeunces,
    eval_action,
    eval_tl,
    get_eval_distance_errors,
    get_train_val_data,
    log_txt_as_img,
)

# ============================================================
# Legibility Regularizer
# ============================================================

def compute_legibility_penalty(pred_texts, vehicle_desc, pedestrian_desc):
    """
    Penalize missing references to visible agents.
    """
    penalty = 0.0
    batch_size = len(pred_texts)

    for i in range(batch_size):
        text = pred_texts[i].lower()

        vehicles = vehicle_desc[i]
        pedestrians = pedestrian_desc[i]

        visible_vehicles = vehicles[:, 0] > 0
        visible_peds = pedestrians[:, 0] > 0

        expected_mentions = int(visible_vehicles.sum() + visible_peds.sum())
        if expected_mentions == 0:
            continue

        mentioned = (
            text.count("car")
            + text.count("vehicle")
            + text.count("pedestrian")
        )

        penalty += max(0.0, 1.0 - mentioned / expected_mentions)

    return penalty / batch_size


# ============================================================
# Custom Trainer with Legibility Loss
# ============================================================

class TrainerWithGeneration(transformers.Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        self.vqa = kwargs.pop("vqa", False)
        super().__init__(*args, **kwargs)
        self.tokenizer = kwargs["data_collator"].tokenizer

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs.loss

        # Decode predictions for legibility regularizer
        with torch.no_grad():
            pred_ids = torch.argmax(outputs.logits, dim=-1)
            pred_texts = self.tokenizer.batch_decode(
                pred_ids, skip_special_tokens=True
            )

        legibility_penalty = compute_legibility_penalty(
            pred_texts,
            inputs["vehicle_descriptors"],
            inputs["pedestrian_descriptors"],
        )

        legibility_weight = 0.3
        total_loss = loss + legibility_weight * loss.new_tensor(legibility_penalty)

        if return_outputs:
            return total_loss, outputs
        return total_loss

    def evaluation_loop(
        self,
        dataloader,
        description,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        prediction_loss_only = False

        eval_output = super().evaluation_loop(
            dataloader,
            description,
            prediction_loss_only,
            ignore_keys,
            metric_key_prefix,
        )

        all_pred_tokens = (
            eval_output.predictions if self.vqa else eval_output.predictions[:, 77:]
        )
        all_pred = decode_generation_seqeunces(self.tokenizer, all_pred_tokens)
        all_label = decode_generation_seqeunces(self.tokenizer, eval_output.label_ids)

        if self.args.process_index != 0:
            return eval_output

        if wandb.run is None:
            self.log({"init": 0})

        images = log_txt_as_img((512, 512), [all_pred[0], all_label[0]])
        wandb.log({"val_logits": wandb.Image(np.concatenate(images, axis=1))})

        wandb.log({
            "val_results": wandb.Table(
                columns=["pred", "label"],
                data=[list(pair) for pair in zip(all_pred, all_label)],
            )
        })

        # Traffic light accuracy
        tl_accuracy = eval_tl(all_pred, all_label)
        wandb.log({"tl_accuracy": tl_accuracy})

        # Distance errors
        self._log_distance(all_pred, all_label, "tl_distance", r"It is (\d+(?:\.\d+)?)m ahead")
        self._log_distance(all_pred, all_label, "car_error", r"observing (\d+(?:\.\d+)?) cars")
        self._log_distance(all_pred, all_label, "ped_error", r"and (\d+(?:\.\d+)?) pedestrians")

        # Action error
        lon_err, lat_err = eval_action(all_pred, all_label)
        wandb.log({
            "control_error_lon": lon_err,
            "control_error_lat": lat_err,
        })

        # Legibility score
        leg_scores = []
        for p, vd, pd in zip(
            all_pred,
            self.eval_dataset["vehicle_descriptors"],
            self.eval_dataset["pedestrian_descriptors"],
        ):
            leg_scores.append(
                1.0 - compute_legibility_penalty([p], vd[None], pd[None])
            )

        wandb.log({"legibility_score": float(np.mean(leg_scores))})

        return eval_output

    def _log_distance(self, all_pred, all_label, name, pattern):
        errors = get_eval_distance_errors(all_pred, all_label, pattern)
        if len(errors) > 0:
            wandb.log({name: float(np.mean(errors))})


# ============================================================
# Training Entry Point
# ============================================================

def train(
    base_model: str = "meta-llama/Llama-2-7b-hf",
    data_path: str = "data/vqa_train_legible.pkl",
    batch_size: int = 128,
    micro_batch_size: int = 32,
    num_epochs: int = 5,
    learning_rate: float = 3e-4,
    val_set_size: int = 100,
    eval_steps: int = 50,
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: Tuple = ("q_proj", "k_proj", "v_proj", "o_proj"),
    wandb_project: str = "llm-driver-legibility",
    resume_from_checkpoint: str = "models/weights/stage1_pretrained_model/",
    augment_times: int = 0,
    output_dir: Optional[str] = None,
    vqa: bool = False,
    eval_items: List[str] = DEFAULT_EVAL_ITEMS,
    mode: str = "train",
    val_data_path: str = "data/vqa_test_legible.pkl",
):

    if output_dir is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = tempfile.mkdtemp(prefix=f"legible_lora_{ts}_")

    gradient_accumulation_steps = batch_size // micro_batch_size

    os.environ["WANDB_PROJECT"] = wandb_project

    model = load_model(
        base_model=base_model,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
        resume_from_checkpoint=resume_from_checkpoint,
    )

    model.print_trainable_parameters()
    tokenizer = load_llama_tokenizer(base_model)

    train_data, val_data = get_train_val_data(
        data_path,
        tokenizer,
        val_data_path=val_data_path,
        val_set_size=val_set_size,
        augment_times=augment_times,
        vqa=vqa,
        eval_only=mode == "eval",
        eval_items=eval_items,
    )

    trainer = TrainerWithGeneration(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.Seq2SeqTrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=eval_steps,
            save_steps=500,
            output_dir=output_dir,
            save_total_limit=3,
            report_to="wandb",
            predict_with_generate=True,
            generation_max_length=384,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, padding=True
        ),
        vqa=vqa,
    )

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))

    logging.set_verbosity_info()

    if mode == "train":
        trainer.train()
        model.save_pretrained(output_dir)
    else:
        trainer.evaluate()


if __name__ == "__main__":
    fire.Fire(train)
