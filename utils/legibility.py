import torch
import torch.nn.functional as F

def compute_legibility_loss(
    logits,
    tokenizer,
    ego_intents,
    observer_mask,
):
    """
    logits: [B, T, V]
    ego_intents: List[str]
    observer_mask: [B, num_agents] (1 if agent present)
    """

    device = logits.device
    loss = 0.0
    count = 0

    for b in range(len(ego_intents)):
        intent_tokens = tokenizer.encode(
            ego_intents[b], add_special_tokens=False
        )
        if len(intent_tokens) == 0:
            continue

        # use final token logits as inference signal
        final_logits = logits[b, -1]

        log_probs = F.log_softmax(final_logits, dim=-1)

        intent_logprob = sum(log_probs[t] for t in intent_tokens)

        num_obs = observer_mask[b].sum().clamp(min=1)

        loss += -intent_logprob * num_obs
        count += num_obs

    return loss / count
