from copy import deepcopy
from typing import Any, Dict, List, Tuple
from collections import deque

import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...util import nethook

from .ft_hparams import FTHyperParams

# Adjust for bloom
import torch
import torch.nn as nn
import torch.optim as optim

def apply_ft_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: FTHyperParams,
    copy=False,
    return_orig_weights=False,
    keep_original_weight=False,
    **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.
    :return: (1) the updated model, (2) the weights that changed
    """
    weights_copy = {}
    if copy:
        model = deepcopy(model)

    if hasattr(model.config, 'model_type') and 'bloom' in model.config.model_type:
        deltas = bloom_ft(model, tok, requests, hparams)
        # deltas = execute_ft(model, tok, requests, hparams)
    else:
        deltas = execute_ft(model, tok, requests, hparams)

    with torch.no_grad():
        for w_name, upd_matrix in deltas.items():
            w = nethook.get_parameter(model, w_name)
            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()

            w[...] += upd_matrix

    print(f"New weights successfully inserted into {list(deltas.keys())}")

    if not keep_original_weight:
        weights_copy = {}

    return model, weights_copy


def execute_ft(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: FTHyperParams,
    **kwargs: Any,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the FT update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """
    device = torch.device(f'cuda:{hparams.device}')
    # model = model.to(device)
    # Update target and print info
    requests = deepcopy(requests)
    for request in requests:
        if request["target_new"] != " ":
            # Space required for correct tokenization
            request["target_new"] = " " + request["target_new"]
        print(
            f"Executing FT algo for: "
            f"[{request['prompt']}] -> [{request['target_new']}]"
        )
    
    # Retrieve weights that user desires to change
    weights = {
        n: p
        for n, p in model.named_parameters()
        for layer in hparams.layers
        if hparams.rewrite_module_tmp.format(layer) in n
    }
    
    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}
    print(f"Weights to be updated: {list(weights.keys())}")

    # Define inputs
    texts = [r["prompt"] for r in requests]
    targets = [r["target_new"] for r in requests]
    
    # Configure optimizer / gradients
    opt = torch.optim.Adam(
        [v for _, v in weights.items()],
        lr=hparams.lr,
        weight_decay=hparams.weight_decay,
    )
    for name, w in model.named_parameters():
        w.requires_grad = name in weights

    # Update loop: intervene at layers simultaneously
    loss_meter = AverageMeter()
    for it in range(hparams.num_steps):
        print(20 * "=")
        print(f"Epoch: {it}")
        print(20 * "=")
        loss_meter.reset()

        for txt, tgt in zip(
            chunks(texts, hparams.batch_size), chunks(targets, hparams.batch_size)
        ):
            inputs = tok(txt, return_tensors="pt", padding=True).to(device)
            target_ids = tok(tgt, return_tensors="pt", padding=True)["input_ids"].to(
                device
            )
            if hparams.objective_optimization == 'prompt_last':
                last_token_inds = inputs["attention_mask"].sum(dim=1) - 1
                loss_mask = target_ids != tok.unk_token_id
            elif hparams.objective_optimization == 'target_new':
                inputs_targets = [txt_ + tgt_ for txt_, tgt_ in zip(txt, tgt)]
                inputs_targets = tok(inputs_targets, return_tensors="pt", padding=True).to(device)
                num_prompt_toks = [int((i != tok.pad_token_id).sum()) for i in inputs['input_ids'].cpu()]
                num_pad_toks = [int((i == tok.pad_token_id).sum()) for i in inputs_targets['input_ids'].cpu()]
                prompt_len = [x + y for x, y in zip(num_pad_toks, num_prompt_toks)]
                prompt_target_len = inputs_targets['input_ids'].size(1)
                label_mask = torch.tensor([[False] * length + [True] * (prompt_target_len - length) for length in prompt_len]).to(device)
            else:
                print(f"{hparams.objective_optimization} has not been supported yet.")
                raise NotImplementedError
            # last_token_inds = inputs["attention_mask"].sum(dim=1) - 1
            # loss_mask = inputs != tok.unk_token_id
            # loss_mask = [:, ]
            opt.zero_grad()
            bs = inputs["input_ids"].shape[0]
            if 't5' in hparams.model_name.lower():
                inputs['decoder_input_ids'] = target_ids
                logits = model(**inputs).logits
                unmasked_log_probs = logits.log_softmax(-1).gather(-1, inputs['decoder_input_ids'].unsqueeze(-1)).squeeze(-1)

                mask = inputs['decoder_input_ids'] != -100
                n_tokens = mask.float().sum()
                avg_log_prob = (unmasked_log_probs * mask.float()).sum() / n_tokens
                nll = -avg_log_prob
                loss = nll
            elif 'chatglm' in hparams.model_name.lower():
                # def get_masks(seq, bos_token_id):
                #     """  code from model_chatglm.py  """
                #     if seq.count(bos_token_id) == 2:
                #         context_length = seq[2:].index(bos_token_id) + 2
                #     else:
                #         context_length = seq.index(bos_token_id)
                #     attention_mask = torch.ones((1, len(seq), len(seq)))
                #     attention_mask.tril_()
                #     attention_mask[..., :context_length] = 1
                #     # attention_mask.unsqueeze_(1)
                #     attention_mask = (attention_mask < 0.5).bool()
                #     return attention_mask

                input_ids = inputs['input_ids'].tolist()
                labels = target_ids.tolist()
                assert len(input_ids) == len(labels)
                len_batches = [len(input_ids[i]) + len(labels[i]) + 1
                                 for i in range(len(input_ids))]
                len_max_batch = max(len_batches)
                batch_input_ids = []
                batch_attention_mask = []
                batch_labels = []
                for x, y in zip(input_ids, labels):
                    len_padding = len_max_batch - len(x) - len(y)
                    if tok.padding_side and tok.padding_side == "left":
                        batch_label = [-100] * len_padding + [-100] * len(x) + y
                        batch_input_id = [0] * (len_padding) + x + y
                    else:
                        batch_label = [-100] * len(x) + y + [-100] * len_padding
                        batch_input_id = x + y + [0] * (len_padding)

                    # tensor_attention_mask = get_masks(batch_input_id, bos_token_id=64792)
                    tensor_input_ids = torch.tensor(batch_input_id, dtype=torch.long)
                    tensor_labels = torch.tensor(batch_label, dtype=torch.long)
                    batch_input_ids.append(tensor_input_ids)
                    # batch_attention_mask.append(tensor_attention_mask)
                    batch_labels.append(tensor_labels)
                # batch_attention_mask = torch.stack(batch_attention_mask).to(device)
                batch_input_ids = torch.stack(batch_input_ids).to(device)
                batch_labels = torch.stack(batch_labels).to(device)
                # loss = model(input_ids=batch_input_ids, labels=batch_labels).loss
                lm_logits = model(input_ids=batch_input_ids)['logits']
                lm_logits = lm_logits.to(torch.float32)
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = batch_labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                loss = loss.to(lm_logits.dtype)
            else:
                if hparams.objective_optimization == 'prompt_last':
                    probs = torch.nn.functional.log_softmax(
                        model(**inputs).logits[torch.arange(bs), last_token_inds], dim=-1
                    )
                    loss = -(torch.gather(probs, 1, target_ids) * loss_mask).sum(
                        1
                    ) / loss_mask.sum(1)
                elif hparams.objective_optimization == 'target_new':
                    logits = model(**inputs_targets).logits
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = inputs_targets['input_ids'][..., 1:].contiguous()
                    loss_fct = CrossEntropyLoss(reduction='none')
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    loss = loss.view(bs, -1)
                    loss = (loss * label_mask[:,1:]).sum(1) / label_mask[:,1:].sum(1)
                    loss = loss.mean()
                else:
                    raise NotImplementedError
            print(f"Batch loss {loss.item()}")
            loss_meter.update(loss.item(), n=bs)

            if loss.item() >= 1e-2:
                loss.backward()
                opt.step()
            else:
                print("Dont update")

            if type(hparams.norm_constraint) is float:
                eps = hparams.norm_constraint
                with torch.no_grad():
                    for k, v in weights.items():
                        v[...] = torch.clamp(
                            v, min=weights_copy[k] - eps, max=weights_copy[k] + eps
                        )

        print(f"Total loss {loss_meter.avg}")

        if loss_meter.avg < 1e-2:
            print("Break FT")
            break

    deltas = {k: (weights[k] - weights_copy[k]).detach() for k in weights}

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}")

    return deltas


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    chunk = []
    for a in arr:
        chunk.append(a)
        if len(chunk) == n:
            yield chunk
            chunk = []
    if len(chunk) > 0:
        yield chunk


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# ====================== MY CODE: ======================

def bloom_ft(model, tokenizer, requests, hparams):
    # Extract the weights to be fine-tuned
    weights_to_finetune = []
    for layer_idx in hparams.layers:
        module_name = hparams.rewrite_module_tmp.format(layer_idx)
        module = model.get_submodule(module_name)
        weights_to_finetune.extend(module.parameters())

    # Create a copy of the original weights
    original_weights = [param.clone() for param in weights_to_finetune]

    # Prepare the input data
    input_ids = []
    attention_mask = []
    target_ids = []
    for request in requests:
        prompt = request["prompt"]
        target_new = request["target_new"]
        encoded_prompt = tokenizer.encode(prompt, return_tensors="pt")
        encoded_target = tokenizer.encode(target_new, return_tensors="pt")
        input_ids.append(encoded_prompt.squeeze(0))
        attention_mask.append(torch.ones_like(encoded_prompt.squeeze(0)))
        target_ids.append(encoded_target.squeeze(0))

    # Set up the optimizer
    optimizer = optim.Adam(weights_to_finetune, lr=hparams.lr, weight_decay=hparams.weight_decay)

    device = torch.device(hparams.device)
    model.to(device)

    # Fine-tuning loop
    for step in range(hparams.num_steps):
        # Shuffle the input data
        indices = torch.randperm(len(input_ids))
        input_ids = [input_ids[i] for i in indices]
        attention_mask = [attention_mask[i] for i in indices]
        target_ids = [target_ids[i] for i in indices]

        # Divide the input data into batches
        batch_indices = torch.arange(0, len(input_ids), hparams.batch_size)
        for batch_start in batch_indices:
            batch_end = min(batch_start + hparams.batch_size, len(input_ids))
            batch_input_ids = nn.utils.rnn.pad_sequence(input_ids[batch_start:batch_end], batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
            batch_attention_mask = nn.utils.rnn.pad_sequence(attention_mask[batch_start:batch_end], batch_first=True, padding_value=0).to(device)
            batch_target_ids = nn.utils.rnn.pad_sequence(target_ids[batch_start:batch_end], batch_first=True, padding_value=tokenizer.pad_token_id).to(device)

            # Forward pass
            outputs = model(batch_input_ids, attention_mask=batch_attention_mask)
            logits = outputs.logits

            # Calculate the loss based on the optimization objective
            if hparams.objective_optimization == "prompt_last":
                last_token_logits = logits[torch.arange(logits.size(0)), batch_attention_mask.sum(1) - 1, :]
                last_token_probs = torch.softmax(last_token_logits, dim=-1)
                last_token_target = batch_target_ids[torch.arange(batch_target_ids.size(0)), batch_attention_mask.sum(1) - 1]
                loss = -torch.log(last_token_probs.gather(1, last_token_target.unsqueeze(1)))
                loss = loss.masked_select(last_token_target != tokenizer.pad_token_id).sum()
            elif hparams.objective_optimization == "target_new":
                shifted_logits = logits[:, :-1, :].contiguous()
                shifted_targets = batch_target_ids[:, 1:].contiguous()
                batch_size, seq_length, vocab_size = shifted_logits.size()

                print(f"shifted_logits shape: {shifted_logits.shape}")
                print(f"shifted_targets shape: {shifted_targets.shape}")

                # Flatten the shifted_targets tensor
                shifted_targets = shifted_targets.view(-1)

                print(f"flattened shifted_targets shape: {shifted_targets.shape}")

                # Create a mask for valid target tokens
                target_mask = (shifted_targets != tokenizer.pad_token_id).float()

                print(f"target_mask shape: {target_mask.shape}")

                # Apply the mask to the shifted_targets
                masked_shifted_targets = shifted_targets * target_mask.long()

                print(f"masked_shifted_targets shape: {masked_shifted_targets.shape}")

                # Reshape the shifted_logits tensor
                shifted_logits = shifted_logits.view(-1, vocab_size)

                print(f"reshaped shifted_logits shape: {shifted_logits.shape}")

                # Apply the mask to the shifted_logits
                masked_shifted_logits = shifted_logits * target_mask.unsqueeze(-1)

                print(f"masked_shifted_logits shape: {masked_shifted_logits.shape}")

                # Calculate the loss only for valid target tokens
                loss_fct = nn.CrossEntropyLoss(reduction="sum")
                loss = loss_fct(masked_shifted_logits, masked_shifted_targets)
                num_valid_targets = target_mask.sum()
                loss = loss / num_valid_targets
            else:
                raise ValueError(f"Invalid optimization objective: {hparams.objective_optimization}")

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Apply norm constraint if specified
            if hparams.norm_constraint is not None:
                torch.nn.utils.clip_grad_norm_(weights_to_finetune, hparams.norm_constraint)

        # Print progress
        print(f"Step {step + 1}/{hparams.num_steps}, Loss: {loss.item():.4f}")

    # Compute the deltas
    deltas = {}
    for param, original_param in zip(weights_to_finetune, original_weights):
        deltas[param] = param.data - original_param.data

    # Restore the original weights
    for param, original_param in zip(weights_to_finetune, original_weights):
        param.data.copy_(original_param.data)

    return deltas