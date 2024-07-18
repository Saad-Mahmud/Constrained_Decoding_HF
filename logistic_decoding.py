import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

def _logistic_decoding_onepass_batched(model, tokenizer, prompt, class_names):
    full_texts = [prompt + class_name for class_name in class_names]
    tokenizer.pad_token = tokenizer.eos_token  # Set the padding token
    inputs = tokenizer(full_texts, return_tensors='pt', padding=True, truncation=True)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    prompt_length = len(tokenizer.encode(prompt, add_special_tokens=False))

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    softmax = torch.nn.functional.log_softmax(logits, dim=-1)
    log_probs = []

    for i, class_name in enumerate(class_names):
        string_ids = tokenizer.encode(class_name, add_special_tokens=False)
        # Adjust the start index for each sequence individually to account for padding
        start_index = (attention_mask[i] == 1).nonzero(as_tuple=True)[0][prompt_length]

        class_log_prob = torch.tensor(0.0, dtype=torch.float64)
        for j, token_id in enumerate(string_ids):
            class_log_prob += softmax[i, start_index + j, token_id].item()

        log_probs.append(class_log_prob.item())

    return log_probs


def _logistic_decoding_onepass(model, tokenizer, prompt, class_name):
    full_text = prompt + class_name
    input_ids = tokenizer.encode(full_text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    softmax = torch.nn.functional.log_softmax(logits, dim=-1)
    string_ids = tokenizer.encode(class_name, add_special_tokens=False)
    start_index = len(tokenizer.encode(prompt, add_special_tokens=False))
    log_probs = torch.tensor(0.0, dtype=torch.float64)
    for i, token_id in enumerate(string_ids):
        log_probs += softmax[0, start_index + i, token_id].item()

    return log_probs.item()


def _logistic_decoding_multipass(model, tokenizer, prompt, class_name):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    string_ids = tokenizer.encode(class_name, add_special_tokens=False)
    log_probs = torch.tensor(0.0, dtype=torch.float64)
    for token_id in string_ids:
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits
        softmax = torch.nn.functional.log_softmax(logits, dim=-1)
        token_log_prob = softmax[0, -1, token_id].item()
        log_probs += token_log_prob
        
        # Update input_ids with the current token
        input_ids = torch.cat([input_ids, torch.tensor([[token_id]])], dim=1)

    return log_probs

def logistic_decoding(model, tokenizer, prompt, class_names, alg="onepass"):
    assert alg in ["onepass", "multipass"]
    log_probs = []
    for class_name in class_names:
        if alg == 'onepass':
            log_probs.append(_logistic_decoding_onepass(model, tokenizer, prompt, class_name))
        else:
            log_probs.append(_logistic_decoding_multipass(model, tokenizer, prompt, class_name))
    probs = np.exp(log_probs)
    probs = (probs) / (sum(probs))
    return probs


def logistic_decoding_batched(model, tokenizer, prompt, class_names, batch_size=2):
    log_probs = []

    for i in range(0, len(class_names), batch_size):
        batch_class_names = class_names[i:i + batch_size]
        log_probs.extend(_logistic_decoding_onepass_batched(model, tokenizer, prompt, batch_class_names))
        
    probs = np.exp(log_probs)
    probs = probs / np.sum(probs)
    return probs
