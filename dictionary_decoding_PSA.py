import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import math
from tqdm import tqdm

class DictionaryGenerator:
    def __init__(self, keys, value_range, delta, nb):
        self.keys = keys
        self.value_range = value_range
        self.delta = delta
        self.nb = nb

    def create_dictionary(self):
        return {key: random.randint(*self.value_range) for key in self.keys}

    def dictionary_to_string(self, dictionary):
        return '\n'.join([f'{key}: {value}' for key, value in dictionary.items()])

    def mutate_dictionary(self, dictionary):
        key = random.choice(list(dictionary.keys()))
        cs = [i*self.delta for i in range(-self.nb,self.nb+1)]
        new_value = dictionary[key] + random.choice(cs)
        # Truncate the new value to the specified range
        new_value = max(self.value_range[0], min(new_value, self.value_range[1]))
        dictionary[key] = new_value
        return dictionary


def calculate_log_probabilities_batched(model, tokenizer, prompt, dictionaries):
    full_texts = [prompt + dictionary for dictionary in dictionaries]
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

    for i, dictionary in enumerate(dictionaries):
        string_ids = tokenizer.encode(dictionary, add_special_tokens=False)
        # Adjust the start index for each sequence individually to account for padding
        start_index = (attention_mask[i] == 1).nonzero(as_tuple=True)[0][prompt_length]

        class_log_prob = torch.tensor(0.0, dtype=torch.float64)
        for j, token_id in enumerate(string_ids):
            class_log_prob += softmax[i, start_index + j, token_id].item()

        log_probs.append(class_log_prob.item())

    return log_probs

def simulated_annealing_batched(model, tokenizer, prompt, sample_generator, B=10, initial_temp=100, min_temp=1, alpha=0.99, max_iter=1000):
    chains = [sample_generator.create_dictionary() for _ in range(B)]
    chain_strings = [sample_generator.dictionary_to_string(chain) for chain in chains]
    chain_probs = calculate_log_probabilities_batched(model, tokenizer, prompt, chain_strings)
    
    best_dicts = chains[:]
    best_probs = chain_probs[:]
    temp = initial_temp

    for i in tqdm(range(max_iter)):
        new_chains = [sample_generator.mutate_dictionary(chain.copy()) for chain in chains]
        new_chain_strings = [sample_generator.dictionary_to_string(chain) for chain in new_chains]
        new_chain_probs = calculate_log_probabilities_batched(model, tokenizer, prompt, new_chain_strings)
        
        for j in range(B):
            delta_prob = new_chain_probs[j] - chain_probs[j]
            if delta_prob > 0 or math.exp(delta_prob / temp) > random.random():
                chains[j] = new_chains[j]
                chain_probs[j] = new_chain_probs[j]
                if new_chain_probs[j] > best_probs[j]:
                    best_dicts[j] = new_chains[j]
                    best_probs[j] = new_chain_probs[j]

        temp *= alpha
        if temp < min_temp:
            break

    return best_dicts, best_probs

def prompt_conditoned_dict_PSA(model, tokenizer, prompt, keys, rang, delta, nb, B=10):
    dict_gen = DictionaryGenerator(keys, rang, delta, nb)
    best_dicts, best_probs = simulated_annealing_batched(model, tokenizer, prompt, dict_gen, B=B)
    return best_dicts, best_probs