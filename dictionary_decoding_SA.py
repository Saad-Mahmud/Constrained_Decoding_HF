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


def calculate_log_probability(model, tokenizer, prompt, dictionary_string):
    full_text = prompt + dictionary_string
    input_ids = tokenizer(full_text, return_tensors='pt').input_ids
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    softmax = torch.nn.functional.log_softmax(logits, dim=-1)
    string_ids = tokenizer.encode(dictionary_string, add_special_tokens=False)
    prompt_length = len(tokenizer.encode(prompt, add_special_tokens=False))

    log_probs = torch.tensor(0.0, dtype=torch.float64)
    for i, token_id in enumerate(string_ids):
        log_probs += softmax[0, prompt_length + i, token_id].item()

    return log_probs.item()


def simulated_annealing(model, tokenizer, prompt, sample_generator, initial_temp=100, min_temp=1, alpha=0.99, max_iter=1000):
    current_dict = sample_generator.create_dictionary()
    current_string = sample_generator.dictionary_to_string(current_dict)
    current_prob = calculate_log_probability(model, tokenizer, prompt, current_string)
    best_dict = current_dict
    best_prob = current_prob
    temp = initial_temp

    for i in tqdm(range(max_iter)):
        new_dict = sample_generator.mutate_dictionary(current_dict.copy())
        new_string = sample_generator.dictionary_to_string(new_dict)
        new_prob = calculate_log_probability(model, tokenizer, prompt, new_string)
        delta_prob = new_prob - current_prob

        if delta_prob > 0 or math.exp(delta_prob / temp) > random.random():
            current_dict = new_dict
            current_prob = new_prob
            if new_prob > best_prob:
                best_dict = new_dict
                best_prob = new_prob

        temp *= alpha
        if temp < min_temp:
            break

    return best_dict, best_prob

def prompt_conditoned_dict_SA(model, tokenizer, prompt, keys, rang, delta, nb):
    dict_gen = DictionaryGenerator(keys,rang,delta, nb)
    best_dict, best_prob = simulated_annealing(model, tokenizer, prompt, dict_gen)
    return best_dict, best_prob