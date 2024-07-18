from logistic_decoding import logistic_decoding, logistic_decoding_batched
from tqdm import tqdm
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import time
from prompt_condition_dict import prompt2dict
from guidance import models
from logistic_decoding_gd import logistic_decoding_gd
from prompt_condition_dict_gd import prompt_conditoned_dict_GD
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def test_dict(model_name, q_config = None, alg = "SA"):
    if q_config is not None:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True, quantization_config=q_config)
        model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=True, quantization_config=q_config)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=True)

    keys = ["key1", "key2", "key3", "key4"]
    value_range = (1, 1000)
    prompt = "Generate a dictionary of numbers:\n"
    best_dict, best_prob = prompt2dict(model, tokenizer, prompt, keys, value_range, {'delta': 0.2, 'nb': 3, 'B':10}, alg = alg)
    print("Best Dictionary Found:")
    print(best_dict)
    print("Log Probability:")
    print(best_prob)



def test_logistic(model_name, q_config = None):
    if q_config is not None:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True, quantization_config=q_config)
        model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=True, quantization_config=q_config)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=True)
    prompt = "Once upon a time"
    class_names = [
        " there was a",
        " in a faraway land",
        " a little boy",
        " who loved adventures"
    ]
    
    start_time = time.time()
    output1 = logistic_decoding(model, tokenizer, prompt, class_names, "onepass")
    onepass = time.time() - start_time

    
    start_time = time.time()
    output2 = logistic_decoding(model, tokenizer, prompt, class_names, "multipass")
    multipass = time.time() - start_time
    
    
    start_time = time.time()
    output3 = logistic_decoding_batched(model, tokenizer, prompt, class_names, batch_size=2)
    b2 = time.time() - start_time

    start_time = time.time()
    output4 = logistic_decoding_batched(model, tokenizer, prompt, class_names, batch_size=4)
    b4 = time.time() - start_time

    print(f'Output of Onepass: {output1}, and it took: {onepass} second.')
    print(f'Output of Multipass: {output2}, and it took: {multipass} second')
    print(f'Output of Onepass Batched = 2: {output3}, and it took: {b2} second.')
    print(f'Output of Onepass Batched = 4: {output4}, and it took: {b4} second')


def test_logistic_gd(model_name, q_config = None):
    
    prompt = "Once upon a time"
    class_names = [
        " there was a",
        " in a faraway land",
        " a little boy",
        " who loved adventures"
    ]
    if q_config is not None:
        model = models.Transformers(model_name, quantization_config = q_config)
    else:
        model = models.Transformers(model_name)
    start_time = time.time()
    output = logistic_decoding_gd(model, prompt, class_names, temp=0.7)
    tm = time.time() - start_time

    
    print(f'Output of GD: {output}, and it took: {tm} second')

def test_dict_gd(model_name, q_config = None):
    
    key_ranges = {
    '1st number': (100.0, 200.0),
    '2nd number': (100.0, 200.0),
    '3rd number': (100.0, 200.0)
    }
    prompt= "Generate 3 random floating point numbers between 100.0 to 200.0: "
    if q_config is not None:
        model = models.Transformers(model_name, quantization_config = q_config)
    else:
        model = models.Transformers(model_name)
    start_time = time.time()
    output = prompt_conditoned_dict_GD(model, prompt, key_ranges, max_tokens=5, temp=0.7)
    tm = time.time() - start_time

    
    print(f'Output of GD: {output}, and it took: {tm} second')


if __name__ == "__main__":
    '''
    #if you have GPUs use quantization
    login(token="YOUR HF TOKEN")
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True, quantization_config=nf4_config)
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=True, quantization_config=nf4_config)
    '''
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=True)
    
    #test_logistic(model, tokenizer)
    #test_dict(model, tokenizer,"PSA")
    
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    #test_logistic_gd(model_name, nf4_config)
    #test_logistic(model_name, nf4_config)
    test_dict_gd(model_name, nf4_config)