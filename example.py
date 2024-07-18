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


#creating dictionary using Simulated Anneling 
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
    #delta is step size
    #nb is neighbourhood size in the search
    #B is the batch size
    #alg "PSA" for batched and "SA" for non batched
    best_dict, best_prob = prompt2dict(model, tokenizer, prompt, keys, value_range, {'delta': 0.2, 'nb': 3, 'B':10}, alg = alg)
    print("Best Dictionary Found:")
    print(best_dict)
    print("Log Probability:")
    print(best_prob)

# creates dictionary using the guidance library faster than SA
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
    # max token to control floating point precision
    output = prompt_conditoned_dict_GD(model, prompt, key_ranges, max_tokens=5, temp=0.7)    
    print(f'Output of GD: {output}')

#selects a given choice conditioned on prompt using my mehtod faster than guidance library comes with probabilities
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
    output = logistic_decoding_batched(model, tokenizer, prompt, class_names, batch_size=4)
    print(f'Output of Onepass Batched = 4: {output}')

#selects a given choice conditioned on prompt using my mehtod faster than guidance library comes with probabilities
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
    # use temp to make system stochastic 
    output = logistic_decoding_gd(model, prompt, class_names, temp=0.7)
    print(f'Output of GD: {output}')



if __name__ == "__main__":
    '''
    #You might need this if you are using models that needs permission
    login(token="YOUR HF TOKEN")
    '''
    
    #use this if you do not have gpus
    model_name = "gpt2"
    test_logistic_gd(model_name)
    test_logistic(model_name)
    test_dict_gd(model_name)
    test_dict(model_name)

    #use this if you have gpus
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    test_logistic_gd(model_name, nf4_config)
    test_logistic(model_name, nf4_config)
    test_dict_gd(model_name, nf4_config)
    test_dict(model_name,nf4_config)