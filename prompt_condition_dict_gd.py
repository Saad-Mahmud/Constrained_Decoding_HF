from guidance import gen
import sys,os
import contextlib

@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def generate_floating_point_dict(prompt, key_ranges, max_tokens = 5,  temp=0.8):
    key_gens = []
    for key, (low, high) in key_ranges.items():
        if low < 0 and high < 0:
            regex = f"(-[0-9]+(\\.[0-9]+)?)"
        elif low < 0 and high >= 0:
            regex = f"(-?[0-9]+(\\.[0-9]+)?)"
        else:
            regex = f"([0-9]+(\\.[0-9]+)?)"
        key_gens.append(f'"{key}": {gen(key, regex=regex, max_tokens=max_tokens, temperature = temp)}, ')
    key_gens[-1] =key_gens[-1][:-2] 
    keys_string = "".join(key_gens)
    
    prompt= prompt+f"""\
    {{{keys_string}}}"""
    return prompt



def prompt_conditoned_dict_GD(model, prompt, dict, max_tokens = 5, temp = 0.8):
    prompt = generate_floating_point_dict(prompt, dict, max_tokens, temp)
    with suppress_stdout():
        ans = model+prompt
        ret = {key: ans[key] for key in dict.keys()}
        return ret