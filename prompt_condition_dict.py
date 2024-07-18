from dictionary_decoding_SA import prompt_conditoned_dict_SA
from dictionary_decoding_PSA import prompt_conditoned_dict_PSA


def prompt2dict(model, tokenizer, prompt, keys, rang, arg , alg = "SA"):
    assert alg in ["SA","PSA"]
    if alg == "SA":
        assert 'delta' in arg
        assert 'nb' in arg
        return prompt_conditoned_dict_SA(model, tokenizer, prompt, keys, rang, arg['delta'], arg['nb'])
    elif alg == "PSA":
        assert 'delta' in arg
        assert 'nb' in arg
        assert 'B' in arg
        return prompt_conditoned_dict_PSA(model, tokenizer, prompt, keys, rang, arg['delta'], arg['nb'], B = arg['B'])