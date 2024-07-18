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

def logistic_decoding_gd(model, prompt, class_names, temp = 0.8):
    with suppress_stdout():
        regex = "("+class_names[0]
        for class_name in class_names[1:]:
            regex= regex+"|"+class_name
        regex+=")$"
        ans = model+prompt+ gen(regex = '(1|2|3|4|5)$', temperature = temp, name = 'text')
        return int(ans['text'])