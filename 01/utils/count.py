from .function import add

def count_word(str, char):
    n = 0
    for i in str:
        if i == char:
            n = add(n, 1)
    return n