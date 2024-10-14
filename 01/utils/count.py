from .function import add

def count_word(s, c):
    assert isinstance(s, str)
    assert isinstance(c, str) and len(c) == 1
    n = 0
    for i in s:
        if i == c:
            n = add(n, 1)
    return n