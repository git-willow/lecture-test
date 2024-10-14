def count_word(str, char):
    int n = 0
    for i in str:
        if i == char:
            n += 1
    return n