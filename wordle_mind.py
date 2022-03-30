import numpy as np

def read_dico(path: str):
    with open(path) as f:
        dico = f.read().splitlines()
    return dico

if __name__ == '__main__':
    file_path = "./dico.txt"
    dico = read_dico(file_path)
    print(dico)
    # solve()
    