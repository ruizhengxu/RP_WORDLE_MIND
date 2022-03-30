import numpy as np

def read_dico(path: str):
    dico = open(path, "r")
    return dico.read()

if __name__ == '__main__':
    file_path = "./dico.txt"
    dico = read_dico(file_path)
    print(type(dico))
    # solve()
    