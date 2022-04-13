import random
import string
import time
import copy
import matplotlib.pyplot as plt

"""
Solver based on methods of Constraint Satisfaction Problem
this class takes below inputs :
    - dico : list
        list of all available words
    - N : int
        size of secret word which the program should find
"""


class CSP():

    def __init__(self, dico: list, secretWord: str):
        self.dico = dico
        self.secretWord = secretWord
        self.vars = dict.fromkeys([i for i in range(len(self.secretWord))])
        self.domains = [list(string.ascii_lowercase)
                        for _ in range(len(self.secretWord))]
        self.n = 0  # Number of tested word

    """
    Backtracking method consists to initialize each variables with potentials values
    then check if the current solution is cosistent, if not, it intialize with next values and so on,
    until find the solution.
    """

    def backTracking(self, verbose=False):
        I = self.vars.copy()
        V = self.vars.copy()
        D = self.domains.copy()
        if self.n != 0:
            self.n = 0

        def solve(I: dict, V: dict, D: list):
            if len(V) == 0:
                return I
            else:
                x = next(iter(V))  # choose a variable (0, 1, 2, ...)
                # for all values of domain of x ('a', 'b', 'c', ...)
                for v_x in D[x]:
                    tmp_I = I.copy()
                    tmp_I[x] = v_x
                    if self.local_consistent(tmp_I):
                        tmp_V = V.copy()
                        tmp_V.pop(x)
                        instance = solve(tmp_I, tmp_V, D)
                        if instance is not None:
                            self.n += 1
                            if self.toWord(instance) == self.secretWord:
                                return instance

        instance = solve(I, V, D)
        if verbose:
            print("Number of words tested :", self.n)
        return self.toWord(instance)

    """
    Check if instance I "could" be a potential solution
    Example : if the word "ye" is in dictionnary, then local_consistent({0: "y", 1: None}) return True
              if "ze" not in dictionnary, then local_consistent({0: "z", 1: None}) return False
    """

    def local_consistent(self, I: dict):
        reduced_dico = [w for w in self.dico if len(w) == len(I)]
        for word in reduced_dico:
            consistent = True
            for i, letter in enumerate(word):
                if I[i] is not None and I[i] != letter:
                    consistent = False
            if consistent:
                return True
        return False

    """
    Forward Checking (FC) method consists to reduce domain of each variables until find the solution.
    """

    def forwardChecking(self, verbose=False):
        I = self.vars.copy()
        V = self.vars.copy()
        D = self.domains.copy()
        if self.n != 0:
            self.n = 0

        def solve(V: dict, I: dict, D: dict):
            if len(V) == 0:
                if self.toWord(I) in self.dico:
                    self.doesExist = True
                else:
                    self.doesExist = False
                return I
            else:
                x = next(iter(V))  # choose a variable (0, 1, 2, ...)
                # for all values of domain of x ('a', 'b', 'c', ...)
                for v_x in D[x]:
                    # remove variable x from V
                    value = V.pop(x)
                    tmp_D = copy.deepcopy(D)
                    if self.check_forward(x, v_x, V, tmp_D):
                        tmp_I = I.copy()
                        tmp_I[x] = v_x
                        instance = solve(V, tmp_I, tmp_D)
                        if instance is not None and self.doesExist:
                            self.n += 1
                            if self.toWord(instance) == self.secretWord:
                                return instance
                    # restore variable x from V
                    V[x] = value

        instance = solve(V, I, D)
        if verbose:
            print("Number of words tested :", self.n)
        return self.toWord(instance)

    def check_forward(self, x_k: int, v: str, V: dict, tmp_D: dict):

        consistant = True

        for x_j in V:
            if not consistant:
                break
            D_j = tmp_D[x_j].copy()
            for v_ in D_j:
                # check if x
                if not self.in_dict(x_k, x_j, v, v_):
                    tmp_D[x_j].remove(v_)
            if len(tmp_D[x_j]) == 0:
                consistant = False
        return consistant

    def in_dict(self, x_k, x_j, v, v_):
        # print(x_k, x_j, v, v_)
        reduced_dico = [w for w in self.dico if len(w) == len(self.secretWord)]
        for word in reduced_dico:
            if word[x_k] == v and word[x_j] == v_:
                return True
        return False

    def eval():
        pass

    def toWord(self, instance: dict):
        return ''.join([c for c in instance.values()])


"""
Solver based on Genetic Algorithm
this class takes below inputs :
    - dico : list
        list of all available words
    - N : int
        size of secret word which the program should find
"""


class GeneticAlgorithm():

    def __init__(self, dico: list, secretWord: str):
        self.secretWord = secretWord
        self.vars = ["" for _ in range(len(self.secretWord))]

    def selection():
        pass

    def croissing():
        pass

    def mutation():
        pass

    def getNeighbours():
        pass

    def evaluate():
        pass

    def solve():
        pass


#################################
def read_dico(path: str):
    with open(path) as f:
        dico = f.read().splitlines()
    return dico

#################################


def choose_secretWord(dico, N):
    return random.choice([w for w in dico if len(w) == N])





############ main ###############
if __name__ == '__main__':
    # Initialize
    file_path = "./dico.txt"
    dico = read_dico(file_path)
    secretWord = choose_secretWord(dico, 4)
    print("Secret word is :", secretWord)

    ## Solve
    csp = CSP(dico, secretWord)
    print("Running back tracking method : ....")
    start = time.time()
    w = csp.backTracking(verbose=True)
    stop = time.time()
    print("Word found with BackTracking :", w, "in ", stop-start, "seconds")

    print("\nRunning back tracking method with Forward checking : ....")
    start = time.time()
    w = csp.forwardChecking(verbose=True)
    stop = time.time()
    print("Word found with Forward checking :",
          w, "in ", stop-start, "seconds")
