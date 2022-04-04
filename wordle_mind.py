import random, string
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
        self.domains = [list(string.ascii_lowercase) for _ in range(len(self.secretWord))]
    
    """
    Backtracking method consists to initialize each variables with potentials values
    then check if the current solution is cosistent, if not, it intialize with next values and so on,
    until find the solution.
    """
    def backTracking(self):
        I = self.vars.copy()
        V = self.vars.copy()
        D = self.domains.copy()
        print([w for w in self.dico if len(w) == len(I)])
        
        def solve(I: dict, V: dict, D: list):
            if len(V) == 0:
                return I
            else:
                x = next(iter(V)) # choose a variable (0, 1, 2, ...)
                for v_x in D[x]: # for all values of domain of x ('a', 'b', 'c', ...)
                    tmp_I = I.copy()
                    tmp_I[x] = v_x
                    if self.local_consistent(tmp_I):
                        tmp_V = V.copy()
                        tmp_V.pop(x)
                        instance = solve(tmp_I, tmp_V, D)
                        if instance is not None and self.toWord(instance) == secretWord:
                            return instance

        instance = solve(I, V, D)
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
    Backtracking with Forward Checking (FC) method is based on Backtracking method, but
    it also check the avalable words in the dictionnary of words if words are consistent with
    solution or not, if not, then remove them, and repeat until finding solution.
    """
    def backTrackingWithFC():
        pass
    
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
    
    # Solve
    csp = CSP(dico, secretWord)
    w = csp.backTracking()
    print("Secret word was :", secretWord)
    print("Word found with BackTracking :", w)
    