import random
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
    
    def __init__(self, dico: list, N: int):
        self.N = N
        self.secretWord = random.choice([w for w in dico if len(w) == N])
    
    """
    Backtracking method consists to initialize each variables with potentials values
    then check if the current solution is cosistent, if not, then remove values from the domain of variables
    which are not consistent, until finding solution.
    """
    def backTracking():
        pass
    
    """
    Backtracking with Forward Checking (FC) method is based on Backtracking method, but
    it also check the avalable words in the dictionnary of words if words are consistent with
    solution or not, if not, then remove them, and repeat until finding solution.
    """
    def backTrackingWithFC():
        pass
    
    def solve():
        pass


"""
Solver based on Genetic Algorithm
this class takes below inputs :
    - dico : list
        list of all available words
    - N : int
        size of secret word which the program should find
"""
class GeneticAlgorithm():
    
    def __init__(self, dico: list, N: int):
        self.N = N
        self.secretWord = random.choice([w for w in dico if len(w) == N])

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

############ main ###############
if __name__ == '__main__':
    file_path = "./dico.txt"
    dico = read_dico(file_path)
    csp = CSP(dico, 4)
    ga = GeneticAlgorithm(dico, 4)
    print(csp.secretWord)
    print(ga.secretWord)
    # solve()
    