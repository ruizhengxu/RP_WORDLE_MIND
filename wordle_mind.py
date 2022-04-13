import random, string, time, copy
from difflib import get_close_matches
import matplotlib.pyplot as plt
import numpy as np

###################################################
#
# Solver with CSP
#
###################################################

"""
Solver based on methods of Constraint Satisfaction Problem
this class takes below inputs :
    - dico : list
        list of all available words
    - secretWord : str
"""
class CSP:
    
    def __init__(self, dico: list, secretWord: str):
        self.dico = dico
        self.secretWord = secretWord
        self.vars = dict.fromkeys([i for i in range(len(self.secretWord))])
        self.domains = [list(string.ascii_lowercase) for _ in range(len(self.secretWord))]
        self.n = 0 # Number of tested word
    
    ##########################################################################
    
    """
    Backtracking method consists to initialize each variables with potentials values
    then check if the current solution is cosistent, if not, it intialize with next values and so on,
    until find the solution.
    """
    def backTracking(self, verbose=False):
        I = self.vars.copy()
        V = self.vars.copy()
        D = self.domains.copy()
        if self.n != 0: self.n = 0
        
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
                        if instance is not None:
                            self.n += 1
                            if self.toWord(instance) == self.secretWord:
                                return instance

        instance = solve(I, V, D)
        if verbose: print("Number of words tested :", self.n)
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
                
    ##########################################################################            
    
    """
    Forward Checking (FC) method consists to reduce domain of each variables until find the solution.
    """
    def forwardChecking(self, verbose=False):
        I = self.vars.copy()
        V = self.vars.copy()
        D = self.domains.copy()
        if self.n != 0: self.n = 0
        
        def solve(V: dict, I: dict, D: dict):
            if len(V) == 0:
                return I
            else:
                x = next(iter(V)) # choose a variable (0, 1, 2, ...)
                for v_x in D[x]: # for all values of domain of x ('a', 'b', 'c', ...)
                    # remove variable x from V
                    value = V.pop(x)
                    tmp_D = copy.deepcopy(D)
                    if self.check_forward(x, v_x, V, tmp_D):
                        tmp_I = I.copy()
                        tmp_I[x] = v_x
                        instance = solve(V, tmp_I, tmp_D)
                        if instance is not None:
                            self.n += 1
                            if self.toWord(instance) == self.secretWord:
                                return instance
                    # restore variable x from V
                    V[x] = value
        
        instance = solve(V, I, D)
        if verbose: print("Number of words tested :", self.n)
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
            if len(tmp_D[x_j]) == 0: consistant = False
        return consistant
                
    def in_dict(self, x_k, x_j, v, v_):
        # print(x_k, x_j, v, v_)
        reduced_dico = [w for w in self.dico if len(w) == len(self.secretWord)]
        for word in reduced_dico:
            if word[x_k] == v and word[x_j] == v_:
                return True
        return False    
    
    ##########################################################################
    
    def toWord(self, instance: dict):
        return ''.join([c for c in instance.values()])



###################################################
#
# Solver with Genetic Algorithm
#
###################################################

"""
Solver based on Genetic Algorithm
this class takes below inputs :
    - dico : list
        list of all available words
    - secretWord : str
    - max_size : int
        maximum size of E (set of compatible words), the algo stops when the size
        of E reaches max_size
    - max_gen : int
        maximum of generation of the algorithm, the algo stops when the number
        of generation reaches max_gen
"""
class GeneticAlgorithm:
    
    def __init__(self, dico: list, secretWord: str, max_size: int, max_gen: int, popsize: int):
        self.dico = [w for w in dico if len(w) == len(secretWord)]
        self.secretWord = secretWord
        self.max_size = max_size
        self.max_gen = max_gen
        self.popsize = popsize
    
    ##########################################################################    
    
    def selection(self, population: list, fitness: list):
        tmp_fitness = copy.deepcopy(fitness)
        tmp_population = copy.deepcopy(population)
        
        pos = self.bestInd(tmp_fitness)
        ind1 = tmp_population[pos]
        tmp_fitness.pop(pos)
        tmp_population.pop(pos)
        ind2 = tmp_population[self.bestInd(tmp_fitness)]
        
        return ind1, ind2
    
    def crossover(self, ind1: str, ind2: str):
        child1 = ""
        child2 = ""
        
        for i in range(len(ind1)):
            pb = random.random()
            if pb > 0.5:
                child1 += ind1[i]
                child2 += ind2[i]
            else:
                child1 += ind2[i]
                child2 += ind1[i]
        
        child1 = self.getClosestWord(child1)
        child2 = self.getClosestWord(child2)
        
        return child1, child2
    
    def mutation(self, ind):
        fit = self.evaluate(ind)
        if len(self.secretWord) - fit[0] == 1: # replace
            tmp_char = random.choice(list(string.ascii_lowercase))
            pos = random.randint(0, len(ind))
            ind = ind[:pos] + tmp_char + ind[pos+1:]
        else: # swap
            pos = random.choices([i for i in range(len(ind))], k=2)
            tmp_ind = list(ind)
            tmp_ind[pos[0]], tmp_ind[pos[1]] = tmp_ind[pos[1]], tmp_ind[pos[0]]
            ind = "".join(tmp_ind)
        
        return self.getClosestWord(ind)

    def isCompatible(self, ind, fitness):
        f = self.evaluate(ind)
        
        for fit in fitness:
            if f[0] <= fit[0] and f[1] <= fit[1]:
                return False
        
        return True
    
    def bestInd(self, fitness):
        pos = 0
        bestFit = fitness[pos][0] * 2 + fitness[pos][1]
        
        for i in range(1, len(fitness)):
            currentFit = fitness[i][0] * 2 + fitness[i][1]
            if currentFit > bestFit:
                bestFit = currentFit
                pos = i
                
        return pos
    
    def getClosestWord(self, word):
        matched_words = get_close_matches(word, self.dico)
        if len(matched_words) == 0:
            return ""
        return matched_words[self.bestInd([self.evaluate(w) for w in matched_words])]
    
    """
    Evaluate return fitness for the word which we want to evaluate.
    Returned fitness contains two numbers (number of well placed words, number of misplaced words)
    
    """
    def evaluate(self, word: str):
        well_placed = 0
        misplaced = 0
        tmp_word = word
        tmp_secretWord = self.secretWord

        # Count well placed words
        i = 0
        while i < len(tmp_word):
            if tmp_secretWord[i] != tmp_word[i]:
                i += 1
            else:
                well_placed += 1
                tmp_secretWord = tmp_secretWord[:i] + tmp_secretWord[i+1:]
                tmp_word = tmp_word[:i] + tmp_word[i+1:]
        
        # Count misplaced words
        for letter in tmp_word:
            if letter in tmp_secretWord:
                misplaced += 1
                tmp_secretWord = tmp_secretWord.replace(letter, "")
        
        return [well_placed, misplaced]
    
    def solve(self, verbose=False):
        
        # randomly initialize a set of words
        population = random.choices([w for w in dico if len(w) == len(self.secretWord)], k=self.popsize)
        fitness = [self.evaluate(ind) for ind in population]
        gen = 0
        
        pos = self.bestInd(fitness)
        bestFit = fitness[pos]
        
        # if the secret word is already in first population
        if population[pos] == self.secretWord:
            print("Gen :", gen)
            print("All population :", population)
            print("All fitness :", fitness)
            return population[pos], bestFit
        
        for _ in range(self.max_gen):
            gen += 1
            new_population = []
            
            for _ in range(int(self.popsize/2)):
                ind1, ind2 = self.selection(population, fitness)
                child1, child2 = self.crossover(ind1, ind2)
                if child1 != "":
                    child1 = self.mutation(child1)
                    if child1 not in new_population and self.isCompatible(child1, fitness):
                        new_population.append(child1)
                if child2 != "":
                    child2 = self.mutation(child2)
                    if child2 not in new_population and self.isCompatible(child2, fitness):
                        new_population.append(child2)
                        
            if len(new_population) > 0:
                population += new_population.copy()
                fitness += [self.evaluate(ind) for ind in new_population]
                i = self.bestInd(fitness)
                if fitness[i] > bestFit:
                    pos = i
                    bestFit = fitness[i]
            
            if population[pos] == self.secretWord:
                print("Gen :", gen)
                print("All population :", population)
                print("All fitness :", fitness)
                return population[pos], bestFit
            
            if len(population) == self.max_size:
                print("Max size of population reached.")
                print("Gen :", gen)
                print("All population :", population)
                print("All fitness :", fitness)
                return population[pos], bestFit
        
        print("Max gen reached.")
        print("All population :", population)
        print("All fitness :", fitness)
        return population[pos], bestFit

    

###################################################
#
# MAIN PROGRAMM
#
###################################################

def read_dico(path: str):
    with open(path) as f:
        dico = f.read().splitlines()
    return dico

def choose_secretWord(dico, N):
    return random.choice([w for w in dico if len(w) == N])

if __name__ == '__main__':
    # Initialize
    file_path = "./dico.txt"
    dico = read_dico(file_path)
    # length of secretWord = [2, 22] !!!
    secretWord = choose_secretWord(dico, 5)
    print("Secret word is :", secretWord)

    # Solve with CSP
    # csp = CSP(dico, secretWord)
    
    # print("Running back tracking method : ....")
    # start = time.time()
    # w = csp.backTracking(verbose=True)
    # stop = time.time()
    # print("Word found with BackTracking :", w, "in", stop-start ,"seconds")
    
    # print("\nRunning forward checking method : ....")
    # start = time.time()
    # w = csp.forwardChecking(verbose=True)
    # stop = time.time()
    # print("Word found with Forward checking :", w, "in", stop-start ,"seconds")
    
    # Solve with Genetic Algorithm
    ga = GeneticAlgorithm(dico, secretWord, max_size=100, max_gen=100, popsize=10)
    
    print("\nRunning genetic algorithm : ....")
    start = time.time()
    w, fit = ga.solve(verbose=True)
    stop = time.time()
    print("Word found with Forward checking :", w, "in", stop-start ,"seconds")
    