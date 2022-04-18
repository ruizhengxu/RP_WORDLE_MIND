import random
import string
import time
import copy
import csv
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
        self.domains = [list(string.ascii_lowercase)
                        for _ in range(len(self.secretWord))]
        self.n = 0  # Number of tested word

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
        self.reduced_dico = [w for w in self.dico if len(w) == len(I)]
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
        for word in self.reduced_dico:
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
        self.reduced_dico = [w for w in self.dico if len(w) == len(self.secretWord)]
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
        
        for word in self.reduced_dico:
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

    def __init__(self, dico: list, secretWord: str, max_size: int, max_gen: int, popsize: int, CXPB: float, MUTPB: float):
        self.dico = [w for w in dico if len(w) == len(secretWord)]
        self.secretWord = secretWord
        self.max_size = max_size
        self.max_gen = max_gen
        self.popsize = popsize
        self.CXPB = CXPB
        self.MUTPB = MUTPB

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

        if random.random() < 0.5:  # two points crossover
            pts = random.sample([i for i in range(5)], k=2)
            pt1 = min(pts)
            pt2 = max(pts)
            child1 = ind1[:pt1] + ind2[pt1:pt2] + ind1[pt2:]
            child2 = ind2[:pt1] + ind1[pt1:pt2] + ind2[pt2:]
        else:  # uniform crossover
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
        if fit[0] >= fit[1]:  # replace
            for _ in range(int(len(self.secretWord)/2)):
                tmp_char = random.choice(list(string.ascii_lowercase))
                pos = random.randint(0, len(ind))
                ind = ind[:pos] + tmp_char + ind[pos+1:]
        else:  # swap
            for _ in range(int(len(self.secretWord)/2)):
                pos = random.sample([i for i in range(len(ind))], 2)
                tmp_ind = list(ind)
                tmp_ind[pos[0]], tmp_ind[pos[1]
                                         ] = tmp_ind[pos[1]], tmp_ind[pos[0]]
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
        matched_words = get_close_matches(word, self.dico, n=3, cutoff=0)
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
        population = random.sample(
            [w for w in dico if len(w) == len(self.secretWord)], self.popsize)
        for ind in population: self.dico.remove(ind)
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
        
        start = time.time()

        while time.time() - start < 300: # define 5 minutes as the limit of time
            gen += 1
            new_population = []

            # for _ in range(int(self.popsize/2)):
            child1, child2 = self.selection(population, fitness)
            if random.random() > self.CXPB:
                child1, child2 = self.crossover(child1, child2)
            if random.random() > self.MUTPB:
                if child1 != "":
                    child1 = self.mutation(child1)
                if child2 != "":
                    child2 = self.mutation(child2)
            
            if child1 in self.dico: self.dico.remove(child1)
            if child2 in self.dico: self.dico.remove(child2)
            
            if child1 not in new_population:
                if self.isCompatible(child1, fitness):
                    new_population.append(child1)
            if child2 not in new_population:
                if self.isCompatible(child2, fitness):
                    new_population.append(child2)
            
            if len(new_population) > 0:
                population += new_population.copy()
                fitness += [self.evaluate(ind) for ind in new_population]
                i = self.bestInd(fitness)
                if fitness[i] > bestFit:
                    pos = i
                    bestFit = fitness[i]

            if verbose:
                print("Generation :", gen, ":", population)

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

            if gen == self.max_gen: print("Max gen reached at", time.time() - start, "seconds")
            
        print("\nEnd of program, secret word ("+self.secretWord+") founded ?", population[pos] == self.secretWord)
        print("All population :", population)
        print("All fitness :", fitness)
        return population[pos], bestFit

###################################################
#
# Test
#
###################################################


def test_4_letters():
    avrg_time_BT, avrg_time_FC = [], []
    f = open('Tests/src/4_letters_opti.csv', 'w')
    writer = csv.writer(f)
    head = ['Iteration', 'Word',
            'Time Back Track (in sec)', 'Time Forward Checking (in sec)']
    writer.writerow(head)
    for i in range(20):
        print(f'\nStart of the {i} iterations.')
        secretWord = choose_secretWord(dico, 4)
        csp = CSP(dico, secretWord)

        body = [i, secretWord]
        print(f'The secret word is {secretWord}.')

        print("Running back tracking method : ....")
        start = time.time()
        _ = csp.backTracking(verbose=True)
        stop = time.time()
        # print(f'Time : {stop - start}')
        body.append(stop - start)

        print("\nRunning back tracking method with Forward checking : ....")
        start = time.time()
        _ = csp.forwardChecking(verbose=True)
        stop = time.time()
        # print(f'Time : {stop - start}')
        body.append(stop - start)

        writer.writerow(body)

    f.close()


def test_n_words(n: int, nb_letter_min: int, nb_letter_max: int):
    avrg_time_BT, avrg_time_FC = [], []
    f = open('Tests/src/avrg_time_up_to_8_letters_3.csv', 'a')
    writer = csv.writer(f)

    for l in range(nb_letter_min, nb_letter_max):
        print(f'Words of {l} letters')
        time_BT, time_FC = [], []
        for i in range(n):
            print(f'\nStart of the {i} iterations.')
            secretWord = choose_secretWord(dico, l)
            csp = CSP(dico, secretWord)
            body = [secretWord, l]

            print(f'The secret word is {secretWord}.')

            print("Running back tracking method : ....")
            start = time.time()
            _ = csp.backTracking(verbose=True)
            stop = time.time()
            print(f'Time : {round(stop - start, 2)} s')
            time_BT.append(stop - start)
            body.append(stop - start)

            print("\nRunning back tracking method with Forward checking : ....")
            start = time.time()
            _ = csp.forwardChecking(verbose=True)
            stop = time.time()
            print(f'Time : {round(stop - start, 2)} s')
            time_FC.append(stop - start)
            body.append(stop - start)
            writer.writerow(body)

        # print(f'Time list BT: {time_BT}')
        # print(f'avrg_time_BT : {np.mean(time_BT)}')
        avrg_time_BT.append(np.mean(time_BT))

        # print(f'Time list FC: {time_FC}')
        # print(f'avrg_time_FC : {np.mean(time_FC)}')
        avrg_time_FC.append(np.mean(time_FC))

    f.close()

    # First figure for Back Tracking
    plt.figure()
    plt.plot(np.arange(nb_letter_min, nb_letter_max),
             avrg_time_BT, color='m', label='Back Tracking')
    plt.plot(np.arange(nb_letter_min, nb_letter_max),
             avrg_time_FC, color='c', label='Forward Checking')
    plt.xlabel("Length of words")
    plt.ylabel("Time in sec")
    plt.legend()
    plt.title(
        "Average time to solve the problem depending on the length of words.")
    plt.savefig(f"Tests/img/{nb_letter_max - 1}_letters")


def read_csv(filename: str):
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        avrg_time_BT, avrg_time_FC = [], []
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(', '.join(row))
                line_count += 1
            elif line_count != 0 and len(row) != 0:
                print(row[2])

                avrg_time_BT.append(float(row[2]))
                avrg_time_FC.append(float(row[3]))

        print(
            f"Avarage time for Back Tracking for 20 words of 4 letters : {np.mean(avrg_time_BT)}.")
        print(
            f'Avarage time for Forward Checking for 20 words of 4 letters : {np.mean(avrg_time_FC)}.')

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
    secretWord = choose_secretWord(dico, 3)
    print("\n=====================\nSecret word is :", secretWord, "\n=====================")

    # Solve with CSP
    csp = CSP(dico, secretWord)

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
    ga = GeneticAlgorithm(dico, secretWord, max_size=50, max_gen=100,
                          popsize=5, CXPB=0.8, MUTPB=0.4)

    # print("\nRunning genetic algorithm : ....")
    # start = time.time()
    # w, fit = ga.solve(verbose=True)
    # stop = time.time()
    # print("Word found with Forward checking :", w, "in", stop-start, "seconds")

    # test_n_words(4, 4, 9)
    # test_4_letters()
    read_csv('Tests/src/4_letters.csv')
    read_csv('Tests/src/4_letters_opti.csv')
