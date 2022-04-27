from QueenEnvironment import cal_heuristic
from QueenEnvironment import size
from primero_en_anchura.n_queens import NQueens
import random
import math
import time
from QueenEnvironment import State
from QueenEnvironment import nCk
import QueenEnvironment
import queue
from QueenEnvironment import graph
from timeit import default_timer
import numpy as np


def convert_state_to_string(my_state):
    my_string = ''
    for column in range(size):
        for row in range(size):
            if my_state.map[row][column] == 1:
                my_string += str(row + 1) + ' '
                break
    return my_string


class AStar:
    def __init__(self, start_state, goal, action):
        self.goal = goal
        self.states = queue.PriorityQueue()
        self.start_state = start_state
        self.action = action
        self.states.put((start_state.c + start_state.h, start_state))

    def is_goal(self, state):
        if self.goal is not None:
            return self.goal(state)

    def push_to_queue(self, new_states):
        for state in new_states:
            self.states.put((state.c + state.h, state))

    def search(self):
        (value, best_state) = self.states.get()
        suma_tiempo = 0
        start = time.process_time()
        count = 1
        tiempo_de_ejecucion = []
        while not self.is_goal(best_state):
            inicio = default_timer()
            neighbor_states = self.action(best_state)
            self.push_to_queue(neighbor_states)
            if not self.states.empty():
                (value, best_state) = self.states.get()
                print('---------------------')
                print('Time %s' % count)
                print('H = %s' % best_state.h)
                print('C = %s' % best_state.c)
                print('Length of queue = %s' % self.states.qsize())
                count += 1
            else:
                return None

            final = default_timer()
            tiempo = (final - inicio) * 1000
            suma_tiempo = suma_tiempo + tiempo
            tiempo_de_ejecucion.append(suma_tiempo)

        axis_x = list(range(len(tiempo_de_ejecucion)))
        axis_y = tiempo_de_ejecucion
        title_axis_x = "Iteraciones"
        title_axis_y = "Tiempo de Ejecución (ms)"
        title = "A*"
        number_iterations = str(len(tiempo_de_ejecucion))

        duration = time.process_time() - start
        print(duration)
        return [best_state, axis_x, axis_y, title_axis_x, title_axis_y, title, number_iterations]


def get_neighbor_highest(states):
    state = states.pop()
    min = cal_heuristic(state)

    for other in states:
        if cal_heuristic(other) < min:
            min = cal_heuristic(other)
            state = other

    return state


def random_selection(population):
    fitnessList = []

    for state in population:
        fitnessList.append(nCk(size, 2) - cal_heuristic(state))

    probaList = []
    total = sum(fitnessList)
    first = 0
    second = 0

    for item in fitnessList:
        probaList.append(item / total)

    for i in range(len(probaList)):
        if i > 0:
            probaList[i] += probaList[i - 1]

    random_value = random.uniform(0, 1)

    for i in range(len(probaList)):
        if random_value < probaList[i]:
            first = i
            break

    random_value = random.uniform(0, 1)

    for i in range(len(probaList)):
        if random_value < probaList[i]:
            second = i
            break

    return population[first], population[second]


def reproduce(father, mother):
    child_h = min([father.h, mother.h])
    child = State()
    for i in range(5):
        child = State()
        c = random.randint(0, size - 1)

        for column in range(c):
            for row in range(size):
                if father.map[row][column] == 1:
                    child.map[row][column] = 1
                    break

        for column in range(c, size):
            for row in range(size):
                if mother.map[row][column] == 1:
                    child.map[row][column] = 1
                    break

        child.h = cal_heuristic(child)

        if child.h < child_h:
            break

    return child

class Node:
    def __init__(self, state, parent=None, action=None, path_cost=0):
        """Create a search tree Node, derived from a parent by an action."""
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def expand(self, problem):
        """List the nodes reachable in one step from this node."""
        return [self.child_node(problem, action)
                for action in problem.actions(self.state)]

    def child_node(self, problem, action):
        """[Figure 3.10]"""
        next_state = problem.result(self.state, action)
        #next_node = Node(next_state, self, action, problem.path_cost(self.path_cost, self.state, action, next_state))
        next_node = Node(next_state, self, action)
        return next_node
# ______________________________________________________________________________

''' USE OTHER SCHEDULE FUNCTION IF YOU WANT TO '''
def schedule(t, k=20, lam=0.005, limit=1000):
    """One possible schedule function for simulated annealing"""
    return (k * math.exp(-lam * t) if t < limit else 0)


''' WRITE THIS FUNCTION: '''

def simulated_annealing(problem):
    """See [Figure 4.5] for the algorithm."""
    current = Node(problem.initial)         #create current tuple -1 * size
    i=0
    tiempo_de_ejecucion = []
    suma_tiempo = 0
    while i<1000000:                        #loop
        inicio = default_timer()
        t = schedule(i)
        if (t==0 or -1 not in current.state):     #stop when t==0 or there is goal
            axis_x = list(range(len(tiempo_de_ejecucion)))
            axis_y = tiempo_de_ejecucion
            title_axis_x = "Iteraciones"
            title_axis_y = "Tiempo de Ejecución (ms)"
            title = "Simulated Annealing"
            number_iterations = str(len(tiempo_de_ejecucion))
            return [current.state, axis_x, axis_y, title_axis_x, title_axis_y, title, number_iterations]                  #( -1 not in state)
        if(current.expand(problem) != []):              #may explain
            n = random.choice(current.expand(problem))      #random child
            deltaE = problem.value(n) - problem.value(current)
            if(deltaE<0):
                print(1)
            if(deltaE>0 or math.exp(deltaE // t) > 0.5):       #accept when ratio >0.5
                current = n
        else:                                           #cannot explain
            current = Node(problem.initial)             #create new node
            i=-1                               #back to the loop wit i=0, random again
        i+=1                            #increase i
        final = default_timer()
        tiempo = (final - inicio) * 1000
        suma_tiempo = suma_tiempo + tiempo
        tiempo_de_ejecucion.append(suma_tiempo)

def mutate(child):
    number_mutate = random.randint(0, size / 2)

    for i in range(number_mutate):
        row = random.randint(0, size - 1)
        column = random.randint(0, size - 1)

        for i in range(size):
            if child.map[i][column] == 1:
                child.map[i][column] = 0
                break
        child.map[row][column] = 1

    return child


class Genetic:
    def __init__(self, states, state_size):
        self.states = states
        self.state_size = state_size

    def is_best_individual(self):
        for state in self.states:
            if cal_heuristic(state) == 0:
                return True

        return False

    def search(self):
        tiempo_de_ejecucion = []
        suma_tiempo = 0
        count = 1
        print(nCk(size, 2))
        start = time.process_time()

        while not self.is_best_individual():
            inicio = default_timer()
            fitness = ''
            new_states = []
            avg = 0
            print('Time %s' % count)
            for i in range(self.state_size):
                father, mother = random_selection(self.states)

                child = reproduce(father, mother)
                if random.uniform(0, 1) < 0.3:
                    child = mutate(child)
                new_states.append(child)
                child.h = cal_heuristic(child)
                avg += child.h
                fitness += str(child.h) + ' '

            avg = avg / self.state_size
            list_childs = []
            for item in new_states:
                if item.h <= avg:
                    list_childs.append(item)

            print(fitness)
            self.states = list_childs
            count += 1
            final = default_timer()
            tiempo = (final - inicio) * 1000
            suma_tiempo = suma_tiempo + tiempo
            tiempo_de_ejecucion.append(suma_tiempo)

        duration = time.process_time() - start
        print(duration)
        for state in self.states:
            if cal_heuristic(state) == 0:

                axis_x = list(range(len(tiempo_de_ejecucion)))
                axis_y = tiempo_de_ejecucion
                title_axis_x = "Iteraciones"
                title_axis_y = "Tiempo de Ejecución (ms)"
                title = "Genetic Algorithm"
                number_iterations = str(len(tiempo_de_ejecucion))

                return [state, axis_x, axis_y, title_axis_x, title_axis_y, title, number_iterations]

        return None

"""
A class for defining an Ant Colony Optimizer for TSP-solving.
The c'tor receives the following arguments:
    Graph: TSP graph 
    Nant: Colony size
    Niter: maximal number of iterations to be run
    rho: evaporation constant
    alpha: pheromones' exponential weight in the nextMove calculation
    beta: heuristic information's (\eta) exponential weight in the nextMove calculation
    seed: random number generator's seed
"""


class ACO(object):
    def __init__(self, dim, Nant, Niter, rho, alpha=1, beta=1, seed=None):
        self.dim = dim
        self.Nant = Nant
        self.Niter = Niter
        self.rho = rho
        self.alpha = alpha
        self.beta = beta
        self.pheromone = np.ones((dim, dim)) / dim
        self.local_state = np.random.RandomState(seed)
        """
        This method invokes the ACO search over the TSP graph.
        It returns the best tour located during the search.
        Importantly, 'all_paths' is a list of pairs, each contains a path and its associated length.
        Notably, every individual 'path' is a list of edges, each represented by a pair of nodes.
        """

    def run(self):
        # Book-keeping: best tour ever
        best_placement = None
        best_path = ("TBD", np.inf)
        for i in range(self.Niter):
            contradictions = np.zeros((self.Nant, self.dim))
            all_paths = self.constructColonyPaths(contradictions)
            self.depositPheronomes(all_paths, contradictions)
            best_placement = min(all_paths, key=lambda x: x[1])
            print(i + 1, ": ", best_placement[1])
            if best_placement[1] < best_path[1]:
                best_path = best_placement
            self.pheromone *= self.rho  # evaporation

        return best_path
        """
        This method deposits pheromones on the edges.
        Importantly, unlike the lecture's version, this ACO selects only 1/4 of the top tours - and updates only their edges, 
        in a slightly different manner than presented in the lecture.
        """

    def depositPheronomes(self, all_paths, contradictions):
        # sorted_paths = sorted(all_paths, key=lambda x: x[1])
        # Nsel = int(self.Nant/4) # Proportion of updated paths
        currPath = 0
        # for path, fitVal in sorted_paths[:Nsel]:
        for path, fitVal in all_paths:
            for move in range(self.dim):
                self.pheromone[path[move]][move] += 1.0 / (contradictions[currPath][move] + 1) ** (self.dim / 2)
            currPath += 1

        """
        This method generates paths for the entire colony for a concrete iteration.
        The input, 'path', is a list of edges, each represented by a pair of nodes.
        Therefore, each 'arc' is a pair of nodes, and thus Graph[arc] is well-defined as the edges' length.
        """

    def evalTour(self, path, contradictions, solution):
        res = 0
        for i in range(len(path)):
            if contradictions[i] != 0:
                res += 1
        if res == 0:
            for i in range(self.dim):
                for j in range(self.dim):
                    if path[i] == j:
                        print(1, end='')
                    else:
                        print(0, end='')
                    print(" ", end='')
                print()

            graph(solution[0], solution[1], solution[2], solution[3], solution[4], solution[5])
            exit(0)
        return res
        #
        """
        This method generates a single Hamiltonian tour per an ant, starting from node 'start'
        The output, 'path', is a list of edges, each represented by a pair of nodes.
        """

    def constructSolution(self, ant, contradictions):
        path = []
        tiempo_de_ejecucion = []
        suma_tiempo = 0
        for i in range(self.dim):
            inicio = default_timer()
            path = self.nextMove(self.pheromone[:][i], path, ant, contradictions)

            final = default_timer()
            tiempo = (final - inicio) * 1000
            suma_tiempo = suma_tiempo + tiempo
            tiempo_de_ejecucion.append(suma_tiempo)

        axis_x = list(range(len(tiempo_de_ejecucion)))
        axis_y = tiempo_de_ejecucion
        title_axis_x = "Iteraciones"
        title_axis_y = "Tiempo de Ejecución (ms)"
        title = "Breadth-First Search"
        number_iterations = str(len(tiempo_de_ejecucion))
        solution = [axis_x, axis_y, title_axis_x, title_axis_y, title, number_iterations]
        return path, self.evalTour(path, contradictions[ant], solution)
        """
        This method generates 'Nant' paths, for the entire colony, representing a single iteration.
        """

    def constructColonyPaths(self, contradictions):
        all_paths = []
        for i in range(self.Nant):
            path, value = self.constructSolution(i, contradictions)
            # constructing pairs: first is the tour, second is its length
            all_paths.append((path, value))
        return all_paths

        """
        This method probabilistically calculates the next move (node) given a neighboring 
        information per a single ant at a specified node.
        Importantly, 'pheromone' is a specific row out of the original matrix, representing the neighbors of the current node.
        Similarly, 'dist' is the row out of the original graph, associated with the neighbors of the current node.
        'visited' is a set of nodes - whose probability weights are constructed as zeros, to eliminate revisits.
        The random generation relies on norm_row, as a vector of probabilities, using the numpy function 'choice'
        """

    def nextMove(self, pheromone, path, ant, contradictions):
        colContr = self.getContradictions(
            path)  # for column k, return pair(num contradictions, vector of with whom contradiction)
        row = pheromone ** self.alpha * ((1.0 / (colContr + 1)) ** self.beta)
        norm_row = row / row.sum()
        dims = range(self.dim)
        move = self.local_state.choice(dims, 1, p=norm_row)[0]
        # changes to path and self.contradictions
        path.append(move)
        if colContr[move] != 0:
            contradictions[ant][len(path) - 1] += 1
        for j in range((len(path) - 1)):  # j = column of path[j], path[j] = row of this element
            if path[j] == move:
                contradictions[ant][j] += 1
            if path[j] + j == len(path) - 1 + move:
                contradictions[ant][j] += 1
            if path[j] - j == move - (len(path) - 1):
                contradictions[ant][j] += 1
        return path

    def getContradictions(self, path):
        colContr = np.zeros(self.dim)
        curCol = len(path)  # current column
        for i in range(self.dim):  # row in curCol
            for j in range(len(path)):  # j = column of path[j], path[j] = row of this element
                if path[j] == i or curCol - i == j - path[j] or curCol + i == j + path[j]:
                    colContr[i] += 1
        return colContr

class NQueensProblem:
    """The problem of placing N queens on an NxN board with none attacking each other.
    A state is represented as an N-element array, where a value of r in the c-th entry means there is a queen at column c,
    row r, and a value of -1 means that the c-th column has not been filled in yet. We fill in columns left to right.

    Sample code: iterative_deepening_search(NQueensProblem(8))
    Result: <Node (0, 4, 7, 5, 2, 6, 1, 3)>
    """

    def __init__(self, N):
        # self.initial = initial
        self.initial = tuple([-1] * N)  # -1: no queen in that column
        self.N = N

    def board_resolve(self, size, vector):
        matrix = []
        for i in range(size):
            matrix.append([0] * size)

        for i in range(size):
            for j in range(size):
                if (vector[j] == i):
                    matrix[i][j] = 1

        for i in range(size):
            for j in range(size):
                print(matrix[i][j], end=" ")
            print("")

    def actions(self, state):
        """In the leftmost empty column, try all non-conflicting rows."""
        if state[-1] != -1:
            return []  # All columns filled; no successors
        else:
            col = state.index(-1)
            # return [(col, row) for row in range(self.N)
            return [row for row in range(self.N)
                    if not self.conflicted(state, row, col)]

    def result(self, state, row):
        """Place the next queen at the given row."""
        col = state.index(-1)
        new = list(state[:])
        new[col] = row
        return tuple(new)

    def conflicted(self, state, row, col):
        """Would placing a queen at (row, col) conflict with anything?"""
        return any(self.conflict(row, col, state[c], c)
                   for c in range(col))

    def conflict(self, row1, col1, row2, col2):
        """Would putting two queens in (row1, col1) and (row2, col2) conflict?"""
        return (row1 == row2 or  # same row
                col1 == col2 or  # same column
                row1 - col1 == row2 - col2 or  # same \ diagonal
                row1 + col1 == row2 + col2)  # same / diagonal

    def value(self, node):
        """Return (-) number of conflicting queens for a given node"""
        num_conflicts = 0
        for (r1, c1) in enumerate(node.state):
            for (r2, c2) in enumerate(node.state):
                if (r1, c1) != (r2, c2):
                    num_conflicts += self.conflict(r1, c1, r2, c2)

        return -num_conflicts

def print_board(state):
    line = ''
    for i in range(size):
        line += ' - '
    print(line)

    for row in range(size):
        line = ''
        for column in range(size):
            if state.map[row][column] == 1:
                line += '|x|'
            else:
                line += '| |'
        print(line)

        line = ''
        for i in range(size):
            line += ' - '
        print(line)


def main():
    bfs_or_sa = False
    env = QueenEnvironment.QueenEnvironment()
    solution = []
    agent = []
    env.current_state = env.random_start_state()
    env.current_state.show()

    print('\n1. Breadth-First Search')
    print('2. A*')
    print('3. Simulated Annealing')
    print('4. Genetic Algorithm')
    print('5. Ant Colony Optimisation')
    print('0. Exit')

    choose = int(input())

    if choose == 1:
        n_queens = NQueens(size)
        bfs_solutions = n_queens.solve_bfs()
        solution = bfs_solutions
        for i, solu in enumerate(bfs_solutions[0]):
            print('BFS Solution %d:' % (i + 1))
            n_queens.print(solu)
        print('Total BFS solutions: %d' % len(bfs_solutions[0]))
        bfs_or_sa = True
    elif choose == 2:
        agent = AStar(env.current_state, env.goal, env.action)
    elif choose == 3:
        problem1 = NQueensProblem(size)
        solution = simulated_annealing(problem1)
        problem1.board_resolve(size, solution[0])
        bfs_or_sa = True
    elif choose == 4:
        state_list = []
        for i in range(20):
            state_list.append(env.random_start_state())

        agent = Genetic(state_list, 30)
    elif choose == 5:
        Niter = 500
        Nant = 200
        n_queens = size
        ant_colony = ACO(n_queens, Nant, Niter, rho=0.95, alpha=1, beta=10)
        solution = ant_colony.run()
        #print("------------------------------------------------")
        #print(solution[1])
        #shortest_path = solution[0]
        #for i in range(n_queens):
        #    for j in range(n_queens):
        #        if shortest_path[0][i] == j:
        #            print(1, end='')
        #        else:
        #            print(0, end='')
        #        print(" ", end='')
        #    print()
        #print("shotest_path: {}".format(shortest_path))

    if(bfs_or_sa != True):
        solution = agent.search()

        state = solution[0]
        state.show()

        result = ''
        for column in range(size):
            for row in range(size):
                if state.map[row][column] == 1:
                    result += str(row + 1) + ' '
                    break
        print(result)

    # axisX  ==================> bfs_solutions[1]
    # axisY  ==================> bfs_solutions[2]
    # titleAxisX ==============> bfs_solutions[3]
    # titleAxisY ==============> bfs_solutions[4]
    # title ===================> bfs_solutions[5]
    # numberIterations ========> bfs_solutions[6]

    graph(solution[1], solution[2], solution[3], solution[4], solution[5], solution[6])

if __name__ == '__main__':
    main()