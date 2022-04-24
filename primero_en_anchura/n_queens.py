from queue import Queue
from timeit import default_timer
import matplotlib.pyplot as plt
import pprint


class NQueens:

    def __init__(self, size):
        self.size = size

    def solve_bfs(self):
        if self.size < 1:
            return []
        solutions = []
        tiempoDeEjecucion = []
        queue = Queue()
        queue.put([])
        i = -1
        while not queue.empty():
            i = i+1
            inicio = default_timer()
            solution = queue.get()
            if self.conflict(solution):
                final = default_timer()
                tiempo = (final - inicio)*1000
                tiempoDeEjecucion.append(tiempo)
                continue
            row = len(solution)
            if row == self.size:
                solutions.append(solution)
                final = default_timer()
                tiempo = (final - inicio)*1000
                tiempoDeEjecucion.append(tiempo)
                continue
            for col in range(self.size):
                queen = (row, col)
                queens = solution.copy()
                queens.append(queen)
                queue.put(queens)
            final = default_timer()
            tiempo = (final - inicio)*1000
            tiempoDeEjecucion.append(tiempo)

        #print(list(tiempoDeEjecucion))
        plt.plot(list(range(len(tiempoDeEjecucion))), tiempoDeEjecucion)
        plt.ylabel('Tiempo de Ejecución (ms)')
        numeroDeIteraciones = "Iteraciones" + " (" + str(len(tiempoDeEjecucion)) + ")"
        plt.xlabel(numeroDeIteraciones)
        plt.title("BFS")
        plt.show()
        return solutions

    def conflict(self, queens):
        for i in range(1, len(queens)):
            for j in range(0, i):
                a, b = queens[i]
                c, d = queens[j]
                if a == c or b == d or abs(a - c) == abs(b - d):
                    return True
        return False

    def print(self, queens):
        for i in range(self.size):
            print(' ---' * self.size)
            for j in range(self.size):
                p = 'Q' if (i, j) in queens else ' '
                print('| %s ' % p, end='')
            print('|')
        print(' ---' * self.size)
