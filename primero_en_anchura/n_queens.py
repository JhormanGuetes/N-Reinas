from queue import Queue
from timeit import default_timer
import matplotlib.pyplot as plt



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
                print(solutions, " solitions")
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
        axisX = list(range(len(tiempoDeEjecucion)))
        axisY = tiempoDeEjecucion
        titleAxisX = "Iteraciones"
        titleAxisY = "Tiempo de Ejecución (ms)"
        title = "Breadth-First Search"
        numberIterations = str(len(tiempoDeEjecucion))
        print("salí")
        return [solutions, axisX, axisY, titleAxisX, titleAxisY, title, numberIterations]

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
            print('[', end='')
            for j in range(self.size):
                p = '1' if (i, j) in queens else '0'

                if (j != self.size - 1): print(' %s,' % p, end='')
                else: print(' %s' % p, end='')

            print(']')
