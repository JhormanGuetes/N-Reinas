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
        tiempo_de_ejecucion = []
        suma_tiempo = 0
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
                suma_tiempo = suma_tiempo + tiempo
                tiempo_de_ejecucion.append(suma_tiempo)
                continue
            row = len(solution)
            if row == self.size:
                solutions.append(solution)
                final = default_timer()
                tiempo = (final - inicio)*1000
                suma_tiempo = suma_tiempo + tiempo
                tiempo_de_ejecucion.append(suma_tiempo)
                continue
            for col in range(self.size):
                queen = (row, col)
                queens = solution.copy()
                queens.append(queen)
                queue.put(queens)
            final = default_timer()
            tiempo = (final - inicio)*1000
            suma_tiempo = suma_tiempo + tiempo
            tiempo_de_ejecucion.append(suma_tiempo)

        #print(list(tiempo_de_ejecucion))
        axis_x = list(range(len(tiempo_de_ejecucion)))
        axis_y = tiempo_de_ejecucion
        title_axis_x = "Iteraciones"
        title_axis_y = "Tiempo de Ejecución (ms)"
        title = "Breadth-First Search"
        number_iterations = str(len(tiempo_de_ejecucion))

        return [solutions, axis_x, axis_y, title_axis_x, title_axis_y, title, number_iterations]

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
