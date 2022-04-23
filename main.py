from primero_en_anchura.n_queens import NQueens
from archivos import numeroDeColumnas


def main():
    print('.: N-Queens Problem :.')
    size = numeroDeColumnas('4x4.txt')
    print_solutions = 'y'
    n_queens = NQueens(size)
    bfs_solutions = n_queens.solve_bfs()
    if print_solutions:
        for i, solution in enumerate(bfs_solutions):
            print('BFS Solution %d:' % (i + 1))
            n_queens.print(solution)
    print('Total BFS solutions: %d' % len(bfs_solutions))


if __name__ == '__main__':
    main()
