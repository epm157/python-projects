from itertools import product

class Solutions:
    SHAPE = 9
    GRID = 3
    EMPTY = '.'
    DIGITS = set([str(num) for num in range(1, SHAPE + 1)])

    def solveSudolu(self, board):
        self.search(board)
        return board

    def is_valid_state(self, board):
        for row in self.get_rows(board):
            if not set(row) == self.DIGITS:
                return False
        for col in self.get_cols(board):
            if not set(col) == self.DIGITS:
                return False
        for grid in self.get_grids(board):
            if not set(grid) == self.DIGITS:
                return False
        return True

    def get_candidates(self, board, row, col):
        used_digits = set()
        used_digits.update(self.get_kth_row(board, row))
        used_digits.update(self.get_kth_col(board, col))
        used_digits.update(self.get_grid_at_row_col(board, row, col))
        used_digits -= set([self.EMPTY])
        candidates = self.DIGITS - used_digits
        return candidates

    def search(self, board):
        if self.is_valid_state(board):
            return True
        for row_idx, row in enumerate(board):
            for col_idx, elm in enumerate(row):
                if elm == self.EMPTY:
                    for candidate in self.get_candidates(board, row_idx, col_idx):
                        board[row_idx][col_idx] = candidate
                        is_solved = self.search(board)
                        if is_solved:
                            return True
                        else:
                            board[row_idx][col_idx] = self.EMPTY
                    return False
        return True

    # helper functions for retrieving rows, cols, and grids
    def get_kth_row(self, board, k):
        return board[k]

    def get_rows(self, board):
        for i in range(self.SHAPE):
            yield board[i]

    def get_kth_col(self, board, k):

        return [
            board[row][k] for row in range(self.SHAPE)
        ]

    def get_cols(self, board):
        for col in range(self.SHAPE):
            ret = [
                board[row][col] for row in range(self.SHAPE)
            ]
            yield ret

    def get_grid_at_row_col(self, board, row, col):
        row = row // self.GRID * self.GRID
        col = col // self.GRID * self.GRID
        return [
            board[r][c] for r, c in
            product(range(row, row + self.GRID), range(col, col + self.GRID))
        ]

    def get_grids(self, board):
        for row in range(0, self.SHAPE, self.GRID):
            for col in range(0, self.SHAPE, self.GRID):
                grid = [
                    board[r][c] for r, c in
                    product(range(row, row + self.GRID), range(col, col + self.GRID))
                ]
                yield grid



def main():
    state = [
        ['.', '.', '.', '.', '.', '.', '.', '.', '.', ],
        ['.', '.', '.', '.', '.', '.', '.', '.', '.', ],
        ['.', '.', '.', '.', '.', '.', '.', '.', '.', ],
        ['.', '.', '.', '.', '.', '.', '.', '.', '.', ],
        ['.', '.', '.', '.', '.', '.', '.', '.', '.', ],
        ['.', '.', '.', '.', '.', '.', '.', '.', '.', ],
        ['.', '.', '.', '.', '.', '.', '.', '.', '.', ],
        ['.', '.', '.', '.', '.', '.', '.', '.', '.', ],
        ['.', '.', '.', '.', '.', '.', '.', '.', '.', ],
    ]
    sudoko = Solutions().solveSudolu(state)
    for row in sudoko:
        print(row)

if __name__ == "__main__":
    main()

