class Solution:
    def __init__(self, n: int):
        self.n = n
        self.solutions = []
        self.state = []

    def solveNQueens(self) -> list[list[str]]:
        self.search(self.state)
        return self.solutions

    def is_solution_state(self, state):
        return len(state) == self.n

    def get_candidates(self, state):
        if not state:
            return range(self.n)

        position = len(state)
        candidates = set(range(self.n))
        for row, col in enumerate(state):
            candidates.discard(col)
            dist = position - row
            candidates.discard(row - dist)
            candidates.discard(row + dist)
        return candidates

    def search(self, state):
        if self.is_solution_state(state):
            self.solutions.append(state.copy())
            return

        for candidate in self.get_candidates(state):
            state.append(candidate)
            self.search(state)
            state.pop()


def main():
    solutions = Solution(n=5).solveNQueens()
    solutions_set = set()
    for solution in solutions:
        print(solution)

if __name__ == "__main__":
    main()
