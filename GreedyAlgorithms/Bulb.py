

class Solution:

    def bulbs(self, A):

        const = 0

        for b in A:
            if const % 2 == 0:
                b = b
            else:
                b = not b

            if b % 2 == 1:
                continue
            else:
                const += 1

        return const


if __name__ == "__main__":
    solution = Solution().bulbs([0, 1, 0, 1])
    print(solution)