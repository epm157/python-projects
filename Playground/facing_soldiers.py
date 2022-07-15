def face_each_other(soldier1: str, soldier2: str) -> bool:
    if soldier1 == '>' and soldier2 == '<':
        return True
    return False


def flip_soldier(soldier: str) -> str:
    if soldier == '<':
        return '>'
    elif soldier == '>':
        return '<'


def soldier_replace(soldiers: str, index: int) -> str:
    return '%s%s%s' % (soldiers[:index], flip_soldier(soldiers[index]), soldiers[index+1:])


def solve(soldiers: str):
    is_solved = False
    steps = 0

    while not is_solved:
        i = 0
        is_solved = True
        print(soldiers)
        while i < len(soldiers) - 1:
            if face_each_other(soldiers[i], soldiers[i+1]):
                is_solved = False
                soldiers = soldier_replace(soldiers, i)
                soldiers = soldier_replace(soldiers, i+1)
                i += 1
            i += 1

        if is_solved:
            return steps

        steps += 1



def sort(array: [int]):
    print(array)
    for i in range(len(array)):
        for j in range(len(array)):
            if array[i] > array[j]:
                array[i], array[j] = array[j], array[i]
        print(array)
    return array

if __name__ == '__main__':
    solve_steps = solve('><><<<><<')
    print(f'Solved in {solve_steps} steps(s)')

    sort([2, 3, 5, 1, 4])








