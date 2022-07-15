import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def getFirstDigit(n):
    while n > 10:
        n = (int)(n/10)
    return n

def readData():
    dfs = pd.read_excel('population.xlsx', sheet_name='SUB-IP-EST2019-ANNRNK')
    return dfs
def getCities(dfs):
    cities = dfs.iloc[:, 1]
    return cities

def getPopulation2019(dfs):
    values = dfs.iloc[:, -1]
    return values
if __name__ == '__main__':
    #print(getFirstDigit(232434343))
    df = readData()
    cities = getCities(df)
    populations = getPopulation2019(df)
    nan_elems = populations.isnull()
    populations = populations[~nan_elems]
    populations = populations.values.tolist()
    populations = populations[1:]
    populations = [getFirstDigit(n) for n in populations]

    dict = {}
    for i in range(1, 10):
        sum = 0
        for element in populations:
            if element == i:
                sum += 1
        dict[i] = sum




    l_2d = [[0, 1, 2], [3, 4, 5]]
    temp1 = np.array(l_2d)
    temp2 = temp1.T

    temp3 = zip(*l_2d)
    l_2d_t_tuple = list(temp3)
    first, second, third = zip(*l_2d)

    l_1d_index = [['Alice', 0], ['Bob', 1], ['Charlie', 2]]
    t = (['Alice', 0])
    index, value = zip(*l_1d_index)
    print(index)


