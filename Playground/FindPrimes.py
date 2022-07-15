import time


N = 1_000_000

start = time.process_time()
primes = []
for num in range(N):
    primes.append(num)
print(time.process_time() - start)
#####################################################

start = time.process_time()
is_prim = [False, False] + [True] * N
for i in range(2, N):
    if is_prim[i]:
        for j in range(i*2, N, i):
            is_prim[j] = False
print(time.process_time() - start)
#print(is_prim)
#####################################################

start = time.process_time()
is_prim = [False]*N
for i in range(2, N):
    if is_prim[i]:
        for j in range(i*2, N, i):
            is_prim[j] = False
print(time.process_time() - start)
#print(is_prim)
#####################################################



start = time.process_time()
primes = [2]
for num in range(3, N, 2):
    isPrime = True
    for prime in primes:
        if num % prime == 0:
            isPrime = False
            break
    if isPrime:
        primes.append(num)
print(time.process_time() - start)


if __name__ == '__main__':
    pass

