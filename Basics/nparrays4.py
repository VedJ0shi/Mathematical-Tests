import numpy as np
from numpy import sin
import time

'''compactly implement loop operations (i.e. sums) by taking advantage
of the vectorized broadcasting property of arrays'''

#calculating a series using loop:
s = 0
start = time.perf_counter()
for n in range(1, 101):
    s = s + (1/n)*sin(n)
end = time.perf_counter()
print('Sum:', s)
print('Runtime:', end-start)

#vectorized:
start = time.perf_counter()
first_terms = 1/(np.arange(1, 101, 1))
second_terms = sin(np.arange(1, 101, 1))
end = time.perf_counter()
print('Sum:', sum(first_terms*second_terms)) #sum is built-in function that expects an iterable
print('Runtime:', end-start) #faster by factor of 10