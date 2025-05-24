import numpy as np
import scipy as sp
import random as rd
import torch as pt


def Monte_Carlo_Integrate(func,a,b,n=1000):
    """
    Monte Carlo integration of a function func in the interval [a,b] with n points
    """
    print("Monte Carlo Integration")
    device = pt.device("cuda" if pt.cuda.is_available() else "cpu")
    x = np.linspace(a,b,n)
    y = func(x)
    # if np.isnan(y).all():
    #     print("Error: y is all NaN")
    #     return 0
    y_max = max(y)
    if np.isinf(y_max):
        y_max = max(y[y!=np.inf])
    # if np.isnan(y_max):
    #     y_max = max([b  for b in y  if  not np.isnan(b)])
    y_min = min(y)
    # if np.isnan(y_min):
    #     y_min = min([b  for b in y  if  not np.isnan(b)])
    lower_bound = min(y_min,0)
    print(f"y_max = {y_max}",f'lower_bound = {lower_bound}')
    x_rand = np.random.uniform(a,b,n)
    y_rand = np.random.uniform(lower_bound,y_max,n)
    y_rand = y_rand[y_rand<y_max]
    y_rand = y_rand[y_rand>lower_bound]
    x_rand = x_rand[:len(y_rand)]
    y_rand = y_rand[:len(x_rand)]
    y_rand_func = func(x_rand)
    y_rand_1 = y_rand[y_rand<y_rand_func]
    y_rand_1 = y_rand_1[0<y_rand_1]
    y_rand_2 = y_rand[y_rand>y_rand_func]
    y_rand_2 = y_rand_2[0>y_rand_2]
    N=len(y_rand_1)-len(y_rand_2)
    integral = (b-a)*(y_max-lower_bound)*N/n
    return integral

def test_func(x):
    return np.sin(x)

