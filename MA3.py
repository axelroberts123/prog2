""" MA3.py

Student:
Mail:
Reviewed by:
Date reviewed:

"""
import random
import matplotlib.pyplot as plt
import math as m
import concurrent.futures as future
from statistics import mean 
from time import perf_counter as pc
from numba import njit

# Exc1

def approximate_pi(n):
    print('n:' + str(n))
    n_c=0
    x_plot_c=[]
    y_plot_c=[]
    x_plot=[]
    y_plot=[]
    for i in range(n):
        x=random.uniform(-1,1)
        y=random.uniform(-1,1)
        if x**2 + y**2 < 1:
            n_c +=1
            x_plot_c.append(x)
            y_plot_c.append(y)
        else:
            x_plot.append(x)
            y_plot.append(y)
    pi=4*n_c/n
    print('pi approximation:' + str(pi))
    # plt.scatter(x_plot_c,y_plot_c,color='red')
    # plt.scatter(x_plot,y_plot,color='blue')
    # plt.show()
    return pi

# Exc2, approximation
def sphere_volume(n, d): 
    #print('n:' + str(n) + ', Dimension(s):' + str(d))

    coord=[]
    for j in range(n):
        point=[]
        for i in range(d):
            x=random.uniform(-1,1)
            point.append(x)
            #point.append((lambda x:x**2)(x))
        distance=list(map(lambda x:x**2,point))
        coord.append(distance)
    in_or_out=[1 for sublist in coord if sum(sublist) <= 1]
    volume = 2**d * len(in_or_out)/n
    #print(volume)
    return volume

    
#Exc2, real value
def hypersphere_exact(d):
    exact = m.pi**(d/2)/m.gamma(d/2 + 1)
    print('exact_volume with dimension(s):' + str(d) + ' = ' + str(exact))
    return exact



@njit
def sphere_volume_numba(n:int, d:int)->float:
    #print('n:' + str(n) + ', Dimension(s):' + str(d))

    coord=[]
    for j in range(n):
        point=[]
        for i in range(d):
            x=random.uniform(-1,1)
            point.append(x)
            #point.append((lambda x:x**2)(x))
        distance=list(map(lambda x:x**2,point))
        coord.append(distance)
    in_or_out=[1 for sublist in coord if sum(sublist) <= 1]
    volume = 2**d * len(in_or_out)/n
    return volume

#Exc4: parallel code - parallelize actual computations by splitting data
def sphere_volume_parallel(n, d, np=10):
    n_split = [n//np for _ in range(np)]
    
    with future.ProcessPoolExecutor() as ex:
        results = ex.map(sphere_volume,n_split,np*[d])
    total_inside=sum(results)
    volume = 2**d * total_inside/n
    # n is the number of points
    # d is the number of dimensions of the sphere
    # np is the number of processes
    return  volume


def main():
    # Exc1
    dots = [1000, 10000, 100000]
    for n in dots:
        approximate_pi(n)
    # Exc2
    n = 100000
    d = 2
    sphere_volume(n, d)
    print(f"Actual volume of {d} dimentional sphere = {hypersphere_exact(d)}")

    n = 100000
    d = 11
    sphere_volume(n, d)
    print(f"Actual volume of {d} dimentional sphere = {hypersphere_exact(d)}")

    #Exc3
    n = 1000000
    d = 11
    start = pc()
    sphere_volume(n, d)
    stop = pc()
    print(f"Exc3: Sequential time of {d} and {n}: {stop-start}")
    print("What is numba time?")
    start2=pc()
    sphere_volume_numba(n,d)
    stop2=pc()
    print(stop2-start2)
    start3=pc()
    sphere_volume_numba(n,d)
    stop3=pc()
    print(stop3-start3)
    start4=pc()
    sphere_volume_numba(n,d)
    stop4=pc()
    print(stop4-start4)


    #Exc3: Sequential time of 11 and 1000000: 5.387973125
# What is numba time?
# 2.2220376670000004
# 1.3982136250000003
# 1.4197842919999992


    # Exc4
    n = 1000000
    d = 11
    start = pc()
    sphere_volume(n, d)
    stop = pc()
    print(f"Exc4: Sequential time of {d} and {n}: {stop-start}")
    print("What is parallel time?")

    start = pc()
    sphere_volume_parallel(n, d, 10)
    stop = pc()
    print(f"Exc4: Parallel time of {d} and {n}: {stop-start}")
    
    

if __name__ == '__main__':
	main()
