__author__ = 'msbcg452'

import numpy as np
from scipy.integrate import ode
from matplotlib import pyplot as plt
from deap import base
from deap import creator
from deap import tools
import random


def evalOneMax(individual):
    a,b,c,d = individual
    if b<0:
        b=0.001
        individual[1]=0.001

    if c<0:
        c = 0
        individual[2]=0

    # print(b*t)
    # input()

    # vals = a * (tt + d) * np.log(b*(tt+d))+c*(tt+d)+e

    vals = a*np.log(b*dtt + c) + d
    return -sum((vals-dyy)**2),


def main():
    # random.seed(64)

    pop = toolbox.population(n=4000)
    CXPB, MUTPB, NGEN = 0.5, 0.2, 200

    print("Start of evolution")

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))

    for ind, fit in zip(pop, fitnesses):

        ind.fitness.values = fit

    print("  Evaluated %i individuals" % len(pop))

    # Begin the evolution
    for g in range(NGEN):
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        print("  Evaluated %i individuals" % len(invalid_ind))

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5

        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)

    print("-- End of (successful) evolution --")

    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
    z = np.linspace(0.001, 1)

    # vz = best_ind[0] * (z + best_ind[3]) * np.log(best_ind[1]*(z + best_ind[3]))+best_ind[2]*(z + best_ind[3])+best_ind[4]
    a,b,c,d = best_ind
    vz = a*np.log(b*z + c) + d
    plt.plot(z,vz)

    plt.plot(dtt, dyy)
    plt.show()


def f(c, y):
    w = y[0]

    f2 = (2*c* (1 - 2 * c + w) * (-1 + w)) / (2*(c**2 -1) *(c - w))
    f5 = 5*c**4 * (1-2*c+w) * (-1+w)/ (2*(c**5-1)*(c-w))
    f1 = (1-2*c+w) * (-1+w)/ (2*(c-1)*(c-w))
    f3 = (3*c**2 * (1 - 2 * c + w) * (-1 + w)) / (2*(c**3 -1) *(c - w))
    # f0 = (w-2*c+1)*(w-1) /( (c-1)*(c-w))
    return f2
t = np.zeros((1000))
y = np.zeros((1000))
w0 = .998
y0 = [w0]
t0 = .999
t1 = .001
dt = -0.001
r = ode(f)
r.set_initial_value(y0,t0)
r.set_integrator("dop853",nsteps=1000)
i = 0
while r.successful() and r.t > t1:
    r.integrate(r.t+dt)
    # print("%g %g" % (r.t, r.y))
    t[i] = r.t
    y[i] = r.y
    i+=1
w = np.where(y<0,0,y)
plt.plot(t,w)

plt.show()

print(y[::-1])



