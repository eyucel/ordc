#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.

import random
from numpy.polynomial.polynomial import polyval
from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial import distance
from deap import base
from deap import creator
from deap import tools
from scipy import stats


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
# Attribute generator
toolbox.register("attr_bool", random.uniform, -10, 10)
# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual, 
    toolbox.attr_bool, 2)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

k = 100000
n = 2
costs = np.random.rand(k, n)
p = np.random.rand(k, 1)
bidseq = np.zeros((k,n))
bidsneq = np.zeros((k,n))
averageneq = np.zeros(n)
p = np.random.rand(k,1)


def evalOneMax(individual):

    bidseq = polyval(costs, individual[2:4])
    sorted_bid_owners = np.argsort(bidseq, axis=1)
    lowest = sorted_bid_owners == 0
    p_clearing = bidseq <= p
    bids = polyval(costs, individual)
    sorted_bid_owners = np.argsort(bids, axis=1)
    lowest = sorted_bid_owners == 0
    p_clearing = bids <= p
    mask = lowest*p_clearing
    # print(mask)

    payoff = (p-costs)*mask
    averageeq = np.mean(payoff,axis=0)

    for i in range(0,n):
        bidsneq[:, 0:i] = polyval(costs[:, 0:i], individual[2:4])
        bidsneq[:, i] = polyval(costs[:, i], individual[0:2])
        if i != n-1:
            bidsneq[:, i+1:n] = polyval(costs[:, i+1:n], individual[2:4])
        sorted_bid_owners = np.argsort(bidsneq, axis=1)
        lowest = sorted_bid_owners == 0
        p_clearing = bidsneq <= p
        mask = lowest*p_clearing
        payoff = (p-costs)*mask
        averageneq[i] = np.mean(payoff[i])





    return sum(averageeq-averageneq)-np.sqrt((individual[0]-individual[2])**2 + (individual[1]-individual[3])**2),
    payoff = (p-costs)*mask
    average = np.mean(payoff,axis=0)

    return sum(average )-sum(distance.pdist(average.reshape(n,1))),

# Operator registering
toolbox.register("evaluate", evalOneMax)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    random.seed(64)
    
    pop = toolbox.population(n=300)
    CXPB, MUTPB, NGEN = 0.5, 0.2, 40
    
    print("Start of evolution")
    
    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        print(ind)
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
    t = np.linspace(0, 1)
    plt.plot(t, polyval(t, best_ind))
    plt.show()
if __name__ == "__main__":
    main()