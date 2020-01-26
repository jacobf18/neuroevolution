import random
from deap import creator, base, tools, algorithms
from random import shuffle
from matplotlib import pyplot as plt
import multiprocessing


def evalOneMax(individual):
    # (x1, y1, x2, y2)
    prices = get_prices(individual[0], individual[1], individual[2], individual[3])

    return sum(individual),

def get_prices(x1, y1, x2, y2):
    def make_graph(depth, graph, start, end, turns):
        # add points to graph
        graph.add(start)
        graph.add(end)

        if depth > 0:
            # unpack input values
            fromtime, fromvalue = start
            totime, tovalue = end

            # calcualte differences between points
            diffs = []
            last_time, last_val = fromtime, fromvalue
            for t, v in turns:
                new_time = fromtime + (totime - fromtime) * t
                new_val = fromvalue + (tovalue - fromvalue) * v
                diffs.append((new_time - last_time, new_val - last_val))
                last_time, last_val = new_time, new_val

            # add 'brownian motion' by reordering the segments
            shuffle(diffs)

            # calculate actual intermediate points and recurse
            last = start
            for segment in diffs:
                p = last[0] + segment[0], last[1] + segment[1]
                make_graph(depth - 1, graph, last, p, turns)
                last = p
            make_graph(depth - 1, graph, last, end, turns)

    depth = 2
    graph = set()
    make_graph(depth, graph, (0, 0.6), (1, 1), [(x1, y1), (x2, y2)])
    pt = plt.plot(*zip(*sorted(graph)))
    #plt.gcf().set_size_inches(30,10)
    #plt.show()
    prices = pt[0].get_data()[1]
    return prices

def main():
    creator.create("FitnessMin", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    PARAMS = 4
    # enable multiprocessing
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    # register the type of parameters
    toolbox.register("attr_float", random.random)

    # register the individuals
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=PARAMS)

    # register the population
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evalOneMax)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # set the population size
    population = toolbox.population(n=300)

    NGEN=10
    for gen in range(NGEN):
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        population = toolbox.select(offspring, k=len(population))
    top10 = tools.selBest(population, k=10)
    print(top10)

if __name__ == '__main__':
    main()
