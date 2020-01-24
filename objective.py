import MultiNEAT as NEAT
import pandas as pd
import numpy as np
import pickle
import argparse
import matplotlib.pyplot as plt
import matplotlib.dates as dates
from joblib import Parallel, delayed
import multiprocessing
from backtesting import Backtest, Strategy
import ta
from pandas_datareader import data
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

trade_threshold = 100

# Calculates the fitness of a genome
def fitness_function(positions, prices_train):
    series = get_portfolio_helper(positions, prices_train)
    fitnesses = [series['# Trades'],
                 series['Exposure [%]'] * -1,
                 series['Max. Drawdown [%]'] * -1,
                 series['Avg. Drawdown [%]'] * -1,
                 series['Win Rate [%]'],
                 series['Best Trade [%]'],
                 series['Worst Trade [%]'] * -1,
                 series['Avg. Trade [%]'],
                 series['SQN'],
                 series['Sharpe Ratio'],
                 series['Sortino Ratio']]

    #maximum = max(fitnesses)
    #minimum = min(fitnesses)

    #maxMin = [(f - minimum)/(maximum-minimum) for f in fitnesses]

    # Can return different metrics.  This returns the Sharpe Ratio now.
    if series['# Trades'] < trade_threshold:
        return 0

    return series['Sharpe Ratio']

# Runs a backtest given signals and a testing dataset
def get_portfolio_helper(signals, test):
    # Strategy
    class Neuro(Strategy):
        def init(self):
            self.i = 0
            self.bought = False
            pass
        def next(self):
            if self.bought == False and signals[self.i] >= 0.5:
                self.bought = True
                self.buy()
            # Sell
            elif self.bought == True and signals[self.i] < 0.5:
                self.bought = False
                self.sell()
            self.i += 1
    # Runs Backtest
    bt = Backtest(test, Neuro, cash=100000, commission=0)
    return bt.run()

# Runs a backtest for a genome.
def get_portfolio(genome, signals_test, test):
    signals = get_signals(genome, signals_test)
    return get_portfolio_helper(signals, test)

# Creates trading signals from a genome and a PCA dataset.
def get_signals(genome, df):
    # this creates a neural network (phenotype) from the genome
    net = NEAT.NeuralNetwork()
    genome.BuildPhenotype(net)

    signals = []
    for index, row in df.iterrows():
        row_list = list(row)
        row_list.pop(0)
        net.Input(row_list)
        net.Activate()
        signals.append(net.Output()[0])
    return signals

# Evaluates the fitness of a genome using a training dataset.
def evaluate(genome, train, prices_train):
    signals = get_signals(genome, train)
    fitness = fitness_function(signals, prices_train)
    genome.SetFitness(fitness)
    return [fitness, genome]

def get_datasets(ticker):
    ohlcv = data.DataReader(ticker, start='2006-1-1', end='2019-12-31', data_source='yahoo')
    df_copy = ohlcv.copy(deep=True)
    df_copy = ta.utils.dropna(df_copy)
    df_copy = ta.add_all_ta_features(df_copy, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
    std = MinMaxScaler().fit_transform(df_copy[df_copy.columns])
    pca = PCA(0.95)
    principalComponents = pca.fit_transform(std)
    principalDF = pd.DataFrame(data=principalComponents)
    return [ohlcv, principalDF]

# Main Function
def main():
    params = NEAT.Parameters()
    params.PopulationSize = 512
    iterations = 100
    ticker = 'AAPL'

    # Parse Arguments
    # Example:
    #   python3 seed.py --Population 512 --Iterations 20 --Ticker AAPL
    parser = argparse.ArgumentParser()
    parser.add_argument("--Population", '-p', help="Set Population Size")
    parser.add_argument("--Iterations", '-i', help="Set How Many Iterations")
    parser.add_argument("--Ticker", '-t', help="Set Which Ticker")
    args = parser.parse_args()
    if args.Population:
        params.PopulationSize = int(args.Population)
    if args.Iterations:
        iterations = int(args.Iterations)
    if args.Ticker:
        ticker = args.Ticker
    print(params.PopulationSize, iterations)
    genome = NEAT.Genome(0, 4, 0, 1, False, NEAT.ActivationFunction.UNSIGNED_SIGMOID, NEAT.ActivationFunction.UNSIGNED_SIGMOID, 0, params, 0, 0)
    pop = NEAT.Population(genome, params, True, 1.0, 0) # the 0 is the RNG seed

    # Input Parameters
    pop.Parameters.ActivationFunction_UnsignedSigmoid_Prob = 1/3
    pop.Parameters.ActivationFunction_Tanh_Prob = 1/3
    pop.Parameters.ActivationFunction_UnsignedGauss_Prob = 1/3
    pop.Parameters.MinSpecies = 5
    pop.Parameters.MaxSpecies = 40
    pop.Parameters.YoungAgeFitnessBoost = 1.1
    pop.Parameters.EliteFraction = 0.01
    pop.Parameters.MutateActivationAProb = 0.25
    pop.Parameters.OverallMutationRate = 0.2
    pop.Parameters.MutateAddNeuronProb = 0.03
    pop.Parameters.MutateAddLinkProb = 0.3
    pop.Parameters.MutateWeightsProb = 0.8
    pop.Parameters.CrossoverRate = 0.75
    pop.Parameters.InterspeciesCrossoverRate = 0.001

    # Load and Split Datasets
    [test, train] = get_datasets(ticker)

    split = int(len(test['Close']) * 0.6)
    signals_test = train[split:]
    prices_train = test[:split]
    train = train[:split]
    test = test[split:]

    # Keep track of the best Genome and Fitness
    best_genome = None
    best_fitness = -100

    # Determine how many cores available for parallel processing
    num_cores = multiprocessing.cpu_count()

    for generation in range(iterations):
        # retrieve a list of all genomes in the population
        genome_list = NEAT.GetGenomeList(pop)

        # Get a list of fitnesses for all genomes currently in the population
        results = Parallel(n_jobs=num_cores)(delayed(evaluate)(g, train, prices_train) for g in genome_list)
        fitnesses = [f[0] for f in results]
        m = max(fitnesses)
        if m > best_fitness:
            best_fitness = m
            best_genome = results[fitnesses.index(m)][1]

        # Print the running best fitness
        print(generation, best_fitness)
        pop.Epoch()

    # Write out the best genome to a pickle object.
    filehandler = open('best_genome_' + ticker + '.obj', 'wb')
    pickle.dump(best_genome, filehandler)

    print(get_portfolio(best_genome, signals_test, test))

if __name__ == '__main__':
    main()
