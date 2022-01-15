import json
import os
import pickle
import random
import time

import click
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

INPUTFILE = "cities.json"

NUMBER_OF_TRIBES_MESSAGE = "Number of tribes"
POPULATION_SIZE_MESSAGE = "Number of genomes per tribe"
PATIENCE_MESSAGE = "Steps of local optimization with no improvement after which giving up (patience)"
CITIES_MESSAGE = "Number of cities of optimization (will be ignored if cities.json is in folder)"
EPOCHS_MESSAGE = "Number of epochs in evolutionary algorithm"
INTERCHANGE_MESSAGE = "Number of sub-epochs which each tribe lives isolated"
KILL_RATE_MESSAGE = "Fraction of genomes killed each sub-epoch"
MUTATION_MESSAGE = "Fraction of genomes randomly mutated each epoch"

start = time.time()
distance_map = {}
cache = {}


@click.command("TSP evolutionary computing solver")
@click.option(
    "--epochs",
    type=int,
    required=False,
    default=2 ** 4,
    help=EPOCHS_MESSAGE,
    prompt=EPOCHS_MESSAGE,
)
@click.option(
    "--freq_interchange",
    type=int,
    default=2 ** 5,
    required=False,
    help=INTERCHANGE_MESSAGE,
    prompt=INTERCHANGE_MESSAGE,
)
@click.option(
    "--kill_rate",
    type=float,
    required=False,
    default=0.4,
    help=KILL_RATE_MESSAGE,
    prompt=KILL_RATE_MESSAGE,
)
@click.option(
    "--mutation_rate",
    type=float,
    required=False,
    default=0.035,
    help=MUTATION_MESSAGE,
    prompt=MUTATION_MESSAGE,
)
@click.option(
    "--n_cities",
    type=int,
    required=False,
    default=2 ** 5,
    help=CITIES_MESSAGE,
    prompt=CITIES_MESSAGE,
)
@click.option(
    "--patience",
    type=int,
    required=False,
    default=2 ** 16,
    help=PATIENCE_MESSAGE,
    prompt=PATIENCE_MESSAGE,
)
@click.option(
    "--populations_size",
    type=int,
    required=False,
    default=2 ** 8,
    help=POPULATION_SIZE_MESSAGE,
    prompt=POPULATION_SIZE_MESSAGE,
)
@click.option(
    "--tribes_n",
    type=int,
    required=False,
    default=2 ** 4,
    help=NUMBER_OF_TRIBES_MESSAGE,
    prompt=NUMBER_OF_TRIBES_MESSAGE,
)
def get_best_route(
        epochs,
        freq_interchange,
        kill_rate,
        mutation_rate,
        n_cities,
        patience,
        populations_size,
        tribes_n,
):
    """
    Does a fair job at calculating the optimal route between N cities for the TSP
    Can be started with generated cities or if a file called cities.json is found in the same folder it will use that.
    Note: cities.json ought to be a simple 2D array of dimensions (n_cities, 2)
    """

    assert epochs > 0
    assert freq_interchange > 0
    assert 1.0 > kill_rate >= 0
    assert 1.0 > mutation_rate >= 0
    assert n_cities > 2
    assert patience >= 0
    assert populations_size > 10
    assert tribes_n > 1

    # setup cities
    if INPUTFILE in os.listdir("."):
        print(f"Ignoring {n_cities=} because {INPUTFILE} was provided")
        cities = np.array(json.loads(open(INPUTFILE, "r").read()))
        assert cities.shape[-1] == 2, f"wrong dimensions for cities provided by file {cities.shape}, should be (n x 2)"
        n_cities = cities.shape[0]
    else:
        cities = np.random.uniform(0, 100, size=(n_cities, 2))

    # cache distances between cities
    for i, city_1 in enumerate(cities):
        for j, city_2 in enumerate(cities):
            distance_map[(i, j)] = np.linalg.norm(city_1 - city_2)

    # start with a number of tribes which are seperate populations
    tribes = np.stack(
        [
            np.random.multivariate_normal(
                np.zeros(n_cities), np.eye(n_cities) * 10, size=populations_size
            )
            for _ in range(tribes_n)
        ]
    )

    # for keeping track
    best = (1e9, None)

    # for logging
    best_scores_global = np.array([])
    worst_scores_global = np.array([])

    # run epochs
    for epoch in range(epochs):
        print(f"### {epoch=}/{epochs=}")

        # for selecting best performing tribes
        scores_tribes = []

        # for plotting
        best_scores_epoch = np.zeros((len(tribes), freq_interchange))
        worst_scores_epoch = np.zeros((len(tribes), freq_interchange))

        # let each tribe evolve seperately for a number of turns
        for i, population in enumerate(tqdm(tribes)):
            (
                best_tribe,
                best_scores,
                worst_score,
                population,
                scores,
            ) = run_tribe_evolution(
                freq_interchange, kill_rate, population, populations_size, mutation_rate
            )

            # save
            tribes[i] = population
            scores_tribes.append(best[0])

            # do we have a new best?
            if best_tribe[0] < best[0]:
                best = (best_tribe[0], deepcopy(best_tribe[1]))
                print(f"New {best[0]=} !!!!!!!!!")

            # for plotting
            best_scores_epoch[i, :] = best_scores
            worst_scores_epoch[i, :] = worst_score

        # for plotting
        best_scores_global = np.concatenate(
            (best_scores_global, best_scores_epoch.min(axis=0))
        )
        worst_scores_global = np.concatenate(
            (worst_scores_global, worst_scores_epoch.max(axis=0))
        )

        # kill worst performing tribe
        tribes, _ = select(
            tribes, np.array(scores_tribes), int((1 - kill_rate) * tribes_n)
        )

        # create new tribes
        n_new_tribes = int(kill_rate * tribes_n)
        for _ in range(n_new_tribes):
            mother = random.choice(tribes)
            father = random.choice(tribes)
            halfway = populations_size // 2
            new_tribe = np.concatenate((mother[:halfway], father[halfway:]))
            tribes = np.concatenate((tribes, np.expand_dims(new_tribe, 0)), axis=0)

    # final local search optimization
    order = best[1].argsort()[::-1]
    best, best_scores_global = local_optimization(
        best, best_scores_global, order, patience
    )

    # log results
    print(f"final cities order={cities[best[1]]} (by index={order}) final length = {cities[0]}")
    print(f"{time.time() - start} seconds")

    # plot optimization
    plt.plot(best_scores_global)
    plt.plot(worst_scores_global)
    plt.show()
    plt.scatter(cities[:, 0], cities[:, 1])
    plot_solution(best[1], cities)
    plt.show()


def deepcopy(obj):
    """Deep copies any object faster than builtin"""
    return pickle.loads(pickle.dumps(obj, protocol=-1))


def fitness(genome):
    """fitness function of a vector-representation (genome) solution"""
    order = genome.argsort()[::-1]
    return fitness_order(order)


def fitness_order(order):
    """fitness function of a order of cities"""
    score = 0
    cacher = str(order)
    if cacher in cache:
        return cache[cacher]
    for i in range(len(order) - 1):
        score += distance_map[(order[i], order[i + 1])]
    score += distance_map[(order[0], order[-1])]
    cache[cacher] = score
    return score


def mutate(genome):
    """mutates the genome with added normal noise"""
    return genome + np.random.normal(size=len(genome))


def select(population, scores, n):
    """selects the top n best genomes"""
    mask = scores.argsort()
    return population[mask][:n], scores[mask][:n]


def sample(population, n):
    """samples new genomes given a population based on a multivariate normal distribution"""
    cov = np.cov(population.T)
    mean = np.mean(population, axis=0)
    new_samples = np.random.multivariate_normal(mean, cov, size=n)
    return new_samples


def plot_solution(order, cities):
    """makes a plot of the map"""
    x = []
    y = []
    for index in order:
        city = cities[index]
        x.append(city[0])
        y.append(city[1])
    plt.plot(x, y)


def run_tribe_evolution(
        sub_epochs, kill_rate, population, populations_size, mutation_rate
):
    """runs probabilistic evolutionary optimization for 1 single tribe"""

    # for logging
    best_scores = []
    worst_score = []

    # forkeeping track
    scores = np.array([fitness(genome) for genome in population])
    best = (1e9, None)

    for _ in range(sub_epochs):

        # mutate some random genomes
        for i, genome in enumerate(population):
            if random.random() < mutation_rate:
                mutated = mutate(genome)
                population[i] = mutated
                scores[i] = fitness(mutated)

        # kill weaklings
        population, scores = select(
            population, scores, round(populations_size * (1 - kill_rate))
        )

        # make babies
        new_population = sample(population, round(populations_size * kill_rate))
        new_scores = np.array([fitness(genome) for genome in new_population])
        population = np.concatenate((population, new_population))
        scores = np.concatenate((scores, new_scores))

        # log stuff
        best_ = scores.min()
        best_scores.append(best_)
        worst_ = scores.max()
        worst_score.append(worst_)
        if best_ < best[0]:
            best = (best_, deepcopy(population[np.argmin(scores)]))
    return (
        best,
        best_scores,
        worst_score,
        population,
        scores,
    )


def local_optimization(best, best_global, order, patience):
    best = (best[0], order)
    print("Doing final local optimization on best solution")
    with tqdm(total=patience) as pbar:
        while patience > 0:
            # create new random swap
            new_order = deepcopy(best[1])
            swap_from, swap_to = [
                random.randint(0, len(new_order) - 1) for _ in range(2)
            ]
            temp = new_order[swap_from]
            new_order[swap_from] = new_order[swap_to]
            new_order[swap_to] = temp

            # evaluate score of swap
            new_fitness = fitness_order(new_order)

            # log
            best_global = np.concatenate((best_global, [min(best[0], new_fitness)]))

            # accept if better
            if new_fitness < best[0]:
                best = (new_fitness, new_order)
                print(f"New {best[0]=} !!!!!!!!!")
                continue
            else:
                # reduce patience
                pbar.update(1)
                patience -= 1
    return best, best_global


if __name__ == "__main__":
    get_best_route()
