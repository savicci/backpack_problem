from copy import copy
from typing import List, Tuple

from bitarray import bitarray
import numpy as np

Genome = bitarray
Population = List[Genome]

# global values
POPULATION_SIZE = 10
GENOME_LENGTH = 100
MAX_GENERATIONS = 100
CARRY_FACTOR = 2 / 3
MIN_WEIGHT_BOUND = 10
MAX_WEIGHT_BOUND = 200


def generate_population(size: int, genome_lenth: int) -> Population:
    return [bitarray(np.random.choice([0, 1]) for _ in range(genome_lenth)) for _ in range(size)]


def generate_items(size: int, min_bound: int, max_bound: int) -> Tuple[List[int], int]:
    max_weight = 0
    generated_items = []

    for i in range(size):
        item = np.random.randint(min_bound, max_bound)
        max_weight += item
        generated_items.append(item)

    return generated_items, int(max_weight * CARRY_FACTOR)


def crossover(population: Population) -> Population:
    crossed_population = copy(population)

    while len(population) != 0:
        first_parent = population.pop(np.random.randint(0, len(population)))
        second_parent = population.pop(np.random.randint(0, len(population)))

        crossover_point = np.random.randint(0, len(first_parent))

        crossed_population.append(bitarray(first_parent[0:crossover_point] + second_parent[crossover_point:]))
        crossed_population.append(bitarray(second_parent[0:crossover_point] + first_parent[crossover_point:]))

    return crossed_population


def fitness(first_pretendent: Genome, items: List[int], carry_limit: int):
    curr_weight = 0

    for i in range(len(first_pretendent)):
        if first_pretendent[i] == 1:
            curr_weight += items[i]

            if curr_weight > carry_limit:
                return 0

    return curr_weight


def tournament_selection(candidates: Population, items: List[int], carry_limit: int) -> Population:
    next_gen_population = []

    while len(candidates) != 0:
        first_pretendent = candidates.pop(np.random.randint(0, len(candidates)))
        second_pretendent = candidates.pop(np.random.randint(0, len(candidates)))

        fitness_first = fitness(first_pretendent, items, carry_limit)
        fitness_second = fitness(second_pretendent, items, carry_limit)

        if fitness_first > fitness_second:
            next_gen_population.append(first_pretendent)
        elif fitness_first < fitness_second:
            next_gen_population.append(second_pretendent)
        else:
            if fitness_first != 0:
                next_gen_population.append(first_pretendent)
            else:
                # generate new
                next_gen_population.append(bitarray(np.random.choice([0, 1]) for _ in range(GENOME_LENGTH)))
    return next_gen_population


def mutation(population, probability: float = 0.5):
    for genome in population:
        if np.random.rand() >= probability:
            # mutate
            genome.invert(np.random.randint(0, len(genome)))


if __name__ == '__main__':
    population = generate_population(POPULATION_SIZE, GENOME_LENGTH)
    items, carry_limit = generate_items(GENOME_LENGTH, MIN_WEIGHT_BOUND, MAX_WEIGHT_BOUND)

    # evolve
    for generation_num in range(MAX_GENERATIONS):
        # crossover
        next_generation_candidates = crossover(population)

        # mutation
        mutation(population)

        # selection
        population = tournament_selection(next_generation_candidates, items, carry_limit)

        # check if found solution is best possible
        population = sorted(population, key=lambda genome: fitness(genome, items, carry_limit), reverse=True)

        current_best_fitness = fitness(population[0], items, carry_limit)
        print(current_best_fitness)

        if current_best_fitness == carry_limit:
            break

    population = sorted(population, key=lambda genome: fitness(genome, items, carry_limit), reverse=True)


    print('Finished in {} generations', generation_num)
    print('Carry limit: ', carry_limit)
    print('Best solution: ', fitness(population[0], items, carry_limit))
