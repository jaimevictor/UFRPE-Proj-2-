from typing import List, Tuple, Callable
from random import randint
import matplotlib.pyplot as plt
import numpy as np
from random import sample, random, shuffle, choices
import time


def format_input(q_line):
    """
    :param q_line: The amount of lines from the matrix
    :return: Returns a dictionary of points with their indexes
    """
    points = {}
    for n in range(q_line):
        line = list(map(str, input().split()))
        for j in range(len(line)):
            if line[j] != '0':
                points[line[j]] = (n, j)
    return points


def first_population(positions: dict, size: int) -> list:
    """

    :param positions: The dictionary with every point and indexes
    :param size: the size of the population to be made
    :return: A randomly generated array of routes
    """
    points_without_r = [item for item in positions if item != 'R']
    random_permutation = []
    while len(random_permutation) < size:
        permutation = ''.join(sample(points_without_r, len(points_without_r)))
        if permutation not in random_permutation:
            random_permutation.append(permutation)
    return random_permutation


def caltulate_general_cost(positions: dict):
    """
    :param positions:A dictionary with every point and indexes
    :return:A dictionary with the cost to every possible connection cost
    """
    points = list(item for item in positions)
    cost = {}
    for index, i in enumerate(points):
        for j in range(index + 1, len(points)):
            point_j: str = points[j]
            actual_point = positions.get(i)  # Actual point value
            next_point = positions.get(point_j)  # Next point value
            point_cost = (abs(actual_point[0] - next_point[0]) + abs(actual_point[1] - next_point[1]))
            route = i + point_j  # String containing the two points
            cost[route] = point_cost
            cost[route[::-1]] = point_cost
    return cost


def individual_fitness(route: str, cost_table: dict):
    """
    :param cost_table: Dict with the cost of every connection
    :param route:str with the order of points
    :return: Int containing the fitness value for that route
    """
    cost: int = 0
    for k in range(len(route) - 1):
        point = route[k]
        nextp = route[k + 1]
        cost += cost_table[point + nextp]
    return cost


def fitness_calculation(population: List[str], cost_table: dict) -> List[float]:
    """
    :param cost_table: A dictionary with the cost to every possible connection cost
    :param population: Population to calculate the fitness
    :return:List with the fitness value for each individual
    """
    fitness_list: List[float] = [0] * len(population)
    for i, ind in enumerate(population):
        fitness_list[i] = individual_fitness(ind, cost_table)
    return fitness_list


def reverse_roulette(fitness: List[float]) -> int:
    """
    :param fitness: List with every fitness from a population
    :return: index of the selected individual
    """
    probability = [1 / x for x in fitness]
    prob_sum = sum(probability)
    probs_norm = [p/prob_sum for p in probability]
    chosen_individual = choices(range(len(fitness)), weights=probs_norm)[0]
    return chosen_individual


def mutation_individual(individual: str, mutation_rate: float) -> str:
    """
    :param individual: Individual to be mutated
    :param mutation_rate: The probability of a mutation
    :return: The mutated Individual
    """""
    mutated = individual
    mutated = mutated.replace("R", "")
    mutated = list(mutated)
    for i, s in enumerate(mutated):
        if random() <= mutation_rate:
            random_index = randint(0, len(mutated)-1)
            while random_index == i:
                random_index = randint(0, len(mutated)-1)
            mutated[i], mutated[random_index] = mutated[random_index], mutated[i]
    mutated = "R" + "".join(str(gene) for gene in mutated if gene != -1) + "R"
    return mutated


def mutation(individuals_list: List[str], mutation_rate: float) -> List[str]:
    """
    :param individuals_list: A list with every individual to be mutated
    :param mutation_rate: The probability of a mutation
    :return: The list of individuals after mutation
    """
    for i, ind in enumerate(individuals_list):
        individuals_list[i] = mutation_individual(ind, mutation_rate)
    return individuals_list

def tournament(fitness: List[float]) -> int:
    """
    :param fitness: A list with the fitness of a population
    :return: The best individual from the 2 chosen
    """
    individual1 = randint(0, len(fitness) - 1)
    individual2 = randint(0, len(fitness) - 1)
    return individual1 if fitness[individual1] < fitness[individual2] else individual2


def selection(population: List[str], fitness: List[float], sel_func: Callable) -> List[str]:
    """
    :param population: The population to run the selection on
    :param fitness: The fitness of said population
    :param sel_func: The selection function chosen
    :return: The best individuals found during selection
    """
    individuals_list: List[str] = [None] * len(population)
    for i in range(len(population)):
        best_index = sel_func(fitness)
        individuals_list[i] = population[best_index]
    return individuals_list


def crossover_individual(individual1: str, individual2: str, cross_rate: float) -> Tuple[str, str]:
    """
    :param individual1: The first parent
    :param individual2: The second parent
    :param cross_rate: The probability of a crossover
    :return: The 2 new individuals (sons) generated with the genes from individual1 and individual2
    """
    if individual1.endswith("R") or individual1.startswith("R") or individual2.endswith("R") or individual2.startswith("R"):
        individual1 = individual1.replace("R", "")
        individual2 = individual2.replace("R", "")
    new1 = list(individual1)
    new2 = list(individual2)
    if random() <= cross_rate:
        n = len(individual1)
        cross_point = randint(1, n - 1)
        new1 = list(individual2[:cross_point])
        new2 = list(individual1[:cross_point])
        slice2_new1 = [item for item in individual1 if item not in new1]
        slice2_new2 = [item for item in individual2 if item not in new2]
        new1 += slice2_new1
        new2 += slice2_new2
    new1 = "R" + "".join(str(gene) for gene in new1 if gene != -1) + "R"
    new2 = "R" + "".join(str(gene) for gene in new2 if gene != -1) + "R"
    return new1, new2


def crossover(parents: List[str], cross_rate: float) -> List[str]:
    """
    :param parents: The list of individuals to act as parents for the new generation
    :param cross_rate: The probability of a crossover
    :return: The list of individuals of the new generation
    """
    sons_list: List[str] = [None] * len(parents)
    shuffle(parents)
    for i in range(0, len(parents), 2):
        son1, son2 = crossover_individual(parents[i], parents[i + 1], cross_rate)
        sons_list[i] = son1
        sons_list[i + 1] = son2
    return sons_list


def evolution(population: list, fitness: list, func: Callable, cross_rate: float,
              mutation_rate: float, cost_table: dict):
    """
    :param population: The population to run the evolution
    :param fitness: The fitness of population
    :param func: The selection function
    :param cross_rate: The probability of a crossover
    :param mutation_rate: The probability of a mutation
    :param cost_table: Dict with the cost of every connection
    :return: A new and (hopefully) improved population and its fitness
    """
    best_individuals = selection(population, fitness, func)
    new_individuals = crossover(best_individuals, cross_rate)
    new_population = mutation(new_individuals, mutation_rate)
    new_fitness = fitness_calculation(new_individuals, cost_table)
    return new_population, new_fitness


def AG(num_gen: int, num_individuals: int, func: Callable,
       cross_rate: float, mutation_rate: float, points: dict) -> str:
    """
    :param num_gen: The number of generation to be had
    :param num_individuals: The number of individuals to generate the first population
    :param func: The selection function
    :param cross_rate: The probability of a crossover
    :param mutation_rate: The probability of a mutation
    :param points: A dictionary of points with their indexes
    :return: The best route after running the AG
    """
    parents = first_population(points, num_individuals)  # Generate te first population
    table = caltulate_general_cost(points)  # Calculates the cost for every connection
    fitness = fitness_calculation(parents, table)

    for i in range(num_gen):  # Runs through the amount of generations
        parents, fitness = evolution(population=parents, fitness=fitness, func=func, cross_rate=cross_rate,
                                     mutation_rate=mutation_rate, cost_table=table)

    best = parents[fitness.index(min(fitness))]
    return best


def plot_route(route, allpoints):
    """
    :param route: The route to create the graph with
    :param all_points: all the possible points
    :return: A list with every coordinate to plot
    """
    x = []
    y = []
    route_size = len(route)-1
    for k in range(route_size):
        actual_point = route[k]
        next_point = route[k+1]
        actual_point = allpoints.get(actual_point)
        next_point = allpoints.get(next_point)
        x.append(actual_point[1])
        y.append(actual_point[0])
        x.append(actual_point[1])
        y.append(next_point[0])
        if k == route_size:
            x.append(next_point[1])
            y.append(next_point[0])
    x.append(allpoints["R"][1])
    y.append(allpoints["R"][0])
    x = np.array(x)
    y = np.array(y)
    return x, y


def create_graph(route, all_points):
    """
    :param route: The route to create the graph with
    :param all_points: all the possible points
    :return: Nothing, it shows the graph
    """

    x_axis, y_axis = plot_route(route, all_points)
    graph, (plot1) = plt.subplots(1)
    plot1.plot(x_axis, y_axis, "-o")
    plot1.invert_yaxis()

    plot1.set_title("Best Route")
    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    plt.grid()

    for b in all_points:
        point = all_points[b]
        x = point[1]
        y = point[0]
        plt.text(x, y, b, horizontalalignment='left')

    plt.show()


while True:
    q_line, q_column = list(map(int, input().split()))  # Reading input
    start_time = time.time() # Start timer
    all_points = format_input(q_line)  # Get every point in the table
    best_route = AG(num_gen=150, num_individuals=20, func=reverse_roulette,
                    cross_rate=0.9, mutation_rate=0.01, points=all_points)
    print("%s seconds" % (time.time() - start_time))
    print(f'After many generations, the best route found was {best_route}')
    create_graph(best_route, all_points)
