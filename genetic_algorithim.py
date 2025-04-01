import numpy as np
import random
import matplotlib.pyplot as plt

# Define the fitness function
def fitness(x):
    return x ** 2

# Generate initial population
def generate_population(size, lower_bound, upper_bound):
    return np.random.randint(lower_bound, upper_bound, size)

# Selection - Tournament Selection
def select_parents(population, fitness_values, num_parents):
    selected = np.random.choice(population, size=num_parents, replace=False, p=fitness_values/fitness_values.sum())
    return selected

# Crossover - Single Point
def crossover(parents):
    midpoint = len(parents) // 2
    offspring = []
    for i in range(0, len(parents), 2):
        if i+1 < len(parents):
            cross_point = random.randint(1, len(bin(parents[i]))-2)
            parent1_bin = bin(parents[i])[2:].zfill(8)
            parent2_bin = bin(parents[i+1])[2:].zfill(8)
            child1 = int(parent1_bin[:cross_point] + parent2_bin[cross_point:], 2)
            child2 = int(parent2_bin[:cross_point] + parent1_bin[cross_point:], 2)
            offspring.extend([child1, child2])
    return np.array(offspring)

# Mutation - Bit Flip Mutation
def mutate(offspring, mutation_rate=0.1):
    for i in range(len(offspring)):
        if random.random() < mutation_rate:
            bit_flip = random.randint(0, 7)
            offspring[i] ^= (1 << bit_flip)  # Flip a random bit
    return offspring

# Genetic Algorithm
def genetic_algorithm(generations=50, population_size=10, lower_bound=-10, upper_bound=10):
    population = generate_population(population_size, lower_bound, upper_bound)
    best_fitness = []
    
    for generation in range(generations):
        fitness_values = np.array([fitness(ind) for ind in population])
        best_fitness.append(max(fitness_values))
        
        parents = select_parents(population, fitness_values, population_size//2)
        offspring = crossover(parents)
        offspring = mutate(offspring)
        
        population = np.concatenate((parents, offspring))
    
    # Plot fitness over generations
    plt.plot(best_fitness)
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.title("Genetic Algorithm Optimization of f(x) = x^2")
    plt.show()
    
    return population[np.argmax(fitness_values)]

best_solution = genetic_algorithm()
print("Best solution found:", best_solution)
