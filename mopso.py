import os
import random
import numpy as np
import pandas as pd
import time
import math
from pandas import read_excel, notnull



# This project included simulation optimization. At this point, just the optimization section is provided, 
# and a function named "simulation (iteration, particle_index)" is constructed, 
# but it is excluded for clarity in understanding the simulation.



# Function to calculate fitness values from the error file
def calculate_fitness(error_file):
    f1 = []
    f2 = []
    f3 = []
    with open(error_file, 'r') as f:
        error = [line.strip().split() for line in f.readlines()[1:]]
        error = np.array(error, dtype=float)
        f1 = np.mean(error[:, 0:6])
        f2 = np.mean(error[:, 6:8])
        f3 = np.mean(error[:, 8])
    return f1, f2, f3


# Particle class
class Particle:
    def __init__(self, dimensions):
        # Initialize self.position with the correct shape
        self.position = np.zeros((12, 6))  # Initialize as a 12x6 matrix
        self.velocity = np.zeros((12, 6))  # Initialize as a 12x6 matrix
        self.pbest_position = self.position.copy()
        self.pbest_fitness = [float('inf')] * 3  # Initialize as a list with 9 elements

    def update(self, gbest_position, inertia_weight, cognitive_coefficient, social_coefficient):
        # Update velocity
        self.velocity = (inertia_weight * self.velocity) + \
                       (cognitive_coefficient * random.random() * (self.pbest_position - self.position)) + \
                       (social_coefficient * random.random() * (gbest_position - self.position))

        self.position = self.position + self.velocity

# Multi-Objective PSO class
class MultiObjectivePSO:
    def __init__(self, dimensions, fitness_functions, n_particle, n_iter,
                 inertia_weight=0.7, cognitive_coefficient=2.0, social_coefficient=2.0):
        self.dimensions = dimensions
        self.fitness_functions = fitness_functions
        self.n_particle = n_particle
        self.n_iter = n_iter
        self.inertia_weight = inertia_weight
        self.cognitive_coefficient = cognitive_coefficient
        self.social_coefficient = social_coefficient

        # Initialize particles
        self.particles = [Particle(dimensions) for _ in range(self.n_particle)]
        self.gbest_position = None
        self.gbest_fitness = [float('inf')] * 3  # Initialize as a list with 9 elements

    def run(self, n_iter_sim, n_particle_sim):
        for iteration in range(self.n_iter):            
            for particle_index in range(self.n_particle):
                
                simulation(iteration, particle_index)
                with open("Rm6.txt", 'r') as f:
                    positions = [line.strip().split() for line in f.readlines()]
                positions = np.array(positions, dtype=float)

                self.particles[particle_index].position = positions
                self.particles[particle_index].pbest_position = positions.copy()
                
                self.particles[particle_index].pbest_fitness = self.fitness_functions("error_TSI_S.txt")
                print(f"particle: {particle_index+1}, pBest Fitness: {self.particles[particle_index].pbest_fitness}")
                # Update gbest based on Pareto dominance
                if self.is_dominated(self.particles[particle_index].pbest_fitness, self.gbest_fitness):
                    # Update gbest_position only when a better solution is found
                    self.gbest_position = self.particles[particle_index].position.copy()
                    self.gbest_fitness = self.particles[particle_index].pbest_fitness

                # Call the update method on the correct particle object
                self.particles[particle_index].update(self.gbest_position, self.inertia_weight,
                               self.cognitive_coefficient, self.social_coefficient)

            print(f"Iteration: {iteration+1}, GBest Fitness: {self.gbest_fitness}")

        return self.gbest_position, self.gbest_fitness, iteration

    def is_dominated(self, fitness_values1, fitness_values2):
        """Check if fitness_values1 dominates fitness_values2."""
        better_in_at_least_one = False
        for i in range(len(fitness_values1)):
            if fitness_values1[i] < fitness_values2[i]:
                better_in_at_least_one = True
            elif fitness_values1[i] > fitness_values2[i]:
                return False

        return better_in_at_least_one


dimensions = 12 * 6
n_particle = 6
n_iter = 2

pso = MultiObjectivePSO(dimensions, calculate_fitness, n_particle, n_iter)

gbest_position, gbest_fitness, iteration = pso.run(n_iter, n_particle)

print("GBest Position:", gbest_position)
print("GBest Fitness:", gbest_fitness)
print("Optimal Iteration:", iteration)

optimal_particle_index = None
for i, particle in enumerate(pso.particles):
    if np.array_equal(particle.position, pso.gbest_position):
        optimal_particle_index = i
        break

print("Optimal Particle Index:", optimal_particle_index)
