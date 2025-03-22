from scanner import Scanner
from network import Network
from time import sleep
import numpy as np
import pykeyboard
import random
import copy

class Generation:
    def __init__(self):
        self.population_size = 50  # Increased population
        self.genomes = [Network() for _ in range(self.population_size)]
        self.best_fitness = 0
        self.generation = 0
        
    def evolve(self):
        """Evolve the population"""
        # Sort by fitness
        self.genomes.sort(key=lambda x: x.fitness, reverse=True)
        
        # Keep track of best fitness
        if self.genomes[0].fitness > self.best_fitness:
            self.best_fitness = self.genomes[0].fitness
            print(f"New best fitness: {self.best_fitness}")
        
        # Keep top 20%
        elite_size = self.population_size // 5
        new_population = self.genomes[:elite_size]
        
        # Create rest through crossover and mutation
        while len(new_population) < self.population_size:
            if random.random() < 0.8:  # 80% crossover
                parent1 = random.choice(self.genomes[:elite_size])
                parent2 = random.choice(self.genomes[:elite_size])
                child = self.crossover(parent1, parent2)
            else:  # 20% mutation of elite
                child = copy.deepcopy(random.choice(self.genomes[:elite_size]))
            
            self.mutate(child)
            new_population.append(child)
        
        self.genomes = new_population
        self.generation += 1

    def crossover(self, parent1, parent2):
        """Perform crossover between two parents"""
        child = Network()
        
        # Crossover for each layer
        for i in range(len(parent1.W1)):
            if random.random() < 0.5:
                child.W1[i] = parent1.W1[i]
            else:
                child.W1[i] = parent2.W1[i]
        
        for i in range(len(parent1.W2)):
            if random.random() < 0.5:

                child.W2[i] = parent1.W2[i]
            else:
                child.W2[i] = parent2.W2[i]
            
        for i in range(len(parent1.W3)):
            if random.random() < 0.5:
                child.W3[i] = parent1.W3[i]
            else:
                child.W3[i] = parent2.W3[i]
            
        for i in range(len(parent1.W4)):
            if random.random() < 0.5:
                child.W4[i] = parent1.W4[i]
            else:
                child.W4[i] = parent2.W4[i]
        
        return child

    def mutate(self, genome):
        """Mutate the genome"""
        mutation_power = 0.2  # Hyperparameter: mutation strength
        mutation_rate = 0.2   # Probability of mutation
        
        if random.random() < mutation_rate:
            genome.W1 += np.random.randn(*genome.W1.shape) * mutation_power
            genome.W2 += np.random.randn(*genome.W2.shape) * mutation_power
            genome.W3 += np.random.randn(*genome.W3.shape) * mutation_power
            genome.W4 += np.random.randn(*genome.W4.shape) * mutation_power
