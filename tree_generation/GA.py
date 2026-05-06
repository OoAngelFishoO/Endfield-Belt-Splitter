import random
import numpy as np
from typing import List, Tuple, Optional
from fractions import Fraction
import dataclasses
from conveyer_tree import Node

@dataclasses.dataclass
class Config:
    # tree
    INPUT_VAL: Fraction = Fraction(1, 1)
    MAX_DEPTH: int = 5
    
    # target
    TARGET_VAL: float = 0.4
    
    # GA parameters
    POPULATION_SIZE: int = 100
    MAX_GENERATIONS: int = 500
    CROSSOVER_RATE: float = 0.7
    MUTATION_RATE: float = 0.3
    TOURNAMENT_SIZE: int = 3
    ELITISM_COUNT: int = 5

class Individual:
    def __init__(self, chromosome: Node):
        self.chromosome = chromosome
        self.fitness: Optional[float] = None
    
    def evaluate(self) -> None:
        error = abs(float(self.chromosome.get_output()) - Config.TARGET_VAL)
        cost = self.chromosome.get_cost()
        
        if cost == 0:
            self.fitness = float('inf')
        else:
            self.fitness = error * cost
    
    def __lt__(self, other) -> bool:
        if not isinstance(other, Individual):
            raise TypeError('cannot compare between `Individual` and `%s`' % type(other).__name__)
        
        if self.fitness is None:
            return False
        if other.fitness is None:
            return True
        
        return self.fitness < other.fitness

class SingleObjectiveGA:
    def __init__(self):
        self.population: List[Individual] = []
        self.generation: int = 0
        self.best_individual: Optional[Individual] = None
        self.best_fitness_history: List[float] = []
        self.avg_fitness_history: List[float] = []
    
    def initialize_population(self) -> None:
        self.population = []
        
        while len(self.population) < Config.POPULATION_SIZE:
            random_tree = Node.create_random_tree(
                input_val=Config.INPUT_VAL, 
                max_depth=Config.MAX_DEPTH
            )
            individual = Individual(random_tree)
            individual.evaluate()
            self.population.append(individual)
        
        self.best_individual = min(self.population)
    
    def tournament_selection(self, tournament_size: int = Config.TOURNAMENT_SIZE) -> Individual:
        candidates = random.sample(self.population, k=tournament_size)
        return min(candidates)
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        child1_tree, child2_tree = Node.crossover_trees(
            parent1.chromosome, 
            parent2.chromosome
        )
        child1 = Individual(child1_tree)
        child2 = Individual(child2_tree)
        return child1, child2
    
    def mutation(self, individual: Individual) -> Individual:
        tree = individual.chromosome.deep_copy()
        tree.mutate(max_depth=Config.MAX_DEPTH)
        mutated = Individual(tree)
        return mutated
    
    def generate_offspring(self) -> List[Individual]:
        offspring = []
        
        sorted_population = sorted(self.population)
        offspring.extend([Individual(ind.chromosome.deep_copy()) for ind in sorted_population[:Config.ELITISM_COUNT]])
        
        while len(offspring) < Config.POPULATION_SIZE:
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()
            
            while parent1 is parent2:
                parent2 = self.tournament_selection()
            
            if random.random() < Config.CROSSOVER_RATE:
                child1, child2 = self.crossover(parent1, parent2)
            else:
                child1 = Individual(parent1.chromosome.deep_copy())
                child2 = Individual(parent2.chromosome.deep_copy())
            
            child1 = self.mutation(child1)
            child2 = self.mutation(child2)
            
            child1.evaluate()
            child2.evaluate()
            
            offspring.extend([child1, child2])
        
        return offspring[:Config.POPULATION_SIZE]
    
    def selection(self, offspring: List[Individual]) -> None:
        combined = self.population + offspring
        
        combined.sort()
        self.population = combined[:Config.POPULATION_SIZE]
        
        current_best = self.population[0]
        if self.best_individual is None or current_best < self.best_individual:
            self.best_individual = current_best
    
    def run_generation(self) -> None:
        offspring = self.generate_offspring()
        
        self.selection(offspring)
        
        fitness_values = [ind.fitness for ind in self.population if ind.fitness is not None]
        self.best_fitness_history.append(min(fitness_values))
        self.avg_fitness_history.append(float(np.mean(fitness_values)))
        
        self.generation += 1
    
    def run(self) -> None:
        print(f"Starting Single-Objective GA with target: {Config.TARGET_VAL}")
        print(f"Objective: minimize error/cost")
        print(f"Population size: {Config.POPULATION_SIZE}")
        print(f"Max generations: {Config.MAX_GENERATIONS}")
        print(f"Crossover rate: {Config.CROSSOVER_RATE}")
        print(f"Mutation rate: {Config.MUTATION_RATE}")
        print("-" * 60)
        
        self.initialize_population()
        
        for generation in range(Config.MAX_GENERATIONS):
            self.run_generation()
            
            if generation % 10 == 0 or generation == Config.MAX_GENERATIONS - 1:
                best = self.best_individual
                if best and best.fitness is not None:
                    error = abs(float(best.chromosome.get_output()) - Config.TARGET_VAL)
                    cost = best.chromosome.get_cost()
                    
                    print(f"Generation {generation}:")
                    print(f"  Best fitness (error/cost): {best.fitness:.6f}")
                    print(f"  Best error: {error:.6f}")
                    print(f"  Best cost: {cost}")
                    print(f"  Best output: {float(best.chromosome.get_output()):.6f}")
                    print(f"  Avg fitness: {self.avg_fitness_history[-1]:.6f}")
                    print()
    
    def print_results(self) -> None:        
        if not self.best_individual:
            print("No results available.")
            return
        
        best = self.best_individual
        error = abs(float(best.chromosome.get_output()) - Config.TARGET_VAL)
        cost = best.chromosome.get_cost()
        output = float(best.chromosome.get_output())
        
        print(f"Best Solution Found:")
        print(f"  Fitness (error/cost): {best.fitness:.10f}")
        print(f"  Error: {error:.10f}")
        print(f"  Cost: {cost}")
        print(f"  Output: {output:.10f}")
        print(f"  Target: {Config.TARGET_VAL}")
        print(f"  Relative error: {error/Config.TARGET_VAL*100:.4f}%")
        print()
        print(f"Tree details:")
        print(best.chromosome)
        print()
        
        if len(self.best_fitness_history) > 0:
            print(f"Convergence history:")
            print(f"  Initial best fitness: {self.best_fitness_history[0]:.6f}")
            print(f"  Final best fitness: {self.best_fitness_history[-1]:.6f}")
            print(f"  Improvement: {(self.best_fitness_history[0] - self.best_fitness_history[-1])/self.best_fitness_history[0]*100:.2f}%")
            
            if len(self.best_fitness_history) > 10:
                recent_best = self.best_fitness_history[-10:]
                max_change = max(recent_best) - min(recent_best)
                if max_change < self.best_fitness_history[-1] * 0.01: 
                    print(f"  Converged around generation {len(self.best_fitness_history) - 10}")
        
        unique_trees = len(set(repr(ind.chromosome) for ind in self.population))
        print(f"Population diversity (unique trees): {unique_trees}/{len(self.population)}")
        
        print("\nTop 5 solutions:")
        sorted_population = sorted(self.population)[:5]
        for i, ind in enumerate(sorted_population):
            if ind.fitness is None:
                continue
            ind_error = abs(float(ind.chromosome.get_output()) - Config.TARGET_VAL)
            ind_cost = ind.chromosome.get_cost()
            print(f"{i+1}. Fitness: {ind.fitness:.6f}, Error: {ind_error:.6f}, Cost: {ind_cost}, Output: {float(ind.chromosome.get_output()):.6f}")


if __name__ == '__main__':
    ga = SingleObjectiveGA()
    ga.run()
    ga.print_results()
