from typing import Callable, List, Optional, Tuple
from fractions import Fraction
import numpy as np
import dataclasses
import random

from conveyer_tree import Node

@dataclasses.dataclass
class Config(object):
    # tree
    INPUT_VAL: Fraction = Fraction(1, 1)
    MAX_DEPTH: int = 5

    # target
    # TARGET_VAL: float = (np.sqrt(5) - 1) / 2
    TARGET_VAL: float = 1 / np.e
    # TARGET_VAL: float = 3/7
    OBJECTIVES: int = 2

    # NSGA-II
    MUTATION_RATE = 0.8
    CROSSOVER_RATE = 0.8
    TOURNAMENT_SIZE = 5

# Individual representing one solution
class Individual:
    __slots__ = ['chromosome', 'objectives', 'rank', 'crowding_distance', 
                 'domination_count', 'dominated_set']
    
    chromosome: Node                 # Decision variables
    objectives: np.ndarray           # Objective values
    rank: int                        # Non-dominated rank
    crowding_distance: float         # Crowding distance
    domination_count: int            # Number of solutions dominating this
    dominated_set: List["Individual"] # Solutions dominated by this
    
    def __init__(self, chromosome: Node):
        self.chromosome = chromosome
    
    def __lt__(self, other) -> bool:
        # For sorting: first compare rank, then crowding distance
        if not isinstance(other, Individual):
            raise TypeError(...)
        
        if self.rank != other.rank:
            return self.rank < other.rank
        
        if self.crowding_distance == float('inf') and other.crowding_distance == float('inf'):
            return False
        if self.crowding_distance == float('inf'):
            return True
        if other.crowding_distance == float('inf'):
            return False
        
        return self.crowding_distance > other.crowding_distance

# NSGA-II main class
class NSGA2(object):
    def __init__(self, 
                 population_size: int,
                 max_generations: int,
                 crossover_rate: float,
                 mutation_rate: float):
        
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.population: List[Individual] = []
        self.generation = 0

    def get_progress_snapshot(self) -> dict:
        snapshot = {
            "generation": self.generation,
            "pareto_front_size": 0,
            "front_count": len(getattr(self, "fronts", [])),
            "avg_error": None,
            "avg_cost": None,
        }
        if self.fronts and self.fronts[0]:
            first_front_objs = [ind.objectives for ind in self.fronts[0]]
            avg_obj = np.mean(first_front_objs, axis=0)
            snapshot["pareto_front_size"] = len(self.fronts[0])
            snapshot["avg_error"] = float(avg_obj[0])
            snapshot["avg_cost"] = float(avg_obj[1])
        return snapshot
        
    # Initialize random population
    def initialize_population(self) -> None:
        assert not self.population, "population has already been initialized. "

        while len(self.population) < self.population_size:
            random_tree = Node.create_random_tree(
                input_val=Config.INPUT_VAL, 
                max_depth=Config.MAX_DEPTH
            )
            individual = Individual(random_tree)
            self.population.append(individual)
    
    # Evaluate all individuals
    def evaluate_population(self, population: List[Individual]) -> None:
        for individual in population:
            error = abs(individual.chromosome.get_output() - Config.TARGET_VAL)
            cost  = individual.chromosome.get_cost()
            individual.objectives = np.array([
                # error / cost, 
                error, 
                cost
            ])
    
    # def non_dominated_sort(self) -> None:
    #     self.fronts: List[List[Individual]] = []

    #     first_front = []
    #     for individual_p in self.population:
    #         individual_p.dominated_set = []
    #         individual_p.domination_count = 0
    #         for individual_q in self.population:
    #             if self.dominates(individual_p, individual_q):
    #                 individual_p.dominated_set.append(individual_q)
    #             elif self.dominates(individual_q, individual_p):
    #                 individual_p.domination_count += 1
            
    #         if individual_p.domination_count == 0:
    #             individual_p.rank = 0
    #             first_front.append(individual_p)
        
    #     self.fronts.append(first_front)

    #     i = 0
    #     while self.fronts[i]:
    #         next_front = []
    #         for individual_p in self.fronts[i]:
    #             for individual_q in individual_p.dominated_set:
    #                 individual_q.domination_count -= 1
    #                 if individual_q.domination_count == 0:
    #                     individual_q.rank = i + 1
    #                     next_front.append(individual_q)
    #         i += 1
    #         if len(self.fronts) == i:
    #             self.fronts.append(next_front)
    #         else:
    #             self.fronts[i] = next_front
        
    #     self.fronts.pop()

    # Non-dominated sort, vectorized
    def non_dominated_sort(self) -> None:
        self.fronts = []
        
        pop_size = len(self.population)
        obj_matrix = np.zeros((pop_size, Config.OBJECTIVES))
        
        for i, ind in enumerate(self.population):
            obj_matrix[i] = ind.objectives
        
        dominated_count = np.zeros(pop_size, dtype=int)
        dominated_sets = [[] for _ in range(pop_size)]
        
        for i in range(pop_size):
            less_equal = np.all(obj_matrix[i] <= obj_matrix, axis=1)
            less = np.any(obj_matrix[i] < obj_matrix, axis=1)
            dominates_j = less_equal & less
            
            for j in range(pop_size):
                if i != j and dominates_j[j]:
                    dominated_sets[i].append(j)
                    dominated_count[j] += 1
        
        current_rank = 0
        remaining = set(range(pop_size))
        
        while remaining:
            front_indices = [i for i in remaining if dominated_count[i] == 0]
            front = [self.population[i] for i in front_indices]
            
            for i in front_indices:
                self.population[i].rank = current_rank
            
            self.fronts.append(front)
            
            for i in front_indices:
                for j in dominated_sets[i]:
                    dominated_count[j] -= 1
            
            remaining -= set(front_indices)
            current_rank += 1
                
    # Calculate crowding distance
    def calculate_crowding_distance(self, front: List[Individual]) -> None:
        l = len(front)
        for i in range(l):
            front[i].crowding_distance = 0
        for m in range(Config.OBJECTIVES):
            sorted_front = sorted(front, key=lambda x: x.objectives[m])

            sorted_front[0].crowding_distance = np.inf
            sorted_front[-1].crowding_distance = np.inf

            f_min = sorted_front[0] .objectives[m]
            f_max = sorted_front[-1].objectives[m]

            for i in range(1, l - 1):
                sorted_front[i].crowding_distance = sorted_front[i].crowding_distance + (
                    sorted_front[i + 1].objectives[m] - sorted_front[i - 1].objectives[m]
                ) / (f_max - f_min + 1e-6)
    
    # # Check if individual_1 dominates individual_2
    # def dominates(self, individual_1: Individual, individual_2: Individual) -> bool:
    #     return bool(
    #         np.all(individual_1.objectives <= individual_2.objectives) and 
    #         np.any(individual_1.objectives < individual_2.objectives)
    #     )
    
    # Tournament selection
    def tournament_selection(self, tournament_size: int = Config.TOURNAMENT_SIZE) -> Individual:
        candidates = random.sample(self.population, k=tournament_size)
        return min(candidates)
    
    # Crossover
    def crossover(self, parent_1: Individual, parent_2: Individual) -> Tuple[Individual, Individual]:
        child_1, child_2 = Node.crossover_trees(parent_1.chromosome, parent_2.chromosome)
        return Individual(child_1), Individual(child_2)
    
    # mutation
    def mutation(self, individual: Individual) -> Individual:
        tree = individual.chromosome.deep_copy()
        tree.mutate(max_depth=Config.MAX_DEPTH)
        return Individual(tree)
    
    # Generate offspring
    def generate_offspring(self) -> List[Individual]:
        offspring = []
        
        while len(offspring) < self.population_size:
            parent_1 = self.tournament_selection()
            parent_2 = self.tournament_selection()
            
            while parent_1 is parent_2:
                parent_2 = self.tournament_selection()
            
            if random.random() < self.crossover_rate:
                child_1, child_2 = self.crossover(parent_1, parent_2)
            else:
                child_1 = Individual(parent_1.chromosome.deep_copy())
                child_2 = Individual(parent_2.chromosome.deep_copy())
            
            if random.random() <= self.mutation_rate:
                child_1 = self.mutation(child_1)
            if random.random() <= self.mutation_rate:
                child_2 = self.mutation(child_2)
        
        
            if Node.is_valid(child_1.chromosome) and Node.is_valid(child_2.chromosome):
                offspring.extend([child_1, child_2])
            elif Node.is_valid(child_1.chromosome):
                offspring.append(child_1)
            elif Node.is_valid(child_2.chromosome):
                offspring.append(child_2)
        
        return offspring[:self.population_size]
    
    # Select next generation
    def environmental_selection(self, offspring: List[Individual]) -> None:
        self.evaluate_population(offspring)
        self.population += offspring

        self.non_dominated_sort()
        
        for front in self.fronts:
            self.calculate_crowding_distance(front)
        
        next_population = []
        remaining = self.population_size
        
        for front in self.fronts:
            if len(front) <= remaining:
                next_population.extend(front)
                remaining -= len(front)
            else:
                front.sort()
                next_population.extend(front[:remaining])
                break
        
        self.population = next_population
    
    # Main loop
    def run(
        self,
        progress_callback: Optional[Callable[[dict], None]] = None,
        progress_interval: int = 10,
    ) -> None:
        self.initialize_population()
        self.evaluate_population(self.population)
        
        self.non_dominated_sort()
        for front in self.fronts:
            self.calculate_crowding_distance(front)
        
        for generation in range(self.max_generations):
            self.generation = generation
            offspring = self.generate_offspring()
            self.environmental_selection(offspring)
            
            if generation % 10 == 0:
                print(f"Generation {generation}: ")
                print(f"  Pareto front size: {len(self.fronts[0]) if self.fronts else 0}")
                print(f"  Number of fronts: {len(self.fronts)}")
                
                if self.fronts and self.fronts[0]:
                    first_front_objs = [ind.objectives for ind in self.fronts[0]]
                    avg_obj = np.mean(first_front_objs, axis=0)
                    print(f"  Avg objectives in front 0: {avg_obj}")

            if progress_callback and (generation % progress_interval == 0 or generation == self.max_generations - 1):
                progress_callback(self.get_progress_snapshot())

    def get_sorted_unique_front(self) -> List[Individual]:
        if not self.fronts:
            return []

        front = self.fronts[0]
        if not front:
            return []

        deduplicated_front: List[Individual] = []
        existing_tree = set()
        for individual in front:
            tree = individual.chromosome
            tree_string = tree.to_string()
            if tree_string not in existing_tree and Node.is_valid(tree):
                deduplicated_front.append(individual)
                existing_tree.add(tree_string)

        return sorted(deduplicated_front, key=lambda ind: (ind.objectives[0], ind.objectives[1]))

    def print_pareto_front(self, max_solutions=10):
        sorted_front = self.get_sorted_unique_front()
        if not sorted_front:
            print("Pareto front is empty.")
            return

        print(f"Pareto Front (sorted by cost, then error) - showing {min(max_solutions, len(sorted_front))} out of {len(sorted_front)} solutions:")
        for i, ind in enumerate(sorted_front[:max_solutions]):
            print(f"Solution {i+1}:")
            # print(f"  Efficiency: {ind.objectives[0]:.6f}")
            # print(f"  Error:      {ind.objectives[1]:.6f}")
            # print(f"  Cost:       {ind.objectives[2]}")
            print(f"  Error:  {ind.objectives[0]:.6f}")
            print(f"  Cost:   {ind.objectives[1]}")
            print(f"  Output: {float(ind.chromosome.get_output()):.6f}")
            # print(f"  TreeStr:{ind.chromosome.to_string()}")
            print(f"  Tree:   {ind.chromosome.format(indent=10)}")
            # print(f"  Tree:       {ind.chromosome.format(indent=14)}")
            print()

if __name__ == '__main__':
    nsga_2 = NSGA2(200, 400, Config.CROSSOVER_RATE, Config.MUTATION_RATE)
    nsga_2.run()
    nsga_2.print_pareto_front()
