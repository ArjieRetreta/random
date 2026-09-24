# ====================================================
# Genetic Algorithm Optimization
# Computational Science Laboratory
# ====================================================

import random
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------
# USER INPUT
# ----------------------------------------------------

population_size = int(input("Population Size: "))
generations = int(input("Number of Generations: "))
mutation_rate = float(input("Mutation Rate (0-1): "))

# ----------------------------------------------------
# SEARCH SPACE
# ----------------------------------------------------

LOWER_BOUND = -1
UPPER_BOUND = 2

# ----------------------------------------------------
# OBJECTIVE FUNCTION
# ----------------------------------------------------

def fitness(x):
    return x * np.sin(10 * np.pi * x) + 1

# ----------------------------------------------------
# CREATE INITIAL POPULATION
# ----------------------------------------------------

population = [
    random.uniform(LOWER_BOUND, UPPER_BOUND)
    for _ in range(population_size)
]

best_history = []

# ----------------------------------------------------
# EVOLUTION
# ----------------------------------------------------

for generation in range(generations):
    scores = [fitness(x) for x in population]
    
    best_index = np.argmax(scores)
    best_solution = population[best_index]
    best_score = scores[best_index]
    
    # Appending best score so best_history has values to plot
    best_history.append(best_score)
    
    print(f"Generation {generation + 1:3d} "
          f"Best x={best_solution:.5f} "
          f"Fitness={best_score:.5f}")
    
    # ------------------------------------------------
    # Selection
    # ------------------------------------------------
    
    selected = random.choices(
        population,
        weights=scores,
        k=population_size
    )
    
    # ------------------------------------------------
    # Crossover
    # ------------------------------------------------
    
    new_population = []
    
    for i in range(0, population_size, 2):
        parent1 = selected[i]
        
        if i + 1 < population_size:
            parent2 = selected[i + 1]
        else:
            parent2 = selected[0]
            
        child1 = (parent1 + parent2) / 2
        child2 = (parent1 + parent2) / 2
        
        # Mutation
        if random.random() < mutation_rate:
            child1 += random.uniform(-0.2, 0.2)
            
        if random.random() < mutation_rate:
            child2 += random.uniform(-0.2, 0.2)
            
        child1 = np.clip(child1, LOWER_BOUND, UPPER_BOUND)
        child2 = np.clip(child2, LOWER_BOUND, UPPER_BOUND)
        
        new_population.extend([child1, child2])
        
    population = new_population[:population_size]

# ----------------------------------------------------
# FINAL RESULT
# ----------------------------------------------------

scores = [fitness(x) for x in population]
best_index = np.argmax(scores)
best_x = population[best_index]
best_fitness = scores[best_index]

print("\nOptimization Complete")
print("----------------------------------------")
print(f"Optimal x : {best_x:.6f}")
print(f"Maximum Fitness : {best_fitness:.6f}")

# ----------------------------------------------------
# PLOT FITNESS EVOLUTION
# ----------------------------------------------------

plt.figure(figsize=(8, 5))
plt.plot(best_history, linewidth=2)
plt.title("Genetic Algorithm Optimization")
plt.xlabel("Generation")
plt.ylabel("Best Fitness")
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------------------------------------------
# PLOT OBJECTIVE FUNCTION
# ----------------------------------------------------

x = np.linspace(LOWER_BOUND, UPPER_BOUND, 500)
y = fitness(x)

plt.figure(figsize=(8, 5))
plt.plot(x, y, linewidth=2)
plt.scatter(
    best_x,
    best_fitness,
    s=120,
    color='red',
    label='Optimal Solution'
)
plt.title("Objective Function")
plt.xlabel("x")
plt.ylabel("Fitness")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
