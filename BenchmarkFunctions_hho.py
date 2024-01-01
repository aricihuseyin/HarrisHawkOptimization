import random
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Solution:
    def __init__(self):
        self.best = float("inf")
        self.bestIndividual = []
        self.convergence = []
        self.executionTime = 0
        self.optimizer = ""
        self.startTime = ""
        self.endTime = ""
        self.objfname = ""

def HHO(objf, lb, ub, dim, SearchAgents_no, Max_iter):
    # dim=30
    # SearchAgents_no=50
    # lb=-100
    # ub=100
    # Max_iter=500

    # initialize the location and Energy of the rabbit
    Rabbit_Location = np.zeros(dim)
    Rabbit_Energy = float("inf")  # change this to -inf for maximization problems

    if not isinstance(lb, list):
        lb = [lb for _ in range(dim)]
        ub = [ub for _ in range(dim)]
    lb = np.asarray(lb)
    ub = np.asarray(ub)

    # Initialize the locations of Harris' hawks
    X = np.asarray([x * (ub - lb) + lb for x in np.random.uniform(0, 1, (SearchAgents_no, dim))])

    # Initialize convergence
    convergence_curve = np.zeros(Max_iter)

    ############################
    s = Solution()

    print("HHO is now tackling  \"" + objf.__name__ + "\"")

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    ############################

    t = 0  # Loop counter

    # Main loop
    while t < Max_iter:
        for i in range(0, SearchAgents_no):

            # Check boundaries
            X[i, :] = np.clip(X[i, :], lb, ub)

            # fitness of locations
            fitness = objf(X[i, :])

            # Update the location of Rabbit
            if fitness < Rabbit_Energy:  # Change this to > for maximization problem
                Rabbit_Energy = fitness
                Rabbit_Location = X[i, :].copy()

        E1 = 2 * (1 - (t / Max_iter))  # factor to show the decreasing energy of rabbit

        # Update the location of Harris' hawks
        for i in range(0, SearchAgents_no):

            E0 = 2 * random.random() - 1  # -1<E0<1
            Escaping_Energy = E1 * (E0)  # escaping energy of rabbit Eq. (3) in the paper

            # -------- Exploration phase Eq. (1) in paper -------------------

            if abs(Escaping_Energy) >= 1:
                # Harris' hawks perch randomly based on 2 strategy:
                q = random.random()
                rand_Hawk_index = math.floor(SearchAgents_no * random.random())
                X_rand = X[rand_Hawk_index, :]
                if q < 0.5:
                    # perch based on other family members
                    X[i, :] = X_rand - random.random() * abs(X_rand - 2 * random.random() * X[i, :])

                elif q >= 0.5:
                    # perch on a random tall tree (random site inside group's home range)
                    X[i, :] = (Rabbit_Location - X.mean(0)) - random.random() * ((ub - lb) * random.random() + lb)

            # -------- Exploitation phase -------------------
            elif abs(Escaping_Energy) < 1:
                # Attacking the rabbit using 4 strategies regarding the behavior of the rabbit

                # phase 1: ----- surprise pounce (seven kills) ----------
                # surprise pounce (seven kills): multiple, short rapid dives by different hawks

                r = random.random()  # probability of each event

                if r >= 0.5 and abs(Escaping_Energy) < 0.5:  # Hard besiege Eq. (6) in paper
                    X[i, :] = (Rabbit_Location) - Escaping_Energy * abs(Rabbit_Location - X[i, :])

                if r >= 0.5 and abs(Escaping_Energy) >= 0.5:  # Soft besiege Eq. (4) in paper
                    Jump_strength = 2 * (1 - random.random())  # random jump strength of the rabbit
                    X[i, :] = (Rabbit_Location - X[i, :]) - Escaping_Energy * abs(
                        Jump_strength * Rabbit_Location - X[i, :])

                # phase 2: --------performing team rapid dives (leapfrog movements)----------

                if r < 0.5 and abs(Escaping_Energy) >= 0.5:  # Soft besiege Eq. (10) in paper
                    # rabbit try to escape by many zigzag deceptive motions
                    Jump_strength = 2 * (1 - random.random())
                    X1 = Rabbit_Location - Escaping_Energy * abs(Jump_strength * Rabbit_Location - X[i, :])
                    X1 = np.clip(X1, lb, ub)

                    if objf(X1) < fitness:  # improved move?
                        X[i, :] = X1.copy()
                    else:  # hawks perform levy-based short rapid dives around the rabbit
                        X2 = Rabbit_Location - Escaping_Energy * abs(
                            Jump_strength * Rabbit_Location - X[i, :]) + np.multiply(np.random.randn(dim),
                                                                                        Levy(dim))
                        X2 = np.clip(X2, lb, ub)
                        if objf(X2) < fitness:
                            X[i, :] = X2.copy()
                if r < 0.5 and abs(Escaping_Energy) < 0.5:  # Hard besiege Eq. (11) in paper
                    Jump_strength = 2 * (1 - random.random())
                    X1 = Rabbit_Location - Escaping_Energy * abs(Jump_strength * Rabbit_Location - X.mean(0))
                    X1 = np.clip(X1, lb, ub)

                    if objf(X1) < fitness:  # improved move?
                        X[i, :] = X1.copy()
                    else:  # Perform levy-based short rapid dives around the rabbit
                        X2 = Rabbit_Location - Escaping_Energy * abs(
                            Jump_strength * Rabbit_Location - X.mean(0)) + np.multiply(np.random.randn(dim),
                                                                                          Levy(dim))
                        X2 = np.clip(X2, lb, ub)
                        if objf(X2) < fitness:
                            X[i, :] = X2.copy()

        convergence_curve[t] = Rabbit_Energy
        if (t % 1 == 0):
            print(['At iteration ' + str(t) + ' the best fitness is ' + str(Rabbit_Energy)])
        t = t + 1

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = convergence_curve
    s.optimizer = "HHO"
    s.objfname = objf.__name__
    s.best = Rabbit_Energy
    s.bestIndividual = Rabbit_Location

    return s


def Levy(dim):
    beta = 1.5
    sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) / (
            math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = 0.01 * np.random.randn(dim) * sigma
    v = np.random.randn(dim)
    zz = np.power(np.absolute(v), (1 / beta))
    step = np.divide(u, zz)
    return step

def run_optimization(obj_function, lb, ub, dim, SearchAgents_no, Max_iter, runs=10):
    results = []

    for run in range(runs):
        result = HHO(obj_function, lb, ub, dim, SearchAgents_no, Max_iter)
        results.append(result)

    return results

def print_results_table(results_list):
    print("\nResults Summary:")
    print("{:<15} {:<15} {:<15} {:<15} {:<15}".format(
        "Function", "Avg Fitness", "Std Fitness", "Avg Execution Time", "Std Execution Time"))
    print("-" * 80)

    for results in results_list:
        fitness_values = [run.best for run in results]
        execution_times = [run.executionTime for run in results]

        avg_fitness = np.mean(fitness_values)
        std_fitness = np.std(fitness_values)
        avg_execution_time = np.mean(execution_times)
        std_execution_time = np.std(execution_times)

        print("{:<15} {:<15.6f} {:<15.6f} {:<15.6f} {:<15.6f}".format(
            results[0].objfname, avg_fitness, std_fitness, avg_execution_time, std_execution_time))

    print("-" * 80)

# Benchmark Fonksiyonları
def rosenbrock(x):
    return sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def rastrigin(x):
    return 10 * len(x) + sum(x**2 - 10 * np.cos(2 * np.pi * x))

def griewank(x):
    return 1 + sum(x**2 / 4000) - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))

def ackley(x):
    n = len(x)
    return -20 * np.exp(-0.2 * np.sqrt(sum(x**2) / n)) - np.exp(sum(np.cos(2 * np.pi * x)) / n) + 20 + np.exp(1)

def sphere(x):
    return sum(x**2)

def plot_best_score_iteration(result, benchmark_name, Max_iter, label=None):
    iterations = list(range(1, Max_iter + 1))
    best_scores = result[0].convergence  # 'best_scores' yerine 'convergence' kullanıldı
    plt.plot(iterations, best_scores, label=label)


def plot_3d_hho(obj_function, solution, title):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    x = np.linspace(lb, ub, 100)
    y = np.linspace(lb, ub, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = obj_function(np.array([X[i, j], Y[i, j]]))

    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax.scatter(solution.bestIndividual[0], solution.bestIndividual[1], solution.best, color='red', s=100, label='Best Solution')

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title(title)
    ax.legend()

    plt.show()

if __name__ == "__main__":
    lb = -5.12
    ub = 5.12
    dim = 2
    SearchAgents_no = 20
    Max_iter = 50

    benchmark_functions = [rosenbrock, ackley, sphere, griewank, rastrigin]

    results_list = []
    for obj_function in benchmark_functions:
        runs = run_optimization(obj_function, lb, ub, dim, SearchAgents_no, Max_iter)
        results_list.append(runs)

    # Ortalamayı hesapla
    all_convergences = np.array([result[0].convergence for result in results_list])
    avg_results = np.mean(all_convergences, axis=0)

    # Ortalama grafiği çiz
    plt.figure(figsize=(10, 6))
    for i, (obj_function, results) in enumerate(zip(benchmark_functions, results_list)):
        plot_best_score_iteration(results, obj_function.__name__, Max_iter, label=f'{obj_function.__name__} - Benchmark Function {i + 1}')

    plt.plot(list(range(1, Max_iter + 1)), avg_results, label='Average', color='black')
    plt.title('Best Score vs. Number of Iterations (Benchmark Functions)')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Best Score')
    plt.legend()
    plt.grid(True)
    plt.show()

    print_results_table(results_list)

    for obj_function, results in zip(benchmark_functions, results_list):
        plot_3d_hho(obj_function, results[0], f'{obj_function.__name__} - HHO')