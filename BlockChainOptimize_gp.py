import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from BlockChain_lib import BlockChain_lib
from scipy.optimize import basinhopping
from skopt import gp_minimize

class BlockChainOptimize_gp:
    def __init__(self, k1=1e-5, k2=0, initial_capacity=1e6):
        self.k1 = k1
        self.k2 = k2
        self.initial_capacity = initial_capacity
        self.capacities = [initial_capacity]
        self.objective_values = []
        self.L_as = []
        self.C_as = []

    def objective_function(self, capacity):
        capacity = float(capacity[-1])
        #print(capacity)
        #print(type(capacity))
        blockchain = BlockChain_lib(lmd=1.0, mean=550, sd=50, time=500000, capacity=capacity)
        L_a = blockchain.getSimulation()
        C_a = self.k1 * capacity + self.k2 * (capacity)**2
        print('目的関数値:{0}'.format(L_a + C_a))
        print('平均系内人数:{0}'.format(L_a ))
        print('コスト:{0}'.format(C_a))
        self.capacities.append(capacity)
        self.objective_values.append(L_a + C_a)
        self.L_as.append(L_a)
        self.C_as.append(C_a)

        return L_a + C_a
    
    def display_results(self, result):
        # Before optimization
        print("Before optimization:")
        print('Initial Capacity:', self.initial_capacity)
        print("Average number in the system:", self.L_as[0])
        print("Objective function value:", self.objective_values[0])
        print("Cost function value:", self.C_as[0])

        # After optimization
        print("\nAfter optimization:")
        optimal_capacity = self.capacities[-1]
        optimal_La = self.L_as[-1]
        optimal_obj_value = self.objective_values[-1]
        optimal_Ca = self.C_as[-1]

        print('Optimal Capacity:', optimal_capacity)
        print("Average number in the system:", optimal_La)
        print("Objective function value:", optimal_obj_value)
        print("Cost function value:", optimal_Ca)

        # 目的関数の推移をグラフ表示
        plt.figure(figsize=(10,5))
        plt.plot(result.func_vals)
        plt.title('Objective Function Value over Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Objective Function Value')
        plt.grid(True)
        plt.show()

        iterations = list(range(len(self.capacities)))
        # Plotting
        plt.figure(figsize=(10,5))
        plt.plot(iterations, self.capacities, marker='o', linestyle='-')
        plt.title('Capacities over Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Capacity')
        plt.grid(True)
        plt.show()

        iterations = list(range(len(self.capacities)-1))
        # Average number in system (L_a) over iterations
        plt.figure(figsize=(10,5))
        plt.plot(iterations, self.L_as, marker='o', linestyle='-')
        plt.title('Average number in system (L_a) over Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('L_a')
        plt.grid(True)
        plt.show()

        # Costs (C_a) over iterations
        plt.figure(figsize=(10,5))
        plt.plot(iterations, self.C_as, marker='o', linestyle='-')
        plt.title('Costs (C_a) over Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('C_a')
        plt.grid(True)
        plt.show()
    
if __name__ == "__main__":
    # ベイズ最適化の実行
    optimizer = BlockChainOptimize_gp()
    result = gp_minimize(optimizer.objective_function,
        [(1e4, 1e10)],
        acq_func="EI",
        n_calls=75,
        n_random_starts=10,
        noise=100**2,
        random_state=123)

    optimizer.display_results(result)
