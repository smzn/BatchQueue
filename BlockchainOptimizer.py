import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from BlockChain_lib import BlockChain_lib
from scipy.optimize import basinhopping

class BlockchainOptimizer:
    def __init__(self, k1=1e-3, k2=0, initial_capacity=1e6):
        self.k1 = k1
        self.k2 = k2
        self.initial_capacity = initial_capacity
        self.capacities = [initial_capacity]
        self.objective_values = []
        self.L_as = []
        self.C_as = []

    def objective_function(self, capacity):
        blockchain = BlockChain_lib(lmd=1.0, mean=550, sd=50, time=500000, capacity=capacity)
        L_a = blockchain.getSimulation()
        C_a = self.k1 * capacity + self.k2 * capacity**2
        print('目的関数値:{0}'.format(L_a + C_a))
        print('平均系内人数:{0}'.format(L_a ))
        print('コスト:{0}'.format(C_a))
        self.capacities.append(capacity)
        self.objective_values.append(L_a + C_a)
        self.L_as.append(L_a)
        self.C_as.append(C_a)
        return L_a + C_a

    def callback(self, xk):
        self.capacities.append(xk[0])
        obj_value = self.objective_function(xk[0])
        self.objective_values.append(obj_value)
        self.L_as.append(obj_value - (self.k1 * xk[0] + self.k2 * xk[0]**2))
        self.C_as.append(self.k1 * xk[0] + self.k2 * xk[0]**2)

    def optimize(self):
        initial_obj_value = self.objective_function(self.initial_capacity)
        self.objective_values.append(initial_obj_value)
        self.L_as.append(initial_obj_value - (self.k1 * self.initial_capacity + self.k2 * self.initial_capacity**2))
        self.C_as.append(self.k1 * self.initial_capacity + self.k2 * self.initial_capacity**2)
        
        #result = minimize(self.objective_function, self.initial_capacity, bounds=[(1e2, 1e10)], callback=self.callback)
        result = minimize(self.objective_function, self.initial_capacity, bounds=[(1e2, 1e10)])
        #result = minimize(self.objective_function, self.initial_capacity, bounds=[(1e1, 1e10)], callback=self.callback, method='Powell')
        #result = basinhopping(self.objective_function, self.initial_capacity, minimizer_kwargs={"bounds": [(1e1, 1e10)]})
        
        self.optimal_capacity = result.x[0]
        self.optimal_obj_value = result.fun
        self.optimal_La = self.optimal_obj_value - (self.k1 * self.optimal_capacity + self.k2 * self.optimal_capacity**2)
        self.optimal_Ca = self.k1 * self.optimal_capacity + self.k2 * self.optimal_capacity**2

    def display_results(self):
        print("Before optimization:")
        print('Initial Capacity:', self.initial_capacity)
        print("Average number in the system:", self.L_as[0])
        print("Objective function value:", self.objective_values[0])
        print("Cost function value:", self.C_as[0])

        print("\nAfter optimization:")
        print('Optimal Capacity:', self.optimal_capacity)
        print("Average number in the system:", self.optimal_La)
        print("Objective function value:", self.optimal_obj_value)
        print("Cost function value:", self.optimal_Ca)

    def plot_against_iterations(self):
        iterations = list(range(len(self.capacities)))

        plt.figure(figsize=(10, 6))
        plt.plot(iterations, self.L_as, label='Average number in the system')
        plt.plot(iterations, self.objective_values, label='Objective function value', linestyle='--')
        plt.plot(iterations, self.C_as, label='Cost function value', linestyle='-.')
        plt.legend()
        plt.title("Evolution over Optimization Iterations")
        plt.xlabel("Number of Iterations")
        plt.grid(True)
        plt.savefig("optimization_evolution_iterations.png")
        plt.show()

# 使用例
optimizer = BlockchainOptimizer()
optimizer.optimize()
optimizer.display_results()
optimizer.plot_against_iterations()
