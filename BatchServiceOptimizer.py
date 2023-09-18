from BatchServiceSimulationWithMB import BatchServiceSimulationWithMB
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

class OptimizeBatchService:
    def __init__(self, lambda_rate, mu, a, b, m, sim_time, c, d, k1, k2):
        self.lambda_rate = lambda_rate
        self.mu = mu
        self.initial_a = a
        self.b = b
        self.m = m
        self.sim_time = sim_time
        self.c = c
        self.d = d
        self.k1 = k1
        self.k2 = k2
        self.objective_values = []
        self.waiting_time_values = []
        self.cost_values = []
        self.index = 0

    def objective_function(self, a):
        print('\n')
        print('繰り返し回数: {0}回目'.format(self.index))
        self.index += 1
        simulation = BatchServiceSimulationWithMB(self.lambda_rate, self.mu, a, self.b, self.m, self.sim_time, self.c, self.d)
        simulation.run_simulation()
        avg_queue_length, _, _ = simulation.calculate_metrics()
        cost = self.k1 * a + self.k2 * (a ** 2)
        total = avg_queue_length + cost
        self.objective_values.append(total)
        self.waiting_time_values.append(avg_queue_length)
        self.cost_values.append(cost)
        return total

    def optimize(self):
        #initial_a = 1  # Initial value of a
        res = minimize_scalar(self.objective_function, self.initial_a, bounds=(1, self.b), method='bounded')
        optimal_a = res.x
        optimal_value = res.fun

        # 最適化の成功を確認
        if res.success:
            print("最適化に成功しました。最小値:", res.fun)
        else:
            print("最適化に失敗しました。エラーコード:", res.status)

        # Show the plot of objective function, waiting time, and cost values
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.plot(self.objective_values)
        plt.xlabel('Iterations')
        plt.ylabel('Objective Function Value')
        plt.title('Convergence of Objective Function')

        plt.subplot(1, 3, 2)
        plt.plot(self.waiting_time_values)
        plt.xlabel('Iterations')
        plt.ylabel('Waiting Time')
        plt.title('Convergence of Waiting Queue')

        plt.subplot(1, 3, 3)
        plt.plot(self.cost_values)
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.title('Convergence of Cost')

        plt.tight_layout()
        plt.savefig("convergence_plot.png")
        plt.show()
        
        plt.figure(figsize=(10, 5))
        
        plt.plot(self.objective_values, label='Objective Function Value')
        plt.plot(self.waiting_time_values, label='Waiting Queue')
        plt.plot(self.cost_values, label='Cost')
        
        plt.xlabel('Iterations')
        plt.ylabel('Value')
        plt.title('Convergence of Optimization')
        plt.legend()

        plt.savefig("convergence_plot_combined.png")
        plt.show()

        # Display the initial and optimal objective function value, waiting time and cost
        initial_value = self.objective_function(self.initial_a)
        initial_waiting_time = initial_value - (self.k1 * self.initial_a + self.k2 * (self.initial_a ** 2))
        optimal_waiting_time = optimal_value - (self.k1 * optimal_a + self.k2 * (optimal_a ** 2))

        print(f"Initial waiting time: {initial_waiting_time}, Initial cost: {self.k1 * self.initial_a + self.k2 * (self.initial_a ** 2)}, Initial objective function value: {initial_value}")
        print(f"Optimal waiting time: {optimal_waiting_time}, Optimal cost: {self.k1 * optimal_a + self.k2 * (optimal_a ** 2)}, Optimal objective function value: {optimal_value}")
        #print(f"Initial waiting time: {initial_waiting_time}, Initial cost: {self.k1 * initial_a + self.k2 * (initial_a ** 2)}, Initial objective function value: {initial_value}")
        #print(f"Optimal waiting time: {optimal_waiting_time}, Optimal cost: {self.k1 * optimal_a + self.k2 * (optimal_a ** 2)}, Optimal objective function value: {optimal_value}")
        print(f"Initial value of a: {self.initial_a}")  # Displaying the optimal value of a
        print(f"Optimal value of a: {optimal_a}")  # Displaying the optimal value of a


        return optimal_a, optimal_value

if __name__ == '__main__':
    # Given parameters
    lambda_rate = 3
    mu = 3
    a = 1
    b = 5
    m = 3
    sim_time = 100000
    c = 3
    d = 5
    k1 = 1
    k2 = 1

    optimizer = OptimizeBatchService(lambda_rate, mu, a, b, m, sim_time, c, d, k1, k2)
    optimal_a, optimal_value = optimizer.optimize()
