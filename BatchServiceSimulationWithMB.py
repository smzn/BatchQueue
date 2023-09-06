# Modifying the class to include Minimum Batch Policy (MB) in service and batch size distribution in arrivals

import numpy as np
import matplotlib.pyplot as plt

class BatchServiceSimulationWithMB:
    def __init__(self, lambda_rate, mu, a, b, m, sim_time, c, d):
        self.lambda_rate = lambda_rate  # Arrival rate of batches
        self.mu = mu  # Service rate
        self.a = a  # Minimum batch size for service to start
        self.b = b  # Maximum batch size for service
        self.m = m  # Number of servers
        self.sim_time = sim_time  # Simulation time
        self.c = c  # Minimum batch size in arrivals
        self.d = d  # Maximum batch size in arrivals
        self.initialize_variables()

    def initialize_variables(self):
        self.event_time = []
        self.event_type = []
        self.num_in_system = []
        self.server_utilization = []
        self.num_servers_busy = 0
        self.total_people = 0
        self.next_arrival = np.random.exponential(1 / self.lambda_rate)
        self.next_departure = np.inf
        self.current_time = 0
        self.max_queue = 0
        self.min_queue = np.inf

    def run_simulation(self):
        while self.current_time < self.sim_time:
            if self.next_arrival < self.next_departure:
                self.handle_arrival_event()
            else:
                self.handle_departure_event()

    def handle_arrival_event(self):
        self.current_time = self.next_arrival
        batch_size = np.random.randint(self.c, self.d + 1)  # Including the effect of batch size distribution in arrivals
        self.total_people += batch_size

        self.event_time.append(self.current_time)
        self.event_type.append('arrival')
        self.num_in_system.append(self.total_people)
        self.server_utilization.append(self.num_servers_busy / self.m)

        if self.num_servers_busy < self.m and self.total_people >= self.a:  # Including MB policy in service
            self.next_departure = self.current_time + np.random.exponential(1 / (self.mu * min(self.total_people, self.b)))
            self.num_servers_busy += 1

        self.next_arrival = self.current_time + np.random.exponential(1 / self.lambda_rate)

    def handle_departure_event(self):
        self.current_time = self.next_departure

        self.event_time.append(self.current_time)
        self.event_type.append(f'departure_server_{self.num_servers_busy}')
        self.num_in_system.append(self.total_people)
        self.server_utilization.append(self.num_servers_busy / self.m)

        if self.total_people >= self.b:
            self.total_people -= self.b
        elif self.total_people >= self.a:
            self.total_people -= self.total_people

        if self.total_people >= self.a:
            self.next_departure = self.current_time + np.random.exponential(1 / (self.mu * min(self.total_people, self.b)))
        else:
            self.next_departure = np.inf
            self.num_servers_busy -= 1

    def plot_results(self, filename):
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.step(self.event_time, self.num_in_system, where='post')
        plt.xlabel('Time')
        plt.ylabel('Number in system')
        plt.title('Number of batches in the system over time')

        plt.subplot(2, 1, 2)
        plt.step(self.event_time, self.server_utilization, where='post')
        plt.xlabel('Time')
        plt.ylabel('Server Utilization')
        plt.title('Server Utilization over time')

        plt.tight_layout()
        plt.savefig(filename)
        plt.show()

    def calculate_metrics(self):
        avg_num_in_system = np.mean(self.num_in_system)
        avg_queue_length = avg_num_in_system - (np.mean(self.server_utilization) * self.m)
        avg_server_utilization = np.mean(self.server_utilization)

        # Calculate the theoretical and simulated system utilization
        rho_theoretical = self.lambda_rate * ((self.a + self.b) / 2) / (self.m * self.mu)
        rho_simulated = avg_num_in_system / self.m
        
        # Print the metrics
        print(f"平均系内人数（シミュレーション）: {avg_num_in_system}")
        print(f"平均待ち人数（シミュレーション）: {avg_queue_length}")
        print(f"平均サーバー利用率（シミュレーション）: {avg_server_utilization}")
        print(f"システム利用率（理論値）: {rho_theoretical}")
        print(f"システム利用率（シミュレーション）: {rho_simulated}")

        return avg_num_in_system, avg_queue_length, avg_server_utilization, rho_theoretical, rho_simulated

if __name__ == "__main__":
    # Parameters
    lambda_rate = 5  # arrival rate (batches per minute)
    mu = 2  # service rate (batches per minute)
    a, b = 3, 5  # min and max batch size for service
    m = 3  # number of servers
    sim_time = 60  # simulation time in minutes
    c, d = 3, 5  # min and max batch size in arrivals

    # Create simulation object
    simulation = BatchServiceSimulationWithMB(lambda_rate, mu, a, b, m, sim_time, c, d)

    # Run the simulation
    simulation.run_simulation()

    # Plot the results
    simulation.plot_results("batch_simulation_with_MB.png")

    # Calculate metrics
    avg_num_in_system, avg_queue_length, avg_server_utilization, rho_theoretical, rho_simulated = simulation.calculate_metrics()
    avg_num_in_system, avg_queue_length, avg_server_utilization, rho_theoretical, rho_simulated
