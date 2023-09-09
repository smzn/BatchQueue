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
        self.cumulative_people_time = 0  # 累積系内人数時間を追跡するための変数
        self.post_time = 0

    def run_simulation(self):
        while self.current_time < self.sim_time:
            if self.next_arrival < self.next_departure:
                self.handle_arrival_event()
            else:
                self.handle_departure_event()

    def handle_arrival_event(self):
        #print('Arrival')
        #time_since_last_event = self.current_time - self.event_time[-1] if self.event_time else 0
        time_since_last_event = self.current_time - self.post_time
        self.cumulative_people_time += time_since_last_event * self.total_people
        #print('現在時刻:{0}'.format( self.current_time))
        #print('Event時刻:{0}'.format(self.event_time[-1]  if self.event_time else 0))
        #print('時間間隔:{0}'.format(time_since_last_event))
        #print('系内人数:{0}'.format(self.total_people))
        #print('累積人数時間:{0}'.format(self.cumulative_people_time))
        
        self.post_time = self.current_time
        self.current_time = self.next_arrival
        batch_size = np.random.randint(self.c, self.d + 1)  # Including the effect of batch size distribution in arrivals
        self.total_people += batch_size #総トランザクションはバッチの大きさを足したもの

        self.event_time.append(self.current_time)
        self.event_type.append('arrival')
        self.num_in_system.append(self.total_people)
        self.server_utilization.append(self.num_servers_busy / self.m)

        if self.num_servers_busy < self.m and self.total_people >= self.a:  # Including MB policy in service
            #self.next_departure = self.current_time + np.random.exponential(1 / (self.mu * min(self.total_people, self.b)))
            self.next_departure = self.current_time + np.random.exponential(1 / (self.mu)) #サービスはブロック単位なので、ブロックが[a,b]の大きさ(可変)でもサービス率はμ、トランザクション毎ではない
            self.num_servers_busy += 1

        self.next_arrival = self.current_time + np.random.exponential(1 / self.lambda_rate)

    def handle_departure_event(self):
        #print('Departure')
        #time_since_last_event = self.current_time - self.event_time[-1] if self.event_time else 0
        time_since_last_event = self.current_time - self.post_time
        self.cumulative_people_time += time_since_last_event * self.total_people
        #print('現在時刻:{0}'.format( self.current_time))
        #print('Event時刻:{0}'.format(self.event_time[-1]  if self.event_time else 0))
        #print('時間間隔:{0}'.format(time_since_last_event))
        #print('累積人数時間:{0}'.format(self.cumulative_people_time))
        #print('系内人数:{0}'.format(self.total_people))
        
        self.post_time = self.current_time
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
            #self.next_departure = self.current_time + np.random.exponential(1 / (self.mu * min(self.total_people, self.b)))
            self.next_departure = self.current_time + np.random.exponential(1 / (self.mu)) #バッチサイズによらずサービス率はμ
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
        avg_num_in_system = self.cumulative_people_time / self.current_time
        std_dev_num_in_system = np.std(self.num_in_system)  # 系内人数の標準偏差を計算
        ci_95 = 1.96 * std_dev_num_in_system   # 信頼区間を計算
        avg_queue_length = avg_num_in_system - (np.mean(self.server_utilization) * self.m)
        avg_server_utilization = np.mean(self.server_utilization)

        # Calculate the theoretical and simulated system utilization
        #rho_theoretical = self.lambda_rate * ((self.a + self.b) / 2) / (self.m * self.mu)
        avg_arrival_batch_size = (self.c + self.d) / 2  # Average batch size of arrivals
        avg_service_batch_size = (self.a + self.b) / 2  # Average batch size of service
        rho_theoretical = self.lambda_rate * avg_arrival_batch_size / (self.mu * avg_service_batch_size * self.m) #サービスはブロックサイズに無関係でμだけど、[a,b]の平均量を同時に処理できる
#        rho_simulated = avg_num_in_system / self.m #よくわからない
        
        # Print the metrics
        print(f"平均系内人数（シミュレーション）: {avg_num_in_system}")
        print(f"系内人数の標準偏差: {std_dev_num_in_system}")
        print(f"平均系内人数の信頼区間(95%): ({avg_num_in_system - ci_95}, {avg_num_in_system + ci_95})")
        print(f"平均待ち人数（シミュレーション）: {avg_queue_length}")
        print(f"平均サーバー利用率（シミュレーション）: {avg_server_utilization}")
        print(f"システム利用率（理論値）: {rho_theoretical}")
#        print(f"システム利用率（シミュレーション）: {rho_simulated}")

        return avg_num_in_system, avg_queue_length, avg_server_utilization, rho_theoretical

if __name__ == "__main__":
    # Parameters
    lambda_rate = 3  # arrival rate (batches per minute)
    mu = 3  # service rate (batches per minute)
    a, b = 3, 5  # min and max batch size for service
    m = 3  # number of servers
    sim_time = 6000  # simulation time in minutes
    c, d = 3, 5  # min and max batch size in arrivals

    # Create simulation object
    simulation = BatchServiceSimulationWithMB(lambda_rate, mu, a, b, m, sim_time, c, d)

    # Run the simulation
    simulation.run_simulation()

    # Plot the results
    simulation.plot_results("batch_simulation_with_MB.png")

    # Calculate metrics
    avg_num_in_system, avg_queue_length, avg_server_utilization, rho_theoretical = simulation.calculate_metrics()
