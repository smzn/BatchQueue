from mpi4py import MPI
import numpy as np
from BatchServiceSimulationWithMB import BatchServiceSimulationWithMB

def main():
    # MPIの初期化
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Parameters
    lambda_rate = 3
    mu = 3
    a, b = 3, 5
    m = 3
    sim_time = 100000
    c, d = 3, 5

    # シミュレーションの実施
    simulation = BatchServiceSimulationWithMB(lambda_rate, mu, a, b, m, sim_time, c, d)
    simulation.run_simulation()
    avg_num_in_system, _, _, _ = simulation.calculate_metrics()

    # rank0で結果を集約
    all_avg_nums = comm.gather(avg_num_in_system, root=0)
    all_num_in_systems = comm.gather(simulation.num_in_system, root=0)

    if rank == 0:
        mean_avg = np.mean(all_avg_nums)
        std_avg = np.std(all_avg_nums)
        ci_95 = 1.96 * (std_avg / np.sqrt(size))

        print(f"使用したMPIプロセス数（並列数）: {size}")
        print(f"平均系内人数の平均: {mean_avg}")
        print(f"平均系内人数の標準偏差: {std_avg}")
        print(f"平均系内人数の信頼区間(95%): ({mean_avg - ci_95}, {mean_avg + ci_95})")

        # 時系列変化のグラフ描画
        import matplotlib.pyplot as plt
        plt.plot(simulation.event_time, all_num_in_systems[0], label='Process 0')
        
        # 信頼区間の追加
        plt.axhline(mean_avg + ci_95, color='gray', linestyle='--', label='Upper 95% CI')
        plt.axhline(mean_avg - ci_95, color='gray', linestyle='--', label='Lower 95% CI')

        plt.xlabel('Simulation Time')
        plt.ylabel('Number in System for Process 0')
        plt.title('Time Series of Number in System for Process 0 with 95% CI')
        plt.legend()
        plt.savefig('time_series_simulation.png')
        plt.close()

        # 各プロセスの平均系内人数をヒストグラムで表示
        plt.hist(all_avg_nums, bins='auto', alpha=0.7)
        plt.xlabel('Average Number in System')
        plt.ylabel('Frequency')
        plt.title('Histogram of Average Number in System Across All Processes')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('histogram_avg_num_in_system.png')
        plt.close()

if __name__ == "__main__":
    main()
