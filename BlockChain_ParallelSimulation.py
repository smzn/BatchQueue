from mpi4py import MPI
import numpy as np
from BlockChain_lib import BlockChain_lib

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Parameters for BlockChain_lib
lmd = 1.0 #トランザクションの到着率
mean = 550 #トランザクションの大きさの平均
sd = 50 #トランザクションの大きさの標準偏差
time = 500000 #上限時間

# Each process will run a simulation
blockchain = BlockChain_lib(lmd, mean, sd, time)
avg_queue_length = blockchain.getSimulation()

# Gather all average queue lengths at rank 0
all_avg_queue_lengths = comm.gather(avg_queue_length, root=0)

# Rank 0 will compute the mean, standard deviation and confidence interval
if rank == 0:
    mean_avg_queue_length = np.mean(all_avg_queue_lengths)
    std_dev_avg_queue_length = np.std(all_avg_queue_lengths)
    
    # Compute the 95% confidence interval
    z_value = 1.96  # for 95% confidence
    margin_of_error = z_value * (std_dev_avg_queue_length / np.sqrt(size))
    confidence_interval = (mean_avg_queue_length - margin_of_error, mean_avg_queue_length + margin_of_error)
    
    # Display results
    print('並列数: {0}'.format(size))
    print(f"Mean Average Queue Length: {mean_avg_queue_length}")
    print(f"Standard Deviation of Average Queue Length: {std_dev_avg_queue_length}")
    print(f"95% Confidence Interval: {confidence_interval}")

    # Display graph for rank 0
    blockchain.getGraph()
