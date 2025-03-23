from mpi4py import MPI
import numpy as np
import math
import matplotlib.pyplot as plt
from datetime import datetime
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  
size = comm.Get_size()  

ROWS, COLUMNS = 1000, 1000
MAX_TEMP_ERROR = 0.01
rows_per_pe = ROWS // size
local_temperature = np.empty(( rows_per_pe+2, COLUMNS+2 ))
local_temperature_last = np.empty(( rows_per_pe+2 ,COLUMNS+2  ))

def initialize_temperature(temp):
    temp[:,:] = 0
   
    #Set right side boundary condition
    for i in range(0,rows_per_pe+1):
        temp[ i , COLUMNS+1 ] = 100 * math.sin( ( (3.14159/2) /ROWS    ) * (i+rows_per_pe* rank ))

    #Set bottom boundary condition
    if rank == (size -1):
        for i in range(COLUMNS+1):
            temp[ rows_per_pe+1 , i ]    = 100 * math.sin( ( (3.14159/2) /COLUMNS ) * i )

    
        

#plot
def output(data):
    plt.imshow(data)
    plt.savefig("temperature3.png")  


initialize_temperature(local_temperature_last)

if rank == 0:
    max_iterations = int(input("Max iterations? "))
else:
    max_iterations = None
max_iterations = comm.bcast(max_iterations, root=0)

dt = 100
iteration = 1
start_time = datetime.now().replace(microsecond=0)
while (dt > MAX_TEMP_ERROR) and (iteration < max_iterations):
    # exchange ghost cells
    if rank > 0:
        comm.Send(local_temperature_last[1,:], dest=rank - 1, tag=1)
        comm.Recv(local_temperature_last[0,:], source=rank - 1, tag=0)
        print(f"PE {rank}", iteration, "send up")
    if rank < size - 1:
        comm.Send(local_temperature_last[rows_per_pe, :], dest=rank + 1, tag=0)
        comm.Recv(local_temperature_last[rows_per_pe+1, :], source=rank + 1, tag=1)
        print(f"PE {rank}", iteration, "send down")
    
    comm.Barrier()

    # calculate
    for i in range(1, rows_per_pe+1):
        for j in range(1, COLUMNS + 1):
            local_temperature[i, j] = 0.25 * (local_temperature_last[i + 1, j] + local_temperature_last[i - 1, j] +
                                              local_temperature_last[i, j + 1] + local_temperature_last[i, j - 1])

    # dt
    local_dt = 0
    for i in range(1, rows_per_pe+1):
        for j in range(1, COLUMNS + 1):
            local_dt = max(local_dt, abs(local_temperature[i, j] - local_temperature_last[i, j]))
            local_temperature_last[i, j] = local_temperature[i, j]

    print(f"PE {rank}, local_dt = {local_dt}", flush=True)  
    comm.Barrier()  
    dt = comm.reduce(local_dt, op=MPI.MAX, root=0)  
    dt = comm.bcast(dt, root=0)  
    print(f"PE {rank}, Iteration {iteration}, global dt = {dt}", flush=True)
    iteration += 1

print("out")   
comm.barrier()
#form final output
if rank >0:
    comm.Isend(local_temperature_last[1:rows_per_pe+1,:], dest=0,tag= rank)
    print("rank >0 send")
if rank == (size-1):
    comm.Isend(local_temperature_last[rows_per_pe+1,:], dest=0,tag= 200)
    print("rank 4 send") 

if rank == 0:
    final_temperature = np.empty((ROWS+2, COLUMNS+2))
    final_temperature [0:rows_per_pe+1,:] = local_temperature_last[0:rows_per_pe+1,:]
    MPI.COMM_WORLD.Recv(final_temperature [ROWS+1,:],source = size-1, tag= 200) 
    for i in range(1,size):
        MPI.COMM_WORLD.Recv(final_temperature[1+i*rows_per_pe:(i+1)*rows_per_pe+1,:] ,source = i, tag= i)#local_temperature_last[1:rows_per_pe+1,:]
    print("output out")
    end_time = datetime.now().replace(microsecond=0)
    print('it takes {} time and {} loops to converge'.format(end_time-start_time, iteration)) if rank == 0 else None
    output(final_temperature)