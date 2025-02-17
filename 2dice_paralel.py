"""
Double Dice with Paralel Computing

@author: José Rodrigues, nº2019246536

# mpirun --use-hwthread-cpus -n {p} python3 2dice_paralel.py
# {p}: number of processes to be used
# use --oversubscribe to use more threads
"""

#%% IMPORTS
from mpi4py import MPI
from random import randint

#%% CODE

comm = MPI.COMM_WORLD
size=comm.Get_size() #nº de processadores
rank = comm.Get_rank()

m=7 #10**m lançamentos
n = int(10**m//size) # nº de lançamentos por processador

D={}
for i in range(2,13):
    D[i]=0
    
for i in range(n):
    D1=randint(1,6)
    D2=randint(1,6)
    v=D1+D2
    D[v] +=1
    
#%% PARALELIZAÇÃO
if rank == 0:
    print(f'\n{10**m} lançamentos')
    print ('\nNº de processadores:', size,';',n,'lançamentos por processador')
    print('\nNo processador', rank, 'os valores foram:', D) # para o processador master
    
    for p in range(1, size): #para os restantes processadores
        Dp = comm.recv(source=p) #?????
        print('No processador', p, 'os valores foram:', Dp)
        
        for i in range(2, 13):
            D[i] += Dp[i]
    print ('Resultado final:                  ', D)
    
else:
    comm.send(D, dest=0) #?????

#%%COMMENTS
"""
    Com 4 processadores a simulação demora entre 11s e 12s. Com 1 processador
demora cerca de 20s, ou seja a eficiência aumenta para o dobro utilizando computação
paralela neste caso (esperava-se que aumentasse para o quadruplo, dado que são 
utilizados 4 processadores).
    Verificou-se também que aumentando o número de processadores o tempo de computação
não diminuiu significativamente, pelo menos no computador utilizado para a simulação,
o que se explica pelo facto de ao ser utilizado um maior nº de threads é também
utilizada uma maior parte da memória RAM do computador.
"""