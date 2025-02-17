"""
MULTIPLICAÇÃO DE MATRIZES UTILIZANDO 4 PROCESSADORES
DIVIDINDO APENAS A MATRIZ A e ENVIANDO A MATRIZ B NA TOTALIDADE

@author: José Rodrigues, Nº 2019246536
"""

#%% IMPORTS
from mpi4py import MPI
from sys import argv
from numpy import array,linspace,reshape,zeros,empty,dot,split
from random import shuffle
from time import time

#%% CODE

comm = MPI.COMM_WORLD
size=comm.Get_size() #nº de processadores
rank = comm.Get_rank()

a=int(argv[1])
n=2**a

#%% PARALELIZAÇÃO

ti=time()
if rank==0:
    print(f'\nDimensões das matrizes: {n}x{n}')
    A=linspace(1,n**2,n**2)
    shuffle(A)
    A=A.reshape(n,n)

    B=linspace(1,n**2,n**2)
    shuffle(B)
    B=B.reshape(n,n)
    
    print('\nMatrizes iniciais:')
    print('A=\n',A)
    print('\nB=\n',B,'\n')
    
    AB=empty(n**2) #produto de matrizes
    
else:
    A=empty(n**2) # criação de espaço para guardar os dados
    B=empty(n**2).reshape(n,n) # criação de espaço para guardar os dados
    AB=None
   
    
Ai=empty(4**(a-1)) #espaço para receber as linhas; nº de elementos para cada processador
comm.Scatter(A,Ai,root=0) #envio dos conjuntos de linhas para os espaços criados acima
Ai=array(split(Ai,2**(a-2)))
Ai=Ai.reshape(2**(a-2),n)
# print(rank,Ai) #linha recebida por cada processador

comm.Bcast(B,root=0)

comm.Gather(dot(Ai,B),AB)
tf=time()

if rank==0:
    AB=AB.reshape(n,n)
    print('\nA.B=\n',AB)
    # print('Tempo de processamento =',tf-ti,'s')
    # ti=time()
    # result=dot(A,B)
    # tf=time()
    # print('\nResultado sem paralelização:\n',result)
    # print('Tempo de processamento =',tf-ti,'s')
    
#%% COMMENTS
"""
Repare-se que a função intrínseca A@B tem um tempo de processamento
cerca de 5 ordem de grandezas inferior ao do programa com paralelização.
"""
