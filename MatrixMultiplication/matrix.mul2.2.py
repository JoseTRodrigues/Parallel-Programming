"""
MULTIPLICAÇÃO DE MATRIZES UTILIZANDO N PROCESSADORES
DIVINDO AS MATRIZES A e B PELOS PROCESSADORES

@author: José Rodrigues, Nº 2019246536
"""

#%% IMPORTS
from mpi4py import MPI
from sys import argv
from numpy import linspace,reshape,empty,split,sqrt,transpose,ascontiguousarray,block
from random import shuffle
from time import time

#%% CODE

comm = MPI.COMM_WORLD
numProcess=comm.Get_size() #nº de processadores
rank = comm.Get_rank()

n=int(argv[1]) #dimensão da matriz
assert n%4==0, print('O argumento (dimensão da matriz) tem de ser múltiplo de 4!!!')
assert n**2>=numProcess, print(f'As matrizes têm que ter um número de elementos\
superior ao nº de processadores: {n**2} < {numProcess}!')

d=int(sqrt(numProcess)) #nº de divisões
nl=n//d #nº de linhas/colunas por processador
np=nl*n #nº de elementos utilizados por processador

A=linspace(1,n**2,n**2)
shuffle(A)
A=A.reshape(n,n)
Ai=split(A,d) #divisão da matriz pelo nº de processadores

B=linspace(1,n**2,n**2)
shuffle(B)
B=B.reshape(n,n)
Bt=ascontiguousarray(transpose(B)) #transposta de B; é preciso transformar num array contíguo
Bi=split(Bt,d) #divisão da matriz pelo nº de processadores

#%% PARALELIZAÇÃO

ti=time()
if rank==0:
    
    print(f'\nDimensões das matrizes: {n}x{n}')
    print('\nMatrizes iniciais:')
    print('A=\n',A)
    print('\nB =\n',B,'\n')
    
    Aip=Ai[0]
    Bip=Bi[0]
    # print(Bip)
    Bip=Bip.reshape(nl,n)
    Bip=transpose(Bip)
    
    # print(rank,Aip,'\n',Bip)
    # print(rank,dot(Aip,Bip))
    
    """
    Envio das partes das matrizes
    """
    p_start=1
    j=1
    for i in range(d):
        for p in range(p_start,d*(i+1)):
            comm.Send(Ai[i],dest=p,tag=1)
            comm.Send(Bi[j],dest=p,tag=2)
            j+=1
        j=0
        if p_start==1: 
            p_start+=(d-1)
        else: 
            p_start+=d
            
else:
    Aip=empty(np) #espaço para receber as linhas; nº de elementos para cada processador
    # print('ok1')
    comm.Recv(Aip,source=0,tag=1)
    # print('ok2')
    Aip=Aip.reshape(nl,n)
    
    Bip=empty(np)
    comm.Recv(Bip,source=0,tag=2)
    Bip=Bip.reshape(nl,n)
    Bip=transpose(Bip)
    
    # print(rank,Aip,'\n',Bip)
    # print(rank,dot(Aip,Bip))
    
AB=empty(n**2) #produto de matrizes
comm.Gather(Aip@Bip,AB) #o símbolo @ é o produto interno entre matrizes

if rank==0:  
    """
    Reorganizção dos resultados
    """
    ABf=[]
    ne=nl**2 #nº de elementos calculados por cada processador
    # print(f'\nd={d},nl={nl},np={np},ne={ne}')
    i=0
    for y in range(d):
        ABf.append([])
        for x in range(d):
            ABf[y].append((AB[i*ne:(i+1)*ne]).reshape(nl,nl))
            i+=1
    AB=block(ABf)
    tf=time()
    
    print('\nA.B=\n',AB)
    # print('Tempo de processamento =',tf-ti,'s')
    # ti=time()
    #print('\nResultado sem paralelização:\n',A@B)
    # tf=time()
    # print('Tempo de processamento =',tf-ti,'s')
    
    # if bool(sum(sum(result-AB)))==False:
    #     print("\nCORRECTO!!!")
    # else:
    #     print('\nINCORRECTO...')
    
#%% COMMENTS
