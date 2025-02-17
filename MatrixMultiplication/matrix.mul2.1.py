"""
MULTIPLICAÇÃO DE MATRIZES UTILIZANDO 4 PROCESSADORES
DIVINDO AS MATRIZES A e B PELOS PROCESSADORES

@author: José Rodrigues, Nº 2019246536
"""

#%% IMPORTS
from mpi4py import MPI
from sys import argv
from numpy import array,linspace,reshape,zeros,empty,dot,split,sqrt,concatenate,\
    transpose,ascontiguousarray,block
from random import shuffle
from time import time

#%% CODE

comm = MPI.COMM_WORLD
numProcess=comm.Get_size() #nº de processadores
rank = comm.Get_rank()

a=int(argv[1])
assert a>=2, print('O argumento tem que ser igual ou superior a 2')
n=2**a #dimensão da matriz

d=int(sqrt(numProcess)) #nº de divisões
nl=2*n//numProcess #nº de linhas/colunas por processador
np=nl*n #nº de elementos por processador

A=linspace(1,n**2,n**2)
shuffle(A)
A=A.reshape(n,n)
Ai=split(A,d) #divisão da matriz pelo nº de processadores

B=linspace(1,n**2,n**2)
shuffle(B)
B=B.reshape(n,n)
Bt=ascontiguousarray(transpose(B)) #transposta de B; é prciso transformar num array contíguo
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
    Bip=Bip.reshape(nl,n)
    Bip=transpose(Bip)
    
    # print(rank,Aip,'\n',Bip)
    # print(rank,dot(Aip,Bip))
    
    p_start=1
    for i in range(d):
        for p in range(p_start,d*(i+1)):
            comm.Send(Ai[i],dest=p,tag=1)
            if p_start==1: p_start+=(d-1)
            else: p_start+=d
    
    # for p in range(1,numProcess):
    #     if p<2:
    #         comm.Send(Ai[0],dest=p,tag=1)
    #     else:
    #         comm.Send(Ai[1],dest=p,tag=1)
            
    for p in range(1,numProcess):
        if p%2==0:
            comm.Send(Bi[0],dest=p,tag=2)
        else:
            comm.Send(Bi[1],dest=p,tag=2)
        
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
comm.Gather(dot(Aip,Bip),AB)
tf=time()

if rank==0:  
    """
    Reorganizção dos resultados
    """
    ABf=[]
    ne=np//2 #nº de elementos calculados por cada processador
    i=0
    for y in range(d):
        ABf.append([])
        for x in range(d):
            ABf[y].append((AB[i*ne:(i+1)*ne]).reshape(int(ne**0.5),int(ne**0.5)))
            i+=1
    AB=block(ABf)
    
    print('\nA.B=\n',AB)
    # print('Tempo de processamento =',tf-ti,'s')
    # ti=time()
    result=dot(A,B)
    # tf=time()
    print('\nResultado sem paralelização:\n',result)
    # print('Tempo de processamento =',tf-ti,'s')
    
#%% COMMENTS
