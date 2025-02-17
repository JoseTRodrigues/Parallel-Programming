"""
MULTIPLICAÇÃO DE MATRIZES UTILIZANDO N PROCESSADORES
DIVINDO AS MATRIZES A e B PELOS PROCESSADORES
UTILIZANDO O MÉTODO SCATTERV

@author: José Rodrigues, Nº 2019246536
"""

#%% IMPORTS
from mpi4py import MPI
from sys import argv
from numpy import linspace,empty,sqrt,transpose,ascontiguousarray,block,\
    full,array,reshape
from random import shuffle
from time import time

#%% CODE

comm = MPI.COMM_WORLD
numProcess=comm.Get_size() #nº de processadores
rank = comm.Get_rank()

debug=argv[1][0]

n=int(argv[1]) #dimensão da matriz
assert n%4==0, print('O argumento (dimensão da matriz) tem de ser múltiplo de 4!!!')
assert n**2>=numProcess, print(f'As matrizes têm que ter um número de elementos\
superior ao nº de processadores: {n**2} < {numProcess}!')

def flush_print (*args):
    import datetime
    if debug!='+': return
    now = datetime.datetime.now()
    timestamp = now.strftime('%H:%M:%S.%f')
    print (timestamp,  *args, flush=True)
    return

#%% PARALELIZAÇÃO

ti=time()
if rank==0:
    
    d=int(sqrt(numProcess)) #nº de divisões
    nl=n//d #nº de linhas/colunas por processador
    np=nl*n #nº de elementos utilizados por processador
    
    A=linspace(1,n**2,n**2)
    shuffle(A)
    A=A.reshape(n,n)

    B=linspace(1,n**2,n**2)
    shuffle(B)
    B=B.reshape(n,n)
    Bt=ascontiguousarray(transpose(B)) #transposta de B; é preciso transformar num array contíguo
    
    sendbuf_A = A    
    displ_A = [] 
    for i in range(d):
        for j in range(d):
            displ_A.append(np*i)
    displ_A=array(displ_A)
    
    sendbuf_B = Bt
    displ_B = array([np*i for i in range(d)]*d)

    count = full(numProcess,np,dtype=int)
    
    print(f'\nDimensões das matrizes: {n}x{n}')
    print('\nMatrizes iniciais:')
    print('A=\n',A)
    print('\nB =\n',B,'\n')
    
            
else:
    sendbuf_A=None
    sendbuf_B=None
    count = empty(numProcess, dtype=int)
    displ_A=None
    displ_B=None
    nl=None
    
comm.Bcast(count, root=0)
nl=comm.bcast(nl,root=0)

Aip = empty(count[rank]) #secção de A por processador
comm.Scatterv([sendbuf_A, count, displ_A, MPI.DOUBLE], Aip, root=0)
Aip=Aip.reshape(nl,n)
# print(rank,Aip)

Bip = empty(count[rank]) #secção de B por processador
comm.Scatterv([sendbuf_B, count, displ_B, MPI.DOUBLE], Bip, root=0)
Bip=Bip.reshape(nl,n)
Bip=transpose(Bip)
# print(rank,Bip)

AB=empty(n**2) if rank==0 else None #produto de matrizes
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
    flush_print('Tempo de processamento =',tf-ti,'s')
    ti=time()
    flush_print('\nResultado sem paralelização:\n',A@B)
    tf=time()
    flush_print('Tempo de processamento =',tf-ti,'s')
    
    # if bool(sum(sum(A@B-AB)))==False:
    #     print("\nCORRECTO!!!")
    # else:
    #     print('\nINCORRECTO...')
    
#%% COMMENTS
