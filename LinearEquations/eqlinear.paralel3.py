"""
RESOLUÇÃO DE EQUAÇÕES LINEARES (UTILIZANDO MÉTODO DE ELIMINAÇÃO DE GAUSS)
COM PARALELIZAÇÃO
UTILIZANDO OS MÉTODOS SCATTERV E GATHERV

@author: José Rodrigues, nº 2019246356
"""
#%% Imports
from numpy import array,zeros,random,save,load,transpose,argmax,sqrt,empty
from mpi4py import MPI
from copy import deepcopy
from sys import argv

#%% PROGRAMA

comm = MPI.COMM_WORLD
numProc=comm.Get_size() #nº de processadores
rank = comm.Get_rank()

pivot=str(argv[1][0]) #com pivotagem

# """DADOS 1"""
# save('A2.npy',[[-2**0.5,2,0],[1,-2**0.5,1],[0,2,-2**0.5]])
# save('B2.npy',[1,1,1])
# A=load('A2.npy')
# B=load('B2.npy')

""" DADOS 2"""
# n=int(argv[1]) #dimensão da matriz
# save('A.npy',random.randint(1,n,size=(n,n)))
# save('B.npy',random.randint(1,n,size=(n)))
A = load('A.npy')
B = load('B.npy')

size=len(A)
def gauss(a,a0,b,b0,m):
    """
    Método de eliminação de gauss por linha:\n
    a - linha da matriz A\n
    a0 - linha do pivot\n
    b - elemento de B\n
    b0 - elemento da linha do pivot\n
    m - linha/coluna do pivot
    """
    f=a[m]/a0[m]
    b-=b0*f
    for k in range(m,size): #coluna
        a[k] -= a0[k]*f
        
    return a,b

if rank==0: 
    
    print(f'A={A}\n\nB={B}')
    a=deepcopy(A)
    b=deepcopy(B)
    n=size
    
    nl=n//numProc #nº de linhas/colunas por processador
    np=n*nl #nº de elementos utilizados por processador
    
    sendbuf_A = a
    count_A = [np for i in range(numProc)]
    count_A[0]=np-n
    count_A=array(count_A)
    displ_A = [i*np for i in range(numProc)]
    displ_A[0]=n
    displ_A=array(displ_A)
    
    sendbuf_B = b
    count_B = [nl for i in range(numProc)]
    count_B[0]=nl-1
    count_B=array(count_B)
    displ_B = [i*nl for i in range(numProc)]
    displ_B[0]=1
    displ_B=array(displ_B)

else:
    sendbuf_A = None
    displ_A = None
    count_A = empty(numProc, dtype=int)
    sendbuf_B = None
    displ_B = None
    count_B = empty(numProc, dtype=int)
    a,b=None,None
    
comm.Bcast(count_A, root=0)
comm.Bcast(count_B, root=0)

for m in range(size):
    a0=comm.bcast(a[m],root=0)
    b0=comm.bcast(b[m],root=0)
    if a0[m]==0:
        print('SISTEMA IMPOSSÍVEL...')
        break
    Aip = empty(count_A[rank]) #secção de A por processador
    comm.Scatterv([sendbuf_A, count_A, displ_A, MPI.DOUBLE], Aip, root=0)
    # print(rank,Aip)
    
    Bip = empty(count_B[rank]) #secção de B por processador
    comm.Scatterv([sendbuf_B, count_B, displ_B, MPI.DOUBLE], Bip, root=0)
    # print(rank,Bip)
    
    a,b=gauss(Aip,a0,Bip,b0,m)
    

    
"""CÁLCULO FINAL DOS VALORES DE X"""    
if rank==0: 
    a=array(a)
    b=array(b) 
    # print('\n',a,b)
    x=empty(size)
    for m in range(size-1,-1,-1):
        if m==size-1:
            x[m]=b[m]/a[m][m]
        else:
            x[m]=(b[m]-sum(a[m]*x))/a[m][m] 
    print(f'\nA solução é:\nx = {x}')
    print(f'\nA.x = {A@x}')
            
