"""
RESOLUÇÃO DE EQUAÇÕES LINEARES (UTILIZANDO MÉTODO DE ELIMINAÇÃO DE GAUSS)
COM PARALELIZAÇÃO

-> NOTA: APENAS FUNCIONA SE Nº EQUAÇÕES = Nº DE PROCESSADORES

@author: José Rodrigues, nº 2019246356
"""
#%% Imports
from numpy import array,append,zeros,random,save,load
from mpi4py import MPI
from copy import deepcopy
from sys import argv

#%% PROGRAMA

comm = MPI.COMM_WORLD
numProc=comm.Get_size() #nº de processadores
rank = comm.Get_rank()


"""DADOS"""
save('A1.npy',[[2,2,1,4],[1,-3,2,3],[-1,1,-1,-1],[1,-1,1,2]])
save('B1.npy',[5,2,-1,2])
A = load('A1.npy')
B = load('B1.npy')

# # n=int(argv[1]) #dimensão da matriz
# # save('A.npy',random.randint(1,n,size=(n,n)))
# # save('B.npy',random.randint(1,n,size=(n)))
# A = load('A.npy')
# B = load('B.npy')

if rank==0: print(f'A={A}\n\nB={B}')

a=deepcopy(A)
b=deepcopy(B)
size=len(a)
a=[list(a[j]) for j in range(size)]
b=list(b)

for m in range(size-1): 
    # print(a)
    a0=comm.bcast(a[m],root=0)  #linha do pivot
    ap=comm.scatter(a,root=0)    #linha da matriz A para cada processador
    b0=comm.bcast(b[m])         #valor da matriz B da linha do pivot
    bp=comm.scatter(b,root=0)    #valor da matriz B para cada processador
    
    if rank>m:
        f=ap[m]/a0[m]
        bp-=b0*f
        for k in range(m,size): #coluna
            ap[k] -= a0[k]*f
        
    a=comm.bcast(comm.gather(ap,root=0))
    b=comm.bcast(comm.gather(bp,root=0))
    
if rank==0: 
    a=array(a)
    b=array(b) 
    # print('\n',a,b)
    
    """CÁLCULO FINAL DOS VALORES DE X"""
    x=zeros(size)
    for m in range(size-1,-1,-1):
        if m==size-1:
            x[m]=b[m]/a[m][m]
        else:
            x[m]=(b[m]-sum(a[m]*x))/a[m][m] 
    print(f'\nA solução é:\nx = {x}')
    print(f'\nA.x = {A@x}')
            