"""
RESOLUÇÃO DE EQUAÇÕES LINEARES (UTILIZANDO MÉTODO DE ELIMINAÇÃO DE GAUSS)
COM PARALELIZAÇÃO

NOTA: APENAS FUNCIONA PARA 4 PROCESSADORES

@author: José Rodrigues, nº 2019246356
"""
#%% Imports
from numpy import array,append,empty,random,save,load
from mpi4py import MPI
from copy import deepcopy
from sys import argv

#%% PROGRAMA

comm = MPI.COMM_WORLD
numProc=comm.Get_size() #nº de processadores
rank = comm.Get_rank()


"""DADOS"""
# dump([[2,2,1,4],[1,-3,2,3],[-1,1,-1,-1],[1,-1,1,2]],open('A1.pkl','wb'))
# dump([5,2,-1,2],open('B1.pkl','wb'))
# A=load(open('A1.pkl','rb'))
# B=load(open('B1.pkl','rb'))

# n=int(argv[1]) #dimensão da matriz
# save('A.npy',random.randint(1,n,size=(n,n)))
# save('B.npy',random.randint(1,n,size=(n)))
A = load('A.npy')
B = load('B.npy')
if rank==0: print(f'A={A}\n\nB={B}')

a=deepcopy(A)
b=deepcopy(B)
size=len(a)

for m in range(size-1): 
    # print(a)
    a=array(a)
    b=array(b) 
    a0=comm.bcast(a[m],root=0)          #linha do pivot
    # print(a0)
    ap=empty([size//numProc,size],dtype=int)  #conjunto de linhas da matriz A para cada processado
    comm.Scatter(a,ap,root=0)
    b0=comm.bcast(b[m])                 #valor da matriz B da linha do pivot
    bp=empty(size//numProc,dtype=int)      #conjunto de valores da matriz B
    comm.Scatter(b,bp,root=0) 
    
    if rank==0:
        init=1
    else:
        init=0
        
    ap=[list(ap[j]) for j in range(len(ap))]
    bp=list(bp)
    for i in range(init,len(ap)):
        print(a0[m])
        f=ap[i][m]/a0[m]
        bp[i]-=b0*f
        for k in range(m,size): #coluna
            ap[i][k] -= a0[k]*f
    
    a=comm.bcast(comm.gather(ap,root=0))
    b=comm.bcast(comm.gather(bp,root=0))

# if rank==0: 
#     # print('\n',a,b)
    
#     """CÁLCULO FINAL DOS VALORES DE X"""
#     x=zeros(size)
#     for m in range(size-1,-1,-1):
#         if m==size-1:
#             x[m]=b[m]/a[m][m]
#         else:
#             x[m]=(b[m]-sum(a[m]*x))/a[m][m] 
#     print(f'\nA solução é:\nx = {x}')
#     print(f'\nA.x = {A@x}')