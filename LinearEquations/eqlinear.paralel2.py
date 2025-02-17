"""
RESOLUÇÃO DE EQUAÇÕES LINEARES (UTILIZANDO MÉTODO DE ELIMINAÇÃO DE GAUSS)
COM PARALELIZAÇÃO

-> NOTA: APENAS FUNCIONA SE Nº EQUAÇÕES = Nº DE PROCESSADORES !!!

@author: José Rodrigues, nº 2019246356
"""
#%% Imports
from numpy import array,zeros,random,save,load,transpose,argmax,sqrt
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
n=len(A)

if rank==0: print(f'A={A}\n\nB={B}')

a=deepcopy(A)
b=deepcopy(B)
size=len(a)
a=[list(a[j]) for j in range(size)]
b=list(b)

for m in range(size-1): 
    # print(a)
    if pivot=='p':
        """ TROCA DE PARCIAL DE PIVOTS """
        if abs(a[m][m])<0.001: #condição de troca de linhas
            a_t=transpose(abs(array(a)))[m][m+1:size] #encontrar o pivot com maior valor absoluto
            i=argmax(a_t==max(a_t))+(m+1) #encontrar índice do novo pivot
            a[m], a[i] = a[i], a[m] #troca de linhas de a
            b[m], b[i] = b[i], b[m] #troca de linhas de b
        
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
    
    
"""CÁLCULO FINAL DOS VALORES DE X"""    
if rank==0: 
    a=array(a)
    b=array(b) 
    # print('\n',a,b)
    x=zeros(size)
    for m in range(size-1,-1,-1):
        if m==size-1:
            x[m]=b[m]/a[m][m]
        else:
            x[m]=(b[m]-sum(a[m]*x))/a[m][m] 
    print(f'\nA solução é:\nx = {x}')
    print(f'\nA.x = {A@x}')
            

            
