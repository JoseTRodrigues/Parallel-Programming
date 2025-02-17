"""
RESOLUÇÃO DE EQUAÇÕES LINEARES (UTILIZANDO MÉTODO DE ELIMINAÇÃO DE GAUSS)
COM PARALELIZAÇÃO
UTILIZANDO OS MÉTODOS SCATTERV E GATHERV

NOTA: APENAS FUNCIONA PARA N=8 E #PROCESSADORES=4

@author: José Rodrigues, nº 2019246356
"""
#%% Imports
from numpy import array,zeros,random,save,load,transpose,argmax,sqrt,empty,c_
from mpi4py import MPI
from copy import deepcopy
from sys import argv

#%% PROGRAMA

comm = MPI.COMM_WORLD
# numProc=comm.Get_size() #nº de processadores
numProc=4
rank = comm.Get_rank()

# pivot=str(argv[1][0]) #com pivotagem

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

def gauss(a,a0,m):
    """
    Método de eliminação de Gauss por linha:\n
    a - linha da matriz A & B\n
    a0 - linha do pivot\n
    m - coluna do pivot
    """
    f=a[m]/a0[m]
    for k in range(m,size+1): #coluna, incluindo a coluna b!!
        a[k] -= a0[k]*f
    return a


n=size
nl=n//numProc #nº de linhas/colunas por processador
np=n

a_b = c_[A,B] #matriz ampliada do sistema (A|B)
a_b=[list(a_b[j]) for j in range(size)]

count_list=[]
displ_list=[]
for m in range(1,size):
    
    if n/nl+1 > m: #condição para que cada processador tenha pelo menos 1 linha e por tanto seja necessário fazer mais do que 1 ciclo
        
        for w in range(nl): 
            """Escolha das linhas e elementos utilizados pelos processadores"""
            displ = [(i+4*w)*(np+1) for i in range(1,numProc+1)] #!!! O FACTOR DE 4 TEM QUE VER n=8!!!
            count = []
            if w < nl-1:
                for i in range(numProc):
                    if m-w-1 > i:
                        count.append(0)
                    else:
                        count.append(np+1)  
            else:
                for i in range(numProc-1):
                    count.append(np+1)
                count.append(0)
            
            # print(m,displ,count)
            
            # Ap = empty(count[rank]) #linha de A&B recebida por processador
            # comm.Scatterv([array(a_b), array(count), array(displ), MPI.DOUBLE], Ap, root=0)
            
            # gatherbuf = zeros(size) if rank == 0 else None
            # a0 = a_b[m-1] #linha do pivot
            # comm.Gatherv(gauss(Ap,a0,m-1), [gatherbuf, array(count), array(displ), MPI.DOUBLE], root=0)

        
    else: #cada processador apenas calcula uma linha
        w=nl-1
        displ = [(i+4*w)*(np+1) for i in range(1,numProc+1)]
        count=[]
        for i in range(numProc-1):
            if m-i <= 5: #!!! O FACTOR DE 5 TEM QUE VER n=8!!!
                count.append(np+1)
            else:
                count.append(0)
        count.append(0)
        
    count_list.append(count)
    displ_list.append(displ)
        
        # print(m,displ,count)
                
        # Ap = empty(count[rank]) #linha de A&B recebida por processador
        # comm.Scatterv([array(a_b), array(count), array(displ), MPI.DOUBLE], Ap, root=0)
        
        # gatherbuf = zeros(size) if rank == 0 else None
        # a0 = a_b[m-1]#linha do pivot
        # comm.Gatherv(gauss(Ap,a0,m-1), [gatherbuf, array(count), array(displ), MPI.DOUBLE], root=0)

                
    
    
    
    
# """CÁLCULO FINAL DOS VALORES DE X"""
# if rank==0: 
#     a=array(a)
#     b=array(b) 
#     # print('\n',a,b)
#     x=empty(size)
#     for m in range(size-1,-1,-1):
#         if m==size-1:
#             x[m]=b[m]/a[m][m]
#         else:
#             x[m]=(b[m]-sum(a[m]*x))/a[m][m] 
#     print(f'\nA solução é:\nx = {x}')
#     print(f'\nA.x = {A@x}')
            