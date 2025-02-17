"""
RESOLUÇÃO DE EQUAÇÕES LINEARES (UTILIZANDO MÉTODO DE ELIMINAÇÃO DE GAUSS)
COM PARALELIZAÇÃO
UTILIZANDO OS MÉTODOS SCATTERV E GATHERV

NOTA: APENAS FUNCIONA PARA N=8 E 4 PROCESSADORES !!!

@author: José Rodrigues, nº 2019246356
"""
#%% Imports
from numpy import array,zeros,random,save,load,transpose,argmax,sqrt,empty,c_
from mpi4py import MPI
# from copy import deepcopy
from sys import argv

#%% PROGRAMA

comm = MPI.COMM_WORLD
# numProc=comm.Get_size() #nº de processadores
numProc=4
rank = comm.Get_rank()

# pivot=str(argv[1][0]) #com pivotagem

"""DADOS 1"""
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

n=size
nl=n//numProc #nº de linhas/colunas por processador
np=n #nº de elementos por processador

def gauss(a,a0,m):
    """
    Método de eliminação de Gauss por linha:\n
    a - linha da matriz A & B\n
    a0 - linha do pivot\n
    m - coluna do pivot
    """
    f=a[m]/a0[m]
    for k in range(m,size+1): #coluna, incluindo a matriz coluna b!!
        a[k] -= a0[k]*f
    return a


#%% PARALELIZAÇÃO

if rank==0:
    
    a_b = c_[A,B] #matriz ampliada do sistema (A|B)
    a_b=[list(a_b[j]) for j in range(size)] #os numpy.array não são eficientes para operações recursivas
    
    """ Construção das listas dos count's e displ's (utilizados posteriormente)"""
    count_list=[]
    displ_list=[]
    for m in range(1,size):
        final = int(n/nl+1) #ponto a partir do qual cada processador só aplica a eliminação gauss a uma linha: ver condição m > final
        for w in range(nl): 
            displ = [(i+(final-1)*w)*(np+1)+(m-1) for i in range(1,numProc+1)] #(np+1)+(m+1): é o elemento inicial que cada processador utiliza;
                                                                               #(final-1)*w: tem que ver com o facto de se utilizarem nl secções diferente da matriz
            displ_list.append(displ)
            count = []
            if w < nl-1:
                for i in range(numProc):
                    if m-w-1 > i:
                        count.append(0)
                    else:
                        count.append(np+1-(m-1))  
            else:
                for i in range(numProc-1):
                    count.append(np+1-(m-1))
                count.append(0)
                
                if m > final:
                    for i in range(m-final):
                        count[i]=0
            
            count_list.append(count)
            # print(m,displ,count)
        
    count_list=array(count_list)
    displ_list=array(displ_list)


else:
    count_list = array((n*nl-2)*[list(zeros(numProc))],dtype=int) #retiram-se duas linhas porque não se efectuam cálculos na primeira e a última não tem pivot
    displ_list = array((n*nl-2)*[list(zeros(numProc))],dtype=int)



comm.Bcast(count_list,root = 0)
comm.Bcast(displ_list,root = 0)
# print(rank,count_list)
# print(rank,displ_list)

for i in range(len(count_list)):
    count = count_list[i]
    displ = displ_list[i]
    
    Ap = empty(count_list[i][rank]) #linha de A&B recebida por cada processador
    comm.Scatterv([array(a_b), array(count), array(displ), MPI.DOUBLE], Ap, root=0)
    print(rank,Ap)

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

#%% COMENTÁRIOS
"""
Apesar da lista displ contemplar todas as possibilidades de cálculo para todas
as linhas, mesmo para aquelas que estão acima ou na linha do pivot, a lista count
tem em conta apenas o nº correto de elementos que devem efectuar operações em
cada linha. Verifique-se portanto que na matriz cd=c_[displ_list,count_list] existem
posições da matriz que têm 0 elementos correspondentes
"""