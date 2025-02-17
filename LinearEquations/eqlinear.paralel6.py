"""
RESOLUÇÃO DE EQUAÇÕES LINEARES (UTILIZANDO MÉTODO DE ELIMINAÇÃO DE GAUSS)
COM PARALELIZAÇÃO
UTILIZANDO OS MÉTODOS SCATTERV E GATHERV

NOTA: APENAS FUNCIONA PARA N=8 E 4 PROCESSADORES !!!

@author: José Rodrigues, nº 2019246356
"""
#%% Imports
from numpy import array,zeros,random,save,load,transpose,argmax,sqrt,empty,c_,r_,\
                  float64,append
from mpi4py import MPI
from copy import deepcopy
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

""" DADOS 2"""
# save('A.npy',random.randint(1,n,size=(n,n)))
# save('B.npy',random.randint(1,n,size=(n)))

A = load('A.npy')
B = load('B.npy')
size=len(A)

# n=int(argv[1]) #dimensão da matriz
n=size
nl=n//numProc #nº de linhas/colunas por processador
np=n #nº de elementos por processador
final = int(n/nl+1) #ponto a partir do qual cada processador só aplica a eliminação gauss a uma linha: ver condição m > final

def gauss(a,a0,m):
    """
    Método de eliminação de Gauss por linha:\n
    a - linha da matriz A & B\n
    a0 - linha do pivot\n
    m - coluna do pivot
    """
    if a==[]:
        return array([])
        # return(zeros(np+1))
    else:
        f=a[m]/a0[m]
        for k in range(m,size+1): #coluna, incluindo a matriz coluna b!!
            a[k] -= a0[k]*f
            
        return array(a)


debug = True
# debug = argv[1][0]
def flush_print (*args):
    import datetime
    if not debug: return
    # if debug!='+': return
    now = datetime.datetime.now()
    timestamp = now.strftime('%H:%M:%S.%f')
    print (timestamp,  *args, flush=True)
    return

#%% PARALELIZAÇÃO

""" CONSTRUÇÃO DAS LISTAS DISPL E COUNT UTILIZADOS NO MÉTODO DE ELIMINAÇÃO DE GAUSS"""

if rank==0:
    
    sendbuf = array(c_[A,B],dtype=float64) #matriz ampliada do sistema (A|B)
    sendbuf0 = sendbuf[0] #primeira linha da matriz
    print(sendbuf,'\n')

    count_list=[]
    displ_list=[]
    for m in range(1,size):
        if m < final:
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
                
                count_list.append(count)
                # print(m,displ,count)
            
        else: #cada processador apenas calcula uma linha
            w=nl-1
            displ = [(i+(final-1)*w)*(np+1)+(m-1) for i in range(1,numProc+1)]
            displ_list.append(displ)
            count=[]
            for i in range(numProc-1):
                if m-i <= final:
                    count.append(np+1-(m-1))
                else:
                    count.append(0)
            count.append(0)
            count_list.append(count)
            
    count_list=array(count_list)
    displ_list=array(displ_list)
    count_init=deepcopy(count_list) #count inicial
    displ_init=deepcopy(displ_list) #displ inicial
    dc=c_[displ_init,count_init]  #displ e count
    print(dc)
    
    ab_triang=[]
    
else:
    
    sendbuf = None
    sendbuf0 = empty(n+1)
    
    count_list = array((n*nl-2)*[list(empty(numProc))],dtype=int) #retiram-se duas linhas porque não se efectuam cálculos na primeira e a última não tem pivot
    displ_list = array((n*nl-2)*[list(empty(numProc))],dtype=int)



"""" MÉTODO DE ELIMINAÇÃO DE GAUSS PARALELIZADO """


comm.Bcast(count_list,root = 0)
comm.Bcast(displ_list,root = 0)
# print(rank,count_list)
# print(rank,displ_list)

# for m in range(1,size):
for m in range(1,3):
    
    if m < final: #cada processador aplica a eliminação de gauss a nl linhas
        comm.Bcast(sendbuf0,root=0)
        flush_print(f'rank{rank} received pivot line:',sendbuf0)
    
        for i in range(nl):
            count = count_list[i+2*(m-1)]
            displ = displ_list[i+2*(m-1)]
    
            Ap = empty(count[rank],dtype=float64) #linha de A&B recebida por cada processador. Ap e sendbuf têm que ter o mesmo dtype!
            comm.Scatterv([sendbuf, count , displ , MPI.DOUBLE], Ap, root=0)
            flush_print(m,i,f'rank{rank} received data:',Ap)
            Ap = Ap.tolist() #os numpy.array não são eficientes para operações recursivas
        
            gatherbuf = empty((np+1-(m-1))*(numProc-i)) if rank==0 else None
            displ=array([sum(count[:p]) for p in range(numProc)])
            
            result=gauss(Ap,list(sendbuf0),m-1)
            flush_print(m,i,f'rank{rank} sent data:',result)

            comm.Gatherv(result,[gatherbuf, count, displ, MPI.DOUBLE], root=0)
        
            if rank==0: 
                flush_print(m,i,f'rank{rank} final result:',gatherbuf)
                ab_triang = ab_triang+list(gatherbuf)
                
        if rank==0: 
            ab_triang=array(ab_triang).reshape(np-m,np+1-(m-1))
            ab_triang=r_[[sendbuf0],ab_triang]
            flush_print(m,i,f'rank{rank} all:',ab_triang)
            
            sendbuf = deepcopy(ab_triang)
            sendbuf0 = sendbuf[m]
            ab_triang = []
        
    # if rank==0: 
    #     ab_triang=ab_triang.reshape(np-1,np-1) 
    #     flush_print(m,i,f'rank{rank} all:',ab_triang)
    
    
            # else:
            #     result=gauss(Ap,Ap[m-1],m-1)
            #     comm.Gatherv(result, [gatherbuf, count, displ, MPI.DOUBLE], root=0)
                
# if rank==0: print(gatherbuf)
            

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
"""
