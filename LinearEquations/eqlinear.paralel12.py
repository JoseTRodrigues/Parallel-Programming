"""
RESOLUÇÃO DE EQUAÇÕES LINEARES (UTILIZANDO MÉTODO DE ELIMINAÇÃO DE GAUSS)
COM TROCA DE PIVOTS
COM PARALELIZAÇÃO
UTILIZANDO OS MÉTODOS SCATTERV E GATHERV

NOTA: APENAS FUNCIONA PARA N=8 E 4 PROCESSADORES !!!

@author: José Rodrigues, nº 2019246356
"""
#%% Imports
from numpy import array,zeros,random,save,load,argmax,empty,c_,float64,count_nonzero,delete
from mpi4py import MPI
from copy import deepcopy
from sys import argv

#%% PROGRAMA

comm = MPI.COMM_WORLD
numProc=comm.Get_size() #nº de processadores
rank = comm.Get_rank()

n=int(argv[1]) #dimensão da matriz
pivot=str(argv[2]) #com pivotagem: colocar p na linha de comando
debug = str(argv[1][0]) #com prints intermédios: colocar + antes do n
# debug='+'

"""DADOS"""
# save('A.npy',[[2,2,1,4],[1,-3,2,3],[-1,1,-1,-1],[1,-1,1,2]])
# save('B.npy',[5,2,-1,2])

"""DADOS 1"""
# save('A2.npy',[[-2**0.5,2,0],[1,-2**0.5,1],[0,2,-2**0.5]])
# save('B2.npy',[1,1,1])

""" DADOS 2"""
# save('A.npy',random.randint(1,n,size=(n,n)))
# save('B.npy',random.randint(1,n,size=(n)))

A = load('A.npy')
B = load('B.npy')
size=len(A)

# n=size
nl=n//numProc #nº de linhas/colunas por processador
np=n #nº de elementos por processador
final = size+1-numProc #ponto a partir do qual cada processador só aplica a eliminação gauss a uma linha: ver condição m > final

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

def flush_print (*args):
    import datetime
    if debug!='+': return
    now = datetime.datetime.now()
    timestamp = now.strftime('%H:%M:%S.%f')
    print (timestamp,  *args, flush=True)
    return

#%% PARALELIZAÇÃO

"""### CONSTRUÇÃO DAS LISTAS DISPL E COUNT UTILIZADOS NO MÉTODO DE ELIMINAÇÃO DE GAUSS ###"""

if rank==0: 

    sendbuf = array(c_[A,B],dtype=float64) #matriz ampliada do sistema (A|B)
    sendbuf0 = sendbuf[0] #primeira linha da matriz
    print(f'\nA =\n{sendbuf}\n\nB =\n{B}')

    def split(input, n):
        return [input[i:i+n] for i in range(0, len(input), n)]
    
    count_list=[]
    displ_list=[]
    for m in range(1,size):
        c=[]
        if m < final: #cada processador (exceto o de maior rank) inicialmente faz neste passo nl operações
            for w in range(nl): 
                count=[]
                if w < nl-1: #cálculo das primeiras size-(numProc-1) linhas
                    for i in range(numProc):
                        if m-1-nl*w > i:
                            count.append(0)
                        else:
                            count.append(np+1) 
    
                else: #cálculo das ultimas (numProc-1) linhas
                    for i in range(numProc-1):
                        count.append(np+1)
                    count.append(0)
                
                count_list.append(count)
                c.append(count)
            # print(c)
                
            c = [ item for elem in c for item in elem]
            # print(c)
            
            displ = [sum(c[:p]) for p in range(1,len(c)+1)]
            # print(displ)
            for i in range(nl):
                displ_list.append(split(displ,numProc)[i])
            
        else: #cada processador inicialmente apenas calcula uma linha
            w=nl-1
            count=[]
            for i in range(numProc-1):
                if m-i <= final:
                    count.append(np+1)
                else:
                    count.append(0)
            count.append(0)
            count_list.append(count)
            c=count
            # print(c)
            
            displ = [sum(c[:p]) for p in range(1,len(c)+1)]
            displ_list.append(displ)

    # print(count_list)
    # print(displ_list)
            
    count_list=array(count_list)
    displ_list=array(displ_list)
    count_init=deepcopy(count_list) #count inicial
    # displ_init=deepcopy(displ_list) #displ inicial
    # dc=c_[displ_init,count_init]  #displ e count
    # print('\n',dc)
    
    ab_triang=[]
    triang_final=[sendbuf0]
    
else:
    
    sendbuf = None
    sendbuf0 = zeros(np+1)
    
    count=zeros(numProc,dtype=int)
    displ=zeros(numProc,dtype=int)
    displ_result=zeros(numProc,dtype=int)
    
    m = 0


"""" ############ MÉTODO DE ELIMINAÇÃO DE GAUSS PARALELIZADO ############### """

for m in range(1,size):
    comm.bcast(m,root=0)
    if pivot=='p' and rank==0:
        """ TROCA DE PARCIAL DE PIVOTS """
        if abs(sendbuf[0][m-1])<0.001: #condição de troca de linhas
            sendbuf_t=abs(sendbuf).T[m-1][m:size] #encontrar o pivot com maior valor absoluto
            i=argmax(sendbuf_t==max(sendbuf_t))+m #encontrar índice do novo pivot
            sendbuf[m-1], sendbuf[i] = sendbuf[i], sendbuf[m-1] #troca de linhas de a
    
    if rank==0: flush_print(f'---------------------------------------------------------- m = {m} ----------------------------------------------------------\n')
    
    if m < final: #cada processador aplica a eliminação de gauss a nl linhas
        comm.Bcast(sendbuf0,root=0)
        flush_print(f'rank{rank} received pivot line:',sendbuf0)
        
        for i in range(nl):
            if rank==0:
                count = count_list[i+2*(m-1)]
                displ = displ_list[i+2*(m-1)]
                
                if i==nl-1:
                    displ_result = array([sum(count[:p]) for p in range(1,numProc+1)])-(np+1)
                else:
                    displ_result = displ - count
                    
            comm.Bcast(count,root=0)
            comm.Bcast(displ,root=0)
            comm.Bcast(displ_result,root=0)
    
            Ap = zeros(count[rank],dtype=float64) #linha de A&B recebida por cada processador. Ap e sendbuf têm que ter o mesmo dtype!
            flush_print(f'{i} rank{rank} {displ} {count} expecting {count[rank]} data:',Ap)
            comm.Scatterv([sendbuf, count , displ , MPI.DOUBLE], Ap, root=0)
            flush_print(i,f'rank{rank} received data:',Ap.round(2))
            Ap = Ap.tolist() #os numpy.array não são eficientes para operações recursivas
            
            result=gauss(Ap,list(sendbuf0),m-1)
            flush_print(i,f'rank{rank} sent data:',result)
            gatherbuf = zeros((np+1)*count_nonzero(count==(np+1))) if rank==0 else None    
            comm.Gatherv(result,[gatherbuf, count, displ_result, MPI.DOUBLE], root=0)
        
            if rank==0: 
                flush_print(i,'all:',gatherbuf.round(2))
                ab_triang = ab_triang + list(gatherbuf)
                
        if rank==0: 
            ab_triang=array(ab_triang).reshape(np-m,np+1)
            # ab_triang=r_[[sendbuf0],ab_triang]
            flush_print('\nfinal result:\n',ab_triang.round(2))
            
            sendbuf = array(deepcopy(ab_triang),dtype=float64)
            sendbuf0 = sendbuf[0]
            ab_triang = []
            
            triang_final.append(sendbuf0)
            
            # if m==4: print(f'\nRESULTADO DA ELIMINAÇÃO:\n{array(triang_final).round(2)}\n')


    else:
        
        if rank==0:
            count = count_list[m-np]
            displ = displ_list[m-np]
            displ_result = displ - count
            
        comm.Bcast(count,root=0)
        comm.Bcast(displ,root=0)
        comm.Bcast(displ_result,root=0)
            
        # if rank==0: flush_print(f'\ndispl={displ}\ncount={count}\n')
        
        comm.Bcast(sendbuf0,root=0)
        flush_print(f'rank{rank} received pivot line:',sendbuf0)
        
        Ap = zeros(count[rank],dtype=float64)
        flush_print(f'rank{rank} {displ} {count} expecting {count[rank]} data:',Ap)
        comm.Scatterv([sendbuf, count , displ , MPI.DOUBLE], Ap, root=0)
        flush_print(i,f'rank{rank} received data:',Ap.round(2))
        Ap = Ap.tolist() #os numpy.array não são eficientes para operações recursivas
    
        result=gauss(Ap,list(sendbuf0),m-1)
        flush_print(i,f'rank{rank} sent data:',result)
        gatherbuf = zeros((np+1)*count_nonzero(count==(np+1))) if rank==0 else None   
        comm.Gatherv(result,[gatherbuf, count, displ_result, MPI.DOUBLE], root=0)
    
        if rank==0: 
            flush_print('all:',gatherbuf.round(2))
            ab_triang = ab_triang + list(gatherbuf)
            
            ab_triang=array(ab_triang).reshape(np-m,np+1)
            # ab_triang=r_[[sendbuf0],ab_triang]
            flush_print('\nfinal result:\n',ab_triang.round(2))
            
            sendbuf = array(deepcopy(ab_triang),dtype=float64)
            sendbuf0 = sendbuf[0]
            ab_triang = []
            
            triang_final.append(sendbuf0)


"""CÁLCULO FINAL DOS VALORES DE X"""
if rank==0:
    triang_final=array(triang_final)
    print(f'\nRESULTADO DA ELIMINAÇÃO DE GAUSS:\n{triang_final.round(2)}\n')
    
    a=delete(triang_final,-1,1)
    b=triang_final.T[-1]
    # print(a)
    # print(b)
    
    x=zeros(size)
    for m in range(size-1,-1,-1):
        if m==size-1:
            x[m]=b[m]/a[m][m]
        else:
            x[m]=(b[m]-sum(a[m]*x))/a[m][m] 
    print(f'\nA solução é:\nx = {x}')
    Ax=A@x
    print(f'\nA.x = {Ax}')
    flush_print(f'B = {B}')
    # print(B-Ax)
    if sum(B-Ax) <= 1**-5:
        print('CORRECTO!!!')
    else:
        print('INCORRECTO...')

#%% COMENTÁRIOS
"""

"""