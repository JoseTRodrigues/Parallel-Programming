"""
Double Dice with Paralel Computing
Com Análise Estatística

@author: José Rodrigues, nº2019246536
"""

#%% IMPORTS
from mpi4py import MPI
from random import randint
from numpy import array
from pickle import dump,load

#%% CODE

comm = MPI.COMM_WORLD
size=comm.Get_size() #nº de processadores
rank = comm.Get_rank()

"###########################  VALORES ESPERADOS ##########################"
cp=[] #casos possíveis
for i in range(1,7):
    for j in range(1,7):
        cp.append(i+j)
p={} 
for i in range(2,13):
    p[i]=cp.count(i)/len(cp)
prob=array(list(p.values())) #probabilidade de cada valor
"##########################################################################"

"###################### VALORES OBTIDOS NOS LANÇAMENTOS ######################"
def D(n):
    "REGISTO DOS n LANÇAMENTOS"
    d={}
    for i in range(2,13):
        d[i]=0
    for i in range(n):
        D1=randint(1,6)
        D2=randint(1,6)
        v=D1+D2
        d[v] +=1
    return d

def valor(d,n):
    "VALOR OBSERVADO DE FREQUÊNCIA PARA n LANLÇAMENTOS"
    return array(list(d.values()))/n

def s(v):
    return 1/11*(sum(((v-prob)/prob)**2))**0.5

#%% PARALELIZAÇÃO

n=2**9
np=1
while np!=0:
    if rank == 0:
        n*=2
        np=n//size
        
        if n==1024:
            print(f'\nn={n} lançamentos; {np} por processador')
            D0=D(np) #dicionário no master
            print(f'Para n={np}, no processador 0 os valores foram: {D0}') #({sum(list(D0.values()))})')
        
        else:
            print(f'\nn={n} lançamentos; {np//2} por processador')
            D0=D(np//2) #simulação para metade dos lançamentos
            print(f'Para n={np}, no processador 0 os valores foram: {D0}') #({sum(list(D0.values()))})')
            D0_load=load(open('dict.pkl','rb')) #utilização dos lançamentos anteriores guardados (outra metdade)
            for i in range(2, 13):
                D0[i] += D0_load[i]
                    
        for p in range(1, size): #para os restantes processadores
            comm.send(np,dest=p,tag=p)
            # print(f'{np} sent to slave')
            Dpp=comm.recv(source=p,tag=p+10*p)
            # print('recieved from slave')
            print(f'Para n={np}, no processador {p} os valores foram: {Dpp}') #({sum(list(Dpp.values()))})')
            for i in range(2, 13):
                D0[i] += Dpp[i]
                   
        # print(sum(list(D0.values())))
        dump(D0, open('dict.pkl', 'wb')) #D0 é guardado num ficheiro para uso no próximo n
        valor_0=valor(D0,n)
        print(f'No total: {D0}') #({sum(list(D0.values()))})')
        # print(s(valor_0))
        
        if s(valor_0)<0.001:
            print('\n----------------- FIM -----------------')
            print(f'\n{n} LANÇAMENTOS NO TOTAL\nResultados: {D0}\nσ = {s(valor_0)}\n')
            np=0
            for p in range(size):
                comm.send(np,dest=p,tag=p)
                
    else:
        npp=comm.recv(source=0,tag=rank)
        # print('recieved from master')
        if npp!=0:
            if npp*size==1024:
                Dp=D(npp)
            else:
                Dp=D(npp//2)
            comm.send(Dp, dest=0, tag=rank+10*rank)
        else:
            break
        # print('sent to master')
    

#%%COMMENTS
"""
Note-se que para n > 1024 cada processador simula na verdade np/2 lançamentos 
uma vez que são reaproveitados os resultados da simulação anterior, ou seja, 
apenas é necessário simular metade dos lançamentos inicialmente referidos por n
"""