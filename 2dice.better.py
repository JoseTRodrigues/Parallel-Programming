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

#%% PARALELIZAÇÃO

if rank == 0:
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
    
    n=2**9
    while n > 0:
        n*=2
        np=n//size      
            
        if n==1024:
            print(f'\nn={n} lançamentos; {np} por processador')
            D0=D(np) #dicionário no master
            print(f'No processador 0 os valores foram: {D0}') #({sum(list(D0.values()))})')
        else:
            print(f'\nn={n} lançamentos; {np//2} por processador')
            D0=D(np//2) #simulação para metade dos lançamentos
            print(f'No processador 0 os valores foram: {D0}') #({sum(list(D0.values()))})')
            D0_load=load(open('dict.pkl','rb')) #utilização dos lançamentos anteriores guardados (outra metdade)
            for i in range(2, 13):
                D0[i] += D0_load[i]
                
        for p in range(1, size): #para os restantes processadores
            comm.send(n,dest=p,tag=p+10*p+100*p)
            comm.send(np,dest=p,tag=p)
            Dpp=comm.recv(source=p,tag=p+10*p)
            print(f'No processador {p} os valores foram: {Dpp}') #({sum(list(Dpp.values()))})')
            for i in range(2, 13):
                D0[i] += Dpp[i]

        dump(D0, open('dict.pkl', 'wb')) #D0 é guardado num ficheiro para uso no próximo n
        print(f'No total: {D0} ({sum(list(D0.values()))})')
        valor_0 = valor(D0,n)
        sigm = 1/11*(sum(((valor_0-prob)/prob)**2))**0.5
        # print(sigm)
        
        if sigm < 0.001:
            print(f'\n{n} LANÇAMENTOS NO TOTAL\n{D0}\nσ = {sigm}')
            for p in range(1,size):
                comm.send(0,dest=p,tag=rank+10*rank+100*rank) #PARA TERMINAR O PROGRAMA
            break
    
else:
    nr=comm.recv(source=0,tag=rank+10*rank+100*rank)
    while nr>0:
        print(nr)        
        npp=comm.recv(source=0,tag=rank)
        if npp*size==1024:
            Dp=D(npp)
        else:
            Dp=D(npp//2)            
        comm.send(Dp, dest=0, tag=rank+10*rank)
    
        # print('sent to master')
    
#%%COMMENTS
"""
Note-se que para n > 1024 cada processador simula na verdade np/2 lançamentos 
uma vez que são reaproveitados os resultados da simulação anterior, ou seja, 
apenas é necessário simular metade dos lançamentos inicialmente referidos por n
Por alguma razão os restantes processadores (que não o master) não recebem os
valores de n, o que implica que o programa nunca chega a terminar e fique a
correr por tempo indefinido.
"""