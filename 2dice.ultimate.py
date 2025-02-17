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
from functools import reduce

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
while True:
    n*=2
    if n==1024: 
        np=comm.bcast(n//size,root=0)
        dump(D(0), open('data.pkl', 'wb')) #dicionário inicial com zeros

    else:
        np=comm.bcast(n//size//2,root=0)
    
    D_0=comm.gather(D(np),root=0) #lista dos resultados dos processadores
    
    if rank==0:
        print(f'\nn={n} lançamentos; {np} por processador')
        
        D_0.append(load(open('data.pkl','rb'))) #utilização dos dados guardados       
        D_final = reduce(lambda x, y: dict((k, v + y[k]) for k, v in x.items()), D_0) #soma dos resultados
        dump(D_final,open('data.pkl','wb')) # resultados guardados num ficheiro externo
        
        for i in range(size): print(f'Para n={np}, no processador {i}, os resultados foram: {D_0[i]}')
        print(f'No total: {D_final}')
        
        sigm=s(valor(D_final,n))
        if sigm<0.001:
            print('\n\n---------------------------------- FIM ----------------------------------')
            print(f'\n{n} LANÇAMENTOS NO TOTAL\nResultados: {D_final}\nσ = {sigm}\n')
            np=comm.bcast(0,root=0)
            break

    else:
        if np==0:
            break

                
#%%COMMENTS
"""
Note-se que para n > 1024 cada processador simula na verdade np/2 lançamentos 
uma vez que são reaproveitados os resultados da simulação anterior, ou seja, 
apenas é necessário simular metade dos lançamentos inicialmente referidos por n.
Este programa está significativamente mais rápido que o da aula anterior, uma
vez que se guardam todos os dicionários dos lançamentos e a soma das contagens
é feita no final, enquanto que no código anterior se somavam as contagens à
medida que as simulações terminavam (em que os dicionários eram criados).
"""