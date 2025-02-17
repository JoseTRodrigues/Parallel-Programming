# -*- coding: utf-8 -*-
"""
Double Dice (com plots)

@author: José Rodrigues, nº2019246536
"""
#%% IMPORTS

from random import randint
from numpy import array,arange
from time import time


#%% PROGRAMA

# start_time = time()
n = 10**7 #nº de lançamentos

D={}
for i in range(2,13):
    D[i]=0
    
for i in range(n):
    D1=randint(1,6)
    D2=randint(1,6)
    v=D1+D2
    D[v] +=1
    

print ('Valores obtidos:',D)
# print("\nTempo Real: %.3f s" % (time() - start_time))


#%% COMPARAÇÃO COM PROBABILIDADES

soma=array(list(D.keys()))
valor=array(list(D.values()))

cp=[] #casos possíveis
for i in range(1,7):
    for j in range(1,7):
        cp.append(i+j)

p={} 
for i in range(2,13):
    p[i]=cp.count(i)/len(cp)
prob=list(p.values()) #probabilidade de cada valor

#%% PLOTS
from matplotlib.pyplot import bar, scatter, xlabel, title, legend,\
    show, xticks
    
bar(soma,valor/n,label='nº de ocorrências / n') #distribuição de ocorrências normalizada
scatter(soma,prob,color='red',label='valor de probabilidade') #valores de probabilidade

legend()
title('Distribuição de Ocorrências')
xticks(arange(2, 13, step=1))
xlabel('Valor da Soma')
show()

#%% COMMENTS
"""
    Como se pode verificar pelo gráfico os valores obtidos nos 10**7 lançamentos 
correspondem praticamente aos valores esperados pela distribuição de probabilidades.
"""

