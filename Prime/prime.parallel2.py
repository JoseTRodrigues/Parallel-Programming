"""
PRIME NUMBERS UP TO n USING PARALLEL COMPUTING
Trying all primes

@author: José Rodrigues, nº 2019246356
"""
#%% IMPORTS
from sys import argv
from mpi4py import MPI
from numpy import array,log10
from time import time

comm = MPI.COMM_WORLD
numProc=comm.Get_size() #nº de processadores
rank = comm.Get_rank()

#%% INPUTS

n = 10**int(argv[1]) # quantidade de números analisados
debug = str(argv[1][0]) #com prints intermédios: colocar + antes do n
# debug='+' #c/ ou s/ debug

#%% PROGRAMA

def flush_print (*args):
    """
    DEBUG PRINT
    """
    import datetime
    if debug!='+': return
    now = datetime.datetime.now()
    # timestamp = now.strftime('%H:%M:%S.%f')
    timestamp = now.strftime('%M:%S.%f')
    print (timestamp,  *args, flush=True)
    return
 

def nprime(prime,prime_test,init,n):
    
    """
    DESCOBRE OS NÚMEROS PRIMOS ENTRE init E n
    ----------
    prime : lista de primos já existente\n
    prime_test : lista de primos utilizados na comparação\n
    init : inteiro em que se inicia a análise\n
    n : inteiro em que termina a análise
    """
    
    for i in range(init,n+1):
            
        for p in prime_test:
            
            if i%p == 0:
                break
                
            elif p == prime_test[-1]: #ponto em que se atinge o último nº primo necessário para a verificação
                prime.append(i)
                
                if i**2 <= n:
                    prime_test.append(i)
                    
    return prime


#%% PARALELIZAÇÃO

ni = 100
while ni < n:
    ti=time()
    if rank==0:
        if ni == 100: #cálculo dos 1º primos até 100
            prime=[2]
            prime_test=[2]
            
            prime = nprime(prime, prime_test, 3, ni) #nº primos até 100;!!! 31 bastaria para o cálculo até 1000!!!
    
        np = ni*10-ni
        np = np//numProc
        
    else:
        prime = None
        np = None

    np = comm.bcast(np,root=0)
    init = ni+rank*np+1
    if rank!=numProc-1: #o último processador tem de fazer mais calculos caso n não seja divisível por numProc
        nf = init+np-1
    else:
        nf = ni*10
        
    flush_print(int(log10(ni)-1),rank,init,nf)
    
    prime_i = comm.bcast(prime,root=0) #primos do passo anterior
    prime_test = prime_i.copy()

    # result = nprime([],prime_test,init,nf)
    prime_f = comm.gather(nprime([],prime_test,init,nf),root=0) #primos calculados
    tf=time()
    
    if rank == 0:
        prime_f.insert(0,prime_i)
        
        prime = [ item for elem in prime_f for item in elem] # pôr os resultados numa unica lista
        
        flush_print('tempo de processamento:',tf-ti)
        
    ni*=10
    
    
if rank==0: 
    flush_print('\n\n',n,array(prime))
    print(f'\n{len(prime)} nº primos')
    
    
#%% COMMENTS
"""
O programa não é nada eficiente ...
Para n = 10**6  o tempo de processamento é ~64m (para 4 cores)
(o programa single demora ~54s)
Muito provavelmente deve-se ao facto de se usarem listas e fazerem operações
com as mesmas
"""