import picos as pc # Optimization lib
import numpy as np # Standard math lib
import qutip as qt # Quantum Mechanics Lib

d=4     # dimensão do conjunto de letras x0x1 -- d = d0d1
d1=2    # dimensão do subsistema do beta
d2=2    # dimensão do subsistema do beta_

N=2     # tamanho da palavra -- quantidade de letras
fatorNormalizacao = 1/(N*d**2)
Pc = 0.5*(1 + 1/d) # Classical probability of success

# ---- criação da hadamard ----
def create_hadamard(d):
    hadamard_d = np.zeros((d, d), dtype=complex)        # criação da matriz H_d de acordo com a dimensão
    w = (2 * np.pi * 1j)/d
    for i in range(d):
        for j in range(d):
            hadamard_d[i][j] = np.exp(j*w*i)
        hadamard_d[i] = hadamard_d[i]/np.sqrt(d)
    return hadamard_d

hadamard_4 = create_hadamard(4)
hadamard_2 = create_hadamard(2)

# ---- criação das bases de medida ----

# Lists for first phase
Comp_basis = [] # Computational basis
Fourrier_basis = [] # Fourier basis
sigma0 = [[0 for _ in range(d)] for _ in range(d)] # pure state to be optimized

for x0 in range(d):
    a = qt.basis(d,x0)*qt.basis(d,x0).dag() # 1st measurement operator -- |x_0><x_0|
    
    for x1 in range(d):
    
        ketx=qt.Qobj(hadamard_4[:, x1]) # Constructiong one of the MUBs
        b = ketx*ketx.dag()  # 2nd measurement operator -- |x_1><x_1|

        # Creating the initial pure state -- equal superposition of the two basis states
        # Divinding Hilbert space in two qubits -- product states
        psi=(ketx+qt.basis(d,x0)) #(x1+x0)/sqrt(2)
        sigma0[x0][x1]=psi*psi.dag()/((psi*psi.dag()).tr())     # <psi|psi> = tr(|psi><psi|) -- fator de normalização
        sigma0[x0][x1].dims = [[2, 2], [2, 2]]      # divide o sistema em duas partes (2 qubits)
        
        ketx_f = qt.Qobj(hadamard_2[:, x1%2], dims=[[d2], [1]]) # Constructiong one of the MUBs for d=2 -- para formar a base de fourrier, usando a hadamard de dimensão 2
        f = ketx_f*ketx_f.dag()  # 2nd measurement operator -- |x_1><x_1|
        Fourrier_basis.append(f)
    
    Comp_basis.append(a)
Fourrier_basis=Fourrier_basis[:d2]

# Criação dos operadores de medida do sistema
M = np.zeros((2, N*N), dtype=object)  
M1 = np.zeros((2, N), dtype=object)   # M1 também
M2 = np.zeros((2, N), dtype=object)   # M2 também

x0=0
for beta0 in range(d1): 
    for beta0_ in range(d2):
        x1=0

        for beta1 in range(d1):
            for beta1_ in range(d2):

                # --- M1 é referente ao beta ---       
                M1[0,beta0] = qt.basis(d1, beta0) * qt.basis(d1, beta0).dag()
                M1[1,beta1] = Fourrier_basis[beta1] 
                
                # --- M2 é referente ao beta_ ---           
                M2[0,beta0_] = qt.basis(d2, beta0_) * qt.basis(d2, beta0_).dag()
                M2[1,beta1_] = Fourrier_basis[beta1_]
                
                # --- produto tensorial para formar os operadores de medida do sistema composto ---
                M[0,x0] = qt.tensor(M1[0,beta0], M2[0,beta0_])
                M[1,x1] = qt.tensor(M1[1,beta1], M2[1,beta1_])
                
                x1+=1
        x0+=1

# for i in range(2):
#     for j in range(N*N):
#         print(f"M[{i}][{j}] =")
#         print(np.round(M[i][j],2))

# for i in range(d):
#     for j in range(d):
#         print(f"sigma0[{i}][{j}] =")
#         print(sigma0[i][j])
# print("Success =", Success)

'''
First, we optimize the states, assuming fixed measurements (computational and fourrier basis)
'''

# OPTIMIZATION FOR THE STATE
F = pc.Problem() #Initiate first-phase solution  
Success=0     # variável para armazenar a soma da função sucesso 

for x0 in range(d):
    for x1 in range(d):
        # State sigma0 that will be optimizied
        sigma0[x0][x1]=pc.HermitianVariable(f"sigma0_{x0}_{x1}", (d, d))

        # Restriction to the state be a state
        F.add_constraint(sigma0[x0][x1]>>0)       # Positive semidefinite
        F.add_constraint(pc.trace(sigma0[x0][x1]) == 1)   # Trace 1

        # Success definition -- termos separados para evitar erro de tipo
        term1 = pc.trace(M[0,x0].full() * sigma0[x0][x1])
        term2 = pc.trace(M[1,x1].full() * sigma0[x0][x1])
        
        Success += fatorNormalizacao * (np.real(term1) + np.real(term2)) # Success formula -- Born's rule

# Our goal: maximize Success
F.set_objective("max", Success)
F.solve(solver = "cvxopt")#, verbosity=1)

# Storing the optimized success 
S=F.value
Pq=F.value/Pc

# Storing the optimized states in a list
SIGMA = [[0 for _ in range(d)] for _ in range(d)]
for x0 in range(d):
    for x1 in range(d):
        SIGMA[x0][x1]=(np.array(sigma0[x0][x1].value))

print("Success_SIGMA_opt = ",S) 

print("--------- Matriz sigma otimizada ---------")
for i in range(d):
    for j in range(d):
        print(f"SIGMA[{i}][{j}] =")
        print(np.round(SIGMA[i][j],3))

print("--------- Operador de medida x0 otimizado ---------")
for j in range(N*N):
    print(f"M[0][{j}] =")
    print(np.round(M[0][j].full(),2))

print("--------- Operador de medida x1 otimizado ---------")
for j in range(N*N):
    print(f"M[1][{j}] =")
    print(np.round(M[1][j].full(),2))

