import numpy as np
import qutip as qt # Standard math lib
import picos as pc # Optimization lib

d = 4;
w =np.exp((2 * np.pi * 1j)/d);
fatorNormalizacao = 1/np.sqrt(d)
Id = np.eye(d) # Defining the Identity operator
Pc = 0.5*(1 + 1/d) # Classical probability of success

# Lists for the first phase (encoding)
Pq = [] # first quantum/classical probability of success
S = [] # 1st quantum success
RHO = []
RHO_0 = []

# Lists for the second phase (decoding)
Pq1 = [] # 2nd quantum/classical probability of success
S1 = [] # 2nd quantum success

# Lists for first phase
Comp_basis = [] # Computational basis
Fourrier_basis = [] # Fourier basis

# ---- criação das bases de medida ----

H = np.array([[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]]) *1/np.sqrt(d)     # Hadamard para dimensão 4

# sigma0 = np.zeros((d, d, d ,d), dtype = complex)
sigma0 = [[0 for _ in range(d)] for _ in range(d)] # pure state to be optimized
for x0 in range(d):
    a = qt.basis(d,x0)*qt.basis(d,x0).dag() # 1st measurement operator -- |x_0><x_0|
    for x1 in range(d):
        ketx=qt.Qobj(H[:, x1]) # Constructiong one of the MUBs
        b = ketx*ketx.dag()  # 2nd measurement operator -- |x_1><x_1|

        # Creating the initial pure state -- equal superposition of the two basis states
        # Divinding Hilbert space in two qubits -- product states
        psi=(ketx+qt.basis(d,x0)) #(x1+x0)/sqrt(2)
        sigma0[x0][x1]=psi*psi.dag()/((psi*psi.dag()).tr())     # <psi|psi> = tr(|psi><psi|) -- fator de normalização
        sigma0[x0][x1].dims = [[2, 2], [2, 2]]      # divide o sistema em duas partes (2 qubits)
        Fourrier_basis.append(b[:])
    Comp_basis.append(a[:])
Fourrier_basis=Fourrier_basis[:d]

# print("Comp_basis")
# print(Comp_basis)
# print("Fourrier_basis") 
# print(np.round(Fourrier_basis,2))
# print("Traço da primeira matriz de sigma:", sigma0[0][0].ptrace(0))
print("pureza = ", (sigma0[0][0].ptrace(0)*sigma0[0][0].ptrace(0)).tr())

'''
First, we optimize the states, assuming fixed measurements (computational and fourrier basis)
'''

# OPTIMIZATION FOR THE STATE
F = pc.Problem() #Initiate first-phase solution   
Success = 0 # Optimization variable
for x0 in range(d):
    for x1 in range(d):
        # State sigma0 that will be optimizied
        sigma0[x0][x1]=pc.HermitianVariable(f"sigma0_{x0}_{x1}", (d, d))
        # Restriction to the state be a state.
        F.add_constraint(sigma0[x0][x1]>>0)       # Positive semidefinite
        F.add_constraint(pc.trace(sigma0[x0][x1]) == 1)   # Trace 1
        # Success definition
        Success += np.real((1/(2*d**2))*(pc.trace(Comp_basis[x0]*sigma0[x0][x1]) + pc.trace(Fourrier_basis[x1]*sigma0[x0][x1]))) # Success formula -- Born's rule

# Our goal: maximize Success
F.set_objective("max", Success)
F.solve(solver = "cvxopt")#, verbosity=1)

# storing the optimized success 
S=F.value
Pq=F.value/Pc

# storing the optimized states in a list
SIGMA = [[0 for _ in range(d)] for _ in range(d)]
for x0 in range(d):
    for x1 in range(d):
        SIGMA[x0][x1]=(np.array(sigma0[x0][x1].value))
print("Success_SIGMA_opt = ",S) 
print("Comp_basis = \n",np.round(Comp_basis,3))
print("Fourrier_basis = \n",np.round(Fourrier_basis,3))
# print("SIGMA = \n",np.round(SIGMA,3))

"""
From now on we optimize the measurement using the optimized states (RHO) from the first phase as input
"""
# OPTIMIZATION FOR THE MEASUREMENT
Comp_basis_2 = [0 for _ in range(d)]
Fourrier_basis_2 = [0 for _ in range(d)]
F1 = pc.Problem() # Initiate second-phase solution

# Defining the measurements.
for i in range(d):
    # Fisrt basis to decode x0
    Comp_basis_2[i]=pc.HermitianVariable(f"Comp_basis_2_{i}", (d, d)) 
    F1.add_constraint(Comp_basis_2[i]>>0)   # Constraint to be a projector
    
    Fourrier_basis_2[i]=pc.HermitianVariable(f"Fourrier_basis_2_{i}", (d, d))   # Second basis to decode x1
    F1.add_constraint(Fourrier_basis_2[i]>>0)   # Constraint to be a projector

# Constraint to that the projectors of A1 (B1) form a complete basis.   
F1.add_constraint(pc.sum(Comp_basis_2)==Id)
F1.add_constraint(pc.sum(Fourrier_basis_2)==Id)

Success1 = 0 # Optimization variable
for x0 in range(d):
    for x1 in range(d):
        Success1 += np.real((1/(2*d**2))*(pc.trace(Comp_basis_2[x0]*SIGMA[x0][x1]) + pc.trace(Fourrier_basis_2[x1]*SIGMA[x0][x1])))

# Our goal: maximize Success   
F1.set_objective("max", Success1) 
F1.solve(solver = "cvxopt")
S1=F1.value

print("Success_Measure_Opt = ",S1)

Am=[]
Bm=[]
for i in range(d):
    Am.append(np.array(Comp_basis_2[i].value))
    Bm.append(np.array(Fourrier_basis_2[i].value))
print("Comp_basis_2 = \n",np.round(Am,3))
print("Fourrier_basis_2 = \n",np.round(Bm,3))
#print("SIGMA = \n",np.round(SIGMA,3))