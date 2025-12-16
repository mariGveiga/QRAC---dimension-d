import picos as pc # Optimization lib
import numpy as np # Standard math lib
import qutip as qt # Quantum Mechanics Lib

d1=3        # Dimension of the beta subsystem (subsystem 1)
d2=3        # Dimension of the beta_ subsystem (subsystem 2)
d=d1*d2     # Dimension of the set of letters x0x1 -- d = d1*d2

N=2     # Word size -- quantity of letters/bases
fatorNormalizacao = 1/(N*d**2) # Normalization factor for the success probability
Pc = 0.5*(1 + 1/d) # Classical probability of success limit

# ---- Creation of the Hadamard Matrix ----
def create_hadamard(dim):
    hadamard_d = np.zeros((dim, dim), dtype=complex)        # Creation of matrix H_d according to dimension
    w = (2 * np.pi * 1j)/dim
    for i in range(dim):
        for j in range(dim):
            hadamard_d[i][j] = np.exp(j*w*i)
    return hadamard_d/np.sqrt(dim)

hadamard_d = create_hadamard(d)
hadamard_d2 = create_hadamard(d2)

def create_operator_optimization(M1, M2):
    # Resulting matrix of PICOS expressions
    M = np.zeros((N, d), dtype=object)
    
    x0 = 0
    for beta0 in range(d1): 
        for beta0_ in range(d2):
            x1 = 0
            for beta1 in range(d1):
                for beta1_ in range(d2):
                    
                    term_1 = M1[0, beta0] 
                    term_2 = M2[0, beta0_]
                    
                    # Converter Qobj para numpy array
                    if isinstance(term_2, qt.Qobj): term_2 = term_2.full()
                    elif isinstance(term_1, qt.Qobj): term_1 = term_1.full()
                    
                    # --- CORREÇÃO AQUI ---
                    # Use pc.kron para Produto Tensorial (funciona com var PICOS e numpy)
                    M[0, x0] = term_1@ term_2                    
                    
                    term_1_b1 = M1[1, beta1]
                    term_2_b1 = M2[1, beta1_]
                    
                    if isinstance(term_2_b1, qt.Qobj): term_2_b1 = term_2_b1.full()
                    elif isinstance(term_1_b1, qt.Qobj): term_1_b1 = term_1_b1.full()
                    
                    # --- CORREÇÃO AQUI ---
                    M[1, x1] = term_1_b1@term_2_b1
                    
                    x1 += 1
            x0 += 1
            
    return M

# ---- Creation of Measurement Bases ----

# Lists for first phase
Comp_basis = [] # Computational basis
Fourrier_basis = [] # Fourier basis
sigma0 = [[0 for _ in range(d)] for _ in range(d)] # pure state to be optimized

for x0 in range(d2):
    ketx0 = qt.Qobj(hadamard_d2[:, x0], dims=[[d2], [1]]) 
    f = ketx0*ketx0.dag()  # 2nd measurement operator -- |x_1><x_1|
    Fourrier_basis.append(f)

for x0 in range(d):
    a = qt.basis(d,x0)*qt.basis(d,x0).dag() # 1st measurement operator -- |x_0><x_0|
    
    for x1 in range(d):    
        ketx=qt.Qobj(hadamard_d[:, x1]) # Constructing one of the Mutually Unbiased Bases (MUBs) vectors
        
        # Creating the initial pure state -- equal superposition of the two basis states
        # Dividing Hilbert space in two qubits -- product states
        psi=(ketx+qt.basis(d,x0)) #(x1+x0)/sqrt(2)
        
        # Normalization factor: <psi|psi> = tr(|psi><psi|)
        sigma0[x0][x1]=psi*psi.dag()/((psi*psi.dag()).tr())     
        sigma0[x0][x1].dims = [[d1, d2], [d1, d2]]      # Divides the system into two parts (2 qubits)
    
    Comp_basis.append(a)

# Creation of system measurement operators
M1 = np.zeros((N, d1), dtype=object)  
M2 = np.zeros((N, d2), dtype=object)  
M = np.zeros((N, d), dtype=object)

x0=0
for beta0 in range(d1): 
    for beta0_ in range(d2):
        x1=0

        for beta1 in range(d1):
            for beta1_ in range(d2):

                # --- M1 refers to beta (Subsystem 1) ---       
                M1[0,beta0] = qt.basis(d1, beta0) * qt.basis(d1, beta0).dag()
                M1[1,beta1] = Fourrier_basis[beta1] 
                
                # --- M2 refers to beta_ (Subsystem 2) ---          
                M2[0,beta0_] = qt.basis(d2, beta0_) * qt.basis(d2, beta0_).dag()
                M2[1,beta1_] = Fourrier_basis[beta1_]

                # --- Tensor product to form the composite system measurement operators ---
                M[0,x0] = qt.tensor(M1[0,beta0], M2[0,beta0_])
                M[1,x1] = qt.tensor(M1[1,beta1], M2[1,beta1_])
                
                x1+=1
        x0+=1

'''
PHASE 1: Optimize the states, assuming fixed measurements (computational and Fourier basis)
'''

# OPTIMIZATION FOR THE STATE
F = pc.Problem() # Initiate first-phase solution  
Success=0     # Variable to store the sum of the success function 

for x0 in range(d):
    for x1 in range(d):
        # State sigma0 that will be optimized
        sigma0[x0][x1]=pc.HermitianVariable(f"sigma0_{x0}_{x1}", (d, d))

        # Restriction for the variable to be a valid quantum state
        F.add_constraint(sigma0[x0][x1]>>0)       # Positive semidefinite
        F.add_constraint(pc.trace(sigma0[x0][x1]) == 1)   # Trace 1

        # Success definition -- separated terms to avoid type errors
        # .full() converts Qobj to numpy array (extracts the complex matrix)
        term1 = pc.trace(M[0,x0].full() * sigma0[x0][x1])       
        term2 = pc.trace(M[1,x1].full() * sigma0[x0][x1])
        
        # Success formula -- Born's rule
        # Using np.real to ensure the solver sees real values in the objective
        Success += fatorNormalizacao * (np.real(term1) + np.real(term2)) 

# Our goal: maximize Success
F.set_objective("max", Success)
F.solve(solver="cvxopt")

# Storing the optimized success 
S=F.value
Pq=F.value/Pc

# Storing the optimized states in a list (converting back to numpy arrays)
SIGMA = [[0 for _ in range(d)] for _ in range(d)]
for x0 in range(d):
    for x1 in range(d):
        SIGMA[x0][x1]=(np.array(sigma0[x0][x1].value))

print("Success_SIGMA_opt = ",S) 

"""
PHASE 2: Optimize Measurement 1 (M1), fixing State (SIGMA) and Measurement 2 (M2)
"""
# M1_2 must have size based on subsystem dimensions, not global N or d
M1_2 = np.zeros((N, d1), dtype=object)
F1 = pc.Problem()

# Defining the measurement variables for subsystem 1
for i in range(d1): 
    # Base 0 (decode x0)
    M1_2[0,i] = pc.HermitianVariable(f"M1_2_x0_{i}", (d1, d1)) 
    F1.add_constraint(M1_2[0,i] >> 0)
    # Base 1 (decode x1)
    M1_2[1,i] = pc.HermitianVariable(f"M1_2_x1_{i}", (d1, d1)) # Attention: subsystem 1 has dimension d1
    F1.add_constraint(M1_2[1,i] >> 0)

# # Completeness Relation (Sum of projectors must be Identity)
F1.add_constraint(sum(M1_2[0,i] for i in range(d1)) - np.eye(d1) << 1e-8*np.eye(d1))
F1.add_constraint(sum(M1_2[1,i] for i in range(d1)) - np.eye(d1) << 1e-8*np.eye(d1))

# Create joint operator (M1_2 is Variable, M2 is Fixed from initialization)
M_2 = create_operator_optimization(M1_2, M2)

Success1 = 0 
for x0 in range(d):
    for x1 in range(d):
        # M_2[0, x0] is the joint operator for answer x0
        term1 = pc.trace(M_2[0,x0] * SIGMA[x0][x1])
        term2 = pc.trace(M_2[1,x1] * SIGMA[x0][x1])
        
        # Using .real to ensure compatibility with solver
        Success1 += fatorNormalizacao * (term1.real + term2.real)

F1.set_objective("max", Success1)
F1.solve(solver="cvxopt")

S1 = F1.value
print("Success_Measure_Opt_Phase2 = ", S1)

# Recovering numerical values after solving
M1_optimal_values = np.zeros((N, d1), dtype=object)
for r in range(2):
    for c in range(d1):
        M1_optimal_values[r,c] = qt.Qobj(M1_2[r,c].value)

"""
PHASE 3: Optimize Measurement 2 (M2), fixing State (SIGMA) and Measurement 1 (M1_optimal)
"""
M2_2 = np.zeros((N, d2), dtype=object)
F2 = pc.Problem()

# Defining variables for M2
for i in range(d2): 
    M2_2[0,i] = pc.HermitianVariable(f"M2_2_x0_{i}", (d2, d2)) 
    F2.add_constraint(M2_2[0,i] >> 0)
    
    M2_2[1,i] = pc.HermitianVariable(f"M2_2_x1_{i}", (d2, d2)) 
    F2.add_constraint(M2_2[1,i] >> 0)

# Completeness relation for M2
F2.add_constraint(sum(M2_2[0,i] for i in range(d2)) - np.eye(d2) << 1e-8*np.eye(d2))
F2.add_constraint(sum(M2_2[1,i] for i in range(d2)) - np.eye(d2) << 1e-8*np.eye(d2))

# Create joint operator (M1_optimal is Fixed, M2_2 is Variable)
M_3 = create_operator_optimization(M1_optimal_values, M2_2)

Success2 = 0 
for x0 in range(d):
    for x1 in range(d):
        term1 = pc.trace(M_3[0,x0] * SIGMA[x0][x1])
        term2 = pc.trace(M_3[1,x1] * SIGMA[x0][x1])
        Success2 += fatorNormalizacao * (term1.real + term2.real)

F2.set_objective("max", Success2)
# Tente relaxar a tolerância relativa (rel_tol) e absoluta (abs_tol)
# O padrão é geralmente muito rigoroso (ex: 1e-8)
F2.solve(solver="cvxopt")
S2 = F2.value

print("Success_Measure_Opt_Phase3 = ", S2)