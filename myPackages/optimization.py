import picos as pc # Optimization lib
import numpy as np # Standard math lib
import qutip as qt # Quantum Mechanics Lib
from myPackages.creation import create_operator_optimization

def clean_matrix(M):
    # Transforms any type of matrix into a numpy array of complex numbers (cleans it for optimization)
    if hasattr(M, 'full'):
        M = M.full()
    if hasattr(M, 'value'):
        M = M.value

    M_clean = [[None for _ in range(len(M))] for _ in range(len(M[0]))]
    for r in range(len(M)):
        for c in range(len(M[0])):
            element = M[r][c]
            
            # Caso 1: É uma variável do PICOS resolvida
            if hasattr(element, 'value'):
                element = element.value
            
            # Caso 2: É um objeto QuTiP
            if hasattr(element, 'full'):
                element = element.full()
            
            # Caso 3: Garante que é um array numpy de complexos
            M_clean[r][c] = np.array(element, dtype=complex)

    return np.array(M_clean)

'''
PHASE 1: Optimize the states, assuming fixed measurements (Computational and Fourier basis)
'''

def optimize_NonLocalStates(sigma, M, D, fatorNormalizacao, Pc):
    # OPTIMIZATION FOR THE STATE
    F = pc.Problem() # Initiate first-phase solution  
    Success=0     # Variable to store the sum of the success function 

    for x0 in range(D):
        for x1 in range(D):
            # State sigma that will be optimized
            sigma[x0][x1]=pc.HermitianVariable(f"sigma_{x0}_{x1}", (D, D))

            # Restriction for the variable to be a valid quantum state
            F.add_constraint(sigma[x0][x1]>>0)       # Positive semidefinite
            F.add_constraint(pc.trace(sigma[x0][x1]) == 1)   # Trace 1  
            # Success definition -- separated terms to avoid type errors
            # .full() converts Qobj to numpy array (extracts the complex matrix)
            term1 = pc.trace(M[0,x0].full() * sigma[x0][x1])       
            term2 = pc.trace(M[1,x1].full() * sigma[x0][x1])

            # Success formula -- Born's rule
            # Using np.real to ensure the solver sees real values in the objective
            Success += fatorNormalizacao * (np.real(term1) + np.real(term2)) 

    # Our goal: maximize Success
    F.set_objective("max", Success)
    try:
        #solver_params = {'primalfactor_tol': 1e-12} # Opcional, dependendo da versão
        F.solve(solver="cvxopt")
    except Exception as e:
        print(f"Erro na otimização: {e}")
        return None, 0

    # Storing the optimized success 
    S=F.value
    Pq=F.value/Pc

    # Storing the optimized states in a list (converting back to numpy arrays)
    SIGMA = np.zeros((D, D), dtype=object)
    for x0 in range(D):
        for x1 in range(D):
            SIGMA[x0][x1]=(np.array(sigma[x0][x1].value))

    return SIGMA, S

def optimize_LocalStates(sigma_fixed, M, d, fatorNormalizacao, subsystem_target):
    D = d**2
    # Fix one of the states and optimize the other
    F = pc.Problem() # Initiate first-phase solution
    Success=0     # Variable to store the sum of the success function    
    sigma_opt = [[None for _ in range(d)] for _ in range(d)]   # pure state to be optimized
    sigma = [[None for _ in range(D)] for _ in range(D)]   # joint state

    for x0 in range(D):
        idx_0 = x0 % d  # Map x0 to subsystem index
        for x1 in range(D):
            idx_1 = x1 % d  # Map x1 to subsystem index
            # State sigma that will be optimized
            sigma_opt[idx_0][idx_1]=pc.HermitianVariable(f"sigma_opt_{x0}_{x1}", (d, d))

            # Restriction for the variable to be a valid quantum state
            F.add_constraint(sigma_opt[idx_0][idx_1]>>0)       # Positive semidefinite
            F.add_constraint(pc.trace(sigma_opt[idx_0][idx_1]) == 1)   # Trace 1

            # Verifies if sigma_fixed elements are Qobj and converts to numpy if necessary
            # hasattr checks if the object has the attribute 'full'
            if hasattr(sigma_fixed[idx_0][idx_1], 'full'): 
                sigma_fixed[idx_0][x1] = sigma_fixed[idx_0][x1].full()

            if subsystem_target == 1:
                # Optimizing sigma0_1 (Variable), fixing sigma0_2 (Constant)
                # sigma = sigma0_1 (x) sigma0_2
                sigma[x0][x1] = pc.kron(sigma_opt[idx_0][idx_1], sigma_fixed[idx_0][idx_1])
            else:
                # Optimizing sigma0_2 (Variable), fixing sigma0_1 (Constant)
                # sigma = sigma0_1 (x) sigma0_2
                sigma[x0][x1] = pc.kron(sigma_fixed[idx_0][idx_1], sigma_opt[idx_0][idx_1])
            

            Op_Comp = clean_matrix(M[0, idx_0])
            Op_Four = clean_matrix(M[1, idx_1])
            
            term1 = pc.trace(pc.Constant(Op_Comp) * sigma[x0][x1])
            term2 = pc.trace(pc.Constant(Op_Four) * sigma[x0][x1])

            # Success formula -- Born's rule
            Success += fatorNormalizacao * (np.real(term1) + np.real(term2)) 

    # Our goal: maximize Success
    F.set_objective("max", Success)

    try:
        #solver_params = {'primalfactor_tol': 1e-12} # Opcional, dependendo da versão
        F.solve(solver="cvxopt")
    except Exception as e:
        print(f"Erro na otimização: {e}")
        return None, 0

    S = F.value
    
    sigma_optimal_values = np.zeros((d, d), dtype=object)   
    for x0 in range(d):
        for x1 in range(d):
            sigma_optimal_values[x0][x1] = np.array(sigma_opt[x0][x1].value)
            
    return sigma, sigma_optimal_values, S

def optimize_LocalMeasurements(M_fixed, sigma, fatorNormalizacao, d, D, N):
    """
    PHASE 2: Optimize Measurement 1 (M_fixed), fixing State (SIGMA) and Measurement 2 (M_opt)
    """
    M_opt = np.zeros((N, d), dtype=object)  # matrix that will be optimized 
    F = pc.Problem()

    # Defining the measurement variables for subsystem 1
    for i in range(d): 
        # Base 0 (decode x0)
        M_opt[0,i] = pc.HermitianVariable(f"M_opt_x0_{i}", (d, d)) 
        F.add_constraint(M_opt[0,i] >> 0)
        # Base 1 (decode x1)
        M_opt[1,i] = pc.HermitianVariable(f"M_opt_x1_{i}", (d, d)) 
        F.add_constraint(M_opt[1,i] >> 0)

    # Completeness Relation (Sum of projectors must be Identity)
    F.add_constraint(sum(M_opt[0,i] for i in range(d)) == np.eye(d))
    F.add_constraint(sum(M_opt[1,i] for i in range(d)) == np.eye(d))

    # Create joint operator (M_opt is Variable, M_fixed is Fixed from initialization)
    M_final = create_operator_optimization(M_opt, M_fixed, d, D, N)
    Success1 = 0 

    for x0 in range(D):
        for x1 in range(D):
            # Verifies if sigma_fixed elements are Qobj and converts to numpy if necessary
            # hasattr checks if the object has the attribute 'full'
            if hasattr(sigma[x0][x1], 'full'): 
                sigma[x0][x1] = sigma[x0][x1].full()
            # M_final[0, x0] is the joint operator for answer x0

            Op_Comp = clean_matrix(M_final[0, x0])
            Op_Four = clean_matrix(M_final[1, x1])

            term1 = pc.trace(pc.Constant(Op_Comp) * sigma[x0][x1])
            term2 = pc.trace(pc.Constant(Op_Four) * sigma[x0][x1])

            # Using .real to ensure compatibility with solver
            Success1 += fatorNormalizacao * (term1.real + term2.real)

    F.set_objective("max", Success1)
    try:
        #solver_params = {'primalfactor_tol': 1e-12} # Opcional, dependendo da versão
        F.solve(solver="cvxopt")
    except Exception as e:
        print(f"Erro na otimização: {e}")
        return None, 0
    S = F.value

    # Recovering numerical values after solving
    M_optimal_values = np.zeros((N, d), dtype=object)
    for r in range(N):
        for c in range(d):
            M_optimal_values[r,c] = qt.Qobj(M_opt[r,c].value)
    

    return M_final, M_optimal_values, S