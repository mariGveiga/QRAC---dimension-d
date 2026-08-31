import myPackages.creation as cs
import myPackages.optimization as opt
import numpy as np
import qutip as qt
import picos as pc

'''
O que eu quero fazer?
Transformar o código de QRAC's com estados e medidas locais para o caso de dimensões diferentes d1 e d2
'''

def build_psi_subsystem(d, idx_0, idx_1, hadamard_d):
    # Mapeia as entradas globais para o subsistema local
    
    # 1. Primeiro termo da fórmula: |x_0>
    ket_x0 = qt.basis(d, idx_0)
    
    # 2. Segundo termo: Somatório da base de Fourier deslocada
    sum_term = qt.Qobj(np.zeros((d, 1))) 
    
    for j in range(d):
        
        # O ket deslocado |j + x_0> com fechamento modular
        shifted_ket = qt.basis(d, (j + idx_0) % d) 
        
        sum_term += hadamard_d[j][idx_1] * shifted_ket
        
    psi = (ket_x0 + sum_term).unit()    
    return psi

# def createLocalStates_d1d2(d1, d2, D, hadamard_d1, hadamard_d2):
#     Comp_basis = [] # Computational basis
#     Fourrier_basis = [] # Fourier basis
    
#     sigma0 = np.zeros((D, D), dtype=object)
#     sigma0_1 = np.zeros((d1, d2), dtype=object)
#     sigma0_2 = np.zeros((d1, d2), dtype=object)

#     for x0 in range(D):
    
#             idx_0 = x0 % d1  # Map x0 to subsystem index
#             a = qt.basis(d1,idx_0)*qt.basis(d1,idx_0).dag() # 1st measurement operator -- |x_0><x_0|
#             Comp_basis.append(a)
            
#             for x1 in range(D):
                
#                 idx_1 = x1 % d2  # Map x1 to subsystem index

#                 psi_1 = build_psi_subsystem(d1, idx_0, idx_1, hadamard_d1)   # State for subsystem 1
#                 psi_2 = build_psi_subsystem(d2, idx_0, idx_1, hadamard_d2)   # State for subsystem 2

#                 sigma0_1[idx_0][idx_1] = psi_1 * psi_1.dag()     
#                 sigma0_2[idx_0][idx_1] = psi_2 * psi_2.dag()
#                 sigma0[x0][x1]=qt.tensor(sigma0_1[idx_0][idx_1], sigma0_2[idx_0][idx_1])   
                
#                 # Constructing one of the MUBs for d=2 to form the Fourier basis using Hadamard of dimension 2
#                 if x0 == 0:     # Forbids duplicates in the Fourier basis
#                     ketx_f = qt.Qobj(hadamard_d2[:, idx_1]) 
#                     f = ketx_f * ketx_f.dag()
#                     Fourrier_basis.append(f)      
    
#     return sigma0, sigma0_1, sigma0_2, Comp_basis, Fourrier_basis

def createLocalStates_d1d2(d1, d2, D, hadamard_d1, hadamard_d2):
    Comp_basis = [] 
    Fourrier_basis = [] 
    
    sigma0 = np.zeros((D, D), dtype=object)
    
    # As matrizes locais devem ter o tamanho exato de cada subsistema
    sigma0_1 = np.zeros((d1, d1), dtype=object)
    sigma0_2 = np.zeros((d2, d2), dtype=object)

    for x0 in range(D):
        # Mapeamento do índice global x0 para cada subsistema
        idx_0_1 = x0 % d1  
        idx_0_2 = x0 % d2  
        
        # Base computacional (mantendo a dimensão d1 como referência)
        a = qt.basis(d1, idx_0_1) * qt.basis(d1, idx_0_1).dag() 
        Comp_basis.append(a)
        
        for x1 in range(D):
            # Mapeamento do índice global x1 para cada subsistema
            idx_1_1 = x1 % d1  
            idx_1_2 = x1 % d2  
            
            # 1. Estado local puro para o Subsistema 1 (Dimensão d1)
            ketx_1 = qt.Qobj(hadamard_d1[:, idx_1_1])
            psi_1 = (ketx_1 + qt.basis(d1, idx_0_1)).unit()
            sigma0_1[idx_0_1][idx_1_1] = psi_1 * psi_1.dag()     
            
            # 2. Estado local puro para o Subsistema 2 (Dimensão d2)
            ketx_2 = qt.Qobj(hadamard_d2[:, idx_1_2])
            psi_2 = (ketx_2 + qt.basis(d2, idx_0_2)).unit()
            sigma0_2[idx_0_2][idx_1_2] = psi_2 * psi_2.dag() 
            
            # 3. Estado conjunto via produto tensorial
            sigma0[x0][x1] = qt.tensor(sigma0_1[idx_0_1][idx_1_1], sigma0_2[idx_0_2][idx_1_2])   
            
            # Construindo a base de Fourier (usando a dimensão d2 como referência)
            if x0 == 0: 
                ketx_f = qt.Qobj(hadamard_d2[:, idx_1_2]) 
                f = ketx_f * ketx_f.dag()
                Fourrier_basis.append(f)      

    return sigma0, sigma0_1, sigma0_2, Comp_basis, Fourrier_basis

def createMeasurementOperators_d1d2(d1, d2, D, Fourrier_basis, N):
    M1 = np.zeros((N, d1), dtype=object)  
    M2 = np.zeros((N, d2), dtype=object)  
    M = np.zeros((N, D), dtype=object)

    x0 = 0
    for beta0 in range(d1): 
        for beta0_ in range(d2):
            x1 = 0

            for beta1 in range(d1):
                for beta1_ in range(d2):
                    # --- M1 refers to beta (Subsystem 1) ---       
                    M1[0, beta0] = qt.basis(d1, beta0) * qt.basis(d1, beta0).dag()   
                    M1[1, beta1] = Fourrier_basis[beta1] 
                    
                    # --- M2 refers to beta_ (Subsystem 2) ---          
                    M2[0, beta0_] = qt.basis(d2, beta0_) * qt.basis(d2, beta0_).dag()
                    M2[1, beta1_] = Fourrier_basis[beta1_]

                    # --- Tensor product to form the composite system measurement operators ---
                    M[0, x0] = qt.tensor(M1[0, beta0], M2[0, beta0_])
                    M[1, x1] = qt.tensor(M1[1, beta1], M2[1, beta1_])
                    
                    x1 += 1
            x0 += 1

    return M1, M2, M

# Creation of measurement operators for optimization (PICOS variables)
def create_operator_optimization(M_opt, M_fixed, d1, d2, D, N, subsystem_target):
    # if subsystem_target == 1:
    #     d_term_1 = d2
    #     d_term_2 = d1
    # else:
    #     d_term_1 = d1
    #     d_term_2 = d2

    # Resulting matrix of PICOS expressions (for the joint system)
    M = np.zeros((N, D), dtype=object)      
    for x in range(N):
        x_i = 0
        for beta in range(d1):
            for beta_ in range(d2):
                
                if subsystem_target == 1:
                    term_1 = M_opt[x, beta]  
                    term_2 = M_fixed[x, beta_] 
                elif subsystem_target == 2:
                    term_1 = M_fixed[x, beta] 
                    term_2 = M_opt[x, beta_] 
                
                # Check if terms are Qobj and convert them to numpy arrays if necessary
                if isinstance(term_1, qt.Qobj): 
                    term_1 = term_1.full()
                if isinstance(term_2, qt.Qobj): 
                    term_2 = term_2.full()
                
                M[x, x_i] = term_1 @ term_2
                x_i += 1
    return M

def optimize_LocalMeasurements_d1d2(M_fixed, sigma, fatorNormalizacao, d1, d2, D, N, subsystem_target):
    if subsystem_target == 1:
        d = d1
    else:
        d = d2

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
    F.add_constraint(sum(M_opt[0,i] for i in range(d)) - np.eye(d) << 1e-10*np.eye(d))
    F.add_constraint(sum(M_opt[1,i] for i in range(d)) - np.eye(d) << 1e-10*np.eye(d))
    
    M_final = create_operator_optimization(M_opt, M_fixed, d1, d2, D, N, subsystem_target)
    # print("M_final:", M_final[0])  # Debug: Check the structure of M_final
    Success1 = 0 

    for x0 in range(D):
        for x1 in range(D):
            # Verifies if sigma_fixed elements are Qobj and converts to numpy if necessary
            # hasattr checks if the object has the attribute 'full'
            if hasattr(sigma[x0][x1], 'full'): 
                sigma[x0][x1] = sigma[x0][x1].full()

            # M_final[0, x0] is the joint operator for answer x0
            term1 = pc.trace(M_final[0, x0] * sigma[x0][x1])
            term2 = pc.trace(M_final[1, x1] * sigma[x0][x1])
            
            # Using .real to ensure compatibility with solver
            Success1 += fatorNormalizacao * (term1.real + term2.real)

    F.set_objective("max", Success1)
    try:
        F.solve(solver="cvxopt")
    except Exception as e:
        print(f"Erro na otimização: {e}")
        return None, None, 0
    
    S = F.value

    # Recovering numerical values after solving
    M_optimal_values = np.zeros((N, d), dtype=object)
    for r in range(N):
        for c in range(d):
            M_optimal_values[r,c] = qt.Qobj(np.array(M_opt[r,c].value))
    
    return M_final, M_optimal_values, S

def optimize_LocalStates_d1d2(sigma_fixed, M, d1, d2, D, fatorNormalizacao, subsystem_target):
    if subsystem_target == 1:
        d = d1
    else:
        d = d2
    # Fix one of the states and optimize the other
    F = pc.Problem() # Initiate first-phase solution
    Success=0     # Variable to store the sum of the success function    
    sigma_opt = [[None for _ in range(d)] for _ in range(d)]   # pure state to be optimized
    sigma = [[None for _ in range(D)] for _ in range(D)]   # joint state

    for i in range(d):
        for j in range(d):
            # State sigma that will be optimized
            sigma_opt[i][j] = pc.HermitianVariable(f"sigma_opt_{i}_{j}", (d, d))
            # Restriction for the variable to be a valid quantum state
            F.add_constraint(sigma_opt[i][j] >> 0)      # Positive semidefinite
            F.add_constraint(pc.trace(sigma_opt[i][j]) == 1)    # Trace 1

    for x0 in range(D):
        for x1 in range(D):

            # // d -- Most Significant Bit (MSB)
            msb_0 = x0 // d2
            msb_1 = x1 // d2

            # % d -- Least Significant Bit (LSB)
            lsb_0 = x0 % d2
            lsb_1 = x1 % d2

            # Direcionamento condicional baseado no alvo da otimização
            if subsystem_target == 1:
                # First qubit will be optmized. It gets the MSBs.
                idx_0_var, idx_1_var = msb_0, msb_1

                # Second qubit is the constant. It gets the LSBs.
                idx_0_fix, idx_1_fix = lsb_0, lsb_1

            elif subsystem_target == 2:
                # First qubit will be optmized. It gets the MSBs.
                idx_0_fix, idx_1_fix = msb_0, msb_1

                # Second qubit is the constant. It gets the LSBs.
                idx_0_var, idx_1_var = lsb_0, lsb_1

            # Verifies if sigma_fixed elements are Qobj and converts to numpy if necessary
            # hasattr checks if the object has the attribute 'full'
            fixed_part = sigma_fixed[idx_0_fix][idx_1_fix]
            if hasattr(fixed_part, 'full'): fixed_part = fixed_part.full()
            elif hasattr(fixed_part, 'value'): fixed_part = fixed_part.value
            fixed_part = np.array(fixed_part, dtype=complex) # ensure it's a numpy array

            if subsystem_target == 1:
                # Optimizing sigma0_1 (Variable), fixing sigma0_2 (Constant)
                # sigma = sigma0_1 (x) sigma0_2
                # sigma[x0][x1] = pc.kron(sigma_opt[idx_0_var][idx_1_var], pc.Constant(fixed_part))
                sigma[x0][x1] = sigma_opt[idx_0_var][idx_1_var] @ pc.Constant(fixed_part)
            else:
                # Optimizing sigma0_2 (Variable), fixing sigma0_1 (Constant)
                # sigma = sigma0_2 (x) sigma0_1
                # sigma[x0][x1] = pc.kron(pc.Constant(fixed_part), sigma_opt[idx_0_var][idx_1_var])
                sigma[x0][x1] = pc.Constant(fixed_part) @ sigma_opt[idx_0_var][idx_1_var]

            # Ensure M elements are numpy arrays for the trace calculation -- picos may not handle Qobj directly
            op_comp_val = M[0, x0]
            if hasattr(op_comp_val, 'value'): op_comp_val = op_comp_val.value

            op_four_val = M[1, x1]
            if hasattr(op_four_val, 'value'): op_four_val = op_four_val.value

            term1 = pc.trace(pc.Constant(op_comp_val) * sigma[x0][x1])
            term2 = pc.trace(pc.Constant(op_four_val) * sigma[x0][x1])
            # Success formula -- Born's rule
            Success += fatorNormalizacao * (np.real(term1) + np.real(term2)) 

    # Our goal: maximize Success
    F.set_objective("max", Success)

    try:
        F.solve(solver="cvxopt")
    except Exception as e:
        print(f"Erro na otimização: {e}")
        return None, None, 0

    S = F.value

    sigma_optimal_values = np.zeros((d, d), dtype=object)   
    for x0 in range(d):
        for x1 in range(d):
            sigma_optimal_values[x0][x1] = np.array(sigma_opt[x0][x1].value)

    sigma_full_numeric = np.zeros((D, D), dtype=object)
    for x0 in range(D):
        for x1 in range(D):
            # extracts the optimized sigma values for the full joint state (after optimization, sigma is expressed in terms of the optimized local states)
            sigma_full_numeric[x0][x1] = np.array(sigma[x0][x1].value)

    return sigma_full_numeric, sigma_optimal_values, S   

def main():
    d1=3         # Dimension of the beta subsystem 1
    d2=2         # Dimension of the beta_ subsystem 2
    D = d1*d2     # Dimension of the set of letters x0x1
    N=2                                 # Word size -- quantity of letters/bases
    fatorNormalizacao = 1/(N*D**2)      # Normalization factor for the success probability

    hadamard_d1 = cs.create_hadamard(d1)
    hadamard_d2 = cs.create_hadamard(d2)
    # print(hadamard_d1)
    # print(hadamard_d2)

    sigma0, sigma0_1, sigma0_2, Comp_basis, Fourrier_basis = createLocalStates_d1d2(d1, d2, D, hadamard_d1, hadamard_d2)

    M1, M2, M = createMeasurementOperators_d1d2(d1, d2, D, Fourrier_basis, N)

    # print("Computational Basis:", Comp_basis)
    # print("\nFourier Basis:", Fourrier_basis)
    # print("\nInitial States:", sigma0)
    # print("\nMeasurement Operators M:", M2)

    # if D<4:
    #     tolerance = 1e-8   
    # else:
    #     tolerance = 1e-7
    tolerance = 1e-8
    t_max = 200
    
    St_inicial = 0
    M_fixed = M2
    sigma = sigma0
    sucess_history = []

    for t in range(t_max):
        M_inicial, M1_optimal_values, S = optimize_LocalMeasurements_d1d2(M_fixed, sigma, fatorNormalizacao, d1, d2, D, N, 1)
        #M_inicial = M1_optimal_values (optimized) and M2 (fixed) -- sigma fixo
        # print(M1_optimal_values)
        
        M_final, M2_optimal_values, S1 = optimize_LocalMeasurements_d1d2(M1_optimal_values, sigma, fatorNormalizacao, d1, d2, D, N, 2)
        # M_final = M1_optimal_values (optimized) and M2_optimal_values (optimized) -- sigma fixo
        # print(M2_optimal_values)

        sigma_inicial, sigma1_opt, S2 = optimize_LocalStates_d1d2(sigma0_2, M_final, d1, d2, D, fatorNormalizacao, 1)
        # sigma_inicial = sigma1_opt (optimized) and sigma0_2 (fixed)
        # print(sigma1_opt)

        sigma_final, sigma2_opt, S_final = optimize_LocalStates_d1d2(sigma1_opt, M_final, d1, d2, D, fatorNormalizacao, 2)
        # sigma_final = sigma1_opt (optimized) and sigma2_opt (optimized)
        # print(sigma2_opt)

        sucess_history.append(S_final)
        print(f"Iteration {t+1}: Total Success = {S_final}")

        if t > 0 and abs(S_final - St_inicial) <= tolerance:
            print("Convergence reached.")

            for x0 in range(D):
                for x1 in range(D):
                    
                    # tr_rho = np.abs(np.vdot(sigma_final[0][x0], sigma_final[1][x1]))**2  # Inner product for density matrices
                    tr_rho = np.trace(np.dot(sigma_final[0][x0], sigma_final[1][x1]))
                    #print(f"  Trace of sigma_final[0][{x0}] and sigma_final[1][{x1}]: {np.round(tr_rho.real, 4)}")

                    # tr_M = np.abs(np.vdot(M_final[0][x0].value, M_final[1][x1].value))**2  # Inner product for density matrices
                    tr_M = np.trace(np.dot(M_final[0, x1].value, M_final[1, x1].value))
                    #print(f"  Trace of M_final[0, {x1}] and M_final[1, {x1}]: {np.round(tr_M.real, 4)}")

                    rho = qt.Qobj(sigma_final[x0][x1], dims=[[d1,d2], [d1,d2]])  # Convert to Qobj for qutip functions
                    rho_p = qt.ptrace(rho, 0)  # Partial trace over subsystem 1
                    if (1-(rho_p * rho_p).tr() > tolerance):  # Trace of rho^2 for purity check
                        # x0=x1=D-1  # Break out of both loops if convergence is not to a local state
                        print("The program was not able to converge to a local state.")
                        break

                    m = qt.Qobj(M_final[1, x1], dims=[[d1,d2], [d1,d2]])  # Measurement operator
                    m_p = qt.ptrace(m, 0)  # Partial trace over subsystem 1
                    if (1-(m_p * m_p).tr() > tolerance):  # Trace of m^2 for measurement check
                        print("The program was not able to converge to a local measurement operator.")
                        break
                        # x0=x1=D-1  # Break out of both loops if convergence is not to a local state
                        
            # Padronização dos resultados para facilitar a leitura e comparação no caso de d=2      
            for i in range(N):
                for j in range(D):

                    # Verifica os traços individuais dos operadores de medida para verificar se estão normalizados
                    print(f"Trace of matrices for x0={i} and x1={j}: {np.round(np.trace(M_final[i, j].value),3)}")

            for i in range(N):

                # Verifica se a soma dos operadores de medida é aproximadamente a identidade (com tolerância)
                soma = sum(M_final[i,j] for j in range(D))
                if ((soma - np.eye(D))<< 1e-10*np.eye(D)):
                    print(f"Completeness relation holds for M_final[{i}]: Sum is approximately Identity.")
            
            for i in range(d1):
                for j in range(d2):
                    # Calculo dos traços de M_final[0, i] e M_final[1, j] para verificar se é = 1/D
                    tr_m = np.trace(np.dot(M_final[0, i].value, M_final[1, j].value))
                    
                    print(f"Trace of M_final[0, {i}] and M_final[1, {j}]: {np.round(tr_m.real, 4)}")

            break

        M_fixed = M2_optimal_values
        sigma0_2 = sigma2_opt
        sigma = sigma_final
        St_inicial = S_final
if __name__ == "__main__":
    main()