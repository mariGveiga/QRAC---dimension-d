import myPackages.creation as cs
import myPackages.optimization as opt
import numpy as np
import qutip as qt

'''
O que eu quero fazer?
Quero criar os estados produto a partir do produto tensorial entre sigma_alpha1 e sigma_alpha2
As dimensões não vão mais partir de d, mas sim de d1 e d2
'''

def main():
    d=2        # Dimension of the beta subsystem (subsystem 1)
    D=d**2     # Dimension of the set of letters x0x1 

    N=2                                 # Word size -- quantity of letters/bases
    fatorNormalizacao = 1/(N*D**2)      # Normalization factor for the success probability
    Pc = 0.5*(1 + 1/D)                  # Classical probability of success limit

    # hadamard_d = create_hadamard(D)
    hadamard_d2 = cs.create_hadamard(d)

    # ---- Creation of Measurement Bases ----
    sigma0, sigma0_1, sigma0_2, Comp_basis, Fourrier_basis = cs.createLocalStates(d, D, hadamard_d2)

    # ---- Creation of system measurement operators ----
    M1, M2, M = cs.createMeasurementOperators(d, D, Fourrier_basis, N)
    Pq = 1/2 *(1 + 1/np.sqrt(D)) # Quantum probability of success limit for d=2
    # see-saw algorithm
    tolerance = 1e-6        # até o 11
    t_max = 50

    St_inicial = 0
    M_fixed = M2
    sigma = sigma0
    
    print("Quantum Probability (ideal case):", Pq)
    print("Classical Probability (ideal case):", Pc)

    for t in range(t_max):
        M_inicial, M1_optimal_values, S = opt.optimize_LocalMeasurements(M_fixed, sigma, fatorNormalizacao, d, D, N, 1)
        # M_inicial = M1_optimal_values (optimized) and M2 (fixed) -- sigma fixo
        M_final, M2_optimal_values, S1 = opt.optimize_LocalMeasurements(M1_optimal_values, sigma, fatorNormalizacao, d, D, N, 2)
        # M_final = M1_optimal_values (optimized) and M2_optimal_values (optimized) -- sigma fixo
        sigma_inicial, sigma1_opt, S2 = opt.optimize_LocalStates(sigma0_2, M_final, d, fatorNormalizacao, 1)
        # sigma_inicial = sigma1_opt (optimized) and sigma0_2 (fixed)
        sigma_final, sigma2_opt, S_final = opt.optimize_LocalStates(sigma1_opt, M_final, d, fatorNormalizacao, 2)
        # sigma_final = sigma1_opt (optimized) and sigma2_opt (optimized)

        print(f"Iteration {t+1}: Total Success = {S_final}")
        # S_final final da iteração t, otimizado em relação a M1, M2, sigma1 e sigma2

        if t > 0 and abs(S_final - St_inicial) <= tolerance:
            print("Convergence reached.")
            
            # Debug: Inspect the structures of the optimized variables
            cs.inspect_matrix_elements(sigma_final, D, D, "sigma_final")  
            cs.inspect_matrix_elements(sigma2_opt, d, d, "sigma2_opt")  
            cs.inspect_matrix_elements(sigma1_opt, d, d, "sigma1_opt")  
            cs.inspect_matrix_elements(M_final, N, D, "M_final")
            cs.inspect_matrix_elements(M2_optimal_values, N, d, "M2_optimal_values") 
            cs.inspect_matrix_elements(M1_optimal_values, N, d, "M1_optimal_values")

            for x0 in range(D):
                for x1 in range(D):
                    rho = qt.Qobj(sigma_final[x0][x1], dims=[[d,d], [d,d]])  # Convert to Qobj for qutip functions
                    rho_p = qt.ptrace(rho, 0)  # Partial trace over subsystem 1
                    if (1-(rho_p * rho_p).tr() > tolerance):  # Trace of rho^2 for purity check
                        # x0=x1=D-1  # Break out of both loops if convergence is not to a local state
                        print("The program was not able to converge to a local state.")
                        break

                    m = qt.Qobj(M_final[1, x1], dims=[[d,d], [d,d]])  # Measurement operator
                    m_p = qt.ptrace(m, 0)  # Partial trace over subsystem 1
                    if (1-(m_p * m_p).tr() > tolerance):  # Trace of m^2 for measurement check
                        print("The program was not able to converge to a local measurement operator.")
                        break
                        # x0=x1=D-1  # Break out of both loops if convergence is not to a local state

            break
        
        M_fixed = M2_optimal_values
        sigma0_2 = sigma2_opt
        sigma = sigma_final
        St_inicial = S_final


if __name__ == "__main__":
    main()

"""
tr parcial 
tr de rho_parcial ^2
"""