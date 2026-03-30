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
    d=3        # Dimension of the beta subsystem (subsystem 1)
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
    tolerance = 1e-4       # d=2 -- tolerance = 1e-6; d=3 -- tolerance = 1e-4
    t_max = 50
    
    St_inicial = 0
    M_fixed = M2
    sigma = sigma0
    
    print(f"QRAC's with Local States and Local Measurements (d={d}):")
    print(f"Quantum Probability (ideal case): {np.round(Pq,3)}")
    print(f"Classical Probability (ideal case): {np.round(Pc,3)}")

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
            # cs.inspect_matrix_elements(sigma_final, D, D, "sigma_final")  
            # cs.inspect_matrix_elements(sigma2_opt, d, d, "sigma2_opt")  
            # cs.inspect_matrix_elements(sigma1_opt, d, d, "sigma1_opt")  
            # cs.inspect_matrix_elements(M_final, N, D, "M_final")
            # cs.inspect_matrix_elements(M2_optimal_values, N, d, "M2_optimal_values") 
            # cs.inspect_matrix_elements(M1_optimal_values, N, d, "M1_optimal_values")


            '''
            rho (0,0) * sigma (1,0)
            rho (0,1) * sigma (1,1)
            '''
            # for x in range(D):
            #     tr_rho = np.trace(np.dot(sigma_final[0][x], sigma_final[1][x]))
            #     print(f"Trace of sigma_final: {np.round(tr_rho.real, 2)}")
            #     tr_m = np.trace(np.dot(M_final[0, x], M_final[1, x]))
            #     print(f"Trace of M_final: {np.round(tr_m.real, 2)}")
            
            for x0 in range(D):
                for x1 in range(D):
                    # tr_rho = np.abs(np.vdot(sigma_final[0][x0], sigma_final[1][x1]))**2  # Inner product for density matrices
                    tr_rho = np.trace(np.dot(sigma_final[0][x0], sigma_final[1][x1]))
                    print(f"  Trace of sigma_final[0][{x0}] and sigma_final[1][{x1}]: {np.round(tr_rho.real, 4)}")

                    # tr_M = np.abs(np.vdot(M_final[0][x0].value, M_final[1][x1].value))**2  # Inner product for density matrices
                    tr_M = np.trace(np.dot(M_final[0, x1].value, M_final[1, x1].value))
                    # print(f"  Trace of M_final[0, {x1}] and M_final[1, {x1}]: {np.round(tr_M.real, 4)}")

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