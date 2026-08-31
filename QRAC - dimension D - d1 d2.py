import myPackages.creation as cs
import myPackages.optimization as opt
import numpy as np
import qutip as qt
import picos as pc

'''
This version of the code is designed to handle QRACs with local states and local measurements, considering two subsystems with dimensions d1 and d2.

The optimization process involves alternating between optimizing the measurement operators and the quantum states for each subsystem, using a see-saw algorithm 
until convergence is reached or a maximum number of iterations is completed.
'''

def main():
    # ----- Hyperparameters -----
    d1=3                     # Dimension of the beta subsystem 1
    d2=5                     # Dimension of the beta_ subsystem 2
    D = d1*d2                # Dimension of the set of letters x0x1
    N=2                      # Word size -- quantity of letters/bases
    tolerance = 1e-8         # Tolerance for convergence
    t_max = 200              # Number of iterations for the optimization loop

    # ----- Statistics for QRAC's -----
    fatorNormalizacao = 1/(N*D**2)            # Normalization factor for the success probability
    Pc = 0.5*(1 + 1/D)                        # Classical probability of success limit
    Pq = 1/2 *(1 + 1/np.sqrt(D))              # Quantum probability of success limit 1
    Pq_ideal1 = 1/4 *(1 + 1/np.sqrt(d1))**2   # Quantum probability of success limit 2 (ideal case)
    Pq_ideal2 = 1/4 *(1 + 1/np.sqrt(d2))**2   # Quantum probability of success limit 2 (ideal case)

    print(f"QRAC's with Local States and Local Measurements (D={D}, d1={d1}, d2={d2}) with Tolerance {tolerance}:")
    print(f"Quantum Probability (ideal case, D={D}): {np.round(Pq,3)}")
    print(f"Quantum Probability (ideal case, d1={d1}): {np.round(Pq_ideal1,3)}")
    print(f"Quantum Probability (ideal case, d2={d2}): {np.round(Pq_ideal2,3)}")
    print(f"Classical Probability (ideal case): {np.round(Pc,3)}")

    # ------ Initial matrices' creation for each subsystem -----
    hadamard_d1 = cs.create_hadamard(d1)
    hadamard_d2 = cs.create_hadamard(d2)

    sigma0, sigma0_1, sigma0_2, Comp_basis, Fourrier_basis = cs.createLocalStates(d1, d2, D, hadamard_d1, hadamard_d2)
    M1, M2, M = cs.createMeasurementOperators(d1, d2, D, Fourrier_basis, N)
    
    St_inicial = 0
    M_fixed = M2
    sigma = sigma0
    sucess_history = []

    # ------ Optimization loop (See-Saw Algorithm) -----
    for t in range(t_max):
        M_inicial, M1_optimal_values, S = opt.optimize_LocalMeasurements(M_fixed, sigma, fatorNormalizacao, d1, d2, D, N, 1)
        #M_inicial = M1_optimal_values (optimized) and M2 (fixed) -- sigma fixo
        
        M_final, M2_optimal_values, S1 = opt.optimize_LocalMeasurements(M1_optimal_values, sigma, fatorNormalizacao, d1, d2, D, N, 2)
        # M_final = M1_optimal_values (optimized) and M2_optimal_values (optimized) -- sigma fixo

        sigma_inicial, sigma1_opt, S2 = opt.optimize_LocalStates(sigma0_2, M_final, d1, d2, D, fatorNormalizacao, 1)
        # sigma_inicial = sigma1_opt (optimized) and sigma0_2 (fixed)

        sigma_final, sigma2_opt, S_final = opt.optimize_LocalStates(sigma1_opt, M_final, d1, d2, D, fatorNormalizacao, 2)
        # sigma_final = sigma1_opt (optimized) and sigma2_opt (optimized)

        sucess_history.append(S_final)
        print(f"Iteration {t+1}: Total Success = {S_final}")

        if t > 0 and abs(S_final - St_inicial) <= tolerance:
            print("Convergence reached.")

            # ----- Verification of convergence to local states and local measurement operators -----
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
                         
            # Padronization of the results to facilitate reading and comparison
            for i in range(N):
                for j in range(D):
                    # Verifies the individual traces of the measurement operators to check if they are normalized
                    print(f"Trace of matrices for x0={i} and x1={j}: {np.round(np.trace(M_final[i, j].value),3)}")

            for i in range(N):
                # Verifies if the sum of the measurement operators is approximately the identity (with tolerance)
                soma = sum(M_final[i,j] for j in range(D))
                if ((soma - np.eye(D))<< 1e-10*np.eye(D)):
                    print(f"Completeness relation holds for M_final[{i}]: Sum is approximately Identity.")
            
            for i in range(d1):
                for j in range(d2):
                    # Final calculation of the traces of the optimized local states to check if they are normalized (approx. 1/D)
                    tr_m = np.trace(np.dot(M_final[0, i].value, M_final[1, j].value))                    
                    print(f"Trace of M_final[0, {i}] and M_final[1, {j}]: {np.round(tr_m.real, 4)}")

            break

        M_fixed = M2_optimal_values
        sigma0_2 = sigma2_opt
        sigma = sigma_final
        St_inicial = S_final
if __name__ == "__main__":
    main()