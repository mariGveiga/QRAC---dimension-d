import myPackages.creation as cs
import myPackages.optimization as opt

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

    # see-saw algorithm
    tolerance = 1e-6
    t_max = 50

    St_inicial = 0
    M_fixed = M2
    sigma = sigma0
    
    for t in range(t_max):
        M_inicial, M1_optimal_values, S = opt.optimize_LocalMeasurements(M_fixed, sigma, fatorNormalizacao, d, D, N)
        M_final, M2_optimal_values, S1 = opt.optimize_LocalMeasurements(M1_optimal_values, sigma, fatorNormalizacao, d, D, N)
        sigma_inicial, sigma1_opt, S2 = opt.optimize_LocalStates(sigma0_2, M_final, d, fatorNormalizacao, 1)
        sigma_final, sigma2_opt, S_final = opt.optimize_LocalStates(sigma1_opt, M_final, d, fatorNormalizacao, 2)

        print(f"Iteration {t+1}: Total Success = {S_final}")

        if t > 0 and abs(S_final - St_inicial) <= tolerance:
            print("Convergence reached.")
            break
        
        M_fixed = M2_optimal_values
        sigma = sigma_final
        St_inicial = S_final

if __name__ == "__main__":
    main()