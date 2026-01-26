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
    # Creation of system measurement operators
    M1, M2, M = cs.createMeasurementOperators(d, D, Fourrier_basis, N)

    sigma_global_, SIGMA1_OPT, S = opt.optimize_LocalStates(sigma0_2, M, d, fatorNormalizacao, 1)
    print("Success_SIGMA_opt = ",S) 

    SIGMA, SIGMA2_OPT, S1 = opt.optimize_LocalStates(SIGMA1_OPT, M, d, fatorNormalizacao, 2)
    print("Success_SIGMA_opt = ",S1) 

    M1_optimal_values, S1 = opt.optimize_LocalMeasurements(M2, SIGMA, fatorNormalizacao, d, D, N)
    print("Success_Measure_Opt_Phase2 = ", S1)

    M2_optimal_values, S2 = opt.optimize_LocalMeasurements(M1_optimal_values, SIGMA, fatorNormalizacao, d, D, N)
    print("Success_Measure_Opt_Phase3 = ", S2)

if __name__ == "__main__":
    main()