
import myPackages.creation as cs
import myPackages.optimization as opt

def main():
    d=2        # Dimension of the beta subsystem (subsystem 1)
    D = d**2     # Dimension of the set of letters x0x1 -- d = d1*d2

    N=2     # Word size -- quantity of letters/bases
    fatorNormalizacao = 1/(N*D**2) # Normalization factor for the success probability
    Pc = 0.5*(1 + 1/D) # Classical probability of success limit

    hadamard_D = cs.create_hadamard(D)
    hadamard_d = cs.create_hadamard(d)

    # ---- Creation of Measurement Bases ----
    sigma0, Comp_basis, Fourrier_basis = cs.createNonLocalStates(d, D, hadamard_D, hadamard_d)

    # Creation of system measurement operators
    M1, M2, M = cs.createMeasurementOperators(d, D, Fourrier_basis, N)

    # PHASE 1: Optimize the states, assuming fixed measurements (Computational and Fourier basis)
    SIGMA, S = opt.optimize_NonLocalStates(sigma0, M, D, fatorNormalizacao, Pc)
    print("Success_Sigma_Opt_Phase1 = ",S) 

    print(type(sigma0))
    print(type(SIGMA))
    print(type(M))
    print(type(M1))
    
    #PHASE 2: Optimize Measurement 1 (M1), fixing State (SIGMA) and Measurement 2 (M2)
    M_, M1_optimal_values, S1 = opt.optimize_LocalMeasurements(M2, SIGMA, fatorNormalizacao, d, D, N)
    print(type(M1_optimal_values))
    print("Success_Measure_Opt_Phase2 = ", S1)

    #PHASE 3: Optimize Measurement 2 (M2), fixing State (SIGMA) and Measurement 1 (M1_optimal)
    M_final, M2_optimal_values, S2 = opt.optimize_LocalMeasurements(M1_optimal_values, SIGMA, fatorNormalizacao, d, D, N)
    print(type(M_final))
    print("Success_Measure_Opt_Phase3 = ", S2)

if __name__ == "__main__":
    main()