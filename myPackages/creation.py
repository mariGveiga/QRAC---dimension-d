import numpy as np # Standard math lib
import qutip as qt # Quantum Mechanics Lib
import picos as pc # Optimization lib

# ---- Creation of the Hadamard Matrix ----
def create_hadamard(dim):
    hadamard_d = np.zeros((dim, dim), dtype=complex)        # Creation of matrix H_d according to dimension
    w = (2 * np.pi * 1j)/dim
    for i in range(dim):
        for j in range(dim):
            hadamard_d[i][j] = np.exp(j*w*i)
    return hadamard_d/np.sqrt(dim)

'''
STATES CREATION
'''
# ------ NON-LOCAL STATES ------
def createNonLocalStates(d, D, hadamard_D, hadamard_d):
    # Lists for first phase
    Comp_basis = [] # Computational basis
    Fourrier_basis = [] # Fourier basis
    # sigma0 = [[None for _ in range(D)] for _ in range(D)] # pure state to be optimized
    sigma0 = np.zeros((D, D), dtype=object)   # pure state to be optimized

    for x0 in range(D):
        a = qt.basis(D,x0)*qt.basis(D,x0).dag() # 1st measurement operator -- |x_0><x_0|
        
        for x1 in range(D):
        
            ketx=qt.Qobj(hadamard_D[:, x1]) # Constructing one of the Mutually Unbiased Bases (MUBs) vectors
            b = ketx*ketx.dag()  # 2nd measurement operator -- |x_1><x_1|

            # Creating the initial pure state -- equal superposition of the two basis states
            # Dividing Hilbert space in two qubits -- product states
            psi=(ketx+qt.basis(D,x0)) #(x1+x0)/sqrt(2)
            
            # Normalization factor: <psi|psi> = tr(|psi><psi|)
            sigma0[x0][x1]=psi*psi.dag()/((psi*psi.dag()).tr())     
            sigma0[x0][x1].dims = [[2, 2], [2, 2]]      # Divides the system into two parts (2 qubits)
            
            # Constructing one of the MUBs for d=2 to form the Fourier basis using Hadamard of dimension 2
            ketx_f = qt.Qobj(hadamard_d[:, x1%2], dims=[[d], [1]]) 
            f = ketx_f*ketx_f.dag()  # 2nd measurement operator -- |x_1><x_1|
            Fourrier_basis.append(f)
        
        Comp_basis.append(a)

    # Slice to ensure correct size for d2 dimension
    Fourrier_basis=Fourrier_basis[:d]

    return sigma0, Comp_basis, Fourrier_basis

# ------ LOCAL STATES ------
def createLocalStates(d, D, hadamard_d):
    # Lists for first phase
    Comp_basis = [] # Computational basis
    Fourrier_basis = [] # Fourier basis
    sigma0 = np.zeros((D, D), dtype=object)
    sigma0_1 = np.zeros((d, d), dtype=object)
    sigma0_2 = np.zeros((d, d), dtype=object)

    for x0 in range(D):

        idx_0 = x0 % d  # Map x0 to subsystem index
        a = qt.basis(d,idx_0)*qt.basis(d,idx_0).dag() # 1st measurement operator -- |x_0><x_0|
        Comp_basis.append(a)
        
        for x1 in range(D):
            
            idx_1 = x1 % d  # Map x1 to subsystem index
            ketx=qt.Qobj(hadamard_d[:, idx_1]) # Constructing one of the Mutually Unbiased Bases (MUBs) vectors
            b = ketx*ketx.dag()  # 2nd measurement operator -- |x_1><x_1|

            # Creating the initial pure state -- equal superposition of the two basis states
            # Dividing Hilbert space in two qubits -- product states
            psi=(ketx+qt.basis(d,idx_0)) #(x1+x0)/sqrt(2)
            
            psi = psi.unit()  # Normalization using .unit() method -- qutip built-in normalization
            # Normalization factor: <psi|psi> = tr(|psi><psi|)
            
            sigma0_1[idx_0][idx_1]=psi*psi.dag()     
            sigma0_2[idx_0][idx_1]=psi*psi.dag() 
            sigma0[x0][x1]=qt.tensor(sigma0_1[idx_0][idx_1], sigma0_2[idx_0][idx_1])   
            
            # Constructing one of the MUBs for d=2 to form the Fourier basis using Hadamard of dimension 2
            if x0 == 0:     # Forbids duplicates in the Fourier basis
                ketx_f = qt.Qobj(hadamard_d[:, idx_1]) 
                f = ketx_f * ketx_f.dag()
                Fourrier_basis.append(f)      

    return sigma0, sigma0_1, sigma0_2, Comp_basis, Fourrier_basis

'''
MEASUREMENT CREATION
'''
# Creation of system measurement operators
def createMeasurementOperators(d, D, Fourrier_basis, N):
    M1 = np.zeros((N, d), dtype=object)  
    M2 = np.zeros((N, d), dtype=object)  
    M = np.zeros((N, D), dtype=object)

    x0=0
    for beta0 in range(d): 
        for beta0_ in range(d):
            x1=0

            for beta1 in range(d):
                for beta1_ in range(d):
                    # --- M1 refers to beta (Subsystem 1) ---       
                    M1[0,beta0] = qt.basis(d, beta0) * qt.basis(d, beta0).dag()   
                    M1[1,beta1] = Fourrier_basis[beta1] 
                    
                    # --- M2 refers to beta_ (Subsystem 2) ---          
                    M2[0,beta0_] = qt.basis(d, beta0_) * qt.basis(d, beta0_).dag()
                    M2[1,beta1_] = Fourrier_basis[beta1_]

                    # --- Tensor product to form the composite system measurement operators ---
                    M[0,x0] = qt.tensor(M1[0,beta0], M2[0,beta0_])
                    M[1,x1] = qt.tensor(M1[1,beta1], M2[1,beta1_])
                    
                    x1+=1
            x0+=1

    return M1, M2, M

# Creation of measurement operators for optimization (PICOS variables)
def create_operator_optimization(M1, M2, d, D, N, subsystem_target):
    # Resulting matrix of PICOS expressions (for the joint system)
    M = np.zeros((N, D), dtype=object)      
    for x in range(N):
        x_i = 0
        for beta in range(d):
            for beta_ in range(d):
                
                term_1 = M1[x,beta]
                term_2 = M2[x,beta_]

                # Check if terms are Qobj and convert them to numpy arrays if necessary
                # This is crucial because PICOS cannot multiply variables by Qobj directly
                if isinstance(term_2, qt.Qobj): term_2 = term_2.full()
                elif isinstance(term_1, qt.Qobj): term_1 = term_1.full()
                
                if subsystem_target == 1:
                    M[x,x_i] = pc.kron(term_1, term_2)  # Optimize subsystem 1
                elif subsystem_target == 2:
                    M[x,x_i] = pc.kron(term_2, term_1)  # Optimize subsystem 2
                x_i+=1
    return M


# Inspection function for debugging purposes
def inspect_matrix_elements(M, N, D, name):
    print(f"\n--- Inspecting Matrix {name} ({N}x{D}) ---")
    for r in range(N):
        for c in range(D):
            item = M[r, c]
            print(f"\nItem [{r}, {c}]:")
            
            # Case 1: Is a PICOS variable (has 'value' attribute)
            if hasattr(item, 'value'):
                if item.value is None:
                    print(f"  Type: PICOS variable (Não resolvida)")
                    print(f"  Name/String: {np.round(item,2)}")
                    # print(f"  Shape: {item.shape}")
                else:
                    print(f"  Type: PICOS variable (Resolvida)")
                    print(f"  Value:\n{np.round(item.value,2)}")
            
            # Case 2: Is a Qobj from QuTiP (has 'full' method)
            elif hasattr(item, 'full'):
                print(f"  Type: Qobj")
                print(f"  Value:\n{np.round(item.full(),2)}")
                
            # Case 3: Is a Numpy array or number
            else:
                print(f"  Type: {type(item)}")
                print(f"  Value:\n{np.round(item,2)}")
