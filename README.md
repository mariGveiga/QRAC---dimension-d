# Quantum Random Access Codes (QRAC) Optimization ⚛️

📌 **Note:** This project is the continuation of my Undergraduate Research (*Iniciação Científica*). The first phase of the project, which covers the theoretical foundations and initial quantum circuit simulations, can be found in the repositories on my profile!

---

## 📖 About the Project

This project focuses on the numerical optimization of **Quantum Random Access Codes (QRACs)**. Rather than just simulating the protocol, the main goal here is to computationally determine the absolute mathematical limits of the protocol by simultaneously finding:
1. The **optimal quantum states** prepared and sent by Alice.
2. The **optimal Positive Operator-Valued Measures (POVMs)** used by Bob to decode the information.

### 🎯 Key Approaches
* **See-Saw Algorithm:** Since optimizing states and measurements simultaneously is a non-linear problem, we use an iterative See-Saw approach. We fix the measurements to optimize the states, then fix the new states to optimize the measurements, repeating this cycle until the success probability converges.

* **Semidefinite Programming (SDP):** The core optimization is modeled as an SDP using `PICOS` and solved via `CVXOPT`, ensuring all physical constraints (e.g., density matrices being positive semidefinite with trace 1) are strictly respected.
* **Product States Constraint:** The code is heavily structured to handle multidimensional systems ($D = d^2$), enforcing Kronecker products ($\rho = \rho_1 \otimes \rho_2$) to ensure the prepared states remain valid product states without introducing artificial entanglement during the local optimization steps.

---

## 📂 Repository Structure

*(Add a brief description of what each file in your project does here)*

* `main.py` - [Brief description of what this does, e.g., Runs the main See-Saw loop]
* `optimize_states.py` - [Brief description]
* `optimize_measurements.py` - [Brief description]
* `utils.py` - [Brief description]
* `...` - [...]

---

## 🛠️ Technologies Used

* **Python 3**
* **PICOS & CVXOPT:** For Semidefinite Programming (SDP) modeling and solving.
* **QuTiP (Quantum Toolbox in Python):** For handling quantum objects, basis creation, and tensor products.
* **NumPy:** For matrix operations and complex array manipulation.
* **Qiskit & Qiskit Aer:** (Inherited from Phase 1) For quantum circuit simulations.

---
