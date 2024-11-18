import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class VQLS:
    """
    Implementation of VQLS with Ising QLSP scaling analysis 
    """
    def __init__(self, n_qubits=10, n_layers=4, kappa=10, epsilon=0.01, J=0.1):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.kappa = kappa
        self.epsilon = epsilon
        self.J = J
        self.training_history = {"cost": [], "time_to_solution": []}
        
        # Initialize the QLSP
        self.A, self.b_state = self._initialize_ising_qlsp()

    def _compute_normalization_params(self):
        """
        Compute ζ and η for eigenvalue bounds following Section 2.2.1
        """
        n = self.n_qubits
        J = self.J
        
        # Maximum and minimum eigenvalues of unnormalized Hamiltonian
        max_unnorm = n + J*(n-1)  
        min_unnorm = -(n + J*(n-1))
        
        # Solve normalization equations
        zeta = (max_unnorm - min_unnorm) * self.kappa/(self.kappa - 1)
        eta = -min_unnorm - zeta/self.kappa
        
        return zeta, eta

    def _initialize_ising_qlsp(self):
        """
        Initialize the Ising-inspired QLSP from Section 2.2.1
        Returns A matrix and |b⟩ state
        """
        zeta, eta = self._compute_normalization_params()
        
        # Construct matrix A terms
        terms = []
        
        # Add X_i terms
        for i in range(self.n_qubits):
            pauli_str = ['I'] * self.n_qubits
            pauli_str[i] = 'X'
            terms.append((''.join(pauli_str), 1.0))
        
        # Add ZZ interaction terms
        for i in range(self.n_qubits - 1):
            pauli_str = ['I'] * self.n_qubits
            pauli_str[i] = 'Z'
            pauli_str[i+1] = 'Z'
            terms.append((''.join(pauli_str), self.J))
        
        # Add identity term with η
        terms.append(('I' * self.n_qubits, eta))
        
        # Create sparse operator and normalize
        A = SparsePauliOp.from_list(terms)
        A = A / zeta
        
        # Create |b⟩ state (H⊗n|0⟩)
        qc = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            qc.h(i)
        b_state = Statevector.from_instruction(qc)
        
        return A, b_state

    # def create_hardware_efficient_ansatz(self, parameters):
    #     """
    #     Hardware-efficient ansatz with Ry rotations and CZ gates
    #     """
    #     qc = QuantumCircuit(self.n_qubits)
    #     param_idx = 0
        
    #     for _ in range(self.n_layers):
    #             num_segs = self.n_qubits // 2
    #             for seg_i in range(num_segs):
    #                 qc.cz(2 * seg_i, (2 * seg_i) + 1)

    #             for qi in range(self.n_qubits):
    #                 qc.ry(parameters[param_idx], qi)
    #                 param_idx += 1

    #             for seg_i in range(num_segs):
    #                 # qc.cz((2 * seg_i) + 1, 2 * (seg_i + 1))
    #                 qc.cz(((2 * seg_i) + 1) % self.n_qubits, ((2 * (seg_i + 1))) % self.n_qubits)

    #             for qi in range(self.n_qubits):
    #                 qc.ry(parameters[param_idx], qi)
    #                 param_idx += 1
        
    #     return qc
    
    # def create_QAOA_ansatz(self, parameters):
    #     """
    #     QAOA ansatz
    #     """
    #     qc = QuantumCircuit(self.n_qubits)
    #     param_idx = 0
        
    #     for _ in range(self.n_layers):
    #         # Apply problem Hamiltonian (Z rotations)
    #         for i in range(self.n_qubits):
    #             qc.rz(parameters[param_idx], i)
    #             param_idx += 1

    #         # Apply mixer Hamiltonian (X rotations)
    #         for i in range(self.n_qubits):
    #             qc.rx(parameters[param_idx], i)
    #             param_idx += 1

    #         # Optional entangling gates (controlled-Z)
    #         for i in range(self.n_qubits - 1):
    #             qc.cz(i, i + 1)
        
    #     return qc
    
    def create_ansatz(self, parameters, ans_type=None):
        qc = QuantumCircuit(self.n_qubits)
        param_idx = 0

        if ans_type is None:
            """QAOA ansatz with alternating Hamiltonians."""

            for _ in range(self.n_layers):
                # Apply problem Hamiltonian (Z rotations)
                for i in range(self.n_qubits):
                    qc.rz(parameters[param_idx], i)
                    param_idx += 1

                # Apply mixer Hamiltonian (X rotations)
                for i in range(self.n_qubits):
                    qc.rx(parameters[param_idx], i)
                    param_idx += 1

                # Optional entangling gates (controlled-Z)
                for i in range(self.n_qubits - 1):
                    qc.cz(i, i + 1)

        elif ans_type == "QAOA_ex":
            for _ in range(self.n_layers):
                # Apply problem Hamiltonian (Z rotations)
                for i in range(self.n_qubits):
                    qc.rz(parameters[param_idx], i)
                    param_idx += 1

                # Apply mixer Hamiltonian (X rotations)
                for i in range(self.n_qubits):
                    qc.rx(parameters[param_idx], i)
                    param_idx += 1

                # Optional entangling gates (controlled-Z)
                for i in range(self.n_qubits):
                    qc.cz(i, (i + 1) % self.n_qubits)

        elif ans_type == "QAOA_ex_cx":
            for _ in range(self.n_layers):
                # Apply problem Hamiltonian (Z rotations)
                for i in range(self.n_qubits):
                    qc.rz(parameters[param_idx], i)
                    param_idx += 1

                # Apply mixer Hamiltonian (X rotations)
                for i in range(self.n_qubits):
                    qc.rx(parameters[param_idx], i)
                    param_idx += 1

                # Optional entangling gates (controlled-NOT)
                for i in range(self.n_qubits):
                    qc.cx(i, (i + 1) % self.n_qubits)

        elif ans_type == "QAOA_ex_|cz|_|cx|":
            for _ in range(self.n_layers):
                # Apply problem Hamiltonian (Z rotations)
                for i in range(self.n_qubits):
                    qc.rz(parameters[param_idx], i)
                    param_idx += 1

                # Apply mixer Hamiltonian (X rotations)
                for i in range(self.n_qubits):
                    qc.rx(parameters[param_idx], i)
                    param_idx += 1

                # Optional entangling gates (controlled-Z)
                for i in range(self.n_qubits):
                    qc.cz(i, (i + 1) % self.n_qubits)

                # Optional entangling gates (controlled-NOT)
                for i in range(self.n_qubits):
                    qc.cx(i, (i + 1) % self.n_qubits)

        elif ans_type == "QAOA_ex_|cz_cx|":
            for _ in range(self.n_layers):
                # Apply problem Hamiltonian (Z rotations)
                for i in range(self.n_qubits):
                    qc.rz(parameters[param_idx], i)
                    param_idx += 1

                # Apply mixer Hamiltonian (X rotations)
                for i in range(self.n_qubits):
                    qc.rx(parameters[param_idx], i)
                    param_idx += 1

                # Optional entangling gates (controlled-Z)
                for i in range(self.n_qubits):
                    qc.cz(i, (i + 1) % self.n_qubits)
                    qc.cx(i, (i + 1) % self.n_qubits)

        elif ans_type == "QAOA_ex_|cx|_|cz|":
            for _ in range(self.n_layers):
                # Apply problem Hamiltonian (Z rotations)
                for i in range(self.n_qubits):
                    qc.rz(parameters[param_idx], i)
                    param_idx += 1

                # Apply mixer Hamiltonian (X rotations)
                for i in range(self.n_qubits):
                    qc.rx(parameters[param_idx], i)
                    param_idx += 1

                # Optional entangling gates (controlled-NOT)
                for i in range(self.n_qubits):
                    qc.cx(i, (i + 1) % self.n_qubits)

                # Optional entangling gates (controlled-Z)
                for i in range(self.n_qubits):
                    qc.cz(i, (i + 1) % self.n_qubits)

        elif ans_type == "QAOA_ex_|cx_cz|":
            for _ in range(self.n_layers):
                # Apply problem Hamiltonian (Z rotations)
                for i in range(self.n_qubits):
                    qc.rz(parameters[param_idx], i)
                    param_idx += 1

                # Apply mixer Hamiltonian (X rotations)
                for i in range(self.n_qubits):
                    qc.rx(parameters[param_idx], i)
                    param_idx += 1

                # Optional entangling gates (controlled-Z)
                for i in range(self.n_qubits):
                    qc.cx(i, (i + 1) % self.n_qubits)
                    qc.cz(i, (i + 1) % self.n_qubits)

        elif ans_type == "hardware":
            for _ in range(self.n_layers):
                num_segs = self.n_qubits // 2
                for seg_i in range(num_segs):
                    qc.cz(2 * seg_i, (2 * seg_i) + 1)

                for qi in range(self.n_qubits):
                    qc.rz(parameters[param_idx], qi)
                    param_idx += 1

                for seg_i in range(num_segs):
                    # qc.cz((2 * seg_i) + 1, 2 * (seg_i + 1))
                    qc.cz(((2 * seg_i) + 1) % self.n_qubits, ((2 * (seg_i + 1))) % self.n_qubits)

                for qi in range(self.n_qubits):
                    qc.rz(parameters[param_idx], qi)
                    param_idx += 1

        else:
            print("Error in the input of variable type")

        return qc

    def compute_cost(self, parameters, cost_type="local", ans_type=None):
        """
        Compute cost function (local or global variants)
        """
        # circuit = self.create_hardware_efficient_ansatz(parameters)
        # circuit = self.create_QAOA_ansatz(parameters)
        circuit = self.create_ansatz(parameters, ans_type=ans_type)

        x_state = Statevector.from_instruction(circuit)
        
        # Compute A|x⟩
        Ax = Statevector(np.dot(self.A.to_matrix(), x_state.data))
        norm_Ax = np.linalg.norm(Ax.data)
        
        if cost_type == "local":
            cost = 0
            for j in range(self.n_qubits):
                # Create local projector
                proj = np.zeros((2**self.n_qubits, 2**self.n_qubits))
                for k in range(2**self.n_qubits):
                    if (k >> j) & 1 == 0:
                        proj[k][k] = 1
                
                term = np.abs(np.dot(Ax.conjugate(), np.dot(proj, Ax)))
                cost += term
            
            cost = 1 - cost/self.n_qubits
            
        else:  # global cost
            overlap = np.abs(np.dot(self.b_state.conjugate(), Ax.data))**2
            cost = 1 - overlap / norm_Ax**2
            
        return cost

    def train(self, n_epochs=100):
        """
        Train using SPSA optimization with scaling analysis
        """
        n_params = self.n_qubits * self.n_layers * 2
        parameters = np.random.uniform(0, 2*np.pi, n_params)
        
        start_time = 0  # Track time to solution
        cost_evals = 0  # Track number of cost evaluations
        
        for epoch in range(n_epochs):
            # SPSA optimization
            delta = 2 * np.random.randint(0, 2, n_params) - 1
            params_plus = parameters + 0.01 * delta
            params_minus = parameters - 0.01 * delta
            
            cost_plus = self.compute_cost(params_plus)
            cost_minus = self.compute_cost(params_minus)
            cost_evals += 2
            
            gradient = (cost_plus - cost_minus) * delta / (2 * 0.01)
            parameters -= 0.1 * gradient
            
            # Compute costs
            local_cost = self.compute_cost(parameters, "local")
            global_cost = self.compute_cost(parameters, "global")
            cost_evals += 2
            
            self.training_history["cost"].append(local_cost)
            self.training_history["time_to_solution"].append(cost_evals)
            
            print(f"Epoch {epoch + 1}/{n_epochs}: Local Cost = {local_cost:.6f}, Global Cost = {global_cost:.6f}")
            
            # Check precision
            if local_cost < (self.epsilon ** 2) / (self.kappa ** 2):
                print(f"Desired precision achieved in {cost_evals} cost evaluations")
                break
        
        self.optimal_parameters = parameters
        return parameters, cost_evals

    def analyze_scaling(self, kappa_range, epsilon_range, n_range):
        """
        Analyze scaling with respect to κ, ε, and n
        """
        results = {
            'kappa': [],
            'epsilon': [],
            'n_qubits': [],
            'time_to_solution': []
        }
        
        # Scaling with κ
        for kappa in kappa_range:
            self.kappa = kappa
            _, cost_evals = self.train()
            results['kappa'].append(kappa)
            results['time_to_solution'].append(cost_evals)
            
        # Scaling with ε
        self.kappa = 10  # Reset κ
        for epsilon in epsilon_range:
            self.epsilon = epsilon
            _, cost_evals = self.train()
            results['epsilon'].append(epsilon)
            results['time_to_solution'].append(cost_evals)
            
        # Scaling with n
        self.epsilon = 0.01  # Reset ε
        for n in n_range:
            self.n_qubits = n
            self.A, self.b_state = self._initialize_ising_qlsp()  # Reinitialize QLSP
            _, cost_evals = self.train()
            results['n_qubits'].append(n)
            results['time_to_solution'].append(cost_evals)
        
        return results

    def plot_scaling_results(self, results):
        """
        Plot scaling analysis results
        """
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # κ scaling
        ax1.plot(results['kappa'], results['time_to_solution'][:len(results['kappa'])])
        ax1.set_xlabel('Condition number κ')
        ax1.set_ylabel('Time to solution')
        ax1.set_title('Scaling with κ')
        ax1.grid(True)
        
        # ε scaling
        ax2.plot(results['epsilon'], results['time_to_solution'][len(results['kappa']):len(results['kappa'])+len(results['epsilon'])])
        ax2.set_xlabel('Precision ε')
        ax2.set_ylabel('Time to solution')
        ax2.set_title('Scaling with ε')
        ax2.grid(True)
        
        # n scaling
        ax3.plot(results['n_qubits'], results['time_to_solution'][-len(results['n_qubits']):])
        ax3.set_xlabel('Number of qubits n')
        ax3.set_ylabel('Time to solution')
        ax3.set_title('Scaling with n')
        ax3.grid(True)
        
        plt.tight_layout()
        plt.show()

def main():
    """
    Main function to run VQLS with scaling analysis
    """
    # Initialize solver
    solver = VQLS(n_qubits=10, n_layers=4)
    
    # Run scaling analysis
    kappa_range = [2, 5, 10, 20, 50]
    epsilon_range = [0.1, 0.01, 0.001]
    n_range = [4, 6, 8, 10, 12]
    ans_type = "QAOA_ex"
    
    print("Running scaling analysis...")
    results = solver.analyze_scaling(kappa_range, epsilon_range, n_range)
    
    # Plot results
    solver.plot_scaling_results(results)
    
    # Get final CX count
    # circuit = solver.create_hardware_efficient_ansatz(solver.optimal_parameters)
    # circuit = solver.create_QAOA_ansatz(solver.optimal_parameters)
    circuit = solver.create_ansatz(solver.optimal_parameters, ans_type=ans_type)

    cx_count = circuit.count_ops().get('cx', 0)
    print(f"\nFinal circuit CX gate count: {cx_count}")

def run_vqls(ans_type=None):
    """
    Run VQLS algorithm for the specific problem from Section 2.3:
    A|x⟩ = 0 where A = ∑X_i + 0.1∑Z_iZ_{i+1} + I
    """
    # Initialize parameters
    n_qubits = 10  # 10 qubits as per Section 2.3
    n_qubits = 4
    n_layers = 4
    kappa = 10
    epsilon = 0.01
    J = 0.1

    print(f"\nInitializing VQLS for Section 2.3 implementation...")
    print(f"Number of qubits: {n_qubits}")
    print(f"Number of layers: {n_layers}")
    print(f"Condition number κ: {kappa}")
    print(f"Precision ε: {epsilon}")
    print(f"Coupling strength J: {J}")
    
    # Initialize solver
    solver = VQLS(
        n_qubits=n_qubits,
        n_layers=n_layers,
        kappa=kappa,
        epsilon=epsilon,
        J=J
    )
    
    # Train the circuit
    print("\nStarting training...")
    final_params, cost_evals = solver.train(n_epochs=50)
    
    # Analyze results
    final_cost = solver.compute_cost(final_params)
    print("\nTraining Results:")
    print(f"Final cost: {final_cost:.6f}")
    print(f"Number of cost evaluations: {cost_evals}")
    
    # Create final circuit and analyze gates
    # final_circuit = solver.create_hardware_efficient_ansatz(final_params)
    # final_circuit = solver.create_QAOA_ansatz(final_params)
    final_circuit = solver.create_ansatz(final_params, ans_type=ans_type)
    
    # Display the circuit
    print("\nFinal Circuit:")
    print(final_circuit)
    
    # Detailed gate analysis
    gate_counts = final_circuit.count_ops()
    total_gates = sum(gate_counts.values())
    two_qubit_gates = gate_counts.get('cz', 0) + gate_counts.get('cx', 0)
    
    print("\nCircuit Analysis:")
    print(f"Total gates: {total_gates}")
    print(f"Two-qubit gates: {two_qubit_gates}")
    print("\nGate breakdown:")
    for gate, count in gate_counts.items():
        print(f"- {gate}: {count}")
    
    # Calculate circuit depth
    depth = final_circuit.depth()
    print(f"\nCircuit depth: {depth}")
    
    # Plot training progress
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(solver.training_history["cost"], label="Cost")
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title("VQLS Training Progress")
    plt.legend()
    plt.grid(True)
    
    # Plot circuit using Qiskit's circuit drawer
    plt.subplot(2, 1, 2)
    try:
        from qiskit.visualization import circuit_drawer
        circuit_drawer(final_circuit, output='mpl')
    except Exception as e:
        print(f"Could not draw circuit: {e}")
    
    plt.tight_layout()
    plt.show()
    
    return solver

if __name__ == "__main__":
    solver = run_vqls(ans_type="hardware")