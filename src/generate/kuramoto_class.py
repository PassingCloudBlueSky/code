import numpy as np
import networkx as nx
import os
import datetime
import json
import random
import scipy.optimize as so
from typing import Optional, Tuple, Any
import sys
from base_model import BaseModel

class SecondOrderKuramotoModel(BaseModel):
    """
    Second-order Kuramoto model for power grid applications.

    Attributes
    ----------
    connectivity_matrix : np.ndarray
        Weighted adjacency (coupling) matrix of the network.
    power_vector : np.ndarray
        Static power injection/consumption at each node.
    damping_coefficient : float
        Damping coefficient for the second-order dynamics.
    fixed_point : Optional[np.ndarray]
        Steady-state phase angles (computed after initialization).
    jacobian_matrix : Optional[np.ndarray]
        Jacobian (curly_L) of the linearized system at the fixed point.
    """

    def __init__(
        self,
        connectivity_matrix: np.ndarray,
        power_vector: np.ndarray,
        damping_coefficient: float,
        ode_dimension: Optional[int] = None,
        model_ode_args: Optional[Tuple[Any, ...]] = None,
    ):
        """
        Initialize the Kuramoto model with system parameters.

        Parameters
        ----------
        connectivity_matrix : np.ndarray
            Weighted adjacency (coupling) matrix.
        power_vector : np.ndarray
            Static power injection/consumption at each node.
        damping_coefficient : float
            Damping coefficient for the system.
        ode_dimension : Optional[int]
            Dimension of the ordinary differential equation system.
        """
        super().__init__()
        self.connectivity_matrix = np.array(connectivity_matrix, dtype=float)
        self.power_vector = np.array(power_vector, dtype=float)
        self.damping_coefficient = float(damping_coefficient)
        self.ode_dimension = ode_dimension if ode_dimension is not None else len(power_vector)*2
        self.model_ode_args = model_ode_args

    @classmethod
    def from_random_sparse_graph(
        cls,
        num_nodes: int,
        edge_probability: float = 0.1,
        max_weight: float = 2.0,
        damping_coefficient: float = 0.01,
        seed: Optional[int] = None,
    ) -> "SecondOrderKuramotoModel":
        """
        Create a sparse, weighted, connected graph and initialize the model.

        Parameters
        ----------
        num_nodes : int
            Number of nodes in the network.
        edge_probability : float
            Probability of adding an edge between two nodes.
        max_weight : float
            Maximum weight for the edges.
        damping_coefficient : float
            Damping coefficient for the system.
        seed : Optional[int]
            Random seed for reproducibility.

        Returns
        -------
        SecondOrderKuramotoModel
            Initialized model instance.
        """

        def generate_sparse_weighted_graph(size, edge_probability, max_weight):
            """
            Generates a sparse, weighted, undirected, and connected graph.

            Parameters
            ----------
            size : int
                Number of nodes in the graph.
            edge_probability : float
                Probability of adding an edge between two nodes.
            max_weight : float
                Maximum weight for the edges.

            Returns
            -------
            G : networkx.Graph
                A connected, sparse, weighted graph.
            """
            # Step 1: Create a spanning tree to ensure connectivity
            G = nx.Graph()
            G.add_nodes_from(range(size))

            # Create a spanning tree by connecting each node to a random previous node
            for i in range(1, size):
                u = random.randint(0, i - 1)
                weight = random.random() * max_weight
                G.add_edge(i, u, weight=weight)

            # Step 2: Add additional edges based on edge_probability
            for u in range(size):
                for v in range(u + 1, size):
                    if random.random() < edge_probability and not G.has_edge(u, v):
                        weight = random.random() * max_weight
                        G.add_edge(u, v, weight=weight)

            return G

        # Generate the graph
        G = generate_sparse_weighted_graph(num_nodes, edge_probability, max_weight)
        K = nx.to_numpy_array(G, weight="weight")

        # Generate power vector (sum zero)
        p = np.random.rand(num_nodes) - 0.5
        p -= np.mean(p)

        return cls(K, p, damping_coefficient)

    @classmethod
    def illustrative_8node(cls):
        a=0.01 # 1/s^^-2  # for Xiaozhu a=1
        K= 16*np.array([
                     [0,0,1,0,1,0,0,0],
                     [0,0,0,0,1,0,0,1],
                     [1,0,0,0,0,1,1,1],
                     [0,0,0,0,0,0,1,0],
                     [1,1,0,0,0,0,0,0],
                     [0,0,1,0,0,0,1,1],
                     [0,0,1,1,0,1,0,0],
                     [0,1,1,0,0,1,0,0]]) # pre-factor 100 in case of Xiaozhu
        p = 1*np.array([-1,-1,-1,-1,3,-1,3,-1]) # pre-factor 10 in case of Xiaozhu

        return cls(K, p, a)

    
    @classmethod
    def from_soft_random_geometric_graph(
        cls,
        num_nodes: int,
        radius: float = 0.1,
        #max_weight: float = 2.0,
        damping_coefficient: float = 0.01,
        seed: Optional[int] = None,
    ) -> "SecondOrderKuramotoModel":
        """
        Create a sparse, weighted, connected graph and initialize the model.

        Parameters
        ----------
        num_nodes : int
            Number of nodes in the network.
        radius : float
            Radius for the soft random geometric graph.
        max_weight : float
            Maximum weight for the edges.
        damping_coefficient : float
            Damping coefficient for the system.
        seed : Optional[int]
            Random seed for reproducibility.

        Returns
        -------
        SecondOrderKuramotoModel
            Initialized model instance.
        """
        if seed is None:
            seed = np.random.randint(0,100)

        # Generate the graph
        G = nx.soft_random_geometric_graph(num_nodes, radius, dim=2, pos=None, p=2, p_dist=None, seed=seed)

        nx.planar_layout(G) # planar layout for better visualization, does not change the graph structure
        K = nx.to_numpy_array(G, weight="weight")

        # Generate power vector (sum zero)
        p = np.random.rand(num_nodes) - 0.5
        p -= np.mean(p)

        return cls(K, p, damping_coefficient)




    def compute_fixed_point(self, tol: float = 1e-10, max_iter: int = 1000) -> np.ndarray:
        """
        Compute the fixed point (steady-state phase angles) of the system.

        Parameters
        ----------
        tol : float
            Tolerance for convergence.
        max_iter : int
            Maximum number of iterations.

        Returns
        -------
        np.ndarray
            Fixed point phase angles.
        """
        def exact_f(theta, K, P):
            return P + np.sum(K * np.sin(theta[None, :] - theta[:, None]), axis=1)

        # DC approximation as initial guess
        L = np.diag(np.sum(self.connectivity_matrix, axis=0)) - self.connectivity_matrix
        theta_dc = np.linalg.pinv(L) @ self.power_vector

        # Solve for the exact fixed point
        theta_star = so.fsolve(
            exact_f, theta_dc, args=(self.connectivity_matrix, self.power_vector), xtol=tol, maxfev=max_iter
        )
        self.fixed_point = np.array(theta_star)  # Ensure correct type
        
        # Checking exclusion criterion for linearization approach
        def max_phase_angle_diff(K,theta):
            adj=np.zeros(np.shape(K))
            adj[np.abs(K)>0]=1
            phase_diff_max=np.max(adj*np.abs(theta[None,:]-theta[:,None]))
            return phase_diff_max
        
        if max_phase_angle_diff(self.connectivity_matrix,theta_star)>np.pi/2:
            print("Maximal absolute phase angle difference between two adjacent nodes more than pi/2.")
            sys.exit(1)

        return self.fixed_point

    def compute_jacobian(self) -> np.ndarray:
        """
        Compute the Jacobian (curly_L) at the fixed point.

        Returns
        -------
        np.ndarray
            Jacobian matrix.
        """
        if self.fixed_point is None:
            self.compute_fixed_point() #before this was just a print message saying that it should be computed which did not lead the debugger to flag none
        K = self.connectivity_matrix
        theta_star = self.fixed_point
        
        L = K * np.cos(theta_star[:, None] - theta_star[None, :])
        L -= np.diag(np.sum(L, axis=0))
        self.jacobian_matrix = L
        return L

    def save_parameters(self, base_dir: str = "data/SecondOrderKuramotoModel") -> str:

        """
        Save the current model parameters to a dated folder.

        Parameters
        ----------
        base_dir : str
            Base directory for saving runs.

        Returns
        -------
        str
            Path to the saved folder.
        """
        date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_dir = os.path.join(base_dir, f"Instance_{date_str}")
        os.makedirs(save_dir, exist_ok=True)
        self.current_dir= save_dir
        # Save parameters as .npy and .json for readability
        np.save(os.path.join(save_dir, "connectivity_matrix.npy"), self.connectivity_matrix)
        np.save(os.path.join(save_dir, "power_vector.npy"), self.power_vector)
        params = {
            "damping_coefficient": self.damping_coefficient,
            "ode_dimension": self.ode_dimension,
            "model_ode_args": self.model_ode_args,
        }
        if self.fixed_point is not None:
            np.save(os.path.join(save_dir, "fixed_point.npy"), self.fixed_point)
        if self.jacobian_matrix is not None:
            np.save(os.path.join(save_dir, "jacobian_matrix.npy"), self.jacobian_matrix)
        with open(os.path.join(save_dir, "parameters.json"), "w") as f:
            json.dump(params, f, indent=4)
        return save_dir

    @classmethod
    def load_from_folder(cls, folder_path: str) -> "SecondOrderKuramotoModel":
        """
        Load model parameters from a folder.

        Parameters
        ----------
        folder_path : str
            Path to the folder containing saved parameters.

        Returns
        -------
        SecondOrderKuramotoModel
            Loaded model instance.
        """
        K = np.load(os.path.join(folder_path, "connectivity_matrix.npy"))
        p = np.load(os.path.join(folder_path, "power_vector.npy"))
        with open(os.path.join(folder_path, "parameters.json"), "r") as f:
            params = json.load(f)
        model = cls(K, p, params["damping_coefficient"])
        model.current_dir=folder_path
        # Optionally load fixed point and Jacobian if present
        fixed_point_path = os.path.join(folder_path, "fixed_point.npy")
        jacobian_path = os.path.join(folder_path, "jacobian_matrix.npy")
        if os.path.exists(fixed_point_path):
            model.fixed_point = np.load(fixed_point_path)
        if os.path.exists(jacobian_path):
            model.jacobian_matrix = np.load(jacobian_path)
        return model



    def summary(self) -> None:
        """
        Print a summary of the current model parameters and state.
        """
        print("SecondOrderKuramotoModel Summary")
        print(f"Current instance: {self.current_dir is not None}")
        print(f"Number of nodes: {self.connectivity_matrix.shape[0]}")
        print(f"Damping coefficient: {self.damping_coefficient}")
        print(f"Power vector (first 5): {self.power_vector[:5]}")
        print(f"Connectivity matrix shape: {self.connectivity_matrix.shape}")
        print(f"Fixed point computed: {self.fixed_point is not None}")
        print(f"Jacobian computed: {self.jacobian_matrix is not None}")


    def calculate_response_amplitudes(self, omega: np.ndarray, k=1, S=None) -> np.ndarray:
        """
        Calculate response amplitudes for a range of frequencies.

        Parameters
        ----------
        omega : np.ndarray
            Array of frequencies.
        damping : float
            Damping coefficient.
        k: int
            Node of perturbation with frequency omega.
        S: np.ndarray, optional
            If provided, response amplitudes are calculated based on the shifted Jacobian J+S.

        Returns
        -------
        response_amplitudes : np.ndarray
            Response amplitudes for each node.
        """
        
        # making sure there is a jacobian
        if self.jacobian_matrix is None:
            self.compute_jacobian()

        # shifted and unshifted
        if np.any(S==None):
            eigvals, eigvecs = np.linalg.eigh(self.jacobian_matrix)
        else:
            eigvals,eigvecs=np.linalg.eig(self.jacobian_matrix+S)

        response_amplitudes = np.zeros((len(eigvals), len(omega)))
        # looping over all nodes to calculate the response amplitude for each
        for i in range(len(eigvals)):
            # looping over all nodes to collect all contributions to the node i's response amplitude
            for l in range(len(eigvals)):
                response_amplitudes[i,:]+=np.abs(eigvecs[l,k]*eigvecs[l,i]/(-omega**2+1j*omega*self.damping_coefficient-eigvals[l]))
        return response_amplitudes

    def predict_resonance_frequencies(self, indices=None, S=None, shifted_eigvals=None) -> np.ndarray:
        """
        Predict resonance frequencies based on eigenvalues and damping.

        Parameters
        ----------
        damping : float
            Damping coefficient.
        S : np.ndarray, optional
            If provided, resonance frequencies are predicted based on the shifted Jacobian J+S.

        Returns 
        -------
        resonance_frequencies : np.ndarray
            Predicted resonance frequencies.
        """
        if self.jacobian_matrix is None:
            raise ValueError("Jacobian matrix not computed. Call compute_jacobian() first.")
        
        # shifted and unshifted
        if S is not None:
            eigvals=np.linalg.eigvals(self.jacobian_matrix+S)
            eigvals=np.sort(eigvals)
            if indices is not None:
                eigvals=eigvals[indices]
        elif indices is not None:
            eigvals=np.linalg.eigvals(self.jacobian_matrix)
            eigvals= np.sort(eigvals)[indices]
        elif shifted_eigvals is not None:
            eigvals=np.sort(shifted_eigvals)
        else:
            eigvals = np.sort(np.linalg.eigvalsh(self.jacobian_matrix))
        eigvals = eigvals[np.invert(np.isnan(eigvals))]
        return np.sqrt(np.abs(eigvals) - (self.damping_coefficient**2) / 4)
    
        
    def model_ode(self, t: float, y: np.ndarray,args) -> np.ndarray:
        """
        Compute the time derivative of the state vector y at time t based on the second-order Kuramoto model.

        Parameters
        ----------
        t : float
            Current time.
        y : np.ndarray
            Current state vector (concatenation of phase angles and their derivatives).

        Returns
        -------
        np.ndarray
            Time derivative of the state vector.
        """
        n = len(self.connectivity_matrix)
        if len(y) is not 2*n:
            raise ValueError(f"Dimension of state vector is {len(y)} which does not match system dymension {2*n}.")


        theta = y[:n]
        dtheta_dt = y[n:]

        # Compute the second derivative of theta using the Kuramoto model equations
        d2theta_dt2 = self.power_vector - self.damping_coefficient * dtheta_dt + np.sum(
            self.connectivity_matrix * np.sin(theta[None,:] - theta[:,None]), axis=1) #check axis/broadcasting here

        return np.concatenate((dtheta_dt, d2theta_dt2))



# Example usage
if __name__ == "__main__":
    model = SecondOrderKuramotoModel.from_random_sparse_graph(num_nodes=10, edge_probability=0.4, damping_coefficient=0.2, seed=42)
    #model.compute_fixed_point()
    #print(model.fixed_point)
    model.compute_jacobian()
    #print(model.fixed_point)
    #model.calculate_response_amplitudes()
    save_path = model.save_parameters()
    print(f"Parameters saved to: {save_path}")
    model.summary()
    model_loaded= SecondOrderKuramotoModel.load_from_folder(save_path)
    model_loaded.summary()