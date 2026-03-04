import numpy as np
import os
import json
import scipy.linalg as la
from typing import List, Tuple, Optional
from kuramoto_class import SecondOrderKuramotoModel as sokm 


class ShiftMatrix:
    """
    Class for generating, applying, saving, and loading shift matrices to manipulate eigenvalues
    of a system's Jacobian.
    """

    def __init__(self, jacobian: np.ndarray):
        """
        Initialize the ShiftMatrix class.

        Parameters
        ----------
        jacobian : np.ndarray
            The Jacobian matrix (curly_L) of the system.
        """
        self.jacobian = jacobian
        self.eigenvalues, self.eigenvectors = la.eigh(jacobian)
        self.shift_matrix = None
        self.intermediate_results = {
            "Xs": [],
            "Ys": [],
            "etas": [],
            "nus": [],
            "eigenvalue_indices": [],
            "shifts": [],
        }

    def generate_index_sets(self, eigenvalue_indices: List[int], num_zero_rows: int, num_zero_cols: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Generate index sets X_k and Y_k for constructing the shift matrix.

        Parameters
        ----------
        eigenvalue_indices : list[int]
            Indices of the eigenvalues to be shifted.
        num_zero_rows : int
            Number of rows to remain zero in the shifted matrix.
        num_zero_cols : int
            Number of columns to remain zero in the shifted matrix.

        Returns
        -------
        Xs, Ys : list[np.ndarray]
            Index sets for constructing the shift matrix.
        """
        dim = self.jacobian.shape[0]
        excess_freedoms = dim - len(eigenvalue_indices) - num_zero_rows - num_zero_cols - 1

        if excess_freedoms < 0:
            raise ValueError(f"Insufficient degrees of freedom for the given constraints.")

        # Generate index sets
        Xs, Ys = [], []
        for idx in eigenvalue_indices:
            Xs.append(np.arange(0, idx + 1))
            Ys.append(np.arange(idx, dim))
        self.intermediate_results["Xs"] = Xs
        self.intermediate_results["Ys"] = Ys
        return Xs, Ys

    def calculate_coefficients(self, eigenvalue_indices: List[int], shifts: np.ndarray, zero_indices: np.ndarray, index_sets: List[np.ndarray], right: bool = True) -> List[np.ndarray]:
        """
        Calculate the coefficients for constructing the shift matrix.

        Parameters
        ----------
        eigenvalue_indices : list[int]
            Indices of the eigenvalues to be shifted.
        shifts : np.ndarray
            Desired shifts for the eigenvalues.
        zero_indices : np.ndarray
            Indices of rows/columns to remain zero.
        index_sets : list[np.ndarray]
            Index sets for constructing the shift matrix.
        right : bool
            Whether to calculate coefficients for right spanning vectors.

        Returns
        -------
        coefficients : list[np.ndarray]
            Coefficients for constructing the shift matrix.
        """
        coefficients = []
        for i, idx in enumerate(eigenvalue_indices):
            masked_vectors = self.eigenvectors[zero_indices][:, index_sets[i]]
            null_space = la.null_space(masked_vectors)
            if null_space.size == 0:
                raise ValueError(f"Null space is empty for eigenvalue index {idx}.")
            coeff = null_space[:, 0] * shifts[i]
            coefficients.append(coeff)
        if right:
            self.intermediate_results["nus"] = coefficients
        else:
            self.intermediate_results["etas"] = coefficients
        return coefficients

    def construct_shift_matrix(self, Xs: List[np.ndarray], etas: List[np.ndarray], Ys: List[np.ndarray], nus: List[np.ndarray]) -> np.ndarray:
        """
        Construct the shift matrix from the coefficients and index sets.

        Parameters
        ----------
        Xs, Ys : list[np.ndarray]
            Index sets for constructing the shift matrix.
        etas, nus : list[np.ndarray]
            Coefficients for constructing the shift matrix.

        Returns
        -------
        shift_matrix : np.ndarray
            The constructed shift matrix.
        """
        dim = self.jacobian.shape[0]
        S = np.zeros((dim, dim))
        for i in range(len(Xs)):
            p = self.eigenvectors[:, Xs[i]] @ etas[i]
            q = self.eigenvectors[:, Ys[i]] @ nus[i]
            S += np.outer(p, q)
        self.shift_matrix = S
        return S

    def save_shift_matrix(self, folder_path: str) -> None:
        """
        Save the shift matrix and intermediate results to a folder.

        Parameters
        ----------
        folder_path : str
            Path to the folder where the shift matrix will be saved.
        """
        os.makedirs(folder_path, exist_ok=True)
        if self.shift_matrix is not None:
            np.save(os.path.join(folder_path, "shift_matrix.npy"), self.shift_matrix)
        with open(os.path.join(folder_path, "intermediate_results.json"), "w") as f:
            json.dump(self.intermediate_results, f, indent=4)

    def load_shift_matrix(self, folder_path: str) -> np.ndarray:
        """
        Load the shift matrix from a folder.

        Parameters
        ----------
        folder_path : str
            Path to the folder containing the saved shift matrix.

        Returns
        -------
        shift_matrix : np.ndarray
            The loaded shift matrix.
        """
        self.shift_matrix = np.load(os.path.join(folder_path, "shift_matrix.npy"))
        with open(os.path.join(folder_path, "intermediate_results.json"), "r") as f:
            self.intermediate_results = json.load(f)
        return self.shift_matrix

    def save_intermediate_results(self, key: str, value: np.ndarray) -> None:
        """
        Save intermediate results for reproducibility.

        Parameters
        ----------
        key : str
            Key for the result.
        value : np.ndarray
            Value to save.
        """
        self.intermediate_results[key] = value.tolist()

    def plot_shift_matrix(self, eigenbasis: bool = False) -> None:
        """
        Plot the shift matrix in either the physical basis or the eigenbasis.

        Parameters
        ----------
        eigenbasis : bool
            Whether to plot the shift matrix in the eigenbasis.
        """
        import matplotlib.pyplot as plt

        if self.shift_matrix is None:
            raise ValueError("Shift matrix has not been constructed.")
        S = self.shift_matrix
        if eigenbasis:
            S = self.eigenvectors.T @ S @ self.eigenvectors
        plt.imshow(np.log10(np.abs(S)), cmap="viridis")
        plt.colorbar(label="Log Magnitude")
        plt.title("Shift Matrix (Eigenbasis)" if eigenbasis else "Shift Matrix (Physical Basis)")
        plt.show()


if __name__ == "__main__":
    # Example Jacobian
    #jacobian = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
    model=sokm.from_random_sparse_graph(80, edge_probability=0.05)
    model.compute_jacobian()
    jacobian=model.jacobian_matrix

    shift_matrix_generator = ShiftMatrix(jacobian)

    # Generate shift matrix
    eigenvalue_indices = [50, 51]
    shifts = np.array([-0.5, 0.3])
    Xs, Ys = shift_matrix_generator.generate_index_sets(eigenvalue_indices, 1, 1)
    etas = shift_matrix_generator.calculate_coefficients(eigenvalue_indices, shifts, np.array([2]), Xs, right=False)
    nus = shift_matrix_generator.calculate_coefficients(eigenvalue_indices, shifts, np.array([2]), Ys, right=True)
    S = shift_matrix_generator.construct_shift_matrix(Xs, etas, Ys, nus)

    # Save and load
    folder = "test_shift_matrix"
    shift_matrix_generator.save_shift_matrix(folder)
    loaded_S = shift_matrix_generator.load_shift_matrix(folder)

    # Plot
    shift_matrix_generator.plot_shift_matrix()
    shift_matrix_generator.plot_shift_matrix(eigenbasis=True)
