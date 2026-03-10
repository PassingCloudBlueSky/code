import numpy as np
import os
import json
from param import String
import scipy.linalg as la
from typing import List, Tuple, Optional
from kuramoto_class import SecondOrderKuramotoModel as sokm 



class ShiftMatrix:
    """
    Class for generating, applying, saving, and loading shift matrices to manipulate eigenvalues
    of a system's Jacobian.
    """

    def __init__(self, model=None, jacobian: Optional[np.ndarray] = None, current_dir: Optional[str] = None):
        """
        Initialize the ShiftMatrix class.

        Parameters
        ----------
        jacobian : np.ndarray
            The Jacobian matrix (curly_L) of the system.
        """
        if model is not None:
            if model.jacobian_matrix is None:
                model.compute_jacobian()
            self.jacobian = model.jacobian_matrix
            if model.current_dir is None:
                model.save_parameters()
            self.current_dir=model.current_dir
        elif jacobian is not None and current_dir is not None:
            self.jacobian = jacobian
            self.current_dir = current_dir


        self.eigenvalues, self.eigenvectors = la.eigh(self.jacobian)
        self.shift_matrix: Optional[np.ndarray] = None
        self.Xs: Optional[np.ndarray] = None
        self.Ys: Optional[np.ndarray] = None
        self.etas: Optional[np.ndarray] = None
        self.nus: Optional[np.ndarray] = None
        self.eigenvalue_indices: Optional[np.ndarray] = None
        self.shifts: Optional[np.ndarray] = None





    def save_to_file(self, path: str="") -> str:
        """
        Save the ShiftMatrix object to a file in JSON format.

        Parameters
        ----------
        filename : str
            The name of the file to save the data to.
        """
        # Prepare data for serialization
        def to_serializable(obj):
            """
            Convert a NumPy array to a list if necessary, otherwise return the object as is.
            """
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj  # Return as is if it's not a NumPy array
        
        def serialize_list_of_arrays(array_list):
            """
            Convert a list of NumPy arrays into a JSON-serializable object.

            Parameters
            ----------
            array_list : list of np.ndarray
                A list containing NumPy arrays.

            Returns
            -------
            list
                A list of Python lists, which is JSON-serializable.
            """
            return [arr.tolist() for arr in array_list]


        data = {
            "jacobian": to_serializable(self.jacobian),
            "eigenvalues": to_serializable(self.eigenvalues),
            "eigenvectors": to_serializable(self.eigenvectors),
            "shift_matrix": to_serializable(self.shift_matrix),
            "Xs": serialize_list_of_arrays(self.Xs),
            "Ys": serialize_list_of_arrays(self.Ys),
            "etas": serialize_list_of_arrays(self.etas),
            "nus": serialize_list_of_arrays(self.nus),
            "eigenvalue_indices": to_serializable(self.eigenvalue_indices),
            "shifts": to_serializable(self.shifts),
        }

        # Save to file
        if path == "":
            path = os.path.join(self.current_dir, "shift_matrix.json")
            print(f"Saving shift matrix data to {path}...")
        with open(path, "w", encoding="utf-8") as file:
            json.dump(data, file)
        return path

    @classmethod
    def load_from_file(cls, filename: str):
        """
        Load a ShiftMatrix object from a file.

        Parameters
        ----------
        filename : str
            The name of the file to load the data from.

        Returns
        -------
        ShiftMatrix
            A new instance of the ShiftMatrix class with the loaded data.
        """
        # Load data from file
        with open(filename, "r") as f:
            data = json.load(f)

        def deserialize_list_of_arrays(serialized_list):
            """
            Convert a JSON-serializable list of lists back into a list of NumPy arrays.

            Parameters
            ----------
            serialized_list : list
                A list of Python lists (JSON-serializable).

            Returns
            -------
            list of np.ndarray
                A list of NumPy arrays reconstructed from the serialized data.
            """
            return [np.array(lst) for lst in serialized_list]

        # Reconstruct the object
        jacobian = np.array(data["jacobian"])
        obj = ShiftMatrix(jacobian=jacobian, current_dir=os.path.dirname(filename))
        obj.eigenvalues = np.array(data["eigenvalues"])
        obj.eigenvectors = np.array(data["eigenvectors"])
        obj.shift_matrix = np.array(data["shift_matrix"]) if data["shift_matrix"] is not None else None
        obj.Xs = deserialize_list_of_arrays(data["Xs"]) if data["Xs"] is not None else None
        obj.Ys = deserialize_list_of_arrays(data["Ys"]) if data["Ys"] is not None else None
        obj.etas = deserialize_list_of_arrays(data["etas"]) if data["etas"] is not None else None
        obj.nus = deserialize_list_of_arrays(data["nus"]) if data["nus"] is not None else None
        obj.eigenvalue_indices = np.array(data["eigenvalue_indices"]) if data["eigenvalue_indices"] is not None else None
        obj.shifts = np.array(data["shifts"]) if data["shifts"] is not None else None

        return obj


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
        if max(eigenvalue_indices) >= len(self.eigenvalues):
            raise ValueError(f"Providedigenvalue index {max(eigenvalue_indices)} is out of bounds for the Jacobian matrix with {len(self.eigenvalues)} eigenvalues.")
        
        self.eigenvalue_indices=np.array(eigenvalue_indices)
        dim = self.jacobian.shape[0]
        excess_freedoms = dim - len(eigenvalue_indices) - num_zero_rows - num_zero_cols - 1

        if excess_freedoms < 0:
            raise ValueError(f"Insufficient degrees of freedom for the given constraints, i.e. the number of eigenvalue, zero rows and zero colum indices exceeds system dimension minus one.")

        # Generate index sets
        Xs, Ys = [], []
        for idx in eigenvalue_indices:
            Xs.append(np.arange(0, idx + 1))
            Ys.append(np.arange(idx, dim))
        self.Xs = Xs
        self.Ys = Ys
        return Xs, Ys

    def calculate_coefficients_one_side(self, shifts: np.ndarray, zero_indices: np.ndarray,  right: bool = True) -> List[np.ndarray]:
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

        # checking prerequisites
        if self.Xs== None or self.Ys ==None:
            raise ValueError("Compute Index sets before computing coefficients.")
            
        if np.shape(shifts)!=np.shape(self.eigenvalue_indices):
            raise ValueError(f"Array of shift values is of shape {np.shape(shifts)}, which does not match the provided array of supposedly corresponding eigenvalue indices {np.shape(self.eigenvalue_indices)}.")

        if right:
            index_sets=self.Ys
        else:
            index_sets=self.Xs
        for i, idx in enumerate(self.eigenvalue_indices):
            masked_vectors = self.eigenvectors[zero_indices][:, index_sets[i]]
            null_space = la.null_space(masked_vectors)
            if null_space.size == 0:
                raise ValueError(f"Null space is empty for eigenvalue index {idx}.")
            coeff = null_space[:, 0] * shifts[i]
            coefficients.append(coeff)
        if right:
            self.nus = coefficients
        else:
            self.etas = coefficients
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
    model=sokm.from_random_sparse_graph(8, edge_probability=0.05)
    model.compute_jacobian()
    jacobian=model.jacobian_matrix

    shift_matrix_generator = ShiftMatrix(model=model)

    # Generate shift matrix
    eigenvalue_indices = [2, 3]
    shifts = np.array([-0.5, 0.3])
    Xs, Ys = shift_matrix_generator.generate_index_sets(eigenvalue_indices, 1, 1) 
    etas = shift_matrix_generator.calculate_coefficients_one_side( shifts, np.array([6,7]), right=False)
    nus = shift_matrix_generator.calculate_coefficients_one_side( shifts, np.array([]), right=True)
    S = shift_matrix_generator.construct_shift_matrix(Xs, etas, Ys, nus)

    # Save and load
    path=shift_matrix_generator.save_to_file()
    loaded_S = shift_matrix_generator.load_from_file(path)

    # Plot
    shift_matrix_generator.plot_shift_matrix()
    shift_matrix_generator.plot_shift_matrix(eigenbasis=True)
