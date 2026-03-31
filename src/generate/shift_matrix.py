import numpy as np
import numpy.typing as npt
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
        system_dim = size of the system
        eigenvalue_indices = indices of the eigenvalues that we want to shift
        num_zero_rows = number of zero rows desired
        num_zero_cols = number of zero columns desired
        
        
        generates for a given set of eigenvalues to shift and the number of zero rows and columns 
        desired index sets Xs={X_k|for all k in eigenvalue_indices} and Ys:={Y_k|for all k in eigenvalue_indices}, 
        omitting the index 0 which corresponds to the unity eigenvector. Excess degrees of freedom not
        needed for the realization of constraints are distributed evenly betwen the Xs and Ys.
        """

        self.eigenvalue_indices=np.array(eigenvalue_indices)
        system_dim = self.jacobian.shape[0]

        # calculating the number of excess degrees of freedom that are not needed for the realization of the constraints
        excess_freedoms=system_dim-len(eigenvalue_indices)-num_zero_rows-num_zero_cols-1 #-1 bcs the eigenvectors [1,...,1] is not being used
        if excess_freedoms<0: #catching invalid unputs
            raise ValueError("Network of size {} is too small for {} degrees of freedom.".format(system_dim,excess_freedoms))
        
        # removing eigenvalue_indices from the available indices for X and Y index sets
        masking_out_eigenvalue_indices=np.full(system_dim-1,True)
        masking_out_eigenvalue_indices[eigenvalue_indices]=False
        all_indices=np.arange(0,system_dim-1,dtype=int)
        indices_available=all_indices[masking_out_eigenvalue_indices]
        
        #splitting remaining indices evenly into X and Y base index sets. Simply taking the first x_len indices for X and the rest for Y.
        x_len=num_zero_rows+int(excess_freedoms/2)
        X_base=indices_available[0:x_len] 
        Y_base=indices_available[x_len:system_dim-1-len(eigenvalue_indices)]
       
        # inititalizing Xs and Ys arrays to store the index sets for each eigenvalue index in eigenvalue_indices
        Xs=np.empty(len(eigenvalue_indices),dtype=np.ndarray)
        Ys=np.empty(len(eigenvalue_indices),dtype=np.ndarray)
        
        # creating X and Y index sets for each eigenvalue index in eigenvalue_indices by concatenating the base sets with the appropriate eigenvalue indices 
        # from eigenvalue_indices and storing them in the Xs and Ys arrays
        for i in range(len(eigenvalue_indices)): 
            Xs[i]=np.concatenate((X_base, eigenvalue_indices[:i+1]))
            Ys[i]=np.concatenate((eigenvalue_indices[i:], Y_base))

        self.Xs = Xs
        self.Ys = Ys
        print("Generated Xs {Xs} and Ys {Ys}.".format(Xs=Xs, Ys=Ys))
        return Xs,Ys


    def calculate_coefficients_one_side(self, shifts: np.ndarray, zero_indices: np.ndarray,  right: bool = True) -> List[np.ndarray]:
        """
        Calculate the left or right coefficients for constructing the shift matrix.

        Parameters
        ----------
        eigenvalue_indices : list[int]
            Indices of the eigenvalues to be shifted.
        shifts : np.ndarray
            Desired shifts for the eigenvalues.
        zero_indices : np.ndarray
            Indices of rows/columns to remain zero.
        right : bool
            Whether to calculate coefficients for right spanning vectors.

        Returns
        -------
        coefficients : list[np.ndarray]
            Coefficients for constructing the shift matrix.
        """

        system_dim = self.jacobian.shape[0]
        coefficients = []

        # checking prerequisites
        if not isinstance(self.Xs,np.ndarray) or not isinstance(self.Ys,np.ndarray):
            raise ValueError("Compute Index sets before computing coefficients.")
            
        if np.shape(shifts)!=np.shape(self.eigenvalue_indices):
            raise ValueError(f"Array of shift values is of shape {np.shape(shifts)}, which does not match the provided array of supposedly corresponding eigenvalue indices {np.shape(self.eigenvalue_indices)}.")

        if not isinstance(zero_indices,np.ndarray) or zero_indices.dtype!=int:
            raise ValueError("Zero indices must be provided as a numpy array of integers.")

        # miltipurposing the function for calculating right and left constructing vectors
        if right:
            index_sets=self.Ys
        else:
            index_sets=self.Xs

        # looping over the eigenvalue indices of all desired eigenvalue shifts and calculating the corresponding coefficients for the construction of the shift matrix
        for i, idx in enumerate(self.eigenvalue_indices):
                
                # calculating the solution space for the zero rows/columns constraint
                masked_vectors = self.eigenvectors[zero_indices][:, index_sets[i]]
                null_space = la.null_space(masked_vectors)
                if null_space.size == 0:
                    raise ValueError(f"Null space is empty for eigenvalue index {idx}.")

                # plugging the solution space into the equation for the coefficient vectors
                relevant_row=null_space[self.Xs[i]==idx,:] if not right else null_space[self.Ys[i]==idx,:]
                coeff=null_space@relevant_row.T/np.sum(relevant_row**2)
                if right:
                    coeff=coeff*shifts[i]
                coefficients.append(coeff)
        if right:
            self.nus = coefficients
        else:
            self.etas = coefficients
        return coefficients


    def calculate_shift_matrix_generators(self, in_eigenspace=False) -> List[np.ndarray]:
        """
        Calculate individual shift generating vectors corresponding to each desired eigenvalue shift.  
        If in_eigenspace is True, the individual generating vectors are calculated in the eigenbasis of the Jacobian, 
        otherwise they are calculated in the physical basis. 
        
        Parameters
        ----------
            in_eigenspace : bool
                Whether to calculate the individual shift generating vectors in the eigenbasis of the Jacobian.

        Returns
        -------
            left_generators : list[np.ndarray]
                List of left spanning vectors for the individual shift generating vectors in the chosen basis.
            right_generators : list[np.ndarray]
                List of right spanning vectors for the individual shift generating vectors in the chosen basis.
        """

        # checking prerequisites
        if self.etas is None or self.nus is None:
            raise ValueError("Coefficients must be calculated before constructing individual shift generating vectors.")
        
        left_generators= []
        right_generators= []

        # looping over all individual shift matrices
        for i in range(len(self.eigenvalue_indices)):

            if in_eigenspace:
                dim=self.jacobian.shape[0]
                p=np.zeros(dim)
                q=np.zeros(dim)
                p[self.Xs[i],None]=self.etas[i]
                q[self.Ys[i],None]=self.nus[i]
                left_generators.append(p)
                right_generators.append(q)
            else:
                p = self.eigenvectors[:, self.Xs[i]] @ self.etas[i]
                q = self.eigenvectors[:, self.Ys[i]] @ self.nus[i]
                left_generators.append(p)
                right_generators.append(q)
        
        return left_generators, right_generators


    def construct_shift_matrix(self) -> np.ndarray:
        """
        Construct the final shift matrix by summing the individual shift matrices.

        Returns
        -------
        shift_matrix : np.ndarray
            The constructed shift matrix.
        """
        p, q = self.calculate_shift_matrix_generators()
        self.shift_matrix = sum(np.outer(p_i, q_i) for p_i, q_i in zip(p, q))

        return self.shift_matrix
    
    def construct_from_scratch(self, eigenvalue_indices: List[int], shifts: np.ndarray, zero_rows: npt.NDArray[np.int_], zero_cols: npt.NDArray[np.int_]) -> np.ndarray:
        """
        Construct the shift matrix from scratch given the eigenvalue indices, shifts, and zero row/column constraints.

        Parameters
        ----------
        eigenvalue_indices : list[int]
            Indices of the eigenvalues to be shifted.
        shifts : np.ndarray
            Desired shifts for the eigenvalues.
        num_zero_rows : int
            Number of rows to remain zero in the shifted matrix.
        num_zero_cols : int
            Number of columns to remain zero in the shifted matrix.

        Returns
        -------
        shift_matrix : np.ndarray
            The constructed shift matrix.
        """

        self.generate_index_sets(eigenvalue_indices, len(zero_rows), len(zero_cols))
        self.calculate_coefficients_one_side(shifts, zero_rows, right=False)
        self.calculate_coefficients_one_side(shifts, zero_cols, right=True)
        return self.construct_shift_matrix()


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
    eigenvalue_indices = [1, 2]
    shifts = np.array([-0.5, 0.3])
    zero_rows = np.array([1,2],dtype=int)
    zero_cols = np.array([],dtype=int)
    #zero_rows=np.arange(10,80)
    #zero_cols=np.array([1])
    

    S_2=shift_matrix_generator.construct_from_scratch(eigenvalue_indices, shifts, zero_rows, zero_cols)
    # Save and load
    path=shift_matrix_generator.save_to_file()
    print(path)
    loaded_S = shift_matrix_generator.load_from_file(path)

    # Plot
    shift_matrix_generator.plot_shift_matrix()
    shift_matrix_generator.plot_shift_matrix(eigenbasis=True)
