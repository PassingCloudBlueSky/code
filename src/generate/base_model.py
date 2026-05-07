import os
import json
import numpy as np
from typing import Optional, Any, Type, TypeVar
from abc import ABC, abstractmethod

T = TypeVar('T', bound='BaseModel')

class BaseModel(ABC):
    """
    Abstract base class for dynamical system models.
    Provides common interface for saving/loading parameters and computing Jacobians.
    """

    def __init__(self):
        self.fixed_point: Optional[np.ndarray] = None
        self.jacobian_matrix: Optional[np.ndarray] = None
        self.current_dir: Optional[str] = None
        self.ode_dimension: Optional[int] = None  # Dimension of the ODE system, to be set by subclasses
        model_args: Optional[Tuple[Any, ...]] = None
    @abstractmethod
    def compute_jacobian(self) -> np.ndarray:
        """
        Compute the Jacobian matrix of the system.
        Must be implemented by subclasses.
        """
        pass

    def compute_fixed_point(self) -> np.ndarray:
        """
        Compute the fixed point of the system.
        Subclasses can override this if they have a specific method for finding fixed points.
        """
        #if self.fixed_point is not None:
        #    return self.fixed_point
        # Default implementation: find fixed point by iterating from a random initial condition

        pass

    def save_parameters(self, base_dir: str = "data/BaseModel") -> str:
        """
        Save model parameters to a directory.
        Subclasses should extend this to save additional parameters.
        """
        import datetime
        date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_dir = os.path.join(base_dir, f"Instance_{date_str}")
        os.makedirs(save_dir, exist_ok=True)
        self.current_dir = save_dir

        # Save generic attributes
        if self.fixed_point is not None:
            np.save(os.path.join(save_dir, "fixed_point.npy"), self.fixed_point)
        if self.jacobian_matrix is not None:
            np.save(os.path.join(save_dir, "jacobian_matrix.npy"), self.jacobian_matrix)
        # Save metadata
        with open(os.path.join(save_dir, "meta.json"), "w") as f:
            json.dump({"model_class": self.__class__.__name__}, f, indent=4)
        return save_dir

    @classmethod
    def load_from_folder(cls: Type[T], folder_path: str) -> T:
        """
        Load model parameters from a directory.
        Subclasses should extend this to load additional parameters.
        """
        obj = cls.__new__(cls)
        obj.current_dir = folder_path
        # Load generic attributes if present
        fp_path = os.path.join(folder_path, "fixed_point.npy")
        jac_path = os.path.join(folder_path, "jacobian_matrix.npy")
        if os.path.exists(fp_path):
            obj.fixed_point = np.load(fp_path)
        if os.path.exists(jac_path):
            obj.jacobian_matrix = np.load(jac_path)
        return obj
