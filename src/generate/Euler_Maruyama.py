from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Literal

import numpy as np
from scipy.integrate import solve_ivp

import json
from abc import ABC, abstractmethod
from typing import Any, Literal

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
from numba import njit
from numba.core.dispatcher import Dispatcher as _NumbaDispatcher

from shift_matrix import ShiftMatrix

#from MeineProjekte.PerturbationSourceReconstruction._PerLoco.systems import LinearSystem

def _as_float_array(x: Any, *, ndim: int | None = None) -> np.ndarray:
    """Konvertiere Eingaben in ein Float-Array und prüfe optional die Dimension.

    Parameters
    ----------
    x : Any
        Beliebiges array-artiges Objekt.
    ndim : int | None, optional
        Erwartete Anzahl von Dimensionen. Bei `None` erfolgt keine Dimensionsprüfung.

    Returns
    -------
    np.ndarray
        Arraydarstellung von `x` mit Datentyp `float`.

    Raises
    ------
    ValueError
        Falls `ndim` gesetzt ist und die tatsächliche Arraydimension nicht passt.
    """
    arr = np.asarray(x, dtype=float)
    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"Expected array with ndim={ndim}, got ndim={arr.ndim}.")
    return arr


def _validate_square_matrix(name: str, mat: np.ndarray) -> None:
    """Prüfe, ob eine Matrix quadratisch ist.

    Parameters
    ----------
    name : str
        Anzeigename der Matrix für Fehlermeldungen.
    mat : np.ndarray
        Zu prüfende Matrix.

    Returns
    -------
    None
        Diese Funktion gibt keinen Wert zurück.

    Raises
    ------
    ValueError
        Falls `mat` nicht zweidimensional und quadratisch ist.
    """
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"{name} must be a square matrix, got shape {mat.shape}.")

        
class BaseSystem(ABC):
    """Abstrakte Basisklasse für alle dynamischen Systeme im Projekt."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Liefere die Länge des vollständigen Zustandsvektors.

        Returns
        -------
        int
            Zustandsdimension des Systems.
        """
        pass

    @property
    @abstractmethod
    def state(self) -> np.ndarray:
        """Liefere den aktuellen Systemzustand.

        Returns
        -------
        np.ndarray
            Kopie des aktuellen Zustandsvektors.
        """
        pass

    @state.setter
    @abstractmethod
    def state(self, value: np.ndarray) -> None:
        """Setze den internen Systemzustand.

        Parameters
        ----------
        value : np.ndarray
            Neuer Zustandsvektor der Form ``(dimension,)``.

        Returns
        -------
        None
            Diese abstrakte Setter-Schnittstelle liefert keinen Rückgabewert.
        """
        pass

    @abstractmethod
    def rhs(self, t: float, y: np.ndarray, forcing: np.ndarray | None = None) -> np.ndarray:
        """Berechne die rechte Seite der Dynamik.

        Parameters
        ----------
        t : float
            Auswertezeitpunkt.
        y : np.ndarray
            Zustandsvektor der Form ``(dimension,)``.
        forcing : np.ndarray | None, optional
            Optionale additive Anregung derselben Dimension wie `y`.

        Returns
        -------
        np.ndarray
            Ableitungsvektor der Form ``(dimension,)``.
        """
        pass

    @abstractmethod
    def jacobian(self, y: np.ndarray | None = None) -> np.ndarray:
        """Berechne die Jacobi-Matrix der Dynamik.

        Parameters
        ----------
        y : np.ndarray | None, optional
            Optionaler Zustand, an dem die Jacobimatrix ausgewertet wird.

        Returns
        -------
        np.ndarray
            Quadratische Jacobi-Matrix der Form ``(dimension, dimension)``.
        """
        pass

    @abstractmethod
    def steady_state(self, **kwargs: Any) -> np.ndarray:
        """Bestimme näherungsweise einen stationären Zustand.

        Parameters
        ----------
        **kwargs : Any
            Modellspezifische Parameter der stationären Berechnung.

        Returns
        -------
        np.ndarray
            Zustandsvektor eines approximativen stationären Zustands.
        """
        pass

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialisiere das System in ein Wörterbuch.

        Returns
        -------
        dict[str, Any]
            JSON-kompatible Darstellung des Systems.
        """
        pass

    def to_json(self, path: str) -> None:
        """Schreibe die serialisierte Systemdarstellung in eine JSON-Datei.

        Parameters
        ----------
        path : str
            Zielpfad der JSON-Datei.

        Returns
        -------
        None
            Diese Methode gibt keinen Wert zurück.
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
    
@dataclass
class SecondOrderLinearSystem(BaseSystem):
    """Lineares System zweiter Ordnung in Zustandsraumdarstellung."""
    A: np.ndarray
    B: np.ndarray
    _state: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Validiere Matrizen und Anfangszustand des Systems.

        Returns
        -------
        None
            Diese Methode gibt keinen Wert zurück.

        Raises
        ------
        ValueError
            Falls Dimensionen von `A`, `B` oder `_state` nicht kompatibel sind.
        """
        self.A = _as_float_array(self.A, ndim=2)
        self.B = _as_float_array(self.B, ndim=2)
        _validate_square_matrix("A", self.A)
        _validate_square_matrix("B", self.B)
        n = self.A.shape[0]
        if self.B.shape != (n, n):
            raise ValueError(f"B must have shape ({n}, {n}), got {self.B.shape}.")
        if self._state is None:
            self._state = np.zeros(2 * n, dtype=float)
        else:
            self._state = _as_float_array(self._state, ndim=1)
            if self._state.shape != (2 * n,):
                raise ValueError(f"initial_state must have shape ({2*n},), got {self._state.shape}.")

    @property
    def n_nodes(self) -> int:
        """Liefere die Anzahl physikalischer Knoten bzw. Freiheitsgrade.

        Returns
        -------
        int
            Anzahl der Positionsvariablen.
        """
        return self.A.shape[0]

    @property
    def dimension(self) -> int:
        """Liefere die Zustandsdimension des Systems erster Ordnung.

        Returns
        -------
        int
            Doppelte Knotenzahl für Position und Geschwindigkeit.
        """
        return 2 * self.n_nodes

    @property
    def state(self) -> np.ndarray:
        """Liefere den aktuellen Zustandsvektor.

        Returns
        -------
        np.ndarray
            Zustandsvektor der Form ``(2*n_nodes,)``.
        """
        return self._state.copy()

    @state.setter
    def state(self, value: np.ndarray) -> None:
        """Setze den internen Zustandsvektor.

        Parameters
        ----------
        value : np.ndarray
            Neuer Zustand der Form ``(dimension,)``.

        Returns
        -------
        None
            Diese Methode gibt keinen Wert zurück.

        Raises
        ------
        ValueError
            Falls die Form nicht zur Systemdimension passt.
        """
        value = _as_float_array(value, ndim=1)
        if value.shape != (self.dimension,):
            raise ValueError(f"state must have shape ({self.dimension},), got {value.shape}.")
        self._state = value.copy()

    def rhs(self, t: float, y: np.ndarray, forcing: np.ndarray | None = None) -> np.ndarray:
        """Berechne die erste Ordnungsschreibweise des Systems zweiter Ordnung.

        Parameters
        ----------
        t : float
            Zeitvariable, die hier nicht explizit verwendet wird.
        y : np.ndarray
            Zustandsvektor ``[x, v]`` der Form ``(2*n_nodes,)``.
        forcing : np.ndarray | None, optional
            Additive Anregung für Positions- und Beschleunigungsteil.

        Returns
        -------
        np.ndarray
            Ableitungsvektor ``[dx, dv]`` der Form ``(2*n_nodes,)``.

        Raises
        ------
        ValueError
            Falls Zustands- oder Anregungsvektor nicht die erwartete Form besitzen.
        """
        del t
        y = _as_float_array(y, ndim=1)
        if y.shape != (self.dimension,):
            raise ValueError(f"y must have shape ({self.dimension},), got {y.shape}.")
        if forcing is None:
            forcing_vec = np.zeros(self.dimension, dtype=float)
        else:
            forcing_vec = _as_float_array(forcing, ndim=1)
            if forcing_vec.shape != (self.dimension,):
                raise ValueError(f"forcing must have shape ({self.dimension},), got {forcing_vec.shape}.")

        n = self.n_nodes
        x = y[:n]
        v = y[n:]
        dx = v + forcing_vec[:n]
        acceleration = self.A @ x + self.B @ v + forcing_vec[n:]
        return np.concatenate([dx, acceleration])

    def jacobian(self, y: np.ndarray | None = None) -> np.ndarray:
        """Liefere die konstante Jacobi-Matrix des linearen Systems.

        Parameters
        ----------
        y : np.ndarray | None, optional
            Unbenutzter Parameter zur Schnittstellenkompatibilität.

        Returns
        -------
        np.ndarray
            Jacobi-Matrix der Form ``(2*n_nodes, 2*n_nodes)``.
        """
        del y
        n = self.n_nodes
        jac = np.zeros((2 * n, 2 * n), dtype=float)
        jac[:n, n:] = np.eye(n, dtype=float)
        jac[n:, :n] = self.A
        jac[n:, n:] = self.B
        return jac

    def steady_state(
        self,
        *,
        forcing: np.ndarray | None = None,
        tol: float = 1e-8,
        window: float = 10.0,
        max_windows: int = 200,
    ) -> np.ndarray:
        """Approximiere einen stationären Zustand per wiederholter Integration.

        Parameters
        ----------
        forcing : np.ndarray | None, optional
            Konstante additive Anregung der Form ``(dimension,)``.
        tol : float, default=1e-8
            Toleranz auf die Unendlichkeitsnorm der rechten Seite.
        window : float, default=10.0
            Länge eines Integrationsfensters.
        max_windows : int, default=200
            Maximale Anzahl von Integrationsfenstern.

        Returns
        -------
        np.ndarray
            Approximierter stationärer Zustand der Form ``(dimension,)``.
        """
        forcing_vec = None if forcing is None else _as_float_array(forcing, ndim=1)
        if forcing_vec is not None and forcing_vec.shape != (self.dimension,):
            raise ValueError(f"forcing must have shape ({self.dimension},), got {forcing_vec.shape}.")

        y = self.state
        for _ in range(max_windows):
            sol = solve_ivp(
                lambda t, yy: self.rhs(t, yy, forcing_vec),
                (0.0, window),
                y,
                t_eval=[window],
                rtol=1e-7,
                atol=1e-9,
            )
            y = sol.y[:, -1]
            if np.linalg.norm(self.rhs(0.0, y, forcing_vec), ord=np.inf) < tol:
                self.state = y
                return y.copy()

        self.state = y
        return y.copy()

    def to_dict(self) -> dict[str, Any]:
        """Serialisiere das System in ein Wörterbuch.

        Returns
        -------
        dict[str, Any]
            Wörterbuch mit Typ, Matrizen `A`, `B` und aktuellem Zustand.
        """
        return {
            "system_type": "SecondOrderLinearSystem",
            "A": self.A.tolist(),
            "B": self.B.tolist(),
            "state": self.state.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SecondOrderLinearSystem":
        """Erzeuge ein System zweiter Ordnung aus einer Wörterbuchdarstellung.

        Parameters
        ----------
        data : dict[str, Any]
            Serialisierte Systemdaten.

        Returns
        -------
        SecondOrderLinearSystem
            Rekonstruiertes Systemobjekt.

        Raises
        ------
        ValueError
            Falls `system_type` nicht passt.
        """
        if data.get("system_type") != "SecondOrderLinearSystem":
            raise ValueError("Invalid system_type for SecondOrderLinearSystem.")
        return cls(
            A=np.asarray(data["A"], dtype=float),
            B=np.asarray(data["B"], dtype=float),
            _state=np.asarray(data["state"], dtype=float),
        )

    @classmethod
    def from_json(cls, path: str) -> "SecondOrderLinearSystem":
        """Lade ein System zweiter Ordnung aus einer JSON-Datei.

        Parameters
        ----------
        path : str
            Pfad zur JSON-Datei.

        Returns
        -------
        SecondOrderLinearSystem
            Aus der Datei rekonstruiertes System.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)



@dataclass
class SecondOrderKuramotoSystem(BaseSystem):
    """Kuramoto-System zweiter Ordnung mit Trägheit und Dämpfung."""
    damping: np.ndarray
    inertia: np.ndarray
    Omega: np.ndarray
    coupling_matrix: np.ndarray
    _state: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Validiere Parameter und Anfangszustand nach der Initialisierung.

        Returns
        -------
        None
            Diese Methode gibt keinen Wert zurück.

        Raises
        ------
        ValueError
            Falls Parameterdimensionen nicht kompatibel sind oder Trägheiten null sind.
        """
        self.damping = _as_float_array(self.damping, ndim=1)
        self.inertia = _as_float_array(self.inertia, ndim=1)
        self.Omega = _as_float_array(self.Omega, ndim=1)
        self.coupling_matrix = _as_float_array(self.coupling_matrix, ndim=2)

        n = self.damping.shape[0]
        if self.inertia.shape != (n,) or self.Omega.shape != (n,):
            raise ValueError("damping, inertia, and Omega must all have shape (N,).")
        if np.any(self.inertia == 0.0):
            raise ValueError("inertia entries must be non-zero.")
        if self.coupling_matrix.shape != (n, n):
            raise ValueError(f"coupling_matrix must have shape ({n}, {n}), got {self.coupling_matrix.shape}.")

        if self._state is None:
            self._state = np.zeros(2 * n, dtype=float)
        else:
            self._state = _as_float_array(self._state, ndim=1)
            if self._state.shape != (2 * n,):
                raise ValueError(f"initial_state must have shape ({2*n},), got {self._state.shape}.")

    @property
    def n_nodes(self) -> int:
        """Liefere die Anzahl der Oszillatoren.

        Returns
        -------
        int
            Anzahl der Knoten bzw. Oszillatoren.
        """
        return self.damping.shape[0]

    @property
    def dimension(self) -> int:
        """Liefere die Zustandsdimension des Kuramoto-Systems.

        Returns
        -------
        int
            Doppelte Knotenzahl für Phase und Frequenzabweichung.
        """
        return 2 * self.n_nodes

    @property
    def state(self) -> np.ndarray:
        """Liefere den aktuellen Zustandsvektor.

        Returns
        -------
        np.ndarray
            Zustandsvektor ``[theta, xi]`` der Form ``(dimension,)``.
        """
        return self._state.copy()

    @state.setter
    def state(self, value: np.ndarray) -> None:
        """Setze den internen Zustandsvektor.

        Parameters
        ----------
        value : np.ndarray
            Neuer Zustandsvektor der Form ``(dimension,)``.

        Returns
        -------
        None
            Diese Methode gibt keinen Wert zurück.

        Raises
        ------
        ValueError
            Falls die Form des Zustandsvektors nicht passt.
        """
        value = _as_float_array(value, ndim=1)
        if value.shape != (self.dimension,):
            raise ValueError(f"state must have shape ({self.dimension},), got {value.shape}.")
        self._state = value.copy()

    def rhs(self, t: float, y: np.ndarray, forcing: np.ndarray | None = None, vtn: np.ndarray | None = None, theta_star: np.ndarray | None = None) -> np.ndarray:
        """Berechne die rechte Seite der Kuramoto-Dynamik.

        Parameters
        ----------
        t : float
            Zeitvariable, die hier nicht explizit verwendet wird.
        y : np.ndarray
            Zustandsvektor ``[theta, xi]`` der Form ``(2*n_nodes,)``.
        forcing : np.ndarray | None, optional
            Additive Anregung der Form ``(dimension,)``.
        vtn : np.ndarray | None, optional
            Virtual transmission network connectivity matrix der Form ``(n_nodes, n_nodes)``.
        theta_star : np.ndarray | None, optional
            Fixpoint relative to which the virtual transmission network is operating. 

        Returns
        -------
        np.ndarray
            Ableitungsvektor der Form ``(dimension,)``.

        Raises
        ------
        ValueError
            Falls `y`, `forcing`, oder `theta_star` nicht die korrekte Form besitzen.
        """
        del t
        y = _as_float_array(y, ndim=1)
        if y.shape != (self.dimension,):
            raise ValueError(f"y must have shape ({self.dimension},), got {y.shape}.")
        if forcing is None:
            forcing_vec = np.zeros(self.dimension, dtype=float)
        else:
            forcing_vec = _as_float_array(forcing, ndim=1)
            if forcing_vec.shape != (self.dimension,):
                raise ValueError(f"forcing must have shape ({self.dimension},), got {forcing_vec.shape}.")
        
        n = self.n_nodes
        theta = y[:n]
        xi = y[n:]

        phase_diff = theta[None, :] - theta[:, None]
        coupling = np.sum(self.coupling_matrix * np.sin(phase_diff), axis=1)

        # computing shift if required parameters are provided
        if vtn is None or theta_star is None:
            shift = np.zeros(self.dimension, dtype=float)
        else:
            vtn = _as_float_array(vtn, ndim=2)
            if vtn.shape != (self.dimension, self.dimension):
                    raise ValueError(f"vtn must have shape ({self.dimension}, {self.dimension}), got {vtn.shape}.")
            theta_star = _as_float_array(theta_star, ndim=1)
            if theta_star.shape != (self.dimension,):
                            raise ValueError(f"theta_star must have shape ({self.dimension},), got {theta_star.shape}.")
            shift = vtn @ (theta - theta_star)
        

        dtheta = xi + forcing_vec[:n]
        dxi = (-self.damping * xi + self.Omega + coupling) / self.inertia + forcing_vec[n:] + shift[n:]
        return np.concatenate([dtheta, dxi])

    def jacobian(self, y: np.ndarray | None = None) -> np.ndarray:
        """Berechne die Jacobimatrix der Kuramoto-Dynamik an einem Zustand.

        Parameters
        ----------
        y : np.ndarray | None, optional
            Zustandsvektor der Form ``(dimension,)``. Falls `None`, wird der
            aktuelle interne Zustand verwendet.

        Returns
        -------
        np.ndarray
            Jacobimatrix der Form ``(dimension, dimension)``.

        Raises
        ------
        ValueError
            Falls der Auswertevektor nicht zur Systemdimension passt.
        """
        if y is None:
            y = self.state
        y = _as_float_array(y, ndim=1)
        if y.shape != (self.dimension,):
            raise ValueError(f"y must have shape ({self.dimension},), got {y.shape}.")

        n = self.n_nodes
        theta = y[:n]
        jac = np.zeros((2 * n, 2 * n), dtype=float)

        jac[:n, n:] = np.eye(n, dtype=float)

        cos_mat = np.cos(theta[None, :] - theta[:, None])
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                jac[n + i, j] = self.coupling_matrix[i, j] * cos_mat[i, j] / self.inertia[i]
            jac[n + i, i] = -np.sum(
                self.coupling_matrix[i, np.arange(n) != i] * cos_mat[i, np.arange(n) != i]
            ) / self.inertia[i]

        jac[n:, n:] = -np.diag(self.damping / self.inertia)
        return jac

    def steady_state(
        self,
        *,
        forcing: np.ndarray | None = None,
        tol: float = 1e-8,
        window: float = 10.0,
        max_windows: int = 300,
    ) -> np.ndarray:
        """Approximiere einen stationären Zustand durch wiederholte Integration.

        Parameters
        ----------
        forcing : np.ndarray | None, optional
            Konstante additive Anregung der Form ``(dimension,)``.
        tol : float, default=1e-8
            Toleranz auf die Unendlichkeitsnorm der Dynamik.
        window : float, default=10.0
            Länge eines Integrationsfensters.
        max_windows : int, default=300
            Maximale Anzahl von Integrationsfenstern.

        Returns
        -------
        np.ndarray
            Approximierter stationärer Zustand der Form ``(dimension,)``.
        """
        forcing_vec = None if forcing is None else _as_float_array(forcing, ndim=1)
        if forcing_vec is not None and forcing_vec.shape != (self.dimension,):
            raise ValueError(f"forcing must have shape ({self.dimension},), got {forcing_vec.shape}.")

        y = self.state
        for _ in range(max_windows):
            sol = solve_ivp(
                lambda t, yy: self.rhs(t, yy, forcing_vec),
                (0.0, window),
                y,
                t_eval=[window],
                rtol=1e-7,
                atol=1e-9,
            )
            y = sol.y[:, -1]
            dyn_res = np.linalg.norm(self.rhs(0.0, y, forcing_vec), ord=np.inf)
            if dyn_res < tol:
                self.state = y
                return y.copy()

        self.state = y
        return y.copy()

    def weighted_laplacian(self, theta_star: np.ndarray | None = None) -> np.ndarray:
        """Berechne die gewichtete Laplace-Matrix um eine Phasenkonfiguration.

        Parameters
        ----------
        theta_star : np.ndarray | None, optional
            Referenzphasen der Form ``(n_nodes,)``. Falls `None`, werden die aktuell
            gespeicherten Phasen aus `state` verwendet.

        Returns
        -------
        np.ndarray
            Gewichtete Laplace-Matrix der Form ``(n_nodes, n_nodes)``.

        Raises
        ------
        ValueError
            Falls `theta_star` nicht die erwartete Form besitzt.
        """
        if theta_star is None:
            theta = self.state[: self.n_nodes]
        else:
            theta = _as_float_array(theta_star, ndim=1)
            if theta.shape != (self.n_nodes,):
                raise ValueError(f"theta_star must have shape ({self.n_nodes},), got {theta.shape}.")

        delta = theta[None, :] - theta[:, None]
        C = np.cos(delta)
        L = -self.coupling_matrix * C
        np.fill_diagonal(L, 0.0)
        np.fill_diagonal(L, -np.sum(L, axis=1))
        return L

    def to_dict(self) -> dict[str, Any]:
        """Serialisiere das Kuramoto-System in ein Wörterbuch.

        Returns
        -------
        dict[str, Any]
            Wörterbuch mit Parametern und aktuellem Zustand.
        """
        return {
            "system_type": "SecondOrderKuramotoSystem",
            "damping": self.damping.tolist(),
            "inertia": self.inertia.tolist(),
            "Omega": self.Omega.tolist(),
            "coupling_matrix": self.coupling_matrix.tolist(),
            "state": self.state.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SecondOrderKuramotoSystem":
        """Erzeuge ein Kuramoto-System aus einem serialisierten Wörterbuch.

        Parameters
        ----------
        data : dict[str, Any]
            Serialisierte Systemdaten.

        Returns
        -------
        SecondOrderKuramotoSystem
            Rekonstruiertes Kuramoto-System.

        Raises
        ------
        ValueError
            Falls `system_type` nicht zu diesem System passt.
        """
        if data.get("system_type") != "SecondOrderKuramotoSystem":
            raise ValueError("Invalid system_type for SecondOrderKuramotoSystem.")
        return cls(
            damping=np.asarray(data["damping"], dtype=float),
            inertia=np.asarray(data["inertia"], dtype=float),
            Omega=np.asarray(data["Omega"], dtype=float),
            coupling_matrix=np.asarray(data["coupling_matrix"], dtype=float),
            _state=np.asarray(data["state"], dtype=float),
        )

    @classmethod
    def from_json(cls, path: str) -> "SecondOrderKuramotoSystem":
        """Lade ein Kuramoto-System aus einer JSON-Datei.

        Parameters
        ----------
        path : str
            Pfad zur JSON-Datei.

        Returns
        -------
        SecondOrderKuramotoSystem
            Aus der Datei rekonstruiertes System.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)





@dataclass
class DeterministicForcingSpec:
    """Spezifikation einer deterministischen harmonischen Anregung."""
    target_nodes: list[int]
    frequencies: list[float]
    amplitudes: list[float]
    phases: list[float] | None = None
    component: str | None = None


@dataclass
class ImpulseForcingSpec:
    """Spezifikation einer deterministischen Impulsanregung."""
    target_nodes: list[int]
    pulse_type: Literal["gaussian", "rectangular"]
    amplitude: float
    center: float
    width: float
    component: str | None = None


@dataclass
class StochasticForcingSpec:
    """Spezifikation einer stochastischen Anregung auf ausgewählten Knoten."""
    target_nodes: list[int]
    noise_type: Literal["white", "pink", "brown", "custom", "bandlimited"]
    strength: float
    seed: int | None = None
    component: str | None = None
    beta: float | None = None
    """Potenzspektrum-Exponent (PSD ~ f^-beta) für noise_type='custom'."""
    f_min: float | None = None
    """Untere Grenzfrequenz [Hz] des Bandes für noise_type='bandlimited'."""
    f_max: float | None = None
    """Obere Grenzfrequenz [Hz] des Bandes für noise_type='bandlimited'."""


def evaluate_impulse_profile(
    t: float | np.ndarray,
    *,
    pulse_type: Literal["gaussian", "rectangular"],
    amplitude: float,
    center: float,
    width: float,
) -> np.ndarray:
    """Werte einen Gauss- oder Rechteckimpuls auf einem Zeitgitter aus.

    Parameters
    ----------
    t : float | np.ndarray
        Zeitpunkt oder Zeitvektor.
    pulse_type : Literal["gaussian", "rectangular"]
        Impulsform.
    amplitude : float
        Maximale Impulshoehe.
    center : float
        Zeitliche Mitte des Impulses.
    width : float
        Standardabweichung des Gaussimpulses oder Breite des Rechteckimpulses.

    Returns
    -------
    np.ndarray
        Impulswerte mit derselben Form wie `t`.

    Raises
    ------
    ValueError
        Falls `pulse_type` unbekannt ist oder `width` nicht positiv ist.
    """
    width = float(width)
    if width <= 0.0:
        raise ValueError("width must be > 0.")

    t_arr = np.asarray(t, dtype=float)
    amp = float(amplitude)
    center = float(center)

    if pulse_type == "gaussian":
        return amp * np.exp(-0.5 * ((t_arr - center) / width) ** 2)
    if pulse_type == "rectangular":
        start = center - 0.5 * width
        end = center + 0.5 * width
        return np.where((t_arr >= start) & (t_arr <= end), amp, 0.0)
    raise ValueError("pulse_type must be 'gaussian' or 'rectangular'.")


def _validate_uniform_time(t: np.ndarray) -> float:
    """Prüfe, ob ein Zeitgitter äquidistant ist.

    Parameters
    ----------
    t : np.ndarray
        Eindimensionaler Zeitvektor mit mindestens zwei Einträgen.

    Returns
    -------
    float
        Konstanter Zeitschritt `dt` des Gitters.

    Raises
    ------
    ValueError
        Falls `t` nicht eindimensional ist, zu wenige Einträge besitzt oder nicht
        äquidistant ist.
    """
    if t.ndim != 1 or t.size < 2:
        raise ValueError("Time vector must be one-dimensional with at least two entries.")
    dt = np.diff(t)
    if not np.allclose(dt, dt[0], rtol=1e-10, atol=1e-12):
        raise ValueError("Stochastic solver requires a uniformly spaced time grid.")
    return float(dt[0])


def _normalize_energy_per_column(X: np.ndarray) -> np.ndarray:
    """Normalisiere jede Spalte auf nullen Mittelwert und Einheitsnorm.

    Parameters
    ----------
    X : np.ndarray
        Zweidimensionales Array, dessen Spalten unabhängig normalisiert werden.

    Returns
    -------
    np.ndarray
        Kopie von `X`, bei der jede Spalte nach Mittelwertabzug auf L2-Norm eins
        skaliert wurde. Nullspalten bleiben nach dem Zentrieren unverändert.
    """
    Y = X.copy()
    for i in range(Y.shape[1]):
        col = Y[:, i]
        col = col - np.mean(col)
        norm = np.linalg.norm(col)
        if norm > 0.0:
            Y[:, i] = col / norm
        else:
            Y[:, i] = col
    return Y


def _colored_noise_1d(n_steps: int, beta: float, rng: np.random.Generator) -> np.ndarray:
    """Erzeuge einen eindimensionalen farbigen Rauschprozess per Spektralskalierung.

    Parameters
    ----------
    n_steps : int
        Länge des zu erzeugenden Rauschsignals.
    beta : float
        Exponent des Potenzspektrums. Typische Werte sind `1` für pinkes und `2`
        für braunes Rauschen.
    rng : np.random.Generator
        Zufallszahlengenerator für die Weißrauschprobe.

    Returns
    -------
    np.ndarray
        Reeller Rauschvektor der Länge `n_steps`.
    """
    white = rng.normal(size=n_steps)
    X = np.fft.rfft(white)
    f = np.fft.rfftfreq(n_steps, d=1.0)
    scale = np.ones_like(f)
    nonzero = f > 0
    scale[nonzero] = 1.0 / np.power(f[nonzero], beta / 2.0)
    scale[~nonzero] = 0.0
    x = np.fft.irfft(X * scale, n=n_steps)
    return x


def _bandlimited_noise_1d(
    n_steps: int,
    f_min: float,
    f_max: float,
    rng: np.random.Generator,
    dt: float = 1.0,
) -> np.ndarray:
    """Erzeuge einen eindimensionalen Rauschprozess mit bandbegrenzter PSD.

    Die Leistungsdichte ist außerhalb von `[f_min, f_max]` exakt null; innerhalb
    des Bandes ist sie flach (weißes Rauschen im Band).

    Parameters
    ----------
    n_steps : int
        Länge des zu erzeugenden Rauschsignals.
    f_min : float
        Untere Grenzfrequenz des Bandes (inklusive).
    f_max : float
        Obere Grenzfrequenz des Bandes (inklusive).
    rng : np.random.Generator
        Zufallszahlengenerator für die Weißrauschprobe.
    dt : float, optional
        Abtastintervall, das die Frequenzachse (in Einheiten von `1/dt`)
        skaliert. Standardmäßig `1.0`.

    Returns
    -------
    np.ndarray
        Reeller Rauschvektor der Länge `n_steps`.

    Raises
    ------
    ValueError
        Falls `f_min` negativ ist oder `f_min > f_max` gilt.
    """
    if f_min < 0.0 or f_min > f_max:
        raise ValueError(f"Require 0 <= f_min <= f_max, got f_min={f_min}, f_max={f_max}.")
    white = rng.normal(size=n_steps)
    X = np.fft.rfft(white)
    f = np.fft.rfftfreq(n_steps, d=dt)
    mask = (f >= f_min) & (f <= f_max)
    x = np.fft.irfft(X * mask, n=n_steps)
    return x


def generate_noise_process(
    *,
    noise_type: Literal["white", "pink", "brown", "custom", "bandlimited"],
    n_steps: int,
    n_channels: int,
    seed: int | None,
    beta: float | None = None,
    f_min: float | None = None,
    f_max: float | None = None,
    dt: float = 1.0,
) -> np.ndarray:
    """Erzeuge normalisierte weiße, farbige oder bandbegrenzte Mehrkanal-Rauschprozesse.

    Parameters
    ----------
    noise_type : Literal["white", "pink", "brown", "custom", "bandlimited"]
        Typ des zu generierenden Rauschens. `"custom"` erzeugt farbiges Rauschen
        mit frei wählbarem Potenzspektrum-Exponenten `beta` (PSD ~ f^-beta).
        `"bandlimited"` erzeugt Rauschen, dessen PSD nur im Band
        `[f_min, f_max]` ungleich null ist.
    n_steps : int
        Anzahl der Zeitschritte pro Kanal.
    n_channels : int
        Anzahl paralleler Rauschkanäle.
    seed : int | None
        Optionaler Zufallsseed.
    beta : float | None, optional
        Potenzspektrum-Exponent für `noise_type="custom"`. Wird für andere
        `noise_type`-Werte ignoriert.
    f_min : float | None, optional
        Untere Grenzfrequenz des Bandes für `noise_type="bandlimited"`.
    f_max : float | None, optional
        Obere Grenzfrequenz des Bandes für `noise_type="bandlimited"`.
    dt : float, optional
        Abtastintervall zur Skalierung der Frequenzachse bei
        `noise_type="bandlimited"`. Standardmäßig `1.0`.

    Returns
    -------
    np.ndarray
        Matrix der Form ``(n_steps, n_channels)`` mit spaltenweise normalisierten
        Rauschprozessen.

    Raises
    ------
    ValueError
        Falls ein unbekannter `noise_type` angefordert wird, `beta` für
        `noise_type="custom"` fehlt, oder `f_min`/`f_max` für
        `noise_type="bandlimited"` fehlen.
    """
    rng = np.random.default_rng(seed)
    if noise_type == "white":
        raw = rng.normal(size=(n_steps, n_channels))
    elif noise_type == "pink":
        raw = np.column_stack([_colored_noise_1d(n_steps, beta=1.0, rng=rng) for _ in range(n_channels)])
    elif noise_type == "brown":
        raw = np.column_stack([_colored_noise_1d(n_steps, beta=2.0, rng=rng) for _ in range(n_channels)])
    elif noise_type == "custom":
        if beta is None:
            raise ValueError("beta must be set for noise_type='custom'.")
        raw = np.column_stack([_colored_noise_1d(n_steps, beta=beta, rng=rng) for _ in range(n_channels)])
    elif noise_type == "bandlimited":
        if f_min is None or f_max is None:
            raise ValueError("f_min and f_max must be set for noise_type='bandlimited'.")
        raw = np.column_stack(
            [_bandlimited_noise_1d(n_steps, f_min=f_min, f_max=f_max, rng=rng, dt=dt) for _ in range(n_channels)]
        )
    else:
        raise ValueError(f"Unsupported noise_type '{noise_type}'.")
    return _normalize_energy_per_column(raw)


DriftFn = Callable[[float, np.ndarray, np.ndarray], np.ndarray]
DiffusionFn = Callable[[float, np.ndarray], np.ndarray]
ShiftFn = Callable[[float, np.ndarray], np.ndarray]





def make_second_order_linear_drift(system: SecondOrderLinearSystem) -> DriftFn:
    """Erzeuge die Driftfunktion für ein lineares System zweiter Ordnung.

    Parameters
    ----------
    system : SecondOrderLinearSystem
        System mit Positions- und Geschwindigkeitskomponenten.

    Returns
    -------
    DriftFn
        Numba-kompilierte Funktion ``f(t, y, forcing)``, die die erste
        Ordnungsschreibweise des Systems inklusive additiver Anregung auswertet.
    """
    A = system.A.copy()
    B = system.B.copy()
    n = system.n_nodes

    @njit(cache=False)
    def drift(t: float, y: np.ndarray, forcing: np.ndarray) -> np.ndarray:
        x = y[:n]
        v = y[n:]
        dx = v + forcing[:n]
        acceleration = A @ x + B @ v + forcing[n:]
        return np.concatenate((dx, acceleration))

    return drift


def make_second_order_linear_diffusion(system: SecondOrderLinearSystem) -> DiffusionFn:
    """Erzeuge eine diagonale Einheitsdiffusion für ein System zweiter Ordnung.

    Parameters
    ----------
    system : SecondOrderLinearSystem
        System, dessen Zustandsdimension verwendet wird.

    Returns
    -------
    DiffusionFn
        Numba-kompilierte Funktion ``g(t, y)``, die einen Einsvektor der Länge
        der Zustandsdimension zurückgibt.
    """
    dim = system.dimension
    sigma = np.ones(dim, dtype=float)

    @njit(cache=False)
    def diffusion(t: float, y: np.ndarray) -> np.ndarray:
        return sigma

    return diffusion


def make_kuramoto_drift(system: SecondOrderKuramotoSystem) -> DriftFn:
    """Erzeuge die Driftfunktion für ein Kuramoto-System zweiter Ordnung.

    Parameters
    ----------
    system : SecondOrderKuramotoSystem
        Kuramoto-System mit Dämpfung, Trägheit, Eigenfrequenzen und Kopplungen.

    Returns
    -------
    DriftFn
        Numba-kompilierte Funktion ``f(t, y, forcing)``, die Phasen- und
        Geschwindigkeitsdynamik inklusive Kopplung und externer Anregung berechnet.
    """
    damping = system.damping.copy()
    inertia = system.inertia.copy()
    omega_nat = system.Omega.copy()
    coupling_matrix = system.coupling_matrix.copy()
    n = system.n_nodes

    @njit(cache=False)
    def drift(t: float, y: np.ndarray, forcing: np.ndarray) -> np.ndarray:
        theta = y[:n]
        xi = y[n:]

        phase_diff = theta[None, :] - theta[:, None]
        coupling = np.sum(coupling_matrix * np.sin(phase_diff), axis=1)
        dtheta = xi + forcing[:n]
        dxi = (-damping * xi + omega_nat + coupling) / inertia + forcing[n:]
        return np.concatenate((dtheta, dxi))

    return drift


def make_kuramoto_diffusion(system: SecondOrderKuramotoSystem) -> DiffusionFn:
    """Erzeuge eine diagonale Einheitsdiffusion für ein Kuramoto-System.

    Parameters
    ----------
    system : SecondOrderKuramotoSystem
        System, dessen Zustandsdimension verwendet wird.

    Returns
    -------
    DiffusionFn
        Numba-kompilierte Funktion ``g(t, y)``, die einen Einsvektor mit Länge
        der Zustandsdimension liefert.
    """
    dim = system.dimension
    sigma = np.ones(dim, dtype=float)

    @njit(cache=False)
    def diffusion(t: float, y: np.ndarray) -> np.ndarray:
        return sigma

    return diffusion


def make_linear_shift(shift_matrix_obj: ShiftMatrix) -> ShiftFn:
        """
        Create a non-python/numba compiling linear shift function that applies the shift matrix to the system's state vector.
        
        """
        n = shift_matrix_obj.shift_matrix.shape[0]
        theta_star = shift_matrix_obj.model.compute_fixed_point()

        @njit
        def linear_shift(t: float, y: np.ndarray) -> np.ndarray:
            """
            Apply a linear shift to the system's state vector.

            Parameters
            ----------
            t : float
                Current time (not used in this function).
            y : np.ndarray
                Current state vector of the system.

            Returns
            -------
            np.ndarray
                The shifted state vector.
            """

            theta = y[:n]
            xi = y[n:]
            
            dtheta = np.zeros_like(theta)
            dxi = shift_matrix_obj.shift_matrix @ (theta - theta_star) # is this the right one?
            return np.concatenate((dtheta, dxi))
            

        return linear_shift


def build_system_drift_diffusion(
    system:  SecondOrderLinearSystem | SecondOrderKuramotoSystem ,
) -> tuple[DriftFn, DiffusionFn]:
    """Wähle passende Drift- und Diffusionsfunktionen für ein System aus.

    Parameters
    ----------
    system : SecondOrderLinearSystem | SecondOrderKuramotoSystem
        Dynamisches System, für das numerisch integriert werden soll.

    Returns
    -------
    tuple[DriftFn, DiffusionFn]
        Tupel aus Drift- und Diffusionsfunktion. Für unbekannte Subtypen wird eine
        generische Drift über `system.rhs` und eine diagonale Einheitsdiffusion genutzt.
    """

    if isinstance(system, SecondOrderLinearSystem):
        return make_second_order_linear_drift(system), make_second_order_linear_diffusion(system)
    if isinstance(system, SecondOrderKuramotoSystem):
        return make_kuramoto_drift(system), make_kuramoto_diffusion(system)

    def generic_drift(t: float, y: np.ndarray, forcing: np.ndarray) -> np.ndarray:
        return system.rhs(t, y, forcing=forcing)

    dim = system.dimension
    sigma = np.ones(dim, dtype=float)

    def generic_diffusion(t: float, y: np.ndarray) -> np.ndarray:
        del t, y
        return sigma

    return generic_drift, generic_diffusion


@njit(cache=False)
def _euler_maruyama_core_numba(
    y0: np.ndarray,
    drift_fn,
    diffusion_fn,
    dt: float,
    deterministic_forcing: np.ndarray,
    noise_increments: np.ndarray,
) -> np.ndarray:
    """Numba-kompilierter Euler-Maruyama-Kern für diagonale Diffusion.

    Wird nur für Drift-/Diffusionsfunktionen genutzt, die selbst
    numba-kompiliert sind (z. B. aus `make_second_order_linear_drift` oder
    `make_kuramoto_drift`); die komplette Zeitschleife läuft dann im
    nopython-Modus statt in Python.
    """
    n_steps = deterministic_forcing.shape[0] + 1
    dim = y0.shape[0]
    y = np.empty((n_steps, dim))
    y[0] = y0
    for k in range(n_steps - 1):
        tk = 0.0
        yk = y[k]
        drift = drift_fn(tk, yk, deterministic_forcing[k])
        G = diffusion_fn(tk, yk)
        y[k + 1] = yk + dt * drift + G * noise_increments[k]
    return y


def euler_maruyama_solver(
    t: np.ndarray,
    y0: np.ndarray,
    drift_fn: DriftFn,
    diffusion_fn: DiffusionFn,
    deterministic_forcing: np.ndarray,
    noise_increments: np.ndarray,
) -> np.ndarray:
    """Integriere ein stochastisches System mit dem Euler-Maruyama-Verfahren.

    Parameters
    ----------
    t : np.ndarray
        Äquidistantes Zeitgitter der Form ``(n_steps,)``.
    y0 : np.ndarray
        Anfangszustand der Form ``(dim,)``.
    drift_fn : DriftFn
        Driftfunktion ``f(t, y, forcing)``.
    diffusion_fn : DiffusionFn
        Diffusionsfunktion ``g(t, y)``. Zulässig sind diagonale Ausgaben der Form
        ``(q,)`` oder Matrixausgaben der Form ``(dim, q)``.
    deterministic_forcing : np.ndarray
        Deterministische Inkremente pro Schritt als Matrix der Form ``(n_steps-1, dim)``.
    noise_increments : np.ndarray
        Stochastische Inkremente der Form ``(n_steps-1, q)``.

    Returns
    -------
    np.ndarray
        Simulierte Zustandsmatrix der Form ``(n_steps, dim)``.

    Raises
    ------
    ValueError
        Falls Zeitgitter, Drift- oder Diffusionsausgaben nicht zu den erwarteten
        Dimensionen passen.

    Notes
    -----
    Sind `drift_fn` und `diffusion_fn` numba-kompilierte Dispatcher (wie von
    `make_second_order_linear_drift`/`make_kuramoto_drift` erzeugt) und liefert
    `diffusion_fn` eine diagonale (1D) Diffusion, läuft die komplette
    Zeitschleife über `_euler_maruyama_core_numba` im nopython-Modus. Andernfalls
    (z. B. generische Systeme oder Matrixdiffusion) wird auf die reine
    Python-Schleife zurückgefallen.
    """
    dt = _validate_uniform_time(t)
    n_steps = t.size
    dim = y0.shape[0]
    if deterministic_forcing.shape != (n_steps - 1, dim):
        raise ValueError(
            "deterministic_forcing must have shape (len(t)-1, dim). "
            f"Got {deterministic_forcing.shape}, expected {(n_steps - 1, dim)}."
        )
    if noise_increments.ndim != 2 or noise_increments.shape[0] != (n_steps - 1):
        raise ValueError("noise_increments must have shape (len(t)-1, q).")

    # Schneller Pfad: Drift und Diffusion sind numba-kompilierte Dispatcher
    # (siehe make_second_order_linear_drift/make_kuramoto_drift) und die
    # Diffusion ist diagonal -> gesamte Zeitschleife läuft nopython-kompiliert.
    if isinstance(drift_fn, _NumbaDispatcher) and isinstance(diffusion_fn, _NumbaDispatcher):
        G0 = np.asarray(diffusion_fn(t[0], y0), dtype=float)
        if G0.ndim == 1 and G0.shape[0] == noise_increments.shape[1]:
            return _euler_maruyama_core_numba(
                np.ascontiguousarray(y0, dtype=float),
                drift_fn,
                diffusion_fn,
                dt,
                np.ascontiguousarray(deterministic_forcing, dtype=float),
                np.ascontiguousarray(noise_increments, dtype=float),
            )

    y = np.zeros((n_steps, dim), dtype=float)
    y[0] = y0

    for k in range(n_steps - 1):
        tk = t[k]
        yk = y[k]
        drift = np.asarray(drift_fn(tk, yk, deterministic_forcing[k]), dtype=float)
        if drift.shape != (dim,):
            raise ValueError(f"drift output must have shape ({dim},), got {drift.shape}.")

        G = np.asarray(diffusion_fn(tk, yk), dtype=float)
        if G.ndim == 1:
            if G.shape[0] != noise_increments.shape[1]:
                raise ValueError(
                    "For diagonal diffusion, diffusion length must match noise increment width."
                )
            noise_term = G * noise_increments[k]
        elif G.ndim == 2:
            if G.shape[0] != dim or G.shape[1] != noise_increments.shape[1]:
                raise ValueError(
                    "For matrix diffusion, expected shape "
                    f"({dim}, {noise_increments.shape[1]}), got {G.shape}."
                )
            noise_term = G @ noise_increments[k]
        else:
            raise ValueError("diffusion output must be 1D or 2D.")

        y[k + 1] = yk + dt * drift + noise_term
    return y


class Simulation:
    """Kapselt die Simulation deterministisch und stochastisch angeregter Systeme."""

    def __init__(
        self,
        *,
        system: SecondOrderLinearSystem | SecondOrderKuramotoSystem,
        deterministic_forcings: list[DeterministicForcingSpec] | None = None,
        impulse_forcings: list[ImpulseForcingSpec] | None = None,
        stochastic_forcings: list[StochasticForcingSpec] | None = None,
    ) -> None:
        """Initialisiere eine Simulation für ein dynamisches System.

        Parameters
        ----------
        system : LinearSystem | SecondOrderLinearSystem | SecondOrderKuramotoSystem | GRNMrnaProteinSystem | TippingNormalForms
            Das zu simulierende dynamische System.
        deterministic_forcings : list[DeterministicForcingSpec] | None, optional
            Liste deterministischer harmonischer Anregungen.
        impulse_forcings : list[ImpulseForcingSpec] | None, optional
            Liste deterministischer Impulsanregungen.
        stochastic_forcings : list[StochasticForcingSpec] | None, optional
            Liste stochastischer Anregungen.
        """
        self.system = system
        self.deterministic_forcings = deterministic_forcings or []
        self.impulse_forcings = impulse_forcings or []
        self.stochastic_forcings = stochastic_forcings or []

    def _component_indices(self, component: str | None, target_nodes: list[int]) -> np.ndarray:
        """Mappe Knoten und Komponentennamen auf Zustandsindizes.

        Parameters
        ----------
        component : str | None
            Name der angeregten Komponente. Die zulässigen Werte hängen vom
            Systemtyp ab.
        target_nodes : list[int]
            Global indizierte Knoten, auf die die Anregung wirkt.

        Returns
        -------
        np.ndarray
            Array mit Zustandsindizes innerhalb des vollständigen Zustandsvektors.

        Raises
        ------
        ValueError
            Falls Knotenindizes negativ, außerhalb des gültigen Bereichs oder
            mit der gewählten Komponente inkompatibel sind.
        """
        target = np.asarray(target_nodes, dtype=int)
        if target.size == 0:
            return target
        if np.any(target < 0):
            raise ValueError("target_nodes must be non-negative.")

        if False:
            if isinstance(self.system, LinearSystem):
                if np.any(target >= self.system.dimension):
                    raise ValueError("target_nodes out of bounds for LinearSystem.")
                if component not in (None, "state"):
                    raise ValueError("LinearSystem only supports component=None or 'state'.")
                return target

        if isinstance(self.system, SecondOrderLinearSystem):
            n = self.system.n_nodes
            if np.any(target >= n):
                raise ValueError("target_nodes out of bounds for SecondOrderLinearSystem.")
            comp = "acceleration" if component is None else component
            if comp in ("position", "x"):
                return target
            if comp in ("velocity", "v", "acceleration"):
                return target + n
            raise ValueError(
                "SecondOrderLinearSystem component must be None/'position'/'x'/'velocity'/'v'/'acceleration'."
            )

        n = self.system.n_nodes
        if np.any(target >= n):
            raise ValueError("target_nodes out of bounds for SecondOrderKuramotoSystem.")
        comp = "theta" if component is None else component
        if comp == "theta":
            return target
        if comp == "xi":
            return target + n
        raise ValueError("Kuramoto component must be None/'theta'/'xi'.")

    def _deterministic_forcing_at_time(self, t: float) -> np.ndarray:
        """Berechne den deterministischen Anregungsvektor zu einem Zeitpunkt.

        Parameters
        ----------
        t : float
            Zeitpunkt, an dem die Summe aller harmonischen und impulsartigen
            Anregungen ausgewertet wird.

        Returns
        -------
        np.ndarray
            Zustandsvektor der Länge `system.dimension` mit additiver Anregung.

        Raises
        ------
        ValueError
            Falls Frequenz-, Amplituden- oder Phasenlisten inkonsistente Längen besitzen.
        """
        force = np.zeros(self.system.dimension, dtype=float)
        for spec in self.deterministic_forcings:
            if len(spec.frequencies) != len(spec.amplitudes):
                raise ValueError("frequencies and amplitudes must have same length.")
            phases = spec.phases if spec.phases is not None else [0.0] * len(spec.frequencies)
            if len(phases) != len(spec.frequencies):
                raise ValueError("phases and frequencies must have same length.")
            idx = self._component_indices(spec.component, spec.target_nodes)
            if idx.size == 0:
                continue
            val = 0.0
            for w, a, p in zip(spec.frequencies, spec.amplitudes, phases):
                val += float(a) * np.sin(float(w) * t + float(p))
            force[idx] += val


        for spec in self.impulse_forcings:
            idx = self._component_indices(spec.component, spec.target_nodes)
            if idx.size == 0:
                continue
            val = evaluate_impulse_profile(
                t,
                pulse_type=spec.pulse_type,
                amplitude=spec.amplitude,
                center=spec.center,
                width=spec.width,
            )
            force[idx] += float(np.asarray(val, dtype=float))
        return force

    def _build_deterministic_matrix(self, t: np.ndarray) -> np.ndarray:
        """Baue die deterministische Anregungsmatrix über das gesamte Zeitgitter auf.

        Parameters
        ----------
        t : np.ndarray
            Zeitgitter der Form ``(n_steps,)``.

        Returns
        -------
        np.ndarray
            Matrix der Form ``(n_steps, dimension)`` mit deterministischen Anregungen.
        """
        return np.vstack([self._deterministic_forcing_at_time(tk) for tk in t])

    def _build_stochastic_matrix(self, t: np.ndarray) -> np.ndarray:
        """Baue die stochastischen Anregungen für alle Simulationsschritte auf.

        Parameters
        ----------
        t : np.ndarray
            Zeitgitter der Form ``(n_steps,)``.

        Returns
        -------
        np.ndarray
            Matrix der Form ``(n_steps-1, dimension)`` mit stochastischen
            Kanalwerten vor der Skalierung mit ``sqrt(dt)``.
        """
        n_steps = t.size
        noise = np.zeros((n_steps - 1, self.system.dimension), dtype=float)
        for spec in self.stochastic_forcings:
            idx = self._component_indices(spec.component, spec.target_nodes)
            if idx.size == 0:
                continue
            proc = generate_noise_process(
                noise_type=spec.noise_type,
                n_steps=n_steps - 1,
                n_channels=idx.size,
                seed=spec.seed,
                beta=spec.beta,
                f_min=spec.f_min,
                f_max=spec.f_max,
                dt=float(t[1] - t[0]),
            )
            print("spec.strength: ",spec.strength)
            noise[:, idx] += float(spec.strength) * proc
        return noise

    def _simulate_deterministic(
        self, t: np.ndarray, y0: np.ndarray, rtol: float, atol: float, method: str, max_step: float | 1
    ) -> np.ndarray:
        """Integriere ein rein deterministisch angeregtes System mit `solve_ivp`.

        Parameters
        ----------
        t : np.ndarray
            Zeitgitter der Form ``(n_steps,)``.
        y0 : np.ndarray
            Anfangszustand der Form ``(dimension,)``.
        rtol : float
            Relative Toleranz für `solve_ivp`.
        atol : float
            Absolute Toleranz für `solve_ivp`.
        method : str
            Name des von `solve_ivp` verwendeten Integrationsverfahrens.

        Returns
        -------
        np.ndarray
            Zustandsmatrix der Form ``(n_steps, dimension)``.

        Raises
        ------
        RuntimeError
            Falls `solve_ivp` keine erfolgreiche Lösung findet.
        """
        def f(tt: float, yy: np.ndarray) -> np.ndarray:
            return self.system.rhs(tt, yy, forcing=self._deterministic_forcing_at_time(tt))

        solve_kwargs: dict[str, Any] = {
            "t_eval": t,
            "rtol": rtol,
            "atol": atol,
            "method": method,
        }
        if max_step is not None:
            solve_kwargs["max_step"] = max_step
        sol = solve_ivp(f, (t[0], t[-1]), y0, **solve_kwargs)
        if not sol.success:
            raise RuntimeError(f"solve_ivp failed: {sol.message}")
        return sol.y.T

    def _simulate_stochastic_or_mixed(
        self, t: np.ndarray, y0: np.ndarray, deterministic_matrix: np.ndarray, noise_matrix: np.ndarray
    ) -> np.ndarray:
        """Integriere ein stochastisches oder gemischt angeregtes System.

        Parameters
        ----------
        t : np.ndarray
            Äquidistantes Zeitgitter der Form ``(n_steps,)``.
        y0 : np.ndarray
            Anfangszustand der Form ``(dimension,)``.
        deterministic_matrix : np.ndarray
            Deterministische Anregung über die Zeit.
        noise_matrix : np.ndarray
            Stochastische Kanalwerte pro Zeitschritt.

        Returns
        -------
        np.ndarray
            Zustandsmatrix der Form ``(n_steps, dimension)``.
        """
        dt = _validate_uniform_time(t)
        drift_fn, diffusion_fn = build_system_drift_diffusion(self.system)
        dW = np.sqrt(dt) * noise_matrix
        return euler_maruyama_solver(
            t=t,
            y0=y0,
            drift_fn=drift_fn,
            diffusion_fn=diffusion_fn,
            deterministic_forcing=deterministic_matrix[:-1],
            noise_increments=dW,
        )

    def run(
        self,
        *,
        t_span: tuple[float, float],
        dt: float,
        y0: np.ndarray | None = None,
        method_det: str = "RK45",
        max_step_det: float | None = None,
        rtol: float = 1e-6,
        atol: float = 1e-9,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Führe die Simulation auf einem gleichmäßigen Zeitgitter aus.

        Parameters
        ----------
        t_span : tuple[float, float]
            Start- und Endzeit der Simulation.
        dt : float
            Positiver Zeitschritt des Ausgabegitters.
        y0 : np.ndarray | None, optional
            Optionaler Anfangszustand der Form ``(dimension,)``. Falls `None`,
            wird `system.state` verwendet.
        method_det : str, default="RK45"
            Deterministisches Integrationsverfahren für `solve_ivp`, sofern keine
            stochastische Anregung vorliegt.
        rtol : float, default=1e-6
            Relative Toleranz für `solve_ivp`.
        atol : float, default=1e-9
            Absolute Toleranz für `solve_ivp`.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, dict[str, Any]]
            Tupel ``(t, y, meta)`` mit Zeitgitter, Zustandsmatrix der Form
            ``(n_steps, dimension)`` und einem Metadaten-Wörterbuch zur verwendeten
            Anregung und Numerik.

        Raises
        ------
        ValueError
            Falls `dt`, `t_span` oder `y0` ungültig sind.
        """
        if dt <= 0:
            raise ValueError("dt must be > 0.")
        t0, tf = float(t_span[0]), float(t_span[1])
        if tf <= t0:
            raise ValueError("t_span must satisfy tf > t0.")

        t = np.arange(t0, tf + 0.5 * dt, dt, dtype=float)
        if y0 is None:
            y0_arr = self.system.state
        else:
            y0_arr = np.asarray(y0, dtype=float)
            if y0_arr.shape != (self.system.dimension,):
                raise ValueError(f"y0 must have shape ({self.system.dimension},), got {y0_arr.shape}.")

        deterministic_matrix = self._build_deterministic_matrix(t)
        stochastic_matrix = self._build_stochastic_matrix(t)
        has_stochastic = bool(self.stochastic_forcings)

        if has_stochastic:
            y = self._simulate_stochastic_or_mixed(t, y0_arr, deterministic_matrix, stochastic_matrix)
            solver_used = "Euler-Maruyama-drift-diffusion"
        else:
            y = self._simulate_deterministic(t, y0_arr, rtol, atol, method_det, max_step=max_step_det)
            solver_used = f"solve_ivp-{method_det}"

        self.system.state = y[-1]
        meta = {
            "solver_used": solver_used,
            "dt": dt,
            "n_steps": int(t.size),
            "deterministic_forcings": [asdict(s) for s in self.deterministic_forcings],
            "impulse_forcings": [asdict(s) for s in self.impulse_forcings],
            "stochastic_forcings": [asdict(s) for s in self.stochastic_forcings],
            "deterministic_forcing": deterministic_matrix,
            "stochastic_noise": stochastic_matrix,
            "stochastic_increments": stochastic_matrix * np.sqrt(dt),
            "total_forcing_increment": deterministic_matrix[:-1] * dt + stochastic_matrix * np.sqrt(dt),
            "noise_normalization": "per-channel equal total energy (unit L2 before strength scaling)",
        }
        return t, y, meta

def use_euler_maruyama(model, t_span, dt, y0, pert, shift_matrix_object=None, seed=42):
    """
    Drop in replacement for the scipy ODE solve using Ai generated code. 
    Returns t, y, meta
    """

    print("Proceeding with numerical integration inside the Euler-Maruyama solver.")
    type = pert[0]
    amplitudes, freqs, target_nodes, onset_t = pert[1]

    kur_system = SecondOrderKuramotoSystem(
                            damping=np.array([model.damping_coefficient]*len(model.power_vector)),
                            inertia=np.array([1.] * len(model.power_vector)),
                            Omega=model.power_vector, 
                            coupling_matrix=model.connectivity_matrix,
                            _state=y0.copy())

    if type=="white":
        sim_kur = Simulation(
            system=kur_system,
            stochastic_forcings=[
                    StochasticForcingSpec(
                    target_nodes=target_nodes, noise_type=type, strength=amplitudes, seed=seed, component="xi")
            ]
        )
    else:
        sim_kur = Simulation(
                    system=kur_system,
                    deterministic_forcings=[
                            DeterministicForcingSpec(
                            target_nodes=target_nodes, amplitudes=amplitudes, frequencies=freqs)
                    ]
                )

    return sim_kur.run(t_span=t_span, dt=dt)