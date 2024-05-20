from __future__ import annotations
from typing import Optional, Union
from numbers import Number
from numpy.typing import ArrayLike
import numpy as np
import jax
import jax.numpy as jnp
from quspin.basis import spin_basis_general
from ..symmetry import Symmetry, Identity
from ..utils import ints_to_array, array_to_ints
from ..global_defs import get_default_dtype


_Array = Union[np.ndarray, jax.Array]


class State:
    """Abstract class for quantum states"""

    def __init__(self, symm: Optional[Symmetry] = None):
        """
        Initiate some properties according to the typical data in the inherent class.
        """
        self._symm = symm if symm is not None else Identity()

    @property
    def nsites(self) -> int:
        """Number of sites"""
        return self.symm.nsites

    @property
    def dtype(self) -> np.dtype:
        return get_default_dtype()

    @property
    def symm(self) -> Symmetry:
        return self._symm

    @property
    def nsymm(self) -> int:
        return self.symm.nsymm

    @property
    def basis(self) -> spin_basis_general:
        return self.symm.basis

    @property
    def total_sz(self) -> Optional[int]:
        return self.symm.total_sz

    def __call__(
        self, fock_states: _Array, **kwargs
    ) -> _Array:
        """
        Evaluate the wave function psi(s) = <s|psi>. The input is interpreted as fock
        states with entries -1 and 1.
        """

    def __getitem__(self, basis_ints: ArrayLike) -> _Array:
        """
        Evaluate the wave function psi(s) = <s|psi>. The input is interpreted as basis
        integers.
        """
        fock_states = ints_to_array(basis_ints)
        psi = self(fock_states)
        return psi

    def __array__(self) -> np.ndarray:
        return np.asarray(self.todense().wave_function)

    def __jax_array__(self) -> jax.Array:
        return self.todense().wave_function

    def todense(self, symm: Optional[Symmetry] = None) -> DenseState:
        if symm is None:
            symm = self.symm
        basis = symm.basis
        basis.make()
        basis_ints = basis.states
        wf = self[basis_ints]
        symm_norm = basis.get_amp(basis_ints)
        if np.isrealobj(wf):
            symm_norm = symm_norm.real
        return DenseState(wf / symm_norm, symm)

    def norm(self, ord: Optional[int] = None) -> float:
        """Norm of state. Default to 2-norm: sqrt(sum(|psi|**2))"""
        return np.linalg.norm(self.todense().wave_function, ord=ord).item()

    def __matmul__(self, other: State) -> Number:
        """
        Default quantum state contraction <self|other> when the contraction is not
        customized.
        """
        if not isinstance(other, State):
            return NotImplemented

        if self.symm is other.symm:
            symm = self.symm
        else:
            symm = Identity()
        wf_self = self.todense(symm).wave_function
        wf_other = other.todense(symm).wave_function
        return jnp.vdot(wf_self, wf_other).item()

    def overlap(self, other: State) -> Number:
        """
        Overlap between two states. Equal to contraction if the two states are
        normalized.
        """
        if self.symm is other.symm:
            symm = self.symm
        else:
            symm = Identity()
        wf_self = self.todense(symm).wave_function
        wf_self /= np.linalg.norm(wf_self)
        wf_other = other.todense(symm).wave_function
        wf_other /= np.linalg.norm(wf_other)
        return np.vdot(wf_self, wf_other).item()


class DenseState(State):
    """Dense state with symmetries."""

    def __init__(
        self,
        wave_function: Union[np.ndarray, jax.Array],
        symm: Optional[Symmetry] = None,
    ):
        """
        Constructs a dense state with symmetries.

        Args:
            wave_function: Wave function component.
            symm: The symmetry of the wave function.
        """
        if symm is None:
            symm = Identity()
        super().__init__(symm)
        self.basis.make()
        wave_function = np.asarray(wave_function, dtype=get_default_dtype(), order="C")
        self._wave_function = wave_function.flatten()
        if wave_function.size != self.basis.states.size:
            raise ValueError(
                "Input wave_function size doesn't match the Hilbert space dimension."
            )

    @property
    def wave_function(self) -> np.ndarray:
        """Wave function in self.basis.states order"""
        return self._wave_function

    def __repr__(self) -> str:
        return self.wave_function.__repr__()

    def todense(self, symm: Optional[Symmetry] = None) -> DenseState:
        if symm is None or symm is self.symm:
            return self
        return super().todense(symm)

    def normalize(self) -> DenseState:
        self._wave_function /= self.norm()
        return self

    def __getitem__(self, basis_ints: ArrayLike) -> np.ndarray:
        basis_ints = np.asarray(basis_ints, dtype=self.basis.dtype, order="C")
        batch_shape = basis_ints.shape
        basis_ints = basis_ints.flatten()

        # obtain rescale factor between symmetry and full basis
        # basis_ints updated in-place to representatives
        symm_norm = self.basis.get_amp(basis_ints, mode="full_basis")
        if np.isrealobj(self.wave_function) and np.allclose(symm_norm.imag, 0.0):
            symm_norm = symm_norm.real

        # search for index of representatives from states
        states = self.basis.states[::-1]
        index = np.searchsorted(states, basis_ints)
        # whether basis_ints is found in basis.states
        # index % states.size to avoid out-of-range
        is_found = basis_ints == states[index % states.size]
        index = states.size - 1 - index

        wf = symm_norm * np.where(is_found, self.wave_function[index], 0.0)
        return wf.reshape(batch_shape)

    def __call__(
        self, fock_states: _Array, **kwargs
    ) -> np.ndarray:
        basis_ints = array_to_ints(fock_states)
        return self[basis_ints]

    def __add__(self, other: DenseState) -> DenseState:
        if isinstance(other, DenseState) and self.symm is other.symm:
            return DenseState(self.wave_function + other.wave_function, self._symm)
        else:
            raise RuntimeError("Invalid addition.")

    def __sub__(self, other: DenseState) -> DenseState:
        if isinstance(other, DenseState) and self.symm is other.symm:
            return DenseState(self.wave_function - other.wave_function, self._symm)
        else:
            raise RuntimeError("Invalid subtraction.")

    def __mul__(self, other: Number) -> DenseState:
        return DenseState(self.wave_function * other, self._symm)

    def __rmul__(self, other: Number) -> DenseState:
        return self.__mul__(other)

    def __truediv__(self, other: Number) -> DenseState:
        return DenseState(self.wave_function / other, self._symm)

    def __rtruediv__(self, other: Number) -> DenseState:
        return DenseState(other / self.wave_function, self._symm)
