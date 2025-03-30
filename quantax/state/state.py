from __future__ import annotations
from typing import Optional, Union, Tuple
from jaxtyping import PyTree
from numbers import Number
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
        :param symm: The symmetry of the state, default to `quantax.symmetry.Identity`
        """
        self._symm = symm if symm is not None else Identity()

    @property
    def N(self) -> int:
        """Number of sites"""
        return self.symm.N

    @property
    def nstates(self) -> int:
        return self.symm.nstates

    @property
    def dtype(self) -> np.dtype:
        return get_default_dtype()

    @property
    def symm(self) -> Symmetry:
        """Symmetry of the state"""
        return self._symm

    @property
    def nsymm(self) -> int:
        """Number of symmetry group elements"""
        return self.symm.nsymm

    @property
    def basis(self) -> spin_basis_general:
        """Quspin basis of the state"""
        return self.symm.basis

    @property
    def Nparticle(self) -> Optional[Tuple[int, int]]:
        """Number of particle convervation of the state"""
        return self.symm.Nparticle

    def __call__(self, fock_states: _Array) -> _Array:
        r"""
        Evaluate the wave function :math:`\psi(s) = \left<s|\psi\right>` by ``state(s)``

        :param fock_states: A batch of fock states with entries :math:`\pm 1`
        """
        return NotImplemented

    def __getitem__(self, basis_ints: _Array) -> _Array:
        r"""
        Evaluate the wave function :math:`\psi(s) = \left<s|\psi\right>` by ``state[s]``

        :param basis_ints: A batch of basis integers
        """
        fock_states = ints_to_array(basis_ints)
        psi = self(fock_states)
        return psi

    def init_internal(self, x: jax.Array) -> PyTree:
        return None

    def ref_forward_with_updates(
        self, s: _Array, s_old: jax.Array, nflips: int, internal: PyTree
    ) -> Tuple[jax.Array, PyTree]:
        return self(s), None

    def ref_forward(
        self,
        s: _Array,
        s_old: jax.Array,
        nflips: int,
        idx_segment: jax.Array,
        internal: PyTree,
    ) -> jax.Array:
        return self(s)

    def __array__(self) -> np.ndarray:
        return np.asarray(self.todense().wave_function)

    def __jax_array__(self) -> jax.Array:
        return jnp.asarray(self.todense().wave_function)

    def todense(self, symm: Optional[Symmetry] = None) -> DenseState:
        r"""
        Obtain the `quantax.state.DenseState` corresponding to the current state

        :param symm: The symmetry of the state, default to the current symmetry of the state

        .. warning::

            Users are responsible to ensure that the state satisfies the given ``symm``.
        """
        if symm is None:
            symm = self.symm
        symm.basis_make()
        basis = symm.basis
        basis_ints = basis.states
        wf = self[basis_ints]
        symm_norm = basis.get_amp(basis_ints)
        if np.isrealobj(wf):
            symm_norm = symm_norm.real
        return DenseState(wf / symm_norm, symm)

    def norm(self, ord: Optional[int] = None) -> float:
        r"""
        `Norm <https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html>`_
        of state

        :param ord: Order of the norm, default to 2-norm :math:`\sqrt{\sum_s |\psi(s)|^2}`
        """
        return np.linalg.norm(self.todense().wave_function, ord=ord).item()

    def __matmul__(self, other: State) -> Number:
        r"""
        Compute the contraction :math:`\left< \psi|\phi \right>` by ``self @ other``.
        This is implemented by converting ``self`` and ``other`` to `~quantax.state.DenseState`.
        """
        if not isinstance(other, State):
            return NotImplemented

        if self.symm is other.symm:
            symm = self.symm
        else:
            symm = Identity()
        wf_self = self.todense(symm).wave_function
        wf_other = other.todense(symm).wave_function
        return np.vdot(wf_self, wf_other).item()

    def overlap(self, other: State) -> Number:
        r"""
        Overlap between two states. Equal to ``self @ other`` if the two states are
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
    
    def expectation(self, operator, samples):
        return operator.expectation(self, samples)


class DenseState(State):
    """Dense state in which the full wave function is stored as a numpy array"""

    def __init__(self, wave_function: _Array, symm: Optional[Symmetry] = None):
        """
        :param wave_function: Full wave function given according to the
            `basis.states order in QuSpin
            <https://quspin.github.io/QuSpin/generated/quspin.basis.spin_basis_general.html#quspin.basis.spin_basis_general.states>`_

        :param symm: The symmetry of the wave function, default to `quantax.symmetry.Identity`
        """
        if symm is None:
            symm = Identity()
        super().__init__(symm)
        symm.basis_make()
        self._wave_function = wave_function.astype(get_default_dtype()).flatten()
        if wave_function.size != self.basis.Ns:
            raise ValueError(
                "Input wave_function size doesn't match the Hilbert space dimension."
            )

    @property
    def wave_function(self) -> _Array:
        """Full wave function"""
        return self._wave_function

    def __repr__(self) -> str:
        return self.wave_function.__repr__()

    def todense(self, symm: Optional[Symmetry] = None) -> DenseState:
        """
        Convert the state to a new ``DenseState`` with the given symmetry

        :param symm: The new symmetry. It's default to ``self.symm``, so ``self``
            without copy is returned by default.
        """
        if symm is None or symm is self.symm:
            return self
        return super().todense(symm)

    def normalize(self) -> DenseState:
        """
        Normalize the wave function, and return ``self``.
        """
        self._wave_function /= self.norm()
        return self

    def __getitem__(self, basis_ints: _Array) -> np.ndarray:
        r"""
        Evaluate the wave function :math:`\psi(s) = \left<s|\psi\right>` by ``state[s]``.
        This is done by slicing the full wave function.

        :param basis_ints: A batch of basis integers
        """
        basis_ints = np.asarray(basis_ints, dtype=self.basis.dtype, order="C")
        batch_shape = basis_ints.shape
        basis_ints = basis_ints.flatten()

        symm_norm = self.basis.get_amp(basis_ints, mode="full_basis")
        basis_ints, sign = self.basis.representative(basis_ints, return_sign=True)
        symm_norm[np.isnan(symm_norm)] = 0
        if np.isrealobj(self.wave_function) and np.allclose(symm_norm.imag, 0.0):
            symm_norm = symm_norm.real

        # search for index of representatives from states
        states = self.basis.states[::-1]
        index = np.searchsorted(states, basis_ints)

        # whether basis_ints is found in basis.states
        # index % states.size to avoid out-of-range
        is_found = basis_ints == states[index % states.size]
        index = states.size - 1 - index

        wf = sign * symm_norm * np.where(is_found, self.wave_function[index], 0.0)
        return wf.reshape(batch_shape)

    def __call__(self, fock_states: _Array) -> np.ndarray:
        r"""
        Evaluate the wave function :math:`\psi(s) = \left<s|\psi\right>` by ``state(s)``.
        This is done by converting fock states basis integers and
        slicing the full wave function.

        :param fock_states: A batch of fock states with entries :math:`\pm 1`
        """
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
