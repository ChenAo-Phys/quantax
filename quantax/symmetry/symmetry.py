from __future__ import annotations
from functools import partial
from typing import Sequence, Optional, Union, Tuple
import numpy as np
import jax
import jax.numpy as jnp
from quspin.basis import spin_basis_general
from ..global_defs import get_sites, get_default_dtype, is_default_cpl


def _get_perm(generator: np.ndarray, sector: list) -> Tuple[jax.Array, jax.Array]:
    nsites = generator.shape[1]
    if np.array_equiv(generator, np.arange(nsites)):
        perm = jnp.arange(nsites)[None]
        eigval = jnp.array([1], dtype=get_default_dtype())
        return perm, eigval

    s0 = jnp.arange(nsites, dtype=jnp.uint16)
    perm = s0.reshape(1, -1)
    eigval = jnp.array([1], dtype=get_default_dtype())

    for g, sec in zip(generator, sector):
        g = jnp.asarray(g, dtype=jnp.uint16)
        new_perm = [s0]
        s_perm = g
        while not jnp.array_equal(s0, s_perm):
            new_perm.append(s_perm)
            s_perm = s_perm[g]
        new_perm = jnp.stack(new_perm, axis=0)
        nperms = new_perm.shape[0]
        if not isinstance(sec, int):
            new_eigval = jnp.asarray(sec)
        else:
            if not 0 <= sec < nperms:
                raise ValueError(f"Sector {sec} out of range.")
            if not is_default_cpl():
                if (sec * 2) % nperms != 0:
                    raise ValueError(
                        "Default dtype is real, but got complex characters."
                    )
                character = -1 if sec else 1
            else:
                character = np.exp(-2j * np.pi * sec / nperms)
            new_eigval = character ** jnp.arange(nperms)

        perm = perm[:, new_perm].reshape(-1, nsites)
        eigval = jnp.einsum("i,j->ij", eigval, new_eigval).flatten()

    return perm, eigval


class Symmetry:
    def __init__(
        self,
        generator: Optional[Sequence] = None,
        sector: Union[int, Sequence] = 0,
        spin_inversion: int = 0,
        total_sz: Optional[int] = None,
        perm: Optional[jax.Array] = None,
        eigval: Optional[jax.Array] = None,
    ):
        nsites = get_sites().nsites
        if generator is None:
            generator = np.atleast_2d(np.arange(nsites, dtype=np.uint16))
        else:
            generator = np.atleast_2d(generator).astype(np.uint16)
            if generator.shape[1] != nsites:
                raise ValueError(
                    f"Got a generator with size {generator.shape[1]}, but it should be"
                    f"the same as the system size {nsites}."
                )
        self._generator = generator
            
        self._nsites = self._generator.shape[1]
        self._sector = np.asarray(sector, dtype=np.uint16).flatten().tolist()
        self._spin_inversion = spin_inversion
        if total_sz is not None and (total_sz + self._nsites) % 2:
            raise ValueError("'total_sz' does not match the number of sites.")
        self._total_sz = total_sz
        if perm is not None and eigval is not None:
            self._perm, self._eigval = perm, eigval
        else:
            self._perm, self._eigval = _get_perm(self._generator, self._sector)
        self._basis = None

    @property
    def nsites(self) -> int:
        return self._nsites

    @property
    def eigval(self) -> jax.Array:
        return self._eigval

    @property
    def nsymm(self) -> int:
        return self.eigval.size if self.spin_inversion == 0 else 2 * self.eigval.size

    @property
    def spin_inversion(self) -> int:
        return self._spin_inversion

    @property
    def total_sz(self) -> Optional[int]:
        return self._total_sz
    
    @property
    def Nup(self) -> Optional[int]:
        if self.total_sz is None:
            return None
        return (self.nsites + self.total_sz) // 2

    @property
    def basis(self) -> spin_basis_general:
        if self._basis is not None:
            return self._basis
        
        blocks = dict()
        for i, (g, s) in enumerate(zip(self._generator, self._sector)):
            if not np.allclose(g, np.arange(g.size)):
                blocks[f"block{i}"] = (g, s)
        
        if self.spin_inversion != 0:
            sector = 0 if self._spin_inversion == 1 else 1
            blocks["inversion"] = (-np.arange(self.nsites)-1, sector)

        basis = spin_basis_general(
            self.nsites, self.Nup, pauli=-1, make_basis=False, **blocks
        )
        self._basis = basis
        return self._basis

    @partial(jax.jit, static_argnums=0)
    def get_symm_spins(self, spins: jax.Array) -> jax.Array:
        # batch = spins.shape[:-1]
        # nsites = spins.shape[-1]
        # spins = spins.reshape(-1, nsites)
        spins = spins[..., self._perm]
        if self._spin_inversion != 0:
            spins = jnp.concatenate([spins, -spins], axis=-2)
        #spins = spins.reshape(*batch, self.nsymm, nsites)
        return spins

    @partial(jax.jit, static_argnums=0)
    def symmetrize(self, inputs: jax.Array) -> jax.Array:
        eigval = self._eigval
        if self.spin_inversion != 0:
            eigval = jnp.concatenate([eigval, self.spin_inversion * eigval])
        eigval = (eigval / eigval.size).astype(inputs.dtype)
        return jnp.einsum("...s,s -> ...", inputs, eigval)

    def __add__(self, other: Symmetry) -> Symmetry:
        generator = np.concatenate([self._generator, other._generator], axis=0)
        sector = [*self._sector, *other._sector]
        perm = self._perm[:, other._perm].reshape(-1, self.nsites)
        eigval = jnp.einsum("i,j->ij", self._eigval, other._eigval).flatten()

        if self.spin_inversion == 0:
            spin_inversion = other.spin_inversion
        elif other.spin_inversion == 0:
            spin_inversion = self.spin_inversion
        elif self.spin_inversion == other.spin_inversion:
            spin_inversion = self.spin_inversion
        else:
            raise ValueError("Symmetry with different spin_inversion can't be added")

        if self.total_sz is None:
            total_sz = other.total_sz
        elif other.total_sz is None:
            total_sz = self.total_sz
        elif self.total_sz == other.total_sz:
            total_sz = self.total_sz
        else:
            raise ValueError("Symmetry with different total_sz can't be added")

        new_symm = Symmetry(generator, sector, spin_inversion, total_sz, perm, eigval)
        return new_symm
    
    def __call__(self, state):
        from ..state import DenseState, Variational
        if isinstance(state, DenseState):
            return state.todense(self)
        elif isinstance(state, Variational):
            input_fn = lambda s: state.input_fn(self.get_symm_spins(s))
            output_fn = lambda x: self.symmetrize(state.output_fn(x))
            new_symm = state.symm + self
            return Variational(
                state.models, None, new_symm, state.max_parallel, input_fn, output_fn
            )
        else:
            return NotImplemented
