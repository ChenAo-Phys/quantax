from __future__ import annotations
from typing import Any, Callable, Literal, overload
import numpy as np
from numpy.typing import NDArray
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.typing import DTypeLike
import equinox as eqx
from ..nn import RefModel, RawInputLayer, exp_by_log
from ..symmetry import Translation
from ..symmetry.symmetry import _permutation_sign
from ..global_defs import get_lattice, get_sites, get_subkeys, PARTICLE_TYPE
from ..utils import PsiArray, LogArray
from .fermion_mf import MF_Internal, _standardize_sublattice


def _get_jastrow_index(sublattice: Translation | None) -> NDArray[np.int32]:
    """
    Map every entry of the ``Nmodes x Nmodes`` coupling matrix to a parameter index.
    Entries related by the sublattice translation share the same index, so that the
    reconstructed matrix is invariant under the sublattice translation. Without a
    sublattice every entry is an independent parameter.

    For spinful fermions the spin-up and spin-down modes are given distinct channels, so
    intra-spin and inter-spin couplings are parametrized separately.
    """
    sites = get_sites()
    M = sites.Nmodes

    if sublattice is None:
        return np.arange(M * M, dtype=np.int32).reshape(M, M)

    sub_coord = sublattice.get_sublattice_coord()
    if sites.particle_type == PARTICLE_TYPE.spinful_fermion:
        sub_dn = sub_coord.copy()
        sub_dn[:, 0] += np.max(sub_coord[:, 0]) + 1
        sub_coord = np.concatenate([sub_coord, sub_dn], axis=0)
    sub_shape = np.max(sub_coord, axis=0) + 1

    diff = (sub_coord[:, None, 1:] - sub_coord[None, :, 1:]) % sub_shape[1:]
    c1 = np.repeat(sub_coord[:, None, :1], M, axis=1)
    c2 = np.repeat(sub_coord[None, :, :1], M, axis=0)
    diff = np.concatenate([c1, c2, diff], axis=-1)
    _, index = np.unique(diff.reshape(M * M, -1), return_inverse=True, axis=0)
    return index.astype(np.int32).reshape(M, M)


class Jastrow(eqx.Module):
    r"""
    Traditional two-body Jastrow factor with a symmetric matrix :math:`W` of variational
    parameters. The natural variable depends on the system:

    - Spin systems use the spin-spin form
      :math:`\psi(s) = \exp\!\big(\frac{1}{2}\sum_{ij} W_{ij}\, s_i s_j\big)`,
      where :math:`s_i = \pm 1` is the configuration on mode :math:`i`.

    - Fermion systems use the density-density form
      :math:`\psi(n) = \exp\!\big(\frac{1}{2}\sum_{ij} W_{ij}\, n_i n_j\big)`,
      where :math:`n_i \in \{0, 1\}` is the occupation. The diagonal :math:`W_{ii} n_i`
      acts as a one-body term, and for spinful fermions the spin-up and spin-down modes
      are coupled separately, so the on-site up-down coupling :math:`W_{i\uparrow, i\downarrow}
      n_{i\uparrow} n_{i\downarrow}` realizes a Gutzwiller-type factor.

    With a real ``dtype`` the factor is real and positive, so it is usually combined with
    a reference state that provides the sign structure, e.g. through
    `~quantax.model.GeneralJastrow`.
    """

    W: jax.Array
    sublattice: Translation | None
    dtype: DTypeLike
    holomorphic: bool

    def __init__(
        self,
        sublattice: Translation | tuple[int, ...] | None = None,
        dtype: DTypeLike = jnp.float32,
    ):
        r"""
        :param sublattice:
            The translation symmetry under which the coupling matrix :math:`W` is
            invariant, given as a `~quantax.symmetry.Translation` or a tuple of
            sublattice sizes along each axis. ``None`` leaves every matrix entry free.

        :param dtype:
            The data type of the parameters. A complex ``dtype`` gives a holomorphic
            complex Jastrow factor.
        """
        sites = get_sites()
        self.dtype = dtype
        self.holomorphic = jnp.issubdtype(dtype, jnp.complexfloating)
        self.sublattice = _standardize_sublattice(sublattice)

        index = _get_jastrow_index(self.sublattice)
        nparams = int(np.max(index)) + 1
        # Small init so the factor starts close to 1 and barely perturbs a reference.
        self.W = jr.normal(get_subkeys(), (nparams,), dtype) / sites.Nmodes

    @property
    def W_full(self) -> jax.Array:
        """The full symmetric ``Nmodes x Nmodes`` coupling matrix."""
        W = self.W[_get_jastrow_index(self.sublattice)]
        return (W + W.T) / 2

    def __call__(self, s: jax.Array) -> LogArray:
        r"""
        Evaluate the Jastrow factor on a given configuration. Spin systems use the
        :math:`\pm 1` configuration directly, while fermion systems use the :math:`0/1`
        occupations.
        """
        if get_sites().is_fermion:
            x = jnp.where(s > 0, 1, 0).astype(self.dtype)
        else:
            x = s.astype(self.dtype)
        x = (x @ self.W_full @ x) / 2
        return exp_by_log(x)


def _to_sub_term(x: jax.Array, sublattice: tuple[int, ...]) -> jax.Array:
    remaining_dims = x.shape[1:]
    x = x.reshape(get_lattice().shape[1:] + remaining_dims)
    for axis, subl in enumerate(sublattice):
        x = x.take(np.arange(subl), axis)
    x = x.reshape(-1, *remaining_dims)
    return x


def _get_sublattice_spins(
    s: jax.Array, trans_symm: Translation | None, sublattice: tuple[int, ...] | None
) -> jax.Array:
    if trans_symm is None or sublattice is None:
        return s[..., None, :]

    perm = _to_sub_term(trans_symm._perm, sublattice)

    Nmodes = trans_symm.Nmodes
    batch = s.shape[:-1]
    s = s.reshape(-1, Nmodes)
    s_symm = s[:, perm]
    s_symm = s_symm.reshape(*batch, -1, perm.shape[0], Nmodes)
    s_symm = jnp.swapaxes(s_symm, -3, -2)
    return s_symm.reshape(*batch, perm.shape[0], -1)


def _sub_symmetrize(
    x_sub: PsiArray,
    s: jax.Array,
    trans_symm: Translation | None,
    sublattice: tuple[int, ...] | None,
) -> PsiArray:
    if trans_symm is None or sublattice is None:
        return x_sub[0]

    eigval = _to_sub_term(trans_symm._character, sublattice) / trans_symm.nsymm

    if trans_symm.is_fermion:
        perm = _to_sub_term(trans_symm._perm, sublattice)
        perm_sign = _to_sub_term(trans_symm._perm_sign, sublattice)
        sign = _permutation_sign(s, perm, perm_sign)
        eigval *= sign

    eigval = eigval.astype(x_sub.dtype)
    return (x_sub * eigval).sum()


class _JastrowFermionLayer(RawInputLayer):
    fermion_mf: RefModel
    trans_symm: Translation | None
    sublattice: tuple[int, ...] | None

    def __init__(self, fermion_mf, trans_symm):
        self.fermion_mf = fermion_mf
        self.trans_symm = trans_symm
        if hasattr(fermion_mf, "sublattice"):
            if isinstance(fermion_mf.sublattice, Translation):
                vectors = fermion_mf.sublattice.vectors
                sublattice = [1] * get_lattice().ndim
                for vec in vectors:
                    if np.sum(vec != 0) != 1:
                        raise ValueError(
                            "Sublattice vectors must be along one axis for Jastrow layer."
                        )
                    else:
                        axis = np.flatnonzero(vec)[0]
                        sublattice[axis] = vec[axis].item()
                sublattice = tuple(sublattice)
            else:
                sublattice = fermion_mf.sublattice
        else:
            sublattice = None

        if sublattice is None and trans_symm is not None:
            sublattice = get_lattice().shape[1:]
        self.sublattice = sublattice

    def get_sublattice_spins(self, s: jax.Array) -> jax.Array:
        return _get_sublattice_spins(s, self.trans_symm, self.sublattice)

    def sub_symmetrize(self, x_net: PsiArray, x_mf: PsiArray, s: jax.Array) -> PsiArray:
        if self.trans_symm is None or self.sublattice is None:
            return x_mf[0] * x_net.mean()

        x_net = x_net.reshape(-1, *get_lattice().shape[1:]).mean(axis=0)
        for axis, subl in enumerate(self.sublattice):
            new_shape = x_net.shape[:axis] + (-1, subl) + x_net.shape[axis + 1 :]
            x_net = x_net.reshape(new_shape)
            x_net = x_net.mean(axis=axis)
        return _sub_symmetrize(
            x_mf * x_net.flatten(), s, self.trans_symm, self.sublattice
        )

    def __call__(self, x: PsiArray, s: jax.Array) -> PsiArray:
        if x.size > 1:
            x = x.reshape(-1, get_lattice().ncells).mean(axis=0)

        s_symm = self.get_sublattice_spins(s)
        x_mf = jax.vmap(self.fermion_mf)(s_symm)
        return self.sub_symmetrize(x, x_mf, s)


class GeneralJastrow(RefModel):
    r"""
    A reference state multiplied by a generalized (neural-network) Jastrow factor,
    :math:`\psi(s) = J(s)\, \psi_\mathrm{mf}(s)`, where :math:`J(s)` is the output of an
    arbitrary network ``net`` and :math:`\psi_\mathrm{mf}` is a reference state, usually
    a fermionic mean-field wavefunction.

    When ``trans_symm`` is given, the factor is translation-symmetrized as
    :math:`\psi(s) = \sum_{T} \chi_T\, J_T(s)\, \psi_\mathrm{mf}(T s)`, where the sum runs
    over the translations :math:`T`, :math:`\chi_T` are the corresponding characters, and
    :math:`J_T(s)` is the network output associated with translation :math:`T`. If the
    reference state defines a ``sublattice``, the sum is restricted to it accordingly.
    """

    net: Callable[[jax.Array], PsiArray]
    fermion_layer: _JastrowFermionLayer
    holomorphic: bool
    trans_symm: Translation | None
    sublattice: tuple[int, ...] | None

    def __init__(
        self,
        net: Callable[[jax.Array], PsiArray],
        fermion_mf: RefModel,
        trans_symm: Translation | None = None,
    ):
        r"""
        :param net:
            The network producing the Jastrow factor :math:`J(s)`. To obtain a
            translation-resolved factor :math:`J_T(s)`, build the network with
            ``trans_symm=Identity()`` so that its output is not summed over translations.

        :param fermion_mf:
            The reference state :math:`\psi_\mathrm{mf}(s)`, typically a fermionic
            mean-field wavefunction. Its ``sublattice`` (if any) is reused to restrict
            the symmetrization.

        :param trans_symm:
            The translation symmetry used to symmetrize the product. ``None`` simply
            multiplies the (mean) network output with the reference state.
        """
        self.net = net
        self.fermion_layer = _JastrowFermionLayer(fermion_mf, trans_symm)
        self.trans_symm = trans_symm
        self.sublattice = self.fermion_layer.sublattice
        net_holomorphic = getattr(net, "holomorphic", False)
        mf_holomorphic = getattr(fermion_mf, "holomorphic", False)
        self.holomorphic = net_holomorphic and mf_holomorphic

    def __call__(self, s: jax.Array) -> PsiArray:
        """
        Evaluate the wavefunction on a given configuration.
        """
        x = self.net(s)
        return self.fermion_layer(x, s)

    @property
    def fermion_mf(self) -> RefModel:
        """The reference state :math:`\\psi_\\mathrm{mf}`."""
        return self.fermion_layer.fermion_mf

    def get_sublattice_spins(self, x: jax.Array) -> jax.Array:
        """
        Return the configurations translated over the sublattice, one per translation
        kept in the symmetrization.
        """
        return self.fermion_layer.get_sublattice_spins(x)

    def sub_symmetrize(self, x_net: PsiArray, x_mf: PsiArray, s: jax.Array) -> PsiArray:
        """
        Combine the network output ``x_net`` with the reference amplitudes ``x_mf``
        evaluated on the translated configurations into the symmetrized wavefunction.
        """
        return self.fermion_layer.sub_symmetrize(x_net, x_mf, s)

    def init_internal(self, s: jax.Array) -> tuple[PsiArray, MF_Internal]:
        """
        Return wavefunction and internal values for given input configurations.
        See `~quantax.nn.RefModel` for details.
        """
        x_net = self.net(s)
        s_symm = self.get_sublattice_spins(s)
        x_mf, internal = jax.vmap(self.fermion_mf.init_internal)(s_symm)
        return self.sub_symmetrize(x_net, x_mf, s), internal

    @property
    def required_update_modes(self) -> tuple[str, ...]:
        """
        The required update modes for accelerated ref_forward pass.
        """
        return self.fermion_mf.required_update_modes

    @overload
    def ref_forward(
        self,
        s: jax.Array,
        s_old: jax.Array,
        update_mode: dict[str, Any],
        internal: MF_Internal,
        return_update: Literal[False] = False,
    ) -> PsiArray: ...

    @overload
    def ref_forward(
        self,
        s: jax.Array,
        s_old: jax.Array,
        update_mode: dict[str, Any],
        internal: MF_Internal,
        return_update: Literal[True],
    ) -> tuple[PsiArray, MF_Internal]: ...

    def ref_forward(
        self,
        s: jax.Array,
        s_old: jax.Array,
        update_mode: dict[str, Any],
        internal: MF_Internal,
        return_update: bool = False,
    ) -> PsiArray | tuple[PsiArray, MF_Internal]:
        """
        Accelerated forward pass through local updates and internal quantities.
        The reference state is updated through its own ``ref_forward`` while the network
        factor is recomputed from scratch. See `~quantax.nn.RefModel` for details.
        """
        x_net = self.net(s)
        if x_net.size > 1:
            x_net = x_net.reshape(-1, get_lattice().ncells).mean(axis=0)

        s_symm = self.get_sublattice_spins(s)
        s_old = self.get_sublattice_spins(s_old)
        fn_vmap = eqx.filter_vmap(
            self.fermion_mf.ref_forward, in_axes=(0, 0, None, 0, None)
        )
        if return_update:
            x_mf, internal = fn_vmap(s_symm, s_old, update_mode, internal, True)
            psi = self.sub_symmetrize(x_net, x_mf, s)
            return psi, internal
        else:
            x_mf = fn_vmap(s_symm, s_old, update_mode, internal, False)
            psi = self.sub_symmetrize(x_net, x_mf, s)
            return psi
