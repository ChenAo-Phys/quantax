from typing import Sequence, Any
from jaxtyping import Key
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from .metropolis import Metropolis
from ..state import State
from ..utils import get_replicated_sharding
from ..global_defs import PARTICLE_TYPE, get_sites


def _sample_keys(key: Key, nsamples: int) -> jax.Array:
    """
    One independent key per sample, derived by folding the sample index into ``key``.
    Unlike splitting ``key`` into a batch and slicing it, this keeps the keys aligned
    with the sharded sample axis, so that proposals written per sample and vmapped
    over the batch need no cross-device communication.
    """
    return jax.vmap(jr.fold_in, in_axes=(None, 0))(key, jnp.arange(nsamples))


class LocalFlip(Metropolis):
    """
    Generate Monte Carlo samples by locally flipping spins. This sampler is suitable for
    spin systems with unconserved spin-up and spin-down numbers.
    """

    _n_candidates = 1  # a flip always changes the spins

    @property
    def particle_type(self) -> tuple[PARTICLE_TYPE, ...]:
        return (PARTICLE_TYPE.spin,)

    @property
    def update_mode(self) -> dict[str, Any]:
        return {"nflips": 1}

    @partial(jax.jit, static_argnums=0)
    def propose(self, key: Key, old_spins: jax.Array) -> jax.Array:
        nsamples, N = old_spins.shape

        def flip_one(key: Key, s: jax.Array) -> jax.Array:
            return s.at[jr.choice(key, N)].multiply(-1)

        return jax.vmap(flip_one)(_sample_keys(key, nsamples), old_spins)


def _get_site_neighbors(n_neighbor: int | Sequence[int]) -> jax.Array:
    """
    Get the neighboring sites for each site.
    """
    sites = get_sites()
    n_neighbor = [n_neighbor] if isinstance(n_neighbor, int) else n_neighbor
    neighbors = sites.get_neighbor(n_neighbor)
    neighbors = np.concatenate(neighbors, axis=0)
    neighbor_matrix = np.zeros((sites.Nsites, sites.Nsites), dtype=np.bool_)
    neighbor_matrix[neighbors[:, 0], neighbors[:, 1]] = True
    neighbor_matrix = neighbor_matrix | neighbor_matrix.T
    max_neighbors = np.max(np.sum(neighbor_matrix, axis=1)).item()
    neighbor_matrix = jnp.asarray(neighbor_matrix, dtype=jnp.bool_)
    fn = jax.vmap(lambda x: jnp.flatnonzero(x, size=max_neighbors, fill_value=-1))
    neighbors = fn(neighbor_matrix)
    neighbors = jnp.asarray(
        neighbors, dtype=jnp.int32, device=get_replicated_sharding()
    )
    if sites.particle_type == PARTICLE_TYPE.spinful_fermion:
        neighbors_dn = jnp.where(neighbors == -1, -1, neighbors + sites.Nsites)
        neighbors = jnp.concatenate([neighbors, neighbors_dn], axis=0)
    return neighbors


def _propose_exchange(
    key: Key,
    old_spins: jax.Array,
    hopping_particle: int,
    neighbors: jax.Array,
    mask: jax.Array | None = None,
) -> jax.Array:
    nsamples, Nmodes = old_spins.shape

    def exchange_one(key: Key, s: jax.Array) -> jax.Array:
        key_particle, key_neighbor = jr.split(key)
        p_site = s == hopping_particle
        if mask is not None:
            p_site = p_site & mask
        # uniform choice among the sites with p_site by inverse CDF: cheaper than
        # ``jr.choice(key, Nmodes, p=p_site)``, whose searchsorted is a scan
        count = jnp.cumsum(p_site)
        r = count[-1] * (1 - jr.uniform(key_particle))  # in (0, count[-1]]
        particle_idx = jnp.sum(count < r)
        neighbor_idx = jr.choice(key_neighbor, neighbors[particle_idx])
        # -1 fills the neighbor table of sites with fewer neighbors: no hopping
        neighbor_idx = jnp.where(neighbor_idx == -1, particle_idx, neighbor_idx)
        particle, neighbor = s[particle_idx], s[neighbor_idx]
        return s.at[particle_idx].set(neighbor).at[neighbor_idx].set(particle)

    return jax.vmap(exchange_one)(_sample_keys(key, nsamples), old_spins)


class SpinExchange(Metropolis):
    """
    Generate Monte Carlo samples by exchanging neighbor spins in spin systems.
    This sampler only works when the system has fixed number of spin-up and spin-down particles.
    """

    def __init__(
        self,
        state: State,
        nsamples: int,
        reweight: float = 2.0,
        thermal_steps: int | None = None,
        sweep_steps: int | None = None,
        initial_spins: jax.Array | None = None,
        n_neighbor: int | Sequence[int] = 1,
    ):
        r"""
        :param state:
            The state used for computing the wave function and probability.
            Exchanging neighbor spins conserves the numbers of spin-up and
            spin-down spins, so the `~quantax.sites.Sites` must fix the
            magnetization sector with ``Nparticles=(Nup, Ndown)``.

        :param nsamples:
            Number of samples generated per iteration.
            It should be a multiple of the total number of machines to allow samples
            to be equally distributed on different machines.

        :param reweight:
            The reweight factor n defining the sample probability :math:`|\psi|^n`,
            default to 2.0.

        :param thermal_steps:
            The number of thermalization steps in the beginning of each Markov chain,
            default to be 20 * fock state length.

        :param sweep_steps:
            The number of steps for generating new samples, default to be 2 * fock state length.

        :param initial_spins:
            The initial spins for every Markov chain before the thermalization steps,
            default to be random spins.

        :param n_neighbor:
            The neighbors to be considered in exchanges, default to nearest neighbors.
        """
        sites = get_sites()
        if sites.Nparticles is None or isinstance(sites.Nparticles, int):
            raise ValueError(
                "The number spin-up and spin-down particles should be specified in "
                "sites for `SpinExchange` sampler."
            )

        Nup = sites.Nparticles[0]
        if 2 * Nup <= state.Nmodes:
            self._hopping_particle = 1
        else:
            self._hopping_particle = -1

        self._neighbors = _get_site_neighbors(n_neighbor)

        super().__init__(
            state, nsamples, reweight, thermal_steps, sweep_steps, initial_spins
        )

    @property
    def particle_type(self) -> tuple[PARTICLE_TYPE, ...]:
        return (PARTICLE_TYPE.spin,)

    @property
    def update_mode(self) -> dict[str, Any]:
        return {"nflips": 2}

    @partial(jax.jit, static_argnums=0)
    def propose(self, key: Key, old_spins: jax.Array) -> jax.Array:
        return _propose_exchange(
            key, old_spins, self._hopping_particle, self._neighbors
        )


class ParticleHop(Metropolis):
    """
    Generate Monte Carlo samples by hopping random fermions to neighbor sites.
    This sampler only works when the system has fixed number of fermions.
    """

    #: The spin sector the hopping fermions belong to, 0 for spin-up and 1 for
    #: spin-down. ``None`` means the hopping fermions can be in either sector.
    _sector: int | None = None

    def __init__(
        self,
        state: State,
        nsamples: int,
        reweight: float = 2.0,
        thermal_steps: int | None = None,
        sweep_steps: int | None = None,
        initial_spins: jax.Array | None = None,
        n_neighbor: int | Sequence[int] = 1,
    ):
        r"""
        :param state:
            The state used for computing the wave function and probability.
            Hopping fermions to neighbor sites conserves the total particle
            number, so the `~quantax.sites.Sites` must be defined with a fixed
            ``Nparticles``.

        :param nsamples:
            Number of samples generated per iteration.
            It should be a multiple of the total number of machines to allow samples
            to be equally distributed on different machines.

        :param reweight:
            The reweight factor n defining the sample probability :math:`|\psi|^n`,
            default to 2.0.

        :param thermal_steps:
            The number of thermalization steps in the beginning of each Markov chain,
            default to be 20 * fock state length.

        :param sweep_steps:
            The number of steps for generating new samples, default to be 2 * fock state length.

        :param initial_spins:
            The initial spins for every Markov chain before the thermalization steps,
            default to be random spins.

        :param n_neighbor:
            The neighbors to be considered by particle hoppings, default to nearest neighbors.
        """
        sites = get_sites()
        if self._sector is None:
            if sites.Ntotal is None:
                raise ValueError(
                    "The number of fermions should be specified in sites for "
                    f"`{self.__class__.__name__}` sampler."
                )
            nparticles, nmodes = sites.Ntotal, state.Nmodes
            self._hop_mask = None
        else:
            if not isinstance(sites.Nparticles, tuple):
                raise ValueError(
                    "The numbers of spin-up and spin-down fermions should be "
                    f"specified in sites for `{self.__class__.__name__}` sampler."
                )
            nparticles, nmodes = sites.Nparticles[self._sector], sites.Nsites
            mask = np.zeros(state.Nmodes, dtype=np.bool_)
            lo = self._sector * sites.Nsites
            mask[lo : lo + sites.Nsites] = True
            self._hop_mask = jnp.asarray(mask, device=get_replicated_sharding())

        # Hop particles at low filling and holes at high filling for efficiency.
        self._hopping_particle = 1 if 2 * nparticles <= nmodes else -1
        self._neighbors = _get_site_neighbors(n_neighbor)

        super().__init__(
            state, nsamples, reweight, thermal_steps, sweep_steps, initial_spins
        )

    @property
    def particle_type(self) -> tuple[PARTICLE_TYPE, ...]:
        return (PARTICLE_TYPE.spinful_fermion, PARTICLE_TYPE.spinless_fermion)

    @property
    def update_mode(self) -> dict[str, Any]:
        return {"nflips": 2}

    @partial(jax.jit, static_argnums=0)
    def propose(self, key: Key, old_spins: jax.Array) -> jax.Array:
        return _propose_exchange(
            key, old_spins, self._hopping_particle, self._neighbors, self._hop_mask
        )


class ParticleHopUp(ParticleHop):
    """
    Generate Monte Carlo samples by hopping random spin-up fermions to neighbor
    sites. This sampler only works when the system has fixed numbers of spin-up
    and spin-down fermions.

    Compared to `ParticleHop`, the update mode tells the state that only spin-up
    modes are changed, which allows more efficient low-rank updates in models
    treating the two spin sectors separately (e.g. `~quantax.model.SingletPair`).
    As this sampler never moves spin-down fermions, it should be combined with
    other samplers, e.g. `ParticleHopDn` in a `~quantax.sampler.MixSampler`.
    """

    _sector = 0

    @property
    def particle_type(self) -> tuple[PARTICLE_TYPE, ...]:
        return (PARTICLE_TYPE.spinful_fermion,)

    @property
    def update_mode(self) -> dict[str, Any]:
        return {"nflips": 2, "nflips_up": 2, "nflips_dn": 0}


class ParticleHopDn(ParticleHop):
    """
    Generate Monte Carlo samples by hopping random spin-down fermions to neighbor
    sites. This sampler only works when the system has fixed numbers of spin-up
    and spin-down fermions.

    Compared to `ParticleHop`, the update mode tells the state that only spin-down
    modes are changed, which allows more efficient low-rank updates in models
    treating the two spin sectors separately (e.g. `~quantax.model.SingletPair`).
    As this sampler never moves spin-up fermions, it should be combined with
    other samplers, e.g. `ParticleHopUp` in a `~quantax.sampler.MixSampler`.
    """

    _sector = 1

    @property
    def particle_type(self) -> tuple[PARTICLE_TYPE, ...]:
        return (PARTICLE_TYPE.spinful_fermion,)

    @property
    def update_mode(self) -> dict[str, Any]:
        return {"nflips": 2, "nflips_up": 0, "nflips_dn": 2}


class SiteExchange(Metropolis):
    """
    Generate Monte Carlo samples by exchanging the spinful fermions on neighbor sites.

    .. warning::

        This sampler conserves the number of doublons and holons.
    """

    def __init__(
        self,
        state: State,
        nsamples: int,
        reweight: float = 2.0,
        thermal_steps: int | None = None,
        sweep_steps: int | None = None,
        initial_spins: jax.Array | None = None,
        n_neighbor: int | Sequence[int] = 1,
    ):
        r"""
        :param state:
            The state used for computing the wave function and probability.
            Exchanging the contents of neighbor sites conserves the numbers of
            spin-up and spin-down fermions, so the `~quantax.sites.Sites` must be
            defined with a fixed ``Nparticles``.

        :param nsamples:
            Number of samples generated per iteration.
            It should be a multiple of the total number of machines to allow samples
            to be equally distributed on different machines.

        :param reweight:
            The reweight factor n defining the sample probability :math:`|\psi|^n`,
            default to 2.0.

        :param thermal_steps:
            The number of thermalization steps in the beginning of each Markov chain,
            default to be 20 * fock state length.

        :param sweep_steps:
            The number of steps for generating new samples, default to be 2 * fock state length.

        :param initial_spins:
            The initial spins for every Markov chain before the thermalization steps,
            default to be random spins.

        :param n_neighbor:
            The neighbors to be considered by exchanges, default to nearest neighbors.
        """
        sites = get_sites()
        if sites.Nparticles is None:
            raise ValueError("`Nparticle` should be specified for `SiteExchange`.")

        n_neighbor = [n_neighbor] if isinstance(n_neighbor, int) else n_neighbor
        neighbors = sites.get_neighbor(n_neighbor)
        neighbors = np.concatenate(neighbors, axis=0)
        self._neighbors = jnp.asarray(neighbors, dtype=jnp.int32)

        super().__init__(
            state, nsamples, reweight, thermal_steps, sweep_steps, initial_spins
        )

    @property
    def particle_type(self) -> tuple[PARTICLE_TYPE, ...]:
        return (PARTICLE_TYPE.spinful_fermion,)

    @property
    def update_mode(self) -> dict[str, Any]:
        return {"nflips": 4, "nflips_up": 2, "nflips_dn": 2}

    @partial(jax.jit, static_argnums=0)
    def propose(self, key: Key, old_spins: jax.Array) -> jax.Array:
        nsamples = old_spins.shape[0]
        n_neighbors = self._neighbors.shape[0]
        N = get_sites().Nsites

        def exchange_one(key: Key, s: jax.Array) -> jax.Array:
            i, j = self._neighbors[jr.choice(key, n_neighbors)]
            idx = jnp.array([i, i + N, j, j + N])
            idx_exchanged = jnp.array([j, j + N, i, i + N])
            return s.at[idx].set(s[idx_exchanged])

        return jax.vmap(exchange_one)(_sample_keys(key, nsamples), old_spins)


class SiteFlip(Metropolis):
    """
    Generate Monte Carlo samples by flipping spins of spinful fermions locally.

    .. warning::

        This sampler conserves the number of fermions on each site.
    """

    @property
    def particle_type(self) -> tuple[PARTICLE_TYPE, ...]:
        return (PARTICLE_TYPE.spinful_fermion,)

    @property
    def update_mode(self) -> dict[str, Any]:
        return {"nflips": 2}

    @partial(jax.jit, static_argnums=0)
    def propose(self, key: Key, old_spins: jax.Array) -> jax.Array:
        nsamples, Nmodes = old_spins.shape
        N = Nmodes // 2

        def flip_one(key: Key, s: jax.Array) -> jax.Array:
            i = jr.choice(key, N)
            idx = jnp.array([i, i + N])
            return s.at[idx].set(s[idx[::-1]])

        return jax.vmap(flip_one)(_sample_keys(key, nsamples), old_spins)
