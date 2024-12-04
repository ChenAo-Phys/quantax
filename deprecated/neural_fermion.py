def _get_sublattice_perm(
    trans_symm: Symmetry,
    sublattice: Optional[tuple],
):
    lattice_shape = get_lattice().shape

    perm = trans_symm._perm
    dims = []
    for fulldim, subdim in zip(lattice_shape[1:], sublattice):
        if not fulldim % subdim == 0:
            raise ValueError(
                f"lattice dimension of length {fulldim} is not divisible by"
                f"sublattice dimension of length {subdim}"
            )
        dims.append(fulldim // subdim)
        dims.append(subdim)
    dims.append(perm.shape[-1])

    perm = perm.reshape(dims)

    for i in range(len(sublattice)):
        perm = jnp.take(perm, 0, axis=i)

    perm = perm.reshape(-1, perm.shape[-1])

    if not get_sites().is_fermion:
        perm = jnp.concatenate((perm, perm + perm.shape[-1]), -1)

    return perm


class _FullOrbsLayerPairProduct(RawInputLayer):
    F: jax.Array
    F_hidden: jax.Array
    index: jax.Array
    Nvisible: int
    Nhidden: int
    holomorphic: bool

    def __init__(
        self,
        Nvisible: int,
        Nhidden: int,
        sublattice: Optional[tuple],
        dtype: jnp.dtype = jnp.float64,
    ):
        sites = get_sites()
        N = sites.nsites
        self.Nvisible = Nvisible
        self.Nhidden = Nhidden

        is_dtype_cpl = jnp.issubdtype(dtype, jnp.complexfloating)

        index, nparams = _get_pair_product_indices(sublattice, N)
        self.index = index
        shape = (nparams,)

        is_dtype_cpl = jnp.issubdtype(dtype, jnp.complexfloating)
        self.F_hidden = jnp.eye(Nhidden, dtype=dtype)
        if is_default_cpl() and not is_dtype_cpl:
            shape = (2,) + shape
            self.F_hidden = jnp.concatenate(
                (self.F_hidden[None], jnp.zeros_like(self.F_hidden[None])), axis=0
            )

        self.F = jr.normal(get_subkeys(), shape, dtype)
        self.holomorphic = is_default_cpl() and is_dtype_cpl

    def to_hidden_orbs(self, x: jax.Array) -> jax.Array:
        N = get_sites().nsites
        x = x.reshape(2 * self.Nhidden, -1, N)
        x = jnp.sum(x, axis=1) / np.sqrt(x.shape[1], dtype=x.dtype)
        return jnp.split(x, 2, axis=0)

    @property
    def F_full(self) -> jax.Array:
        F = self.F if self.F.ndim == 1 else jax.lax.complex(self.F[0], self.F[1])
        F_full = F[self.index]
        return F_full

    @property
    def F_hidden_full(self) -> jax.Array:
        if self.F_hidden.ndim == 2:
            F_hidden = self.F_hidden
        else:
            F_hidden = jax.lax.complex(self.F_hidden[0], self.F_hidden[1])

        return F_hidden

    def __call__(self, x: jax.Array, s: jax.Array) -> jax.Array:

        idx = _get_fermion_idx(s, self.Nvisible)

        N = get_sites().nsites

        idx_down, idx_up = jnp.split(idx, 2)
        idx_up = idx_up - N

        F_full = self.F_full

        mat11 = F_full[idx_down, :][:, idx_up]

        xd, xu = self.to_hidden_orbs(x)
        mat21 = xd.T[idx_down, :].astype(mat11.dtype)
        mat12 = xu[:, idx_up].astype(mat11.dtype)

        mat22 = self.F_hidden_full

        full_orbs = jnp.block([[mat11, mat21], [mat12, mat22]])

        return full_orbs


class HiddenPairProduct(Sequential):
    Nvisible: int
    Nhidden: int
    layers: Tuple[eqx.Module]
    holomorphic: bool = eqx.field(static=True)

    def __init__(
        self,
        pairing_net: Union[eqx.Module, int],
        Nvisible: Union[None, int, Sequence[int]] = None,
        Nhidden: Optional[int] = None,
        sublattice: Optional[tuple] = None,
        dtype: jnp.dtype = jnp.float64,
    ):
        if isinstance(pairing_net, int):
            pairing_net = _ConstantPairing(pairing_net, dtype)

        self.Nvisible = _get_Nparticle(Nvisible)
        self.Nhidden = _get_default_Nhidden(pairing_net) if Nhidden is None else Nhidden

        full_orbs_layer = _FullOrbsLayerPairProduct(
            self.Nvisible, self.Nhidden, sublattice, dtype
        )
        scale_layer = Scale(np.sqrt(np.e / (self.Nvisible + self.Nhidden)))
        pfa_layer = eqx.nn.Lambda(lambda x: det(x))

        if isinstance(pairing_net, Sequential):
            layers = pairing_net.layers + (full_orbs_layer, scale_layer, pfa_layer)
        else:
            layers = (pairing_net, full_orbs_layer, scale_layer, pfa_layer)

        if hasattr(pairing_net, "holomorphic"):
            holomorphic = pairing_net.holomorphic and full_orbs_layer.holomorphic
        else:
            holomorphic = False

        Sequential.__init__(self, layers, holomorphic)

    @property
    def pairing_net(self) -> Sequential:
        return self[:-3]

    @property
    def full_orbs_layer(self) -> _FullOrbsLayerPfaffian:
        return self.layers[-3]

    @property
    def scale_layer(self) -> Scale:
        return self.layers[-2]

    def rescale(self, maximum: jax.Array) -> HiddenPfaffian:
        scale = self.scale_layer.scale
        scale /= maximum.astype(scale.dtype) ** (2 / (self.Nvisible + self.Nhidden))
        return eqx.tree_at(lambda tree: tree.layers[-2].scale, self, scale)