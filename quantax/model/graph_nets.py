import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.nn import leaky_relu
from typing import Optional, Tuple, Callable, Sequence, Optional, Union
from ..nn import (
    Sequential,
    apply_he_normal,
    Exp,
    Scale,
    pair_cpl,
    ReshapeSite,
    GraphLayer,
    ConvSymmetrize,
    SquareGconv,
)
from functools import partial
from ..symmetry import Symmetry, Trans2D
from ..global_defs import get_sites, is_default_cpl, get_subkeys


def glorot_init(key, shape, dtype=jnp.float32):
    """Sample from Glorot uniform distribution."""
    fan_in, fan_out = shape[-2], shape[-1] if len(shape) >= 2 else (shape[0], shape[0])
    limit = jnp.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(key, shape, dtype, -limit, limit)

def GTran(
    nblocks: int,
    hidden_channels: int,
    out_channels: int,
    ffn_hidden: list,
    heads: int,
    final_activation: Optional[Callable] = None,
    n_neighbor: int=1,
    dtype: jnp.dtype = jnp.float32,
):
    """
    The convolutional residual network with a summation in the end.

    :param nblocks:
        The number of residual blocks. Each block contains two convolutional layers.

    :param hidden_channels:
        The number of hidden channels. Each site's configuration will be encoded as a hidden_channels shape 1D feature.
    
    :param out_channels:
        The number of channels in the output features.

    :param heads:
        The heads number used in graph attention. Each layer has the same heads number.

    :param final_activation:
        The activation function in the last layer.
        By default, `~quantax.nn.Exp` is used.

    :param dtype:
        The data type of the parameters.

    """

    if nblocks == 1:
        assert hidden_channels == out_channels

    site = get_sites()
    is_fermion = site.is_fermion

    blocks = []
    for i in range(nblocks):
        if i == 0:
            blocks.append(
                GTransBlock(
                    in_channels=2 if is_fermion else 1,
                    out_channels=hidden_channels,
                    ffn_hidden=ffn_hidden,
                    heads=heads,
                    activation=jax.nn.leaky_relu,
                    n_neighbor=n_neighbor,
                )
            )

        elif i == nblocks - 1:
            blocks.append(
                GTransBlock(
                    hidden_channels,
                    out_channels=out_channels,
                    ffn_hidden=ffn_hidden,
                    heads=heads,
                    activation=jax.nn.leaky_relu,
                    n_neighbor=n_neighbor,
                )
            )
    # (4,4)

    scale = Scale(1 / np.sqrt(nblocks + 1))
    layers = [ReshapeSite(dtype), *blocks, scale]

    if is_default_cpl():
        cpl_layer = eqx.nn.Lambda(lambda x: pair_cpl(x))
        layers.append(cpl_layer)

    if final_activation is None:
        final_activation = Exp()
    elif not isinstance(final_activation, eqx.Module):
        final_activation = eqx.nn.Lambda(final_activation)

    layers.append(final_activation)

    return Sequential(layers, holomorphic=False)



class GATv2Conv(GraphLayer):
    # ----------------------
    # Hyperparameters
    # ----------------------
    in_channels: int
    out_channels: int
    heads: int
    concat: bool
    negative_slope: float
    dropout_rate: float
    add_self_loops: bool
    edge_dim: Optional[int]
    residual: bool
    share_weights: bool
    
    # ----------------------
    # Learnable parameters
    # ----------------------
    lin_l: eqx.nn.Linear
    lin_r: eqx.nn.Linear
    lin_edge: Optional[eqx.nn.Linear]
    att: jnp.ndarray            # shape = (heads, out_channels)
    bias: Optional[jnp.ndarray] # shape = (heads*out_channels,) or (out_channels,)
    res: Optional[eqx.nn.Linear]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = False,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = False,
        edge_dim: Optional[int] = None,
        residual: bool = False,
        share_weights: bool = False,
        use_bias: bool = True,
        *,
        key: "jax.random.PRNGKey",
    ):
        """GATv2Conv in JAX + Equinox."""

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout_rate = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.residual = residual
        self.share_weights = share_weights

        # For reproducibility, split keys:
        key_l, key_r, key_e, key_att, key_res, key_bias = jax.random.split(key, 6)

        # lin_l and lin_r: shape (in_channels, heads*out_channels)
        self.lin_l = eqx.nn.Linear(
            in_channels, heads * out_channels,
            use_bias=False,  # PyG sometimes sets bias here or not
            key=key_l,
            # Equinox defaults to a Glorot init for weights, so you could skip
            # a custom initializer. If you want to replicate exactly, do:
            # weight_init=lambda k, shp: glorot_init(k, shp)
        )

        if share_weights:
            self.lin_r = self.lin_l
        else:
            self.lin_r = eqx.nn.Linear(
                in_channels, heads * out_channels,
                use_bias=False,
                key=key_r,
            )

        # Edge projection if needed
        if edge_dim is not None:
            self.lin_edge = eqx.nn.Linear(
                edge_dim, heads * out_channels,
                use_bias=False,
                key=key_e,
            )
        else:
            self.lin_edge = None

        # Learnable attention vector: shape (heads, out_channels)
        # Original PyG is shape (1, heads, out_channels), but we can store (heads, out_channels)
        # and broadcast as needed.
        att_raw = glorot_init(key_att, (heads, out_channels))
        self.att = att_raw  # eqx.Module fields can be jnp arrays

        # Residual linear if requested
        if residual:
            # Maps from in_channels -> heads*out_channels if concat, else out_channels
            out_dim = heads * out_channels if concat else out_channels
            self.res = eqx.nn.Linear(in_channels, out_dim, use_bias=False, key=key_res)
        else:
            self.res = None

        # Optional bias
        if use_bias:
            # If concat=True => shape is heads*out_channels, else out_channels
            out_dim = heads * out_channels if concat else out_channels
            self.bias = jax.random.normal(key_bias, (out_dim,))
        else:
            self.bias = None

    def __call__(
        self,
        x: jnp.ndarray,           # (N, in_channels)
        edge_index: jnp.ndarray,  # (2, E)
        edge_attr: Optional[jnp.ndarray] = None,
        key: Optional["jax.random.PRNGKey"] = None,
    ) -> jnp.ndarray:

        """
        Args:
          x: Node features, (N, in_features)
          edge_index: (2, E) with edge_index[0]=dst, edge_index[1]=src
          edge_attr: (E, edge_dim) or None
          key: PRNG key for dropout (if needed)
          return_attention_weights: If True, return alpha along with outputs.
        Returns:
          out: shape = (N, heads*out_channels) if concat=True,
                       (N, out_channels)       if concat=False
          (optionally) alpha: shape (E, heads)
        """
        if key is None:
            # If no key is provided, we do "no-dropout" or rely on eqx's stateless approach.
            # Alternatively, you could require a key always.
            key = jax.random.PRNGKey(0)

        N = x.shape[0]
        H, C = self.heads, self.out_channels

        # 1) Linear projections:
        #    x_l, x_r each shape (N, heads*out_channels).
        x_l = jax.vmap(self.lin_l)(x)           # (N, H*C)
        x_r = x_l if self.share_weights else jax.vmap(self.lin_r)(x)

        # 2) Reshape to (N, H, C)
        x_l = x_l.reshape(N, H, C)
        x_r = x_r.reshape(N, H, C)

        # 3) Add self-loops if requested
        if self.add_self_loops:
            # Just do a naive approach: append [i, i] edges for all i in [0..N-1].
            # You could also skip duplicates or handle fill_value logic, etc.
            loop_index = jnp.stack([jnp.arange(N), jnp.arange(N)], axis=0)  # (2, N)
            edge_index = jnp.concatenate([edge_index, loop_index], axis=1)
            if edge_attr is not None:
                # If you want "fill_value='mean'", you'd need to compute from existing edges.
                # Here, we do a simple zero-edge or pass an identity for self-loops.
                loop_attr = jnp.zeros((N, self.edge_dim))
                edge_attr = jnp.concatenate([edge_attr, loop_attr], axis=0)

        E = edge_index.shape[1]

        # 4) Compute un-normalized attention logits = leakyrelu(x_i + x_j (+ edge))
        #    We gather x_j, x_i via edge_index, then do alpha = sum(...) * att
        src = edge_index[1]  # indexes x_r
        dst = edge_index[0]  # indexes x_l

        x_i = x_l[dst]  # (E, H, C)
        x_j = x_r[src]  # (E, H, C)

        # sum of node-projected features
        alpha_ij = x_i + x_j  # (E, H, C)

        # If edge_attr:
        if (self.lin_edge is not None) and (edge_attr is not None):
            edge_emb = jax.vmap(self.lin_edge)(edge_attr)          # (E, H*C)
            edge_emb = edge_emb.reshape(E, H, C)         # (E, H, C)
            alpha_ij = alpha_ij + edge_emb               # add edge contribution

        alpha_ij = leaky_relu(alpha_ij, self.negative_slope)  # (E, H, C)

        # Dot with self.att (heads, out_channels) => broadcast multiply + sum over last dim
        # alpha_ij shape: (E, H, C), att shape: (H, C)
        # => (E, H) by an elementwise product and sum over channel dim
        att_expanded = self.att[None, :, :]          # (1, H, C)
        logits = jnp.einsum("ehc, h c->eh", alpha_ij, self.att)  # but simpler is:
        # Equivalent to: jnp.sum(alpha_ij * att_expanded, axis=-1)
        logits = jnp.sum(alpha_ij * att_expanded, axis=-1)  # (E, H)

        # 5) Groupwise softmax over edges that share the same dst
        # We flatten (E, H) => (E*H) to run segment_softmax. Each edge e belongs to
        # node `dst[e]`, but we must account for the head dimension too.
        # Trick: we can do a grouped segment ID = dst[e]*H + head_idx.
        # But simpler is to do it per-head with vmap, or we compute attention per-head:

        def per_head_softmax(logits_1h, head_idx):
            # logits_1h: (E,) => compute softmax over the grouping `dst`.
            return segment_softmax(logits_1h, dst, N)  # shape (E,)

        # We can vmap across heads dimension:
        head_indices = jnp.arange(H)
        alpha = jax.vmap(per_head_softmax, in_axes=(1, 0), out_axes=1)(logits, head_indices)
        # alpha has shape (E, H)

        # 6) Dropout on alpha if needed
        if self.dropout_rate > 0.0:
            # Equinox doesn't do "training mode" automatically. 
            # We'll do a simple deterministic approach if dropout_rate>0 and a key was passed.
            # We'll broadcast a random mask of shape (E, H).
            keep_prob = 1.0 - self.dropout_rate
            rng = jax.random.bernoulli(key, p=keep_prob, shape=alpha.shape)
            alpha = alpha * rng / (keep_prob + 1e-16)

        # 7) Message passing: out_i = sum_{j in neighbors(i)} alpha_ij * x_j
        # For each edge e from j->i, we add alpha[e, h] * x_j[e, h, :]
        # We'll scatter-sum into out array. The shape of out is (N, H, C).
        x_j_times_alpha = x_j * alpha[..., None]  # (E, H, C)

        # We'll define an accumulator:
        out = jnp.zeros((N, H, C), dtype=x.dtype)

        # scatter-sum over edges (the 'dst' node)
        out = out.at[dst].add(x_j_times_alpha)

        # 8) If concat, flatten heads; else average over heads
        if self.concat:
            out = out.reshape(N, H * C)  # (N, heads*out_channels)
        else:
            out = jnp.mean(out, axis=1)  # (N, out_channels)

        # 9) Residual
        if self.res is not None:
            res_connection = self.res(x)  # shape matches (N, heads*C) or (N, C)
            out = out + res_connection

        # 10) Optional bias
        if self.bias is not None:
            out = out + self.bias

        # Return attention weights if requested. We’ll return shape (E, H).
        return out

@partial(jax.jit, static_argnums=(2,))
def segment_softmax(logits: jnp.ndarray, segment_ids: jnp.ndarray, num_segments: int) -> jnp.ndarray:
    """Compute a groupwise softmax over `logits` partitioned by `segment_ids`."""
    # 1) For numerical stability, subtract per-segment max
    max_per_segment = jax.ops.segment_max(logits, segment_ids, num_segments)
    centered = logits - max_per_segment[segment_ids]
    
    # 2) Exponentiate
    exp_vals = jnp.exp(centered)
    
    # 3) Sum within each segment
    sum_per_segment = jax.ops.segment_sum(exp_vals, segment_ids, num_segments)
    
    # 4) Normalize
    return exp_vals / (sum_per_segment[segment_ids] + 1e-16)


class FeedForward(eqx.Module):
    """A simple feedforward network (MLP) with optional dropout."""
    layers: tuple
    activations: tuple
    dropout_rate: float

    def __init__(
        self,
        in_dim: int,
        hidden_dims: Sequence[int],
        out_dim: int,
        activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.relu,
        dropout_rate: float = 0.0,
        *,
        key: "jax.random.PRNGKey",
    ):
        """
        Args:
          in_dim: Size of the input features.
          hidden_dims: A list/tuple of hidden layer sizes, e.g. [64, 64].
          out_dim: Size of the output.
          activation: The activation function to apply between layers.
          dropout_rate: Probability for dropout (0.0 means no dropout).
          key: PRNGKey for weight initialization.
        """
        super().__init__()

        # Split keys for each layer:
        n_layers = len(hidden_dims) + 1  # all hidden + final output layer
        keys = jax.random.split(key, n_layers)

        # Build the layers as a tuple of `eqx.nn.Linear`.
        # The "fan in" of the first layer is in_dim, then each hidden layer,
        # and finally out_dim for the last layer.
        dims = [in_dim] + list(hidden_dims) + [out_dim]
        layer_list = []
        for i in range(n_layers):
            layer_list.append(
                eqx.nn.Linear(
                    dims[i],
                    dims[i + 1],
                    # By default, Equinox uses a Glorot init for `weight_init`,
                    # which is generally good for MLPs.
                    use_bias=True,
                    key=keys[i],
                )
            )

        self.layers = tuple(layer_list)
        # We'll apply the same activation between each layer except after the final:
        # This means we have `n_layers - 1` activations. We'll handle the last layer 
        # separately in the forward pass (no activation by default).
        self.activations = tuple([activation] * (n_layers - 1))
        self.dropout_rate = dropout_rate

    def __call__(
        self,
        x: jnp.ndarray,         # (batch_size, in_dim) or (in_dim,) for single example
        key: Optional["jax.random.PRNGKey"] = None,
        training: bool = False
    ) -> jnp.ndarray:
        """
        Forward pass.

        Args:
          x: input array, shape (..., in_dim).
          key: PRNG key for dropout; if None, no dropout is applied.
          training: whether in training mode (dropout on) or inference mode (dropout off).

        Returns:
          The MLP output, shape (..., out_dim).
        """
        for i, layer in enumerate(self.layers):
            x = jax.vmap(layer)(x)  # eqx.nn.Linear does x @ W + b
            if i < len(self.layers) - 1:
                # Not the final layer => apply activation
                x = self.activations[i](x)

                # Optional dropout
                if self.dropout_rate > 0.0 and training and key is not None:
                    # Typically we combine one per layer for reproducibility:
                    # e.g. split a new key for each layer
                    subkey, key = jax.random.split(key)
                    keep_prob = 1.0 - self.dropout_rate
                    mask = jax.random.bernoulli(subkey, p=keep_prob, shape=x.shape)
                    x = jnp.where(mask, x / keep_prob, 0.0)
            else:
                # Final layer => no activation by default
                pass
        return x


class GTransBlock(eqx.Module):
    in_channels: int
    out_channels: int
    heads: int
    ffn_hidden: list
    activation: Callable
    attn: GATv2Conv
    ffn: FeedForward
    edge_index: jax.Array

    ln1: eqx.nn.LayerNorm
    ln2: eqx.nn.LayerNorm

    affn1: eqx.nn.Linear
    affn2: eqx.nn.Linear

    def __init__(self, 
        in_channels: int,
        out_channels: int,
        heads: int,
        ffn_hidden: Sequence[int],
        activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.relu,
        n_neighbor: int=1,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads,
        self.ffn_hidden = ffn_hidden
        self.activation = activation

        sites = get_sites()
        self.edge_index = sites.get_neighbor(n_neighbor=n_neighbor).T
        
        self.attn = GATv2Conv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            concat=False,
            negative_slope=0.2,
            add_self_loops=False,
            key=get_subkeys()
        )

        self.ffn = FeedForward(
            in_dim=out_channels,
            hidden_dims=ffn_hidden,
            out_dim=out_channels,
            activation=activation,
            key=get_subkeys()
        )

        self.ln1 = eqx.nn.LayerNorm(
            shape=in_channels,
            eps=1e-7
        )

        self.ln2 = eqx.nn.LayerNorm(
            shape=out_channels,
            eps=1e-7
        )

        self.affn1 = eqx.nn.Linear(
            in_features=in_channels,
            out_features=out_channels,
            key=get_subkeys()
        )

        self.affn2 = eqx.nn.Linear(
            in_features=out_channels,
            out_features=out_channels,
            key=get_subkeys()
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x_res = jax.vmap(self.affn1)(x)
        out = jax.vmap(self.ln1)(x)
        out = self.attn(out, self.edge_index)
        out = 0.70710678 * out + 0.70710678 * x_res

        x_res = jax.vmap(self.affn2)(out)
        out = jax.vmap(self.ln2)(out)
        out = 0.70710678 * self.ffn(out) + 0.70710678 * x_res

        return out
    

@partial(jax.jit, static_argnums=(2,))
def scatter_softmax(x: jax.Array, index: jax.Array, out_dim: int) -> jax.Array:
    """
        This function compute the softmax individually according to the group of index.
    """
    out = -jnp.inf * jnp.ones([out_dim]+list(x.shape[1:]))
    out = out.at[index].max(x)
    out = jnp.exp(x - out[index])
    out_sum = jnp.zeros([out_dim]+list(x.shape[1:]))
    out_sum = out_sum.at[index].add(out) + 1e-16

    return out / out_sum[index]


