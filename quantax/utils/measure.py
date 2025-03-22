from typing import Union, Optional, Tuple
from jaxtyping import PyTree
import jax.numpy as jnp

def expectation(
    operator,
    state,
    samples,
    internal: Optional[PyTree] = None,
    return_var: bool = False
) -> Union[float, Tuple[float, float]]:
    r"""
    See `operator.expectation`
    """
    return operator.expectation(state, samples, internal, return_var)
    

def overlap(state1, state2, samples1, samples2, return_ratios: bool = False):
    s1_wf1 = samples1.wave_function
    s2_wf2 = samples2.wave_function
    s1_wf2 = state2(samples1.spins)
    s2_wf1 = state1(samples2.spins)
    ratio1 = jnp.mean(s1_wf2 / s1_wf1 * samples1.reweight_factor)
    ratio2 = jnp.mean(s2_wf1 / s2_wf2 * samples2.reweight_factor)
    if return_ratios:
        return ratio1, ratio2
    else:
        return ratio1 * ratio2
