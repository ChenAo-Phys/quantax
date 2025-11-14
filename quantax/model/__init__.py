from .shallow_nets import SingleDense, RBM_Dense, SingleConv, RBM_Conv
from .conv_nets import ResConv, ResGConv
from .deprecated import ConvTransformer
from .triangular_nets import Triangular_Neighbor_Conv, Triangular_ResConv
from .fermion_mf import (
    GeneralDet,
    RestrictedDet,
    UnrestrictedDet,
    MultiDet,
    GeneralPf,
    SingletPair,
    MultiPf,
    PartialPair,
)
from .jastrow import NeuralJastrow
from .transformer import Transformer, Embedding, MHSA, FFN
