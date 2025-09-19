from .shallow_nets import SingleDense, RBM_Dense, SingleConv, RBM_Conv
from .conv_nets import ResConv, ResGConv
from .transformer import ConvTransformer
from .sign_nets import SgnNet, MarshallSign, StripeSign, Neel120
from .triangular_nets import Triangular_Neighbor_Conv, Triangular_ResSum
from .fermion_mf import (
    GeneralDet,
    RestrictedDet,
    UnrestrictedDet,
    MultiDet,
    GeneralPf,
    SingletPair,
    MultiPf,
)
from .neural_fermion import NeuralJastrow
