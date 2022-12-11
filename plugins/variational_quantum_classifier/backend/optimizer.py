from enum import Enum
from qiskit.algorithms.optimizers import ADAM, AQGD, BOBYQA, COBYLA, NELDER_MEAD, SPSA, POWELL, NFT, TNC


class OptimizerEnum(Enum):
    adam = "ADAM"
    aqgd = "AQGD"
    bobyqa = "BOBYQA"
    cobyla = "COBYLA"
    nelder_mead = "NELDER_MEAD"
    spsa = "SPSA"
    powell = "POWELL"
    nft = "NFT"
    tnc = "TNC"

    def get_optimizer(self, maxitr):
        if self == OptimizerEnum.adam:
            return ADAM(maxiter=maxitr)
        if self == OptimizerEnum.aqgd:
            return AQGD(maxiter=maxitr)
        if self == OptimizerEnum.bobyqa:
            return BOBYQA(maxiter=maxitr)
        if self == OptimizerEnum.cobyla:
            return COBYLA(maxiter=maxitr)
        if self == OptimizerEnum.nelder_mead:
            return NELDER_MEAD(maxiter=maxitr)
        if self == OptimizerEnum.spsa:
            return SPSA(maxiter=maxitr)
        if self == OptimizerEnum.powell:
            return POWELL(maxiter=maxitr)
        if self == OptimizerEnum.nft:
            return NFT(maxiter=maxitr)
        if self == OptimizerEnum.tnc:
            return TNC(maxiter=maxitr)
