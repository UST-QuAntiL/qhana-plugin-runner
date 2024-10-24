# Copyright 2023 QHAna plugin runner contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import Enum
from qiskit.algorithms.optimizers import (
    ADAM,
    AQGD,
    CG,
    COBYLA,
    L_BFGS_B,
    GSLS,
    GradientDescent,
    NELDER_MEAD,
    NFT,
    P_BFGS,
    POWELL,
    SLSQP,
    SPSA,
    # QNSPSA,
    TNC,
    BOBYQA,
    IMFIL,
    SNOBFIT,
)


class OptimizerEnum(Enum):
    adam = "Adam"
    aqgd = "AQGD"
    cg = "CG"
    cobyla = "cobyla"
    l_bfgs_b = "L-BFGS-B"
    gsls = "GSLS"
    gradient_descent = "Gradient Descent"
    nelder_mead = "Nelder Mead"
    nft = "NFT"
    p_bfgs = "P-BFGS"
    powell = "Powell"
    slsqp = "SLSQP"
    spsa = "SPSA"
    # qnspsa = "QNSPSA"
    tnc = "TNC"
    bobyqa = "BOBYQA"
    imfil = "IMFIL"
    snobfit = "SNOBFIT"

    def get_optimizer(self, maxiter: int):
        if self == OptimizerEnum.adam:
            return ADAM(maxiter=maxiter)
        elif self == OptimizerEnum.aqgd:
            return AQGD(maxiter=maxiter)
        elif self == OptimizerEnum.cg:
            return CG(maxiter=maxiter)
        elif self == OptimizerEnum.cobyla:
            return COBYLA(maxiter=maxiter)
        elif self == OptimizerEnum.l_bfgs_b:
            return L_BFGS_B(maxiter=maxiter)
        elif self == OptimizerEnum.gsls:
            return GSLS(maxiter=maxiter)
        elif self == OptimizerEnum.gradient_descent:
            return GradientDescent(maxiter=maxiter)
        elif self == OptimizerEnum.nelder_mead:
            return NELDER_MEAD(maxiter=maxiter)
        elif self == OptimizerEnum.nft:
            return NFT(maxiter=maxiter)
        elif self == OptimizerEnum.p_bfgs:
            return P_BFGS()
        elif self == OptimizerEnum.powell:
            return POWELL(maxiter=maxiter)
        elif self == OptimizerEnum.slsqp:
            return SLSQP(maxiter=maxiter)
        elif self == OptimizerEnum.spsa:
            return SPSA()
        # elif self == OptimizerEnum.qnspsa:
        #     QNSPSA(maxiter=maxiter)
        elif self == OptimizerEnum.tnc:
            return TNC(maxiter=maxiter)
        elif self == OptimizerEnum.bobyqa:
            return BOBYQA(maxiter=maxiter)
        elif self == OptimizerEnum.imfil:
            return IMFIL(maxiter=maxiter)
        elif self == OptimizerEnum.snobfit:
            return SNOBFIT(maxiter=maxiter)
