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
    BOBYQA,
    COBYLA,
    NELDER_MEAD,
    SPSA,
    POWELL,
    NFT,
    TNC,
)


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
