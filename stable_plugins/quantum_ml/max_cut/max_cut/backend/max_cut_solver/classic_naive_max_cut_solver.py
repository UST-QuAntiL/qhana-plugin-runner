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

import numpy as np
import networkx as nx
from itertools import chain, combinations
from typing import Set
from collections.abc import Iterable

from celery.utils.log import get_task_logger


TASK_LOGGER = get_task_logger(__name__)


class ClassicNaiveMaxCutSolver:
    def __init__(self, graph: nx.Graph):
        self.graph = graph

    """
    Solves the max cut problem classically and
    naive (O(2^n)) and returns the
    maximum cut in the format (cutValue, [(node1, node2), ...]),
    i.e. the cut value and the list of edges that
    correspond to the cut.
    """

    def solve(self) -> (np.array, float):
        nodes = set(self.graph.nodes())
        powerset = set(self.__get_powerset(nodes))

        largestCutValue = 0.0
        largestCutSubset = None
        tempCutValue = 0.0

        printMod = 100
        probsize = len(powerset)
        count = 0

        for subset in powerset:
            count = count + 1
            tempCutValue = self.__calculate_cut_value(nodes, set(subset))
            if tempCutValue > largestCutValue:
                largestCutValue = tempCutValue
                largestCutSubset = list(subset)

            # Print output
            if count % printMod == 0:
                TASK_LOGGER.info(str(count) + " / " + str(probsize) + " cuts checked")
            if count >= probsize:
                break

        largestCutSubset = np.array(largestCutSubset)
        cut = np.zeros(len(nodes))
        cut[largestCutSubset] = 1

        return cut, largestCutValue

    def __calculate_cut_edges(self, set: Set, subset: Set):
        diffSubSet = set - subset

        cutEdges = []

        for a in subset:
            for b in diffSubSet:
                cutEdges.append((a, b))

        return cutEdges

    def __calculate_cut_value(self, set: Set, subset: Set) -> float:
        diffSubSet = set - subset

        cutValue = 0.0

        for a in subset:
            for b in diffSubSet:
                cutValue = cutValue + float(self.graph[a][b]["weight"])

        return cutValue

    """
    Returns the powerset of the given set,
    excluding the empty set.
    """

    def __get_powerset(self, nodes: Iterable):
        s = list(nodes)
        return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))
