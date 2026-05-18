# Copyright 2026 QHAna plugin runner contributors.
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

"""Unit and property-based tests for the data-creator dataset generators.

The dataset generators in :mod:`data_creator.backend.datasets` are pure
functions on top of numpy and scikit-learn. They have no Flask, database, or
Celery dependencies, so the tests in this module run as plain pytest cases
without any fixtures.

Two test styles appear here:

* **Example-driven unit tests** assert the documented contract for a fixed
  set of inputs (shape, dimensionality, label range).
* **Property-based tests** with `hypothesis
  <https://hypothesis.readthedocs.io/>`_ generate many random parameter
  combinations and check invariants that should hold for every input
  (output length, label dtype, no NaN or infinity).

The combination catches both regressions in known-good cases and edge cases
the author did not think of when writing the example-driven tests.
"""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from data_creator.backend.datasets import (
    DataTypeEnum,
    blobs,
    blobs_3d,
    checkerboard,
    checkerboard_3d,
    twospirals,
)

# Example-driven unit tests.


@pytest.mark.parametrize("data_type", list(DataTypeEnum), ids=lambda e: e.name)
def test_get_data(data_type):
    """Check ``get_data`` shapes across several ``(n_train, n_test)`` pairs.

    Uses :py:func:`pytest.mark.parametrize` to generate one independent test
    case per :py:class:`DataTypeEnum` member.
    """
    expected_dim = (
        3 if data_type in {DataTypeEnum.checkerboard_3d, DataTypeEnum.blobs_3d} else 2
    )

    for n_train, n_test in [(1, 0), (10, 0), (20, 10), (50, 50), (100, 1)]:
        train_data, train_labels, test_data, test_labels = data_type.get_data(
            n_train, n_test
        )

        assert (
            isinstance(train_data, list)
            and isinstance(train_labels, list)
            and isinstance(test_data, list)
            and isinstance(test_labels, list)
        )
        assert len(train_data) == n_train
        assert len(train_labels) == n_train
        if data_type is DataTypeEnum.two_spirals:
            # twospirals mirrors its input, so total = 2*(n_train + n_test)
            # and the test partition is whatever remains after taking n_train.
            assert len(test_data) == 2 * (n_train + n_test) - n_train
            assert len(test_labels) == 2 * (n_train + n_test) - n_train
        else:
            assert len(test_data) == n_test
            assert len(test_labels) == n_test
        assert len(test_labels) == len(test_data)
        assert all(len(p) == expected_dim for p in train_data)
        assert all(len(p) == expected_dim for p in test_data)


def test_twospirals_doubles_point_count():
    """``twospirals`` mirrors each spiral, returning ``2 * n_points`` rows."""
    np.random.seed(0)
    data, labels = twospirals(50)
    assert data.shape == (100, 2)
    assert labels.shape == (100,)


def test_twospirals_labels_are_balanced_binary():
    """One spiral is labelled ``0``, the mirrored spiral is labelled ``1``."""
    np.random.seed(0)
    _, labels = twospirals(50)
    assert set(np.unique(labels).tolist()) == {0, 1}
    assert int((labels == 0).sum()) == 50
    assert int((labels == 1).sum()) == 50


def test_checkerboard_shape_and_labels():
    np.random.seed(0)
    data, labels = checkerboard(80)
    assert data.shape == (80, 2)
    assert set(np.unique(labels).tolist()).issubset({0, 1})


def test_checkerboard_pushes_points_off_axes():
    """The generator nudges every point at least ``0.2`` away from each axis."""
    np.random.seed(0)
    data, _ = checkerboard(100)
    assert np.all(np.abs(data) >= 0.2)


def test_blobs_label_range_matches_centers():
    np.random.seed(0)
    data, labels = blobs(60, centers=3)
    assert data.shape == (60, 2)
    assert set(np.unique(labels).tolist()) == {0, 1, 2}


def test_checkerboard_3d_returns_three_features():
    np.random.seed(0)
    data, labels = checkerboard_3d(40)
    assert data.shape == (40, 3)
    assert set(np.unique(labels).tolist()).issubset({0, 1})


def test_blobs_3d_returns_three_features():
    np.random.seed(0)
    data, labels = blobs_3d(60, centers=2)
    assert data.shape == (60, 3)
    assert set(np.unique(labels).tolist()) == {0, 1}


# Property-based tests.

# Strategies. Bounds are chosen to keep generation fast and to avoid degenerate
# cases (zero points, NaN-producing turns) that the production code is not
# expected to handle.
_n_points = st.integers(min_value=1, max_value=50)
_noise = st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False)
_turns = st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False)
_centers = st.integers(min_value=1, max_value=8)


@given(n=_n_points, noise=_noise, turns=_turns)
def test_twospirals_invariants(n, noise, turns):
    """For any valid parameters, ``twospirals`` returns ``2*n`` finite 2D points."""
    data, labels = twospirals(n, noise=noise, turns=turns)
    assert data.shape == (2 * n, 2)
    assert labels.shape == (2 * n,)
    assert np.all(np.isfinite(data))
    assert np.issubdtype(labels.dtype, np.integer)


@given(n=_n_points)
def test_checkerboard_invariants(n):
    data, labels = checkerboard(n)
    assert data.shape == (n, 2)
    assert labels.shape == (n,)
    assert np.all(np.isfinite(data))
    assert np.issubdtype(labels.dtype, np.integer)
    assert set(np.unique(labels).tolist()).issubset({0, 1})


@given(n=_n_points, centers=_centers)
def test_blobs_invariants(n, centers):
    data, labels = blobs(n, centers=centers)
    assert data.shape == (n, 2)
    assert labels.shape == (n,)
    assert np.all(np.isfinite(data))
    assert np.issubdtype(labels.dtype, np.integer)
    assert int(labels.min()) >= 0
    assert int(labels.max()) < centers


@given(n=_n_points)
def test_checkerboard_3d_invariants(n):
    data, labels = checkerboard_3d(n)
    assert data.shape == (n, 3)
    assert labels.shape == (n,)
    assert np.all(np.isfinite(data))
    assert set(np.unique(labels).tolist()).issubset({0, 1})


@given(n=_n_points, centers=_centers)
def test_blobs_3d_invariants(n, centers):
    data, labels = blobs_3d(n, centers=centers)
    assert data.shape == (n, 3)
    assert labels.shape == (n,)
    assert np.all(np.isfinite(data))
    assert int(labels.max()) < centers
