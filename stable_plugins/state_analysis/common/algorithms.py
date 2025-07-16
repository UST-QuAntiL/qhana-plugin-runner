from typing import List, Set

import numpy as np

DEFAULT_TOLERANCE = 1e-10


def are_vectors_linearly_dependent(
    vectors: List[np.ndarray], tolerance: float = DEFAULT_TOLERANCE
) -> bool:
    """
    Checks if a list of vectors is linearly dependent.

    Args:
        vectors (List[np.ndarray]): A list of NumPy arrays representing vectors.
        tolerance (float): A small tolerance for numerical stability (default 1e-10).

    Returns:
        bool: True if the vectors are linearly dependent, False otherwise.

    Raises:
        ValueError: If the list of vectors is empty or if the vectors have different shapes.
        TypeError: If any element in the list is not a NumPy array.
    """

    # Set default for tolerance
    if tolerance is None:
        tolerance = DEFAULT_TOLERANCE

    # Validate input
    if not vectors:
        raise ValueError("The input list of vectors is empty.")

    if not all(isinstance(vec, np.ndarray) for vec in vectors):
        raise TypeError("All elements in the input list must be NumPy arrays.")

    if not all(vec.shape == vectors[0].shape for vec in vectors):
        raise ValueError(f"All vectors must have the same shape. All vectors={vectors}")

    if len(vectors) == 1:
        vec = vectors[0]
        if np.allclose(vec, 0, atol=tolerance):
            return True  # Zero vector is linearly dependent
        else:
            return False  # Any other single vector is linearly independent

    try:
        # Convert the list of vectors into a matrix where each vector is a row
        matrix = np.vstack(vectors)

        # Calculate the rank of the matrix
        rank = np.linalg.matrix_rank(matrix, tol=tolerance)

        # If the rank is smaller than the number of vectors, they are linearly dependent
        return rank < len(vectors)
    except np.linalg.LinAlgError as e:
        raise RuntimeError(f"An error occurred while computing the matrix rank: {e}")


def are_vectors_linearly_dependent_inhx(
    states: List[np.ndarray],
    dimHX: int,
    dimHR: int,
    singular_value_tolerance: float = DEFAULT_TOLERANCE,
    linear_independence_tolerance: float = DEFAULT_TOLERANCE,
) -> bool:
    """
    Checks if the collected Schmidt basis vectors in subsystem Hx
    (obtained via SVD from bipartite states in Hx ⊗ Hr) are linearly dependent.

    Args:
        states: List of state vectors (1D NumPy arrays) in Hx ⊗ Hr.
        dimHX:  Dimension of subsystem Hx.
        dimHR:  Dimension of subsystem Hr.
        singular_value_tolerance: Threshold below which singular values are ignored.
        linear_independence_tolerance: Tolerance for rank-based linear dependence checks.

    Returns:
        True if the collected basis vectors are linearly dependent, otherwise False.

    Raises:
        ValueError: If a state has an unexpected size.
        TypeError: If `states` is not a list of NumPy arrays.
        RuntimeError: If an SVD fails to converge for a given state.
    """
    if not isinstance(dimHX, int) or not isinstance(dimHR, int):
        raise TypeError("`dimHX` and `dimHR` must be integers.")
    if dimHX < 0 or dimHR < 0:
        raise ValueError("`dimHX` and `dimHR` must be positive integers.")
    if not isinstance(states, list):
        raise TypeError("`states` must be a list of NumPy arrays.")
    if not all(isinstance(state, np.ndarray) for state in states):
        raise TypeError("All elements in `states` must be NumPy arrays.")
    if not all(state.size == (2 ** (dimHR + dimHX)) for state in states):
        raise ValueError(
            f"Each state must have a size of 2 ** (dimHR + dimHX), but at least one state has an incorrect size."
        )

    if not singular_value_tolerance:
        singular_value_tolerance = DEFAULT_TOLERANCE
    if not linear_independence_tolerance:
        linear_independence_tolerance = DEFAULT_TOLERANCE

    collected_basis = []

    for state in states:
        # Reshape to (dimHX, dimHR)
        state_matrix = state.reshape(2**dimHX, 2**dimHR)

        # SVD -> M = U * S * V^dagger
        try:
            u, s, vh = np.linalg.svd(state_matrix)
        except np.linalg.LinAlgError as e:
            raise RuntimeError(f"SVD failed for state: {e}") from e

        # Filter basis vectors based on singular values
        # Collect the corresponding columns of U as basis vectors in Hx
        # (columns of U are the left singular vectors)
        for index, singular_value in enumerate(s):
            if singular_value > singular_value_tolerance:
                basis = u[:, index]
                collected_basis.append(basis)

    # Check if the collected vectors are linearly dependent
    if not collected_basis:
        # If no vectors, they are trivially independent
        return False
    return are_vectors_linearly_dependent(collected_basis, linear_independence_tolerance)


def are_vectors_orthogonal(
    vec1: np.ndarray, vec2: np.ndarray, tolerance: float = DEFAULT_TOLERANCE
) -> bool:
    """
    Checks whether two NumPy vectors are orthogonal, considering complex conjugates for complex vectors.

    Args:
        vec1 (np.ndarray): The first vector.
        vec2 (np.ndarray): The second vector.
        tolerance (float): The tolerance value for checking orthogonality (default is 1e-10).

    Returns:
        bool: True if the vectors are orthogonal, False otherwise.

    Raises:
        ValueError: If the vectors do not have the same dimension.
        TypeError: If the inputs are not NumPy arrays or cannot be interpreted as such.
    """
    # Set default for tolerance
    if tolerance is None:
        tolerance = DEFAULT_TOLERANCE

    # Input validation
    if not isinstance(vec1, np.ndarray) or not isinstance(vec2, np.ndarray):
        raise TypeError("Both inputs must be NumPy arrays.")

    if vec1.ndim != 1 or vec2.ndim != 1:
        raise ValueError(
            f"Both inputs must be 1-dimensional arrays (vectors). vec1: {vec1}, vec2: {vec2}"
        )

    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have the same dimension.")

    try:
        # Compute the dot product using the complex conjugate of vec1
        dot_product = np.vdot(vec1, vec2)  # Handles complex vectors correctly
    except Exception as e:
        raise RuntimeError(f"An error occurred while computing the dot product: {e}")

    # Check if the dot product is close to zero within the specified tolerance
    is_orthogonal = abs(dot_product) <= tolerance

    return is_orthogonal


def compute_schmidt_rank(
    state: np.ndarray, dimHX: int, dimHR: int, tolerance: float = DEFAULT_TOLERANCE
) -> int:
    """
    Computes the Schmidt rank of a bipartite quantum state using singular value decomposition (SVD).

    Args:
        state (np.ndarray): The state vector representing the bipartite quantum system.
        dimHX (int): Dimension of subsystem Hx.
        dimHR (int): Dimension of subsystem Hr.
        tolerance (float): Threshold below which singular values are considered zero (default: 1e-10).

    Returns:
        int: The Schmidt rank of the state, i.e., the number of non-zero Schmidt coefficients.

    Raises:
        ValueError: If the dimensions `dimHX * dimHR` do not match the size of `state`.
        TypeError: If `state` is not a NumPy array or if dimensions are invalid.
    """
    # Set default value for tolerance
    if tolerance is None:
        tolerance = DEFAULT_TOLERANCE

    # Input validation
    if not isinstance(state, np.ndarray):
        raise TypeError("The input `state` must be a NumPy array.")
    if not isinstance(dimHX, int) or not isinstance(dimHR, int):
        raise TypeError("`dimHX` and `dimHR` must be integers.")
    if dimHX < 0 or dimHR < 0:
        raise ValueError("`dimHX` and `dimHR` must be positive integers.")
    if not state.size == (2 ** (dimHR + dimHX)):
        raise ValueError(
            f"Each state must have a size of 2 ** (dimHR + dimHX), but at least one state has an incorrect size."
        )

    try:
        # Step 1: Reshape the state vector into a matrix of shape (dimHX, dimHR)
        state_matrix = state.reshape(2**dimHX, 2**dimHR)

        # U, S, Vh are the unitary matrices and singular values
        U, S, Vh = np.linalg.svd(state_matrix)

        # Step 3: Count the number of singular values greater than the tolerance
        schmidt_rank = np.sum(S > tolerance)
    except np.linalg.LinAlgError as e:
        raise RuntimeError(f"An error occurred during the SVD computation: {e}")

    return schmidt_rank


def dfs(vectors: List[np.ndarray], visited: Set[int], current: int, tolerance: float):
    """Recursive Depth-First Search (DFS) algorithm that checks if all nodes can be reached."""
    visited.add(current)
    for neighbor in range(len(vectors)):
        if neighbor not in visited and not are_vectors_orthogonal(
            vectors[current], vectors[neighbor], tolerance
        ):
            dfs(vectors, visited, neighbor, tolerance)


def are_vectors_orthogonal_partitioning_resistance(
    vectors: List[np.ndarray], tolerance: float = DEFAULT_TOLERANCE
) -> bool:
    """
    Checks whether the graph is fully connected using DFS without an adjacency list.
    :param vectors: List of vectors representing the graph nodes.
    :param tolerance: Tolerance for the dot product to determine orthogonality.
    :return: True if the graph is fully connected, otherwise False.
    """
    if not vectors:
        return False

    visited = set()
    dfs(vectors, visited, 0, tolerance)

    return len(visited) == len(vectors)


def are_vectors_pairwise_orthogonal(
    vectors: List[np.ndarray], tolerance: float = DEFAULT_TOLERANCE
) -> bool:
    """
    Checks whether all vectors in the list are pairwise orthogonal.

    :param vectors: List of numpy arrays representing vectors.
    :param tolerance: Tolerance for the dot product to determine orthogonality.
    :return: True if all vectors are pairwise orthogonal, otherwise False.
    """
    n = len(vectors)

    for i in range(n):
        for j in range(i + 1, n):
            if not are_vectors_orthogonal(vectors[i], vectors[j], tolerance):
                return False  # Early exit if any pair is not orthogonal

    return True  # If all pairs are orthogonal, return True
