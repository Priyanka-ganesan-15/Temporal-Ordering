"""Spectral ordering via the Fiedler vector of the similarity graph Laplacian."""

from __future__ import annotations

import numpy as np

from chronologic.ordering.nearest_neighbor import validate_similarity_matrix


def spectral_fiedler_ordering(
    similarity_matrix: np.ndarray,
) -> tuple[list[int], float]:
    """Order frames using the Fiedler vector of the graph Laplacian.

    Builds an undirected weighted graph where edge weights are cosine
    similarities.  The Fiedler vector (eigenvector for the second-smallest
    eigenvalue of the normalised Laplacian) captures the primary connectivity
    gradient of the graph.  Sorting frames by their Fiedler coordinate yields
    a 1-D temporal ordering.

    Returns
    -------
    ordering : list[int]
        Frame indices in the predicted temporal order.
    path_score : float
        Total adjacent-similarity score of the returned ordering.
    """
    validate_similarity_matrix(similarity_matrix)
    n = similarity_matrix.shape[0]

    if n == 1:
        return [0], 0.0

    # Adjacency matrix — zero the diagonal so self-loops don't distort the Laplacian
    A = similarity_matrix.copy().astype(np.float64)
    np.fill_diagonal(A, 0.0)

    # Degree matrix and un-normalised Laplacian
    degree = A.sum(axis=1)
    # Guard against isolated nodes (all-zero rows) by clamping to 1
    degree_safe = np.where(degree < 1e-12, 1.0, degree)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degree_safe))

    # Symmetric normalised Laplacian: L_sym = I - D^{-1/2} A D^{-1/2}
    L_sym = np.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt

    # eigh returns eigenvalues in ascending order for symmetric matrices
    eigenvalues, eigenvectors = np.linalg.eigh(L_sym)  # shape: (n,), (n, n)

    # Fiedler vector — column at index 1 (second smallest eigenvalue)
    fiedler_vec = eigenvectors[:, 1]

    ordering = list(np.argsort(fiedler_vec))

    # Compute adjacency path score for comparison with other methods
    path_score = float(
        sum(similarity_matrix[ordering[i], ordering[i + 1]] for i in range(n - 1))
    )

    return ordering, path_score


__all__ = ["spectral_fiedler_ordering"]
