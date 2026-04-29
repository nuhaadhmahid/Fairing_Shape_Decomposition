import os
import sys
import json
import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import svd


# ---------------------------------------------------------------------------
# I/O utilities
# ---------------------------------------------------------------------------

def save_object(obj, path, method):
    """Save obj to path using 'pickle' or 'json'."""
    if method == "pickle":
        with open(path + ".pickle", "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    elif method == "json":
        with open(path + ".json", "w") as f:
            json.dump(obj, f, indent=4)
    else:
        print(f"ERROR: Unknown save method '{method}'")


def load_object(path, method):
    """Load an object from path using 'pickle' or 'json'.

    Handles pickles created with Python 2 by falling back to latin-1 encoding.
    """
    if method == "pickle":
        with open(path + ".pickle", "rb") as f:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=np.exceptions.VisibleDeprecationWarning)
                try:
                    return pickle.load(f)
                except UnicodeDecodeError:
                    try:
                        return pickle.load(f, encoding="latin1")
                    except Exception:
                        print("ERROR: Failed to load pickle file.")
                        return None
    elif method == "json":
        with open(path + ".json", "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        print(f"ERROR: Unknown load method '{method}'")
        return None


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

class FairingData:
    def __init__(
        self,
        case_folder,
        case_number,
        hinge_node=None,
        surface_nodes=None,
        shell_equivalent=None,
    ):
        self.case_folder = case_folder
        self.case_number = case_number
        self.hinge_node = hinge_node or {}
        self.surface_nodes = surface_nodes or {}
        self.shell_equivalent = shell_equivalent or {}


# ---------------------------------------------------------------------------
# SVD shape decomposition helpers
# ---------------------------------------------------------------------------

def compute_svd_modes(dX):
    """Return U, S (diagonal matrix), V_T from SVD of the deformation field."""
    U, s, V_T = svd(dX.reshape(-1, 3), full_matrices=False)
    print("Singular values (normalised):", [f"{v/s[0]:.2f}" for v in s])
    return U, np.diag(s), V_T


def reconstruct_deformation(U, S, V_T, r, shape):
    """Reconstruct deformation using the first r SVD modes."""
    return (U[:, :r] @ S[:r, :r] @ V_T[:r, :]).reshape(shape)


def reconstruction_error(X_approx, X_ref):
    """Mean nodal Euclidean error between X_approx and X_ref."""
    diff = np.asarray(X_approx - X_ref, dtype=np.float64).reshape(-1, 3)
    return np.mean(np.linalg.norm(diff, axis=1))


def plot_mode(ax, X, ndXk_contribution, X1, surface_nodes_coords, r, e, ek):
    """Scatter-plot rank-k SVD contribution vs. original deformed shape."""
    X_approx = np.asarray(X + ndXk_contribution, dtype=np.float64)
    X1_plot  = np.asarray(X1, dtype=np.float64)

    ax.scatter(*X_approx.T, alpha=0.3, color="red",  s=1, label="SVD")
    ax.scatter(*X1_plot.T,  alpha=0.5, color="grey", s=1, label="Original")

    aspect = tuple(np.ptp(surface_nodes_coords[:, i]) for i in range(3))
    ax.set_box_aspect(aspect)
    ax.axis("off")
    ax.set_title(f"k={r},  e={e:.3f},  ek={ek:.3f}", fontsize=9, x=0.5, y=0.8)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # --- Arguments & paths --------------------------------------------------
    CASE_NAME = sys.argv[-2]
    FILE_ID   = sys.argv[-1]

    CASE_DIR = os.path.join(os.path.abspath(os.getcwd()), CASE_NAME)
    DATA_DIR = os.path.join(CASE_DIR, "data")

    # --- Load data ----------------------------------------------------------
    surface_mesh = load_object(os.path.join(DATA_DIR, f"{FILE_ID}_fairing_mesh_data"), "pickle")
    if surface_mesh is None:
        raise RuntimeError("Failed to load surface mesh data.")

    surface_nodes_set    = surface_mesh["surface_nodes"]
    surface_nodes_coords = surface_mesh["surface_nodes_coords"]

    fairing_data = load_object(os.path.join(DATA_DIR, f"{FILE_ID}_fairing_data"), "pickle")
    if fairing_data is None:
        raise RuntimeError("Failed to load fairing data.")

    # --- Extract displacements at final increment ---------------------------
    final_increment = len(list(fairing_data.surface_nodes_U.values())[0]) - 1
    surface_nodes_displacements = np.array(
        [fairing_data.surface_nodes_U[n][final_increment] for n in surface_nodes_set],
        dtype=np.float32,
    )

    # --- SVD of deformation field -------------------------------------------
    X  = surface_nodes_coords         # undeformed
    dX = surface_nodes_displacements  # deformation
    X1 = X + dX                       # deformed

    U, S, V_T = compute_svd_modes(dX)

    # --- Plot rank-1, 2, 3 reconstructions ----------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(9, 3), subplot_kw={"projection": "3d"})

    for idx, r in enumerate([1, 2, 3]):
        ndXk_full        = reconstruct_deformation(U, S, V_T, r, np.shape(dX))
        ndXk_contribution = (U[:, r-1:r] @ S[r-1:r, r-1:r] @ V_T[r-1:r, :]).reshape(np.shape(dX))

        e  = reconstruction_error(X + ndXk_full, X1)
        ek = reconstruction_error(X + ndXk_contribution, X1)

        plot_mode(axes[idx], X, ndXk_contribution, X1, surface_nodes_coords, r, e, ek)

    plt.tight_layout()
    plt.show()

