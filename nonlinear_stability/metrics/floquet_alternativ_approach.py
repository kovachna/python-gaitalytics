import os
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _build_normalized_trajectories(seg: xr.DataArray, M: int):
    """
    Given seg(cycle, time, axis), returns
      com_norm: (n_cycles, M, n_axes)
      vel_norm: (n_cycles, M, n_axes) via central diff
    """
    n_cycles, _, n_axes = seg.shape
    norm_t = np.linspace(0, 1, M)
    com = seg.values
    com_norm = np.zeros((n_cycles, M, n_axes))
    for i in range(n_cycles):
        t = seg.isel(cycle=i).coords["time"].values
        t_n = (t - t[0]) / (t[-1] - t[0])
        for j in range(n_axes):
            com_norm[i, :, j] = np.interp(norm_t, t_n, com[i, :, j])

    dt = 1.0 / (M - 1)
    vel_norm = np.zeros_like(com_norm)
    # central difference
    vel_norm[:, 1:-1, :] = (com_norm[:, 2:, :] - com_norm[:, :-2, :])/(2*dt)
    # endpoints
    vel_norm[:, 0, :]    = (com_norm[:, 1, :] - com_norm[:, 0, :]) / dt
    vel_norm[:, -1, :]   = (com_norm[:, -1, :] - com_norm[:, -2, :]) / dt

    return com_norm, vel_norm


def _estimate_J(dX: np.ndarray, dY: np.ndarray, rcond=1e-6):
    """
    Solve dY = J dX via least-squares, with ridge fallback.
    Returns J (state_dim x state_dim).
    """
    try:
        J = dY @ np.linalg.pinv(dX, rcond=rcond)
    except np.linalg.LinAlgError:
        lam = rcond * np.trace(dX @ dX.T)
        J = dY @ dX.T @ np.linalg.inv(dX @ dX.T + lam * np.eye(dX.shape[0]))
    return J


def _analyze_map(S_from: np.ndarray, S_to: np.ndarray, plot_spectrum=False, title=""):
    """
    Fit Jacobian J for map S_from -> S_to, return (J, lambdas)
    """
    X = S_from[:-1].T    # state_dim x (N-1)
    Y = S_to[1:].T       # state_dim x (N-1)
    Xm = X - X.mean(axis=1, keepdims=True)
    Ym = Y - Y.mean(axis=1, keepdims=True)
    J = _estimate_J(Xm, Ym)
    lambdas, _ = np.linalg.eig(J)
    if plot_spectrum:
        mag = np.abs(lambdas)
        plt.figure()
        plt.plot(np.arange(len(mag)), mag, 'o-')
        plt.ylim(0, 1.2)
        plt.xlabel('Mode #')
        plt.ylabel('|λ|')
        plt.title(f'Floquet spectrum: {title}')
        plt.grid(True)
        plt.show()
    return J, lambdas


def compute_stride_and_step_floquet(data: dict,
                                    channel="COM",
                                    axes=["x","y","z"],
                                    M=101,
                                    plot_spectrum=False):
    """
    For each dataset in data dict:
      - J_stride: Right->Right
      - J_RL, J_LR: half-stride maps
      - J_composite: J_LR @ J_RL
    Returns dict of results.
    """
    results = {}
    for name, seg_path in data.items():
        seg_da = seg_path.sel(channel=channel, axis=axes)
        n_cycles = seg_da.sizes["cycle"]
        state_dim = 2 * len(axes)
        if n_cycles <= state_dim:
            raise ValueError(f"{name}: need >{state_dim} cycles, got {n_cycles}")

        # Right->Right full-stride
        segR = seg_da.sel(context="Right").transpose("cycle","time","axis")
        comR, velR = _build_normalized_trajectories(segR, M)
        S_R = np.hstack([comR[:,0,:], velR[:,0,:]])
        J_stride, lam_stride = _analyze_map(S_R, S_R, plot_spectrum, f"{name} stride")

        # Right->Left half-stride
        segL = seg_da.sel(context="Left").transpose("cycle","time","axis")
        comL, velL = _build_normalized_trajectories(segL, M)
        Np = min(S_R.shape[0], comL.shape[0])
        S_L = np.hstack([comL[:Np,0,:], velL[:Np,0,:]])
        J_RL, lam_RL = _analyze_map(S_R[:Np], S_L, plot_spectrum, f"{name} R→L")

        # Left->Right half-stride
        J_LR, lam_LR = _analyze_map(S_L[:-1], S_R[1:Np+1], plot_spectrum, f"{name} L→R")

        # Composite full-stride
        J_comp = J_LR @ J_RL
        lambdas_comp, _ = np.linalg.eig(J_comp)

        results[name] = {
            'J_stride': J_stride,
            'lambdas_stride': lam_stride,
            'J_R->L': J_RL,
            'lambdas_R->L': lam_RL,
            'J_L->R': J_LR,
            'lambdas_L->R': lam_LR,
            'J_composite': J_comp,
            'lambdas_comp': lambdas_comp
        }
    return results


def plot_poincare(S: np.ndarray, labels: list):
    """
    Plot Poincaré scatter for each component in S.
    S shape: (n_cycles, state_dim)
    labels length = state_dim
    """
    for idx, lab in enumerate(labels):
        xk  = S[:-1, idx]
        xk1 = S[1:,  idx]
        plt.figure(figsize=(5,5))
        plt.scatter(xk, xk1, s=20, alpha=0.6)
        mn, mx = xk.min(), xk.max()
        plt.plot([mn,mx], [mn,mx], 'k--', lw=1)
        plt.xlabel(f'{lab}$_k$')
        plt.ylabel(f'{lab}$_{{k+1}}$')
        plt.title(f'Poincaré plot: {lab}')
        plt.grid(True)
        plt.show()


def main():
    base_dir = Path("data/01")
    datasets = {
        'FWS': xr.open_dataarray(base_dir/'FWS'/'segmented'/'markers.nc'),
        'PWS': xr.open_dataarray(base_dir/'PWS'/'segmented'/'markers.nc'),
        'SWS': xr.open_dataarray(base_dir/'SWS'/'segmented'/'markers.nc')
    }
    # define normalization points
    M = 101

    # compute Floquet maps
    results = compute_stride_and_step_floquet(
        datasets, channel="COM", axes=["x","y","z"], M=M, plot_spectrum=False
    )

    # summarize and Poincaré
    for name, res in results.items():
        print(f"--- {name} ---")
        print(f"Stride max |λ|   : {np.max(np.abs(res['lambdas_stride'])):.4f}")
        print(f"R→L max |λ|      : {np.max(np.abs(res['lambdas_R->L'])):.4f}")
        print(f"L→R max |λ|      : {np.max(np.abs(res['lambdas_L->R'])):.4f}")
        print(f"Composite max |λ|: {np.max(np.abs(res['lambdas_comp'])):.4f}")

        # rebuild and plot Poincaré for S_R
        segR = datasets[name].sel(channel="COM", axis=["x","y","z"]).sel(context="Right").transpose('cycle','time','axis')
        comR, velR = _build_normalized_trajectories(segR, M)
        S_R = np.hstack([comR[:,0,:], velR[:,0,:]])
        labels = ['x','y','z','dot_x','dot_y','dot_z']
        print("Plotting Poincaré for full state at Right-strike... ")
        plot_poincare(S_R, labels)

if __name__ == '__main__':
    main()
