import os
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def compute_floquet(data: xr.DataArray,
                    context: str = "Left",
                    channel: str = "COM",
                    axes: list = ["x","y","z"],
                    normalization_points: int = 101,
                    rcond: float = 1e-6,
                    phase_dependent: bool = False,
                    plot_spectrum: bool = False):
    """
    Compute Floquet multipliers and Poincaré plots for the Poincaré map at each stride.

    - Checks that n_cycles >= state_dim+1 to ensure dX is full rank.
    - Time-normalizes each cycle to `normalization_points` samples.
    - Uses central-difference for velocity estimation.
    - Can return a single max |λ| or the full phase-dependent curve.
    - Optionally plots the eigenvalue spectrum and a Poincaré section.

    Returns:
      float (max |λ|) if phase_dependent=False,
      else np.ndarray of length normalization_points.
    """
    # 1) select and reorder: dims -> (cycle, time, axis)
    seg = data.sel(context=context, channel=channel, axis=axes).transpose('cycle','time','axis')
    com = seg.values  # (n_cycles, n_time, n_axes)
    n_cycles, n_time, n_axes = com.shape

    # 1a) ensure enough cycles for state_dim
    state_dim = 2 * n_axes
    if n_cycles - 1 < state_dim:
        raise ValueError(f"Need ≥{state_dim+1} cycles for state_dim={state_dim}, got {n_cycles}")

    # 2) time-normalize each cycle onto [0,1]
    M = normalization_points
    norm_t = np.linspace(0, 1, M)
    com_norm = np.zeros((n_cycles, M, n_axes))
    for i in range(n_cycles):
        t_cycle = seg.isel(cycle=i).coords['time'].values
        t_norm = (t_cycle - t_cycle[0]) / (t_cycle[-1] - t_cycle[0])
        for j in range(n_axes):
            com_norm[i, :, j] = np.interp(norm_t, t_norm, com[i, :, j])

    dt_norm = 1.0 / (M - 1)

    # 3) central-difference velocity over normalized grid
    vel_norm = np.zeros_like(com_norm)
    vel_norm[:, 1:-1, :] = (com_norm[:, 2:, :] - com_norm[:, :-2, :]) / (2 * dt_norm)
    vel_norm[:, 0, :]   = (com_norm[:, 1, :] - com_norm[:, 0, :])   / dt_norm
    vel_norm[:, -1, :]  = (com_norm[:, -1, :] - com_norm[:, -2, :]) / dt_norm

    # helper to estimate J and its max lambda
    def _estimate_J_and_maxlam(S_pos, S_vel):
        S = np.hstack([S_pos, S_vel])  # (n_cycles, state_dim)
        X = S[:-1, :]
        Y = S[1:, :]
        S_bar = S.mean(axis=0)
        dX = (X - S_bar).T
        dY = (Y - S_bar).T
        try:
            pinv_dX = np.linalg.pinv(dX, rcond=rcond)
            J = dY @ pinv_dX
        except np.linalg.LinAlgError:
            lam = rcond * np.trace(dX @ dX.T)
            J = dY @ dX.T @ np.linalg.inv(dX @ dX.T + lam * np.eye(dX.shape[0]))
        lambdas = np.linalg.eigvals(J)
        if plot_spectrum:
            mag = np.abs(lambdas)
            plt.figure()
            plt.plot(range(len(mag)), mag, marker='o', linestyle='-')
            plt.ylim(0, 1.2)
            plt.xlabel('Mode #')
            plt.ylabel('|λ|')
            plt.title(f'Floquet spectrum ({context}, {channel})')
            plt.grid(True)
            plt.show()
        return lambdas, np.max(np.abs(lambdas))

    # 4) compute either single-phase or phase-dependent
    if not phase_dependent:
        S_pos = com_norm[:, 0, :]
        S_vel = vel_norm[:, 0, :]
        lambdas, max_lam = _estimate_J_and_maxlam(S_pos, S_vel)

        # Poincaré section plot of x-position
        S = np.hstack([S_pos, S_vel])  # (n_cycles, 6)
        xk  = S[:-1, 0]  # x at strike k
        xk1 = S[1:,  0]  # x at strike k+1
        plt.figure(figsize=(5,5))
        plt.scatter(xk, xk1, s=20, alpha=0.6)
        mn, mx = min(xk.min(), xk1.min()), max(xk.max(), xk1.max())
        plt.plot([mn, mx], [mn, mx], 'k--', lw=1)
        plt.xlabel(r'$x_k$ at strike $k$')
        plt.ylabel(r'$x_{k+1}$ at strike $k+1$')
        plt.title('Poincaré plot (COM x-pos)')
        plt.grid(True)
        plt.show()

        return float(max_lam)

    # full phase-dependent curve
    max_vs_phi = np.zeros(M)
    for phi in range(M):
        S_pos = com_norm[:, phi, :]
        S_vel = vel_norm[:, phi, :]
        _, max_vs_phi[phi] = _estimate_J_and_maxlam(S_pos, S_vel)
    phases = np.linspace(0, 100, M)
    plt.figure()
    plt.plot(phases, max_vs_phi)
    plt.xlabel('Gait phase (%)')
    plt.ylabel('Max |λ|')
    plt.title(f'Phase-dependent Floquet ({context}, {channel})')
    plt.grid(True)
    plt.show()
    return max_vs_phi


def main():
    base_dir = Path("data/01")
    datasets = {
        'FWS': xr.open_dataarray(base_dir / 'FWS' / 'segmented' / 'markers.nc'),
        'PWS': xr.open_dataarray(base_dir / 'PWS' / 'segmented' / 'markers.nc'),
        'SWS': xr.open_dataarray(base_dir / 'SWS' / 'segmented' / 'markers.nc')
    }
    for name, data in datasets.items():
        print(f"\n--- {name} --- dims={data.dims}")
        maxlam = compute_floquet(data,
                                 context='Left',
                                 channel='COM',
                                 phase_dependent=False,
                                 plot_spectrum=True)
        print(f"{name}: Max |Floquet λ| = {maxlam:.4f}")

if __name__ == '__main__':
    main()
