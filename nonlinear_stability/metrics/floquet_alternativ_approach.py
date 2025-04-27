import os
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _estimate_J_and_spectrum(S_pos, S_vel, rcond=1e-6, plot_spectrum=False, context='', channel='', suffix=''):
    """
    Helper to estimate J and return eigenvalues and max |λ|.
    """
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
        plt.figure()
        plt.plot(np.arange(len(lambdas)), np.abs(lambdas), marker='o')
        plt.ylim(0, 1.2)
        plt.xlabel('Mode #')
        plt.ylabel('|λ|')
        plt.title(f'Floquet spectrum {suffix} ({context}, {channel})')
        plt.grid(True)
        plt.show()
    return lambdas, np.max(np.abs(lambdas))


def compute_stride_floquet(data: xr.DataArray,
                           context: str = "Right",
                           channel: str = "COM",
                           axes: list = ["x","y","z"],
                           normalization_points: int = 101,
                           rcond: float = 1e-6,
                           plot_spectrum: bool = False) -> float:
    """
    Compute max Floquet multiplier for full-stride (same-foot) map.
    """
    seg = data.sel(context=context, channel=channel, axis=axes).transpose('cycle','time','axis')
    com = seg.values
    n_cycles, n_time, n_axes = com.shape
    state_dim = 2 * n_axes
    if n_cycles - 1 < state_dim:
        raise ValueError(f"Need ≥{state_dim+1} cycles, got {n_cycles}")
    M = normalization_points
    norm_t = np.linspace(0, 1, M)
    com_norm = np.zeros((n_cycles, M, n_axes))
    for i in range(n_cycles):
        t_cycle = seg.isel(cycle=i).coords['time'].values
        t_norm = (t_cycle - t_cycle[0])/(t_cycle[-1] - t_cycle[0])
        for j in range(n_axes):
            com_norm[i,:,j] = np.interp(norm_t, t_norm, com[i,:,j])
    dt = 1/(M-1)
    vel_norm = np.zeros_like(com_norm)
    vel_norm[:,1:-1,:] = (com_norm[:,2:,:] - com_norm[:,:-2,:])/(2*dt)
    vel_norm[:,0,:]    = (com_norm[:,1,:]  - com_norm[:,0,:])/dt
    vel_norm[:,-1,:]   = (com_norm[:,-1,:] - com_norm[:,-2,:])/dt
    S_pos = com_norm[:,0,:]
    S_vel = vel_norm[:,0,:]
    _, maxlam = _estimate_J_and_spectrum(S_pos, S_vel,
                                          rcond=rcond,
                                          plot_spectrum=plot_spectrum,
                                          context=context,
                                          channel=channel,
                                          suffix='stride')
    return float(maxlam)


def compute_step_floquet(data: xr.DataArray,
                          channel: str = "COM",
                          axes: list = ["x","y","z"],
                          normalization_points: int = 101,
                          rcond: float = 1e-6,
                          plot_spectrum: bool = False) -> dict:
    """
    Compute Floquet multipliers for step-to-step maps across both contexts.
    Returns dict with half-step and approximate composite multipliers.
    """
    seg_R = data.sel(context='Right', channel=channel, axis=axes).transpose('cycle','time','axis')
    seg_L = data.sel(context='Left',  channel=channel, axis=axes).transpose('cycle','time','axis')
    com_R, nR = seg_R.values, seg_R.shape[0]
    com_L, nL = seg_L.values, seg_L.shape[0]
    n_pairs = min(nR, nL)
    M = normalization_points; norm_t = np.linspace(0,1,M); dt = 1/(M-1)
    def normalize(arr, seg):
        n_cycles, _, n_axes = arr.shape
        out = np.zeros((n_cycles, M, n_axes))
        for i in range(n_cycles):
            t = seg.isel(cycle=i).coords['time'].values
            t_norm = (t-t[0])/(t[-1]-t[0])
            for j in range(n_axes):
                out[i,:,j] = np.interp(norm_t, t_norm, arr[i,:,j])
        return out
    comR_n = normalize(com_R, seg_R)
    comL_n = normalize(com_L, seg_L)
    velR_n = np.zeros_like(comR_n); velL_n = np.zeros_like(comL_n)
    for arr, vel in [(comR_n, velR_n),(comL_n, velL_n)]:
        vel[:,1:-1,:] = (arr[:,2:,:]-arr[:,:-2,:])/(2*dt)
        vel[:,0,:]    = (arr[:,1,:]-arr[:,0,:]) / dt
        vel[:,-1,:]   = (arr[:,-1,:]-arr[:,-2,:]) / dt
    S_R = np.hstack([comR_n[:n_pairs,0,:], velR_n[:n_pairs,0,:]])
    S_L = np.hstack([comL_n[:n_pairs,0,:], velL_n[:n_pairs,0,:]])
    lambdas_RL, max_RL = _estimate_J_and_spectrum(
        S_R, S_L, rcond=rcond, plot_spectrum=plot_spectrum,
        suffix='step R->L', context='R->L', channel=channel)
    lambdas_LR, max_LR = _estimate_J_and_spectrum(
        S_L[:-1], S_R[1:], rcond=rcond, plot_spectrum=plot_spectrum,
        suffix='step L->R', context='L->R', channel=channel)
    return {
        'step_R->L': max_RL,
        'step_L->R': max_LR,
        'approx_stride': max_RL * max_LR
    }


def main():
    base_dir = Path("data/01")
    datasets = {
        'FWS': xr.open_dataarray(base_dir / 'FWS' / 'segmented' / 'markers.nc'),
        'PWS': xr.open_dataarray(base_dir / 'PWS' / 'segmented' / 'markers.nc'),
        'SWS': xr.open_dataarray(base_dir / 'SWS' / 'segmented' / 'markers.nc')
    }
    for name, data in datasets.items():
        print(f"\n--- {name} --- dims={data.dims}")
        stride_max = compute_stride_floquet(data, context='Right', plot_spectrum=True)
        print(f"{name} stride max |λ|: {stride_max:.4f}")
        step_results = compute_step_floquet(data, plot_spectrum=True)
        print(f"{name} step-to-step max |λ|:")
        for k,v in step_results.items():
            print(f"  {k}: {v:.4f}")

if __name__ == '__main__':
    main()
