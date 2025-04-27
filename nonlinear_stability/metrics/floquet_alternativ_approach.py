# import os
# from pathlib import Path
# import numpy as np
# import xarray as xr
# import matplotlib.pyplot as plt


# def ensure_dir(path: str):
#     os.makedirs(path, exist_ok=True)


# def _build_normalized_trajectories(seg: xr.DataArray, M: int):
#     """
#     Given seg(cycle, time, axis), returns
#       com_norm: (n_cycles, M, n_axes)
#       vel_norm: (n_cycles, M, n_axes) via central diff
#     """
#     n_cycles, _, n_axes = seg.shape
#     norm_t = np.linspace(0, 1, M)
#     com = seg.values
#     com_norm = np.zeros((n_cycles, M, n_axes))
#     for i in range(n_cycles):
#         t = seg.isel(cycle=i).coords["time"].values
#         t_n = (t - t[0]) / (t[-1] - t[0])
#         for j in range(n_axes):
#             com_norm[i, :, j] = np.interp(norm_t, t_n, com[i, :, j])

#     dt = 1.0 / (M - 1)
#     vel_norm = np.zeros_like(com_norm)
#     # central difference
#     vel_norm[:, 1:-1, :] = (com_norm[:, 2:, :] - com_norm[:, :-2, :])/(2*dt)
#     # endpoints
#     vel_norm[:, 0, :]    = (com_norm[:, 1, :] - com_norm[:, 0, :]) / dt
#     vel_norm[:, -1, :]   = (com_norm[:, -1, :] - com_norm[:, -2, :]) / dt

#     return com_norm, vel_norm


# def _estimate_J(dX: np.ndarray, dY: np.ndarray, rcond=1e-6):
#     """
#     Solve dY = J dX via least-squares, with ridge fallback.
#     Returns J (state_dim x state_dim).
#     """
#     try:
#         J = dY @ np.linalg.pinv(dX, rcond=rcond)
#     except np.linalg.LinAlgError:
#         lam = rcond * np.trace(dX @ dX.T)
#         J = dY @ dX.T @ np.linalg.inv(dX @ dX.T + lam * np.eye(dX.shape[0]))
#     return J

# def _estimate_J_and_maxlam(S_pos: np.ndarray, S_vel: np.ndarray, rcond=1e-6, plot_spectrum=False, context='', channel=''):
#     S = np.hstack([S_pos, S_vel])
#     X = S[:-1].T
#     Y = S[1:].T
#     S_bar = S.mean(axis=0)
#     dX = (X - S_bar[:,None])
#     dY = (Y - S_bar[:,None])
#     try:
#         J = dY @ np.linalg.pinv(dX, rcond=rcond)
#     except np.linalg.LinAlgError:
#         lam = rcond * np.trace(dX @ dX.T)
#         J = dY @ dX.T @ np.linalg.inv(dX @ dX.T + lam * np.eye(dX.shape[0]))
#     lambdas = np.linalg.eigvals(J)
#     if plot_spectrum:
#         mag = np.abs(lambdas)
#         plt.figure()
#         plt.plot(np.arange(len(mag)), mag, 'o-')
#         plt.ylim(0, 1.2)
#         plt.xlabel('Mode #')
#         plt.ylabel('|λ|')
#         plt.title(f'Floquet spectrum ({context}, {channel})')
#         plt.grid(True)
#         plt.show()
#     return lambdas, np.max(np.abs(lambdas))

# def _analyze_map(S_from: np.ndarray, S_to: np.ndarray, plot_spectrum=False, title=""):
#     """
#     Fit Jacobian J for map S_from -> S_to, return (J, lambdas)
#     """
#     X = S_from[:-1].T    # state_dim x (N-1)
#     Y = S_to[1:].T       # state_dim x (N-1)
#     Xm = X - X.mean(axis=1, keepdims=True)
#     Ym = Y - Y.mean(axis=1, keepdims=True)
#     J = _estimate_J(Xm, Ym)
#     lambdas, _ = np.linalg.eig(J)
#     if plot_spectrum:
#         mag = np.abs(lambdas)
#         plt.figure()
#         plt.plot(np.arange(len(mag)), mag, 'o-')
#         plt.ylim(0, 1.2)
#         plt.xlabel('Mode #')
#         plt.ylabel('|λ|')
#         plt.title(f'Floquet spectrum: {title}')
#         plt.grid(True)
#         plt.show()
#     return J, lambdas
# def plot_poincare(S: np.ndarray, labels: list):
#     """
#     Plot Poincaré section scatter for each component in S.
#     """
#     for idx, lab in enumerate(labels):
#         xk  = S[:-1, idx]
#         xk1 = S[1:,  idx]
#         plt.figure(figsize=(5,5))
#         plt.scatter(xk, xk1, s=20, alpha=0.6)
#         mn, mx = xk.min(), xk.max()
#         plt.plot([mn,mx], [mn,mx], 'k--', lw=1)
#         plt.xlabel(f'{lab}$_k$')
#         plt.ylabel(f'{lab}$_{{k+1}}$')
#         plt.title(f'Poincaré: {lab}')
#         plt.grid(True)
#         plt.show()


# def plot_delay_embedding(x: np.ndarray, tau: int = 10, title='Delay Embedding'):
#     """
#     Plot a 3D delay-coordinate embedding [x(t), x(t+tau), x(t+2*tau)].
#     """
#     N = len(x)
#     idx0 = np.arange(0, N - 2*tau)
#     X = np.column_stack([x[idx0], x[idx0 + tau], x[idx0 + 2*tau]])
#     fig = plt.figure(figsize=(6,6))
#     ax = fig.add_subplot(111, projection='3d')
#     ax.plot(X[:,0], X[:,1], X[:,2], lw=0.5)
#     ax.set_xlabel('x(t)')
#     ax.set_ylabel(f'x(t+{tau})')
#     ax.set_zlabel(f'x(t+{2*tau})')
#     ax.set_title(title)
#     plt.show()


# def compute_stride_and_step_floquet(data: dict,
#                                     channel="COM",
#                                     axes=["x","y","z"],
#                                     M=101,
#                                     plot_spectrum=False):
#     """
#     For each dataset in data dict:
#       - J_stride: Right->Right
#       - J_RL, J_LR: half-stride maps
#       - J_composite: J_LR @ J_RL
#     Returns dict of results.
#     """
#     results = {}
#     for name, seg_path in data.items():
#         seg_da = seg_path.sel(channel=channel, axis=axes)
#         n_cycles = seg_da.sizes["cycle"]
#         state_dim = 2 * len(axes)
#         if n_cycles <= state_dim:
#             raise ValueError(f"{name}: need >{state_dim} cycles, got {n_cycles}")

#         # Right->Right full-stride
#         segR = seg_da.sel(context="Right").transpose("cycle","time","axis")
#         comR, velR = _build_normalized_trajectories(segR, M)
#         S_R = np.hstack([comR[:,0,:], velR[:,0,:]])
#         J_stride, lam_stride = _analyze_map(S_R, S_R, plot_spectrum, f"{name} stride")

#         # Right->Left half-stride
#         segL = seg_da.sel(context="Left").transpose("cycle","time","axis")
#         comL, velL = _build_normalized_trajectories(segL, M)
#         Np = min(S_R.shape[0], comL.shape[0])
#         S_L = np.hstack([comL[:Np,0,:], velL[:Np,0,:]])
#         J_RL, lam_RL = _analyze_map(S_R[:Np], S_L, plot_spectrum, f"{name} R→L")

#         # Left->Right half-stride
#         J_LR, lam_LR = _analyze_map(S_L[:-1], S_R[1:Np+1], plot_spectrum, f"{name} L→R")

#         # Composite full-stride
#         J_comp = J_LR @ J_RL
#         lambdas_comp, _ = np.linalg.eig(J_comp)

#         results[name] = {
#             'J_stride': J_stride,
#             'lambdas_stride': lam_stride,
#             'J_R->L': J_RL,
#             'lambdas_R->L': lam_RL,
#             'J_L->R': J_LR,
#             'lambdas_L->R': lam_LR,
#             'J_composite': J_comp,
#             'lambdas_comp': lambdas_comp
#         }
#     return results


# def plot_poincare(S: np.ndarray, labels: list):
#     """
#     Plot Poincaré scatter for each component in S.
#     S shape: (n_cycles, state_dim)
#     labels length = state_dim
#     """
#     for idx, lab in enumerate(labels):
#         xk  = S[:-1, idx]
#         xk1 = S[1:,  idx]
#         plt.figure(figsize=(5,5))
#         plt.scatter(xk, xk1, s=20, alpha=0.6)
#         mn, mx = xk.min(), xk.max()
#         plt.plot([mn,mx], [mn,mx], 'k--', lw=1)
#         plt.xlabel(f'{lab}$_k$')
#         plt.ylabel(f'{lab}$_{{k+1}}$')
#         plt.title(f'Poincaré plot: {lab}')
#         plt.grid(True)
#         plt.show()


# def compute_floquet(data: xr.DataArray,
#                     context: str = "Left",
#                     channel: str = "COM",
#                     axes: list = ["x","y","z"],
#                     normalization_points: int = 101,
#                     rcond: float = 1e-6,
#                     phase_dependent: bool = False,
#                     plot_spectrum: bool = False):
#     # ... existing code up to Poincaré plot ...
#     # after Poincaré scatter, add delay embedding of x-position time series
#     print("Plotting delay embedding of COM x-axis over one continuous recording...")
#     # extract continuous COM x time series before segmentation
#     x_cont = data.sel(channel=channel, axis='x').values.flatten()
#     plot_delay_embedding(x_cont, tau=int(normalization_points/10),
#                          title=f'Delay embedding ({context}, {channel}-x)')


#     # 1) select and reorder: dims -> (cycle, time, axis)
#     seg = data.sel(context=context, channel=channel, axis=axes).transpose('cycle','time','axis')
#     com = seg.values  # (n_cycles, n_time, n_axes)
#     n_cycles, n_time, n_axes = com.shape

#     # 1a) ensure enough cycles for state_dim
#     state_dim = 2 * n_axes
#     if n_cycles - 1 < state_dim:
#         raise ValueError(f"Need ≥{state_dim+1} cycles for state_dim={state_dim}, got {n_cycles}")

#     # 2) time-normalize each cycle onto [0,1]
#     M = normalization_points
#     norm_t = np.linspace(0, 1, M)
#     com_norm = np.zeros((n_cycles, M, n_axes))
#     for i in range(n_cycles):
#         t_cycle = seg.isel(cycle=i).coords['time'].values
#         t_norm = (t_cycle - t_cycle[0]) / (t_cycle[-1] - t_cycle[0])
#         for j in range(n_axes):
#             com_norm[i, :, j] = np.interp(norm_t, t_norm, com[i, :, j])

#     dt_norm = 1.0 / (M - 1)

#     # 3) central-difference velocity over normalized grid
#     vel_norm = np.zeros_like(com_norm)
#     vel_norm[:, 1:-1, :] = (com_norm[:, 2:, :] - com_norm[:, :-2, :]) / (2 * dt_norm)
#     vel_norm[:, 0, :]   = (com_norm[:, 1, :] - com_norm[:, 0, :])   / dt_norm
#     vel_norm[:, -1, :]  = (com_norm[:, -1, :] - com_norm[:, -2, :]) / dt_norm


# def main():
#     base_dir = Path("data/01")
#     datasets = {
#         'FWS': xr.open_dataarray(base_dir/'FWS'/'segmented'/'markers.nc'),
#         'PWS': xr.open_dataarray(base_dir/'PWS'/'segmented'/'markers.nc'),
#         'SWS': xr.open_dataarray(base_dir/'SWS'/'segmented'/'markers.nc')
#     }
#     # define normalization points
#     M = 101

#     # compute Floquet maps
#     results = compute_stride_and_step_floquet(
#         datasets, channel="COM", axes=["x","y","z"], M=M, plot_spectrum=False
#     )

#     # summarize and Poincaré
#     for name, res in results.items():
#         print(f"--- {name} ---")
#         print(f"Stride max |λ|   : {np.max(np.abs(res['lambdas_stride'])):.4f}")
#         print(f"R→L max |λ|      : {np.max(np.abs(res['lambdas_R->L'])):.4f}")
#         print(f"L→R max |λ|      : {np.max(np.abs(res['lambdas_L->R'])):.4f}")
#         print(f"Composite max |λ|: {np.max(np.abs(res['lambdas_comp'])):.4f}")

#         # rebuild and plot Poincaré for S_R
#         segR = datasets[name].sel(channel="COM", axis=["x","y","z"]).sel(context="Right").transpose('cycle','time','axis')
#         comR, velR = _build_normalized_trajectories(segR, M)
#         S_R = np.hstack([comR[:,0,:], velR[:,0,:]])
#         labels = ['x','y','z','dot_x','dot_y','dot_z']
#         print("Plotting Poincaré for full state at Right-strike... ")
#         plot_poincare(S_R, labels)

# if __name__ == '__main__':
#     main()





import os
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _build_normalized_trajectories(seg: xr.DataArray, M: int):
    # ... your existing code unchanged ...
    n_cycles, _, n_axes = seg.shape
    norm_t = np.linspace(0, 1, M)
    com = seg.values
    com_norm = np.zeros((n_cycles, M, n_axes))
    for i in range(n_cycles):
        t = seg.isel(cycle=i).coords["time"].values
        t_n = (t - t[0])/(t[-1] - t[0])
        for j in range(n_axes):
            com_norm[i,:,j] = np.interp(norm_t, t_n, com[i,:,j])
    dt = 1/(M-1)
    vel_norm = np.zeros_like(com_norm)
    vel_norm[:,1:-1,:] = (com_norm[:,2:,:] - com_norm[:,:-2,:])/(2*dt)
    vel_norm[:,0,:]    = (com_norm[:,1,:]  - com_norm[:,0,:])/dt
    vel_norm[:,-1,:]   = (com_norm[:,-1,:] - com_norm[:,-2,:])/dt
    return com_norm, vel_norm

def _estimate_J_and_maxlam(S_pos, S_vel, rcond=1e-6, plot_spectrum=False, context='', channel=''):
    # ... your existing code unchanged ...
    S = np.hstack([S_pos, S_vel])
    X = S[:-1].T; Y = S[1:].T
    S_bar = S.mean(axis=0)
    dX = X - S_bar[:,None]
    dY = Y - S_bar[:,None]
    try:
        J = dY @ np.linalg.pinv(dX, rcond=rcond)
    except np.linalg.LinAlgError:
        lam = rcond*np.trace(dX@dX.T)
        J = dY@dX.T@np.linalg.inv(dX@dX.T+lam*np.eye(dX.shape[0]))
    lambdas = np.linalg.eigvals(J)
    if plot_spectrum:
        mag = np.abs(lambdas)
        plt.figure()
        plt.plot(np.arange(len(mag)), mag, 'o-')
        plt.ylim(0,1.2); plt.xlabel('Mode #'); plt.ylabel('|λ|')
        plt.title(f'Floquet spectrum ({context},{channel})'); plt.grid(True)
        plt.show()
    return lambdas, np.max(np.abs(lambdas))

def plot_poincare(S: np.ndarray, labels: list):
    for idx, lab in enumerate(labels):
        xk  = S[:-1, idx]
        xk1 = S[1:,  idx]
        plt.figure(figsize=(5,5))
        plt.scatter(xk, xk1, s=20, alpha=0.6)
        mn, mx = xk.min(), xk.max()
        plt.plot([mn,mx],[mn,mx],'k--',lw=1)
        plt.xlabel(f'{lab}$_k$'); plt.ylabel(f'{lab}$_{{k+1}}$')
        plt.title(f'Poincaré: {lab}'); plt.grid(True)
        plt.show()

def plot_delay_embedding(x: np.ndarray, tau: int = 10, title='Delay Embedding'):
    N = len(x)
    idx0 = np.arange(0, N-2*tau)
    X = np.column_stack([x[idx0], x[idx0+tau], x[idx0+2*tau]])
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(X[:,0], X[:,1], X[:,2], lw=0.5)
    ax.set_xlabel('x(t)'); ax.set_ylabel(f'x(t+{tau})'); ax.set_zlabel(f'x(t+{2*tau})')
    ax.set_title(title); plt.show()

def compute_floquet(data: xr.DataArray,
                    context: str = "Left",
                    channel: str = "COM",
                    axes: list = ["x","z"],
                    normalization_points: int = 101,
                    rcond: float = 1e-6,
                    phase_dependent: bool = False,
                    plot_spectrum: bool = False):
    """
    Compute Floquet multipliers and Poincaré + delay‐embedding plots.

    Parameters:
      data              : xarray.DataArray with dims ('context','cycle','axis','channel','time')
      context           : which foot‐strike to use ('Left' or 'Right')
      channel           : marker channel, e.g. 'COM'
      axes              : spatial axes to include ['x','y','z']
      normalization_points: number of samples per normalized stride
      rcond             : cutoff for pseudoinverse
      phase_dependent   : if True, returns full λ(ϕ) curve instead of scalar
      plot_spectrum     : if True, plots |λ| spectrum

    Returns:
      float max‐|λ|  (if phase_dependent=False)
      np.ndarray     (if phase_dependent=True)
    """
    # 1) select & reorder to (cycle, time, axis)
    seg = data.sel(context=context, channel=channel, axis=axes).transpose('cycle','time','axis')
    com = seg.values
    n_cycles, _, n_axes = com.shape

    # 1a) state‐dimension check
    state_dim = 2 * n_axes
    if n_cycles - 1 < state_dim:
        raise ValueError(f"Need ≥{state_dim+1} cycles; got {n_cycles}")

    # 2) time‐normalize each cycle onto [0,1]
    M = normalization_points
    norm_t = np.linspace(0, 1, M)
    com_norm = np.zeros((n_cycles, M, n_axes))
    for i in range(n_cycles):
        t = seg.isel(cycle=i).coords['time'].values
        t_n = (t - t[0]) / (t[-1] - t[0])
        for j in range(n_axes):
            com_norm[i, :, j] = np.interp(norm_t, t_n, com[i, :, j])

    dt = 1.0 / (M - 1)
    # 3) compute velocities via central‐difference
    vel_norm = np.zeros_like(com_norm)
    vel_norm[:,1:-1,:] = (com_norm[:,2:,:] - com_norm[:,:-2,:])/(2*dt)
    vel_norm[:,0,:]    = (com_norm[:,1,:]  - com_norm[:,0,:])   / dt
    vel_norm[:,-1,:]   = (com_norm[:,-1,:] - com_norm[:,-2,:]) / dt

    # helper: build Jacobian & get eigenvalues
    def _estimate_J_and_maxlam(S_pos, S_vel):
        S = np.hstack([S_pos, S_vel])      # (n_cycles, state_dim)
        X = S[:-1].T                       # state_dim x (n_cycles-1)
        Y = S[1:].T
        S_bar = S.mean(axis=0)
        dX = X - S_bar[:, None]
        dY = Y - S_bar[:, None]
        try:
            J = dY @ np.linalg.pinv(dX, rcond=rcond)
        except np.linalg.LinAlgError:
            lam = rcond * np.trace(dX @ dX.T)
            J = dY @ dX.T @ np.linalg.inv(dX @ dX.T + lam*np.eye(dX.shape[0]))
        eigs = np.linalg.eigvals(J)
        if plot_spectrum:
            mag = np.abs(eigs)
            plt.figure()
            plt.plot(mag, 'o-')
            plt.ylim(0,1.2)
            plt.xlabel('Mode #')
            plt.ylabel('|λ|')
            plt.title(f'Floquet spectrum ({context},{channel})')
            plt.grid(True); plt.show()
        return eigs, np.max(np.abs(eigs))

    # 4) compute at phase=0 (Poincaré)
    S_pos = com_norm[:,0,:]
    S_vel = vel_norm[:,0,:]
    lambdas, max_lam = _estimate_J_and_maxlam(S_pos, S_vel)

    # Poincaré scatter (x‐axis component)
    S = np.hstack([S_pos, S_vel])
    xk  = S[:-1, 0]
    xk1 = S[1:,  0]
    plt.figure(figsize=(5,5))
    plt.scatter(xk, xk1, s=20, alpha=0.6)
    mn, mx = xk.min(), xk.max()
    plt.plot([mn,mx],[mn,mx],'k--')
    plt.xlabel('$x_k$'); plt.ylabel('$x_{k+1}$')
    plt.title('Poincaré (COM x)'); plt.grid(True); plt.show()

    # Delay embedding of continuous COM-x
    print("Plotting delay embedding…")
    x_cont = data.sel(channel=channel, axis='x').values.flatten()
    tau = max(1, M//10)
    N = len(x_cont)
    idx0 = np.arange(0, N-2*tau)
    X = np.column_stack([x_cont[idx0],
                         x_cont[idx0+tau],
                         x_cont[idx0+2*tau]])
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(X[:,0], X[:,1], X[:,2], lw=0.5)
    ax.set_xlabel('x(t)'); ax.set_ylabel(f'x(t+{tau})'); ax.set_zlabel(f'x(t+{2*tau})')
    ax.set_title('Delay embedding'); plt.show()

    if not phase_dependent:
        return float(max_lam)

    # optional: full phase‐dependent λ curve
    max_vs_phi = np.zeros(M)
    for phi in range(M):
        sp = com_norm[:,phi,:]; sv = vel_norm[:,phi,:]
        _, max_vs_phi[phi] = _estimate_J_and_maxlam(sp, sv)
    phases = np.linspace(0,100,M)
    plt.figure()
    plt.plot(phases, max_vs_phi)
    plt.xlabel('Phase %'); plt.ylabel('Max |λ|')
    plt.title('Phase‐dependent Floquet'); plt.grid(True); plt.show()
    return max_vs_phi


def main():
    base_dir = Path("data/01")
    datasets = {
      "FWS": xr.open_dataarray(base_dir/"FWS"/"segmented"/"markers.nc"),
      "PWS": xr.open_dataarray(base_dir/"PWS"/"segmented"/"markers.nc"),
      "SWS": xr.open_dataarray(base_dir/"SWS"/"segmented"/"markers.nc"),
    }
    for name, data in datasets.items():
        print(f"\n--- {name} --- dims={data.dims}")
        compute_floquet(data,
                        context="Left",
                        channel="COM",
                        normalization_points=101,
                        plot_spectrum=True)

if __name__ == "__main__":
    main()
