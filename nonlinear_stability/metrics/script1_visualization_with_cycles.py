import os
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def plotting(data: xr.DataArray,
             dataset_name: str,
             channel: str = "COM",
             output_dir: str = "data/01/plots/segmented"):
    """
    For each axis and context, plots all cycles as separate lines
    on a single figure.
    
    Expects data with dims: ('context','cycle','axis','channel','time')
    """
    ensure_dir(output_dir)
    contexts = data.coords['context'].values
    cycles   = data.coords['cycle'].values
    axes     = data.coords['axis'].values
    time     = data.coords['time'].values

    for context in contexts:
        for axis_val in axes:
            fig, ax = plt.subplots(figsize=(10, 5))
            for cycle in cycles:
                # select 1D series: now dims=(time,)
                series = data.sel(
                    context=context,
                    cycle=cycle,
                    axis=axis_val,
                    channel=channel
                ).values.squeeze()
                
                ax.plot(time, series, label=f"cycle {cycle}")
            
            ax.set_title(f"{dataset_name} — context={context}, axis={axis_val}")
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
            ax.grid(True)
            ax.legend(ncol=4, fontsize="small", loc="upper right")
            
            out_path = Path(output_dir) / f"{dataset_name}_{context}_{axis_val}_cycles.png"
            plt.tight_layout()
            #plt.savefig(out_path, dpi=300)
            #plt.show()
            #print(f"Saved: {out_path}")

# def compute_floquet(data: xr.DataArray,
#                     context: str = "Right",
#                     channel: str = "COM",
#                     axes: list = ["x","y","z"],
#                     rcond: float = 1e-6) -> float:
#     """
#     Compute the max Floquet multiplier (|lambda_max|) for one-step Poincaré map
#     at start-of-stride (t=0) using the specified context and channel.

#     Uses a robust pseudoinverse with rcond and a fallback ridge estimator if SVD fails.
#     Returns:
#         max_abs_lambda: float
#     """
#     # 1) select data
#     seg = data.sel(context=context, channel=channel, axis=axes)
#     com = seg.values  # shape (n_cycles, n_time, n_axes)
#     n_cycles, n_time, n_axes = com.shape
#     dt = float(seg.coords["time"][1] - seg.coords["time"][0])

#     # 2) build state vectors at t=0: positions and velocities
#     S_pos = com[:, 0, :]
#     S_vel = (com[:, 1, :] - com[:, 0, :]) / dt
#     S = np.hstack([S_pos, S_vel])  # shape (n_cycles, 2*n_axes)

#     # 3) form X and Y matrices
#     X = S[:-1, :]  # drop last cycle
#     Y = S[1:, :]   # drop first cycle

#     # 4) center
#     S_mean = S.mean(axis=0)
#     dX = (X - S_mean).T  # (state_dim, n_cycles-1)
#     dY = (Y - S_mean).T

#     # 5) solve for J = dY @ pinv(dX) with robust fallback
#     try:
#         pinv_dX = np.linalg.pinv(dX, rcond=rcond)
#         J = dY @ pinv_dX
#     except np.linalg.LinAlgError:
#         # ridge regression fallback: J = dY * dX^T * inv(dX*dX^T + λI)
#         lam = rcond * np.trace(dX @ dX.T)
#         J = dY @ dX.T @ np.linalg.inv(dX @ dX.T + lam * np.eye(dX.shape[0]))

#     # 6) eigenvalues
#     lambdas = np.linalg.eigvals(J)
#     return float(np.max(np.abs(lambdas)))



# def compute_floquet(data: xr.DataArray,
#                     context: str = "Right",
#                     channel: str = "COM",
#                     axes: list = ["x","y","z"],
#                     normalization_points: int = 101,
#                     rcond: float = 1e-6) -> float:
#     """
#     Compute the max Floquet multiplier (|lambda_max|) for one-step Poincaré map
#     at start-of-stride (t=0) using the specified context and channel.

#     Time-normalizes each cycle to `normalization_points` samples before building
#     state vectors [position; velocity]. Uses robust pseudoinverse with fallback.

#     Returns:
#         max_abs_lambda: float
#     """
#     # 1) select and reorder data: dims (cycle, time, axis)
#     seg = data.sel(context=context, channel=channel, axis=axes)
#     seg = seg.transpose('cycle', 'time', 'axis')

#     # extract array
#     com = seg.values               # shape (n_cycles, n_time, n_axes)
#     n_cycles, n_time, n_axes = com.shape

#     M = normalization_points
#     norm_t = np.linspace(0, 1, M)
#     com_norm = np.zeros((n_cycles, M, n_axes))

#     # 2) time-normalize each cycle using its own time vector
#     for i in range(n_cycles):
#         # get per-cycle time values
#         da_cyc = seg.isel(cycle=i)
#         t_cycle = da_cyc.coords['time'].values
#         # normalize to [0,1]
#         t_norm = (t_cycle - t_cycle[0]) / (t_cycle[-1] - t_cycle[0])
#         for j in range(n_axes):
#             com_norm[i, :, j] = np.interp(norm_t, t_norm, com[i, :, j])

#     # 3) normalized dt
#     dt_norm = 1.0 / (M - 1)

#     # 4) build state vectors at t=0: positions and velocities
#     S_pos = com_norm[:, 0, :]
#     S_vel = (com_norm[:, 1, :] - com_norm[:, 0, :]) / dt_norm
#     S = np.hstack([S_pos, S_vel])  # shape (n_cycles, 2*n_axes)

#     # 5) form X and Y matrices
#     X = S[:-1, :]
#     Y = S[1:, :]

#     # 6) center
#     S_mean = S.mean(axis=0)
#     dX = (X - S_mean).T
#     dY = (Y - S_mean).T

#     # 7) Jacobian with robust pseudoinverse
#     try:
#         pinv_dX = np.linalg.pinv(dX, rcond=rcond)
#         J = dY @ pinv_dX
#     except np.linalg.LinAlgError:
#         lam = rcond * np.trace(dX @ dX.T)
#         J = dY @ dX.T @ np.linalg.inv(dX @ dX.T + lam * np.eye(dX.shape[0]))

#     # 8) eigenvalues
#     lambdas = np.linalg.eigvals(J)
#     return float(np.max(np.abs(lambdas)))

def main():
    base_dir = Path("data/01")
    datasets = {
        'FWS': xr.open_dataarray(base_dir / 'FWS' / 'segmented' / 'markers.nc'),
        'PWS': xr.open_dataarray(base_dir / 'PWS' / 'segmented' / 'markers.nc'),
        'SWS': xr.open_dataarray(base_dir / 'SWS' / 'segmented' / 'markers.nc'),
    }

    for name, data in datasets.items():
        print(f"Loading {name}: dims={data.dims}, coords={list(data.coords)}")
        plotting(data, name)
        # max_fm = compute_floquet(data,
        #                          context="Right",
        #                          channel="COM",
        #                          axes=["x","y","z"],
        #                          normalization_points=101,
        #                          rcond=1e-6)
        # print(f"{name}: Max |Floquet λ| (Right-strikes, COM) = {max_fm:.4f}")




if __name__ == "__main__":
    main()
