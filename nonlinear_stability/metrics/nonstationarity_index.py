import numpy as np
from pathlib import Path
import xarray as xr

def spatial_nsi_from_cycles(cycles):
    """
    Spatial NSI per axis from cycles array.
    
    cycles: np.ndarray, shape (n_strides, n_time, 3)
    Returns: array of length 3 with NSI for [X,Y,Z]
    """
    # mean across time for each stride & axis -> (n_strides, 3)
    cycle_means = cycles.mean(axis=1)
    # flatten all data per axis -> (n_strides*n_time, 3)
    all_data = cycles.reshape(-1, cycles.shape[2])
    return cycle_means.std(axis=0, ddof=1) / all_data.std(axis=0, ddof=1)

def temporal_nsi_cv_from_cycles(cycles, fs):
    """
    Temporal NSI (CV) from cycle durations.
    
    cycles: xarray.DataArray with dims (context, cycle, time, axis)
    fs: sampling rate (Hz)
    """
    # number of time samples per stride (assumed equal)
    n_time = cycles.sizes["time"]
    durations = np.full(cycles.sizes["context"] * cycles.sizes["cycle"],
                        (n_time - 1) / fs)
    # CV = std/mean
    return durations.std(ddof=1) / durations.mean()

def main():
    base = Path("data/01")
    fs = 100  # sampling frequency

    for speed in ("FWS","PWS","SWS"):
        # 1) load segmented markers
        da = xr.open_dataarray(base/speed/"segmented"/"markers.nc")
        com = da.sel(channel="COM").sel(axis=["x","y","z"])
        # dims: (context, cycle, time, axis)

        # 2) stack context+cycle into one 'stride' dimension
        strides = com.stack(stride=("context","cycle"))  # dims: (stride, time, axis)

        # 3) reorder to (stride, time, axis)
        strides = strides.transpose("stride","time","axis")

        # 4) compute NSIs
        spatial_nsi = spatial_nsi_from_cycles(strides.values)
        temporal_nsi = temporal_nsi_cv_from_cycles(com, fs)

        # 5) report
        print(f"\n=== {speed} ===")
        print("Axis | Spatial NSI | Temporal NSI (CV)")
        print("-----|-------------|------------------")
        for ax_label, s_nsi in zip(["X","Y","Z"], spatial_nsi):
            print(f"  {ax_label}   |    {s_nsi:6.4f}   |    {temporal_nsi:6.4f}")

if __name__ == "__main__":
    main()
