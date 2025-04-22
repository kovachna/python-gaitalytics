
"""
This script processes patient marker data without segmentation to compute mean, variance, rolling statistics, and stationarity tests.
For segmented data (e.g., gait cycles), separate scripts/functions should be used for Floquet and NI analyses.
"""
import os
from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from statsmodels.tsa.stattools import adfuller


def ensure_dir(path: str):
    """Ensure output directory exists."""
    os.makedirs(path, exist_ok=True)


def mean_variance(data: xr.DataArray, dataset_name: str, channel: str = "COM"):
    """
    Compute and print mean and variance for each axis of the specified channel.

    Parameters:
        data: xarray.DataArray with dims (axis, channel, time)
        dataset_name: identifier for data (e.g., "FWS")
        channel: channel to analyze (default "COM")
    """
    print(f"\n--- Mean & Variance for {dataset_name} (channel={channel}) ---")
    for axis_val in data.coords['axis'].values:
        available = data.sel(axis=axis_val).coords['channel'].values
        if channel in available:
            subset = data.sel(axis=axis_val, channel=channel)
            mean_val = float(subset.mean(dim='time', skipna=True))
            var_val = float(subset.var(dim='time', skipna=True))
            print(f"{dataset_name} axis={axis_val}: Mean={mean_val:.4f}, Variance={var_val:.4f}")
        else:
            print(f"{dataset_name} axis={axis_val}: channel '{channel}' not found")


def rolling_mean_variance(data: xr.DataArray,
                          dataset_name: str,
                          channel: str = "COM",
                          window_size: int = 100,
                          output_dir: str = "data/01/plots"):
    """
    Plot and save rolling mean & variance for each axis of a given dataset.

    Parameters:
        data: xarray.DataArray with dims (axis, channel, time)
        dataset_name: identifier (e.g., "FWS") for filename prefix
        channel: channel to analyze (default "COM")
        window_size: window length for rolling calculations
        output_dir: directory to save plots
    """
    ensure_dir(output_dir)
    time = data.coords['time'].values
    axes = data.coords['axis'].values
    n_axes = len(axes)

    fig, axs = plt.subplots(n_axes, 1, figsize=(10, 5 * n_axes), sharex=True)
    if n_axes == 1:
        axs = [axs]

    for i, axis_val in enumerate(axes):
        ax = axs[i]
        available = data.sel(axis=axis_val).coords['channel'].values
        if channel not in available:
            print(f"{dataset_name} axis={axis_val}: channel '{channel}' not found")
            continue

        series = pd.Series(
            data.sel(axis=axis_val, channel=channel).values,
            index=time if len(time) == len(data.sel(axis=axis_val, channel=channel).values)
                  else pd.RangeIndex(len(data.sel(axis=axis_val, channel=channel).values))
        )
        roll_mean = series.rolling(window=window_size).mean()
        roll_var = series.rolling(window=window_size).var()
        overall_mean = series.mean()
        overall_var = series.var()
        print(f"{dataset_name} axis={axis_val}: overall mean={overall_mean:.4f}, var={overall_var:.4f}")

        ax.plot(series.index, series, label="Raw Data", alpha=0.5)
        ax.plot(roll_mean.index, roll_mean, label="Rolling Mean")
        ax.axhline(overall_mean, linestyle='--', label=f"Mean={overall_mean:.4f}")
        ax.set_ylabel('Value')
        ax.set_title(f"{dataset_name} axis={axis_val} ({channel}): rolling (window={window_size})")

        ax2 = ax.twinx()
        ax2.plot(roll_var.index, roll_var, label="Rolling Variance", alpha=0.7)
        ax2.axhline(overall_var, linestyle='--', label=f"Var={overall_var:.4f}")
        ax2.set_ylabel('Variance')

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1+h2, l1+l2, loc='upper right')

    plt.xlabel('Time')
    plt.tight_layout()
    out_file = Path(output_dir) / f"{dataset_name}_{channel}_rolling_win{window_size}.png"
    plt.savefig(out_file, dpi=300)
    plt.show()
    print(f"Saved rolling plot: {out_file}")


def high_pass_filter(signal: np.ndarray, cutoff: float, fs: float, order: int = 4) -> np.ndarray:
    """
    Apply a Butterworth high-pass filter to remove drift.
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, signal)


def remove_drift_and_plot(data: xr.DataArray,
                           dataset_name: str,
                           channel: str = "COM",
                           fs: float = 100.0,
                           cutoff: float = 0.1,
                           output_dir: str = "data/01/plots"):
    """
    Filter each axis to remove drift, then plot original vs filtered.
    """
    ensure_dir(output_dir)
    time = data.coords['time'].values

    for axis_val in data.coords['axis'].values:
        available = data.sel(axis=axis_val).coords['channel'].values
        if channel not in available:
            continue
        series = pd.Series(data.sel(axis=axis_val, channel=channel).values, index=time)
        filt = high_pass_filter(series.values, cutoff=cutoff, fs=fs)

        plt.figure(figsize=(10, 5))
        plt.plot(series.index, series, label='Original', alpha=0.5)
        plt.plot(series.index, filt, label='Filtered', linewidth=2)
        plt.title(f"{dataset_name} axis={axis_val} ({channel}): high-pass cutoff={cutoff}Hz")
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.tight_layout()
        out_file = Path(output_dir) / f"{dataset_name}_{channel}_filtered_axis{axis_val}.png"
        plt.savefig(out_file, dpi=300)
        plt.show()
        print(f"Saved filtered plot: {out_file}")


def plotting(data: xr.DataArray, dataset_name: str, channel: str = "COM"):
    """
    Plot concatenated time series for each axis.
    """
    fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    for i, axis_val in enumerate(data.coords['axis'].values):
        series = data.sel(axis=axis_val, channel=channel).values
        axs[i].plot(data.coords['time'].values, np.squeeze(series), label=channel)
        axs[i].set_title(f"{dataset_name} axis={axis_val}")
        axs[i].set_xlabel('Time')
        axs[i].set_ylabel('Value')
        axs[i].grid(True)
        axs[i].legend()
    plt.tight_layout()
    plt.show()
def adf_test(data: xr.DataArray, channel: str = "COM") -> tuple:
    """
    Perform ADF stationarity test for each axis on the specified channel.
    Return (adf_stats, p_values).
    """
    stats, pvals = [], []
    for axis_val in data.coords['axis'].values:
        try:
            series = data.sel(axis=axis_val, channel=channel).values.squeeze()
            adf_stat, p_value = adfuller(series)[:2]
        except Exception as e:
            print(f"axis={axis_val}: Error performing ADF test: {e}")
            adf_stat, p_value = np.nan, np.nan
        else:
            print(f"axis={axis_val}: ADF={adf_stat:.4f}, p={p_value:.4f}")
        stats.append(adf_stat)
        pvals.append(p_value)
    return stats, pvals


def write_adf_results(dataset_name: str,
                      markers_data: xr.DataArray,
                      adf_stats: list,
                      p_values: list,
                      output_dir: str = "data/01/plots"):
    """Save ADF results to a text file."""
    ensure_dir(output_dir)
    out_file = Path(output_dir) / f"{dataset_name}_adf_results.txt"
    with open(out_file, 'w') as f:
        for axis_val, stat, pval in zip(markers_data.coords['axis'].values, adf_stats, p_values):
            f.write(f"axis={axis_val}: ADF={stat:.4f}, p={pval:.4f}\n")
    print(f"Saved ADF results: {out_file}")


def main():
    base_dir = Path("data/01")
    datasets = {
        'FWS': xr.open_dataarray(base_dir / 'FWS/markers.nc'),
        'PWS': xr.open_dataarray(base_dir / 'PWS/markers.nc'),
        'SWS': xr.open_dataarray(base_dir / 'SWS/markers.nc'),
    }

    for name, data in datasets.items():
        print(f"Loading {name} with dims {data.dims} and coords {data.coords}")
        mean_variance(data, name)
        rolling_mean_variance(data, name)
        # remove_drift_and_plot(data, name)
        plotting(data, name)
        adf_stats, p_vals = adf_test(data)
        write_adf_results(name, data, adf_stats, p_vals)


if __name__ == '__main__':
    main()
