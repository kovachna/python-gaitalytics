# this scirpt processes the data form the patient without segementaion
# this means the data is only used to calculate lyapunov and entropies

# for floquet and NI we need segmented data - aka gait cycles
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.stattools import adfuller
from scipy.signal import butter, filtfilt

trial_toread=Path("trial_CoM/01/FWS_Park/markers.nc")
print(xr.open_dataarray(trial_toread))

# define the markers
markers = xr.open_dataarray(trial_toread)

# this is not really needed
selected_marker = markers.sel(channel="COM")

# as next we check mean and variance 
# function to measure now the mean and variance in each cycle

def mean_variance(data, channel="COM"):
    """
    For each axis in the provided array (with dims: "axis", "channel", "time"),
    compute the mean and variance along the time dimension for the specified channel.
    The function prints the results and creates a line plot showing the computed
    means and variances across axes.
    
    Parameters:
        data (xarray.DataArray): DataArray with dimensions ("axis", "channel", "time").
        channel (str): The channel to compute statistics for (default is "COM").
    """

    # Extract time values for the x-axis.
    time = data.coords["time"].values

    # Loop over each axis.
    axes = data.coords["axis"].values
    for axis_val in axes:
        # Check if the desired channel exists for the current axis.
        available_channels = data.sel(axis=axis_val).coords["channel"].values
        if channel in available_channels:
            # Select data for the current axis and channel.
            subset = data.sel(axis=axis_val, channel=channel)
            
            # Compute mean and variance over the time dimension.
            mean_val = subset.mean(dim="time", skipna=True).item()
            var_val = subset.var(dim="time", skipna=True).item()
            print(f"Axis {axis_val}, Channel {channel}: Mean = {mean_val:.4f}, Variance = {var_val:.4f}")
        else:
            print(f"Axis {axis_val}: Channel {channel} not found")



def rolling_mean_variance(data, channel="COM", window_size=100):
    """
    Computes and plots the rolling mean and rolling variance for each axis
    in the provided DataArray (with dimensions: "axis", "channel", "time").
    All axes are plotted in one figure, each in its own subplot.

    Parameters:
        data (xarray.DataArray): DataArray with dimensions ("axis", "channel", "time").
        channel (str): The channel to analyze (default is "COM").
        window_size (int): The window size (number of consecutive data points) for rolling computations.
    """
    # Get time values and axes from the DataArray.
    time = data.coords["time"].values
    axes = data.coords["axis"].values
    n_axes = len(axes)
    
    # Create one figure with n_axes subplots (vertically stacked).
    fig, axs = plt.subplots(n_axes, 1, figsize=(10, 5 * n_axes), sharex=True)
    
    # If there's only one axis, wrap the single subplot in a list.
    if n_axes == 1:
        axs = [axs]
    
    # Loop over each axis and its corresponding subplot.
    for i, axis_val in enumerate(axes):
        # Check if the specified channel exists for the current axis.
        available_channels = data.sel(axis=axis_val).coords["channel"].values
        if channel not in available_channels:
            print(f"Axis {axis_val}: Channel {channel} not found.")
            continue
        
        # Extract the time series for the axis and channel.
        series_data = data.sel(axis=axis_val, channel=channel).values
        
        # Use the actual time values as index if lengths match; otherwise, use a RangeIndex.
        series_index = time if len(time) == len(series_data) else pd.RangeIndex(len(series_data))
        
        # Convert the data to a pandas Series.
        series = pd.Series(series_data, index=series_index)
        
        # Compute rolling mean and variance.
        roll_mean = series.rolling(window=window_size).mean()
        roll_var = series.rolling(window=window_size).var()

        # Compute overall mean and variance.
        overall_mean = series.mean()
        overall_var = series.var()
        print(f"Axis {axis_val}, Channel {channel}: Overall Mean = {overall_mean:.4f}, Overall Variance = {overall_var:.4f}")
        
        # Get the subplot for this axis.
        ax = axs[i]
        
        # Plot raw data and rolling mean on the primary y-axis.
        ax.plot(series.index, series, label="Raw Data", color="blue", alpha=0.5)
        ax.plot(roll_mean.index, roll_mean, label="Rolling Mean", color="green")
        ax.axhline(overall_mean, color="blue", linestyle="--", label=f"Overall Mean = {overall_mean:.4f}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.set_title(f"{channel} - Axis {axis_val}: Rolling Mean & Variance (window={window_size})")
        ax.legend(loc="upper left")
        
        # Plot rolling variance on a secondary y-axis.
        ax2 = ax.twinx()
        ax2.plot(roll_var.index, roll_var, label="Rolling Variance", color="red", alpha=0.7)
        ax2.axhline(overall_var, color="red", linestyle="--", label=f"Overall Variance = {overall_var:.4f}")
        ax2.set_ylabel("Variance")
        
        # Combine legends from both y-axes.
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    
    plt.tight_layout()
    filename = "trial_CoM/01/plots/Rolling_timeseries_all_axes.png"
    plt.savefig(filename, dpi=300)
    plt.show()
    print(f"Saved combined plot for all axes as {filename}")


def high_pass_filter(signal, cutoff, fs, order=4):
    """
    Applies a Butterworth high-pass filter to remove low-frequency drift.
    
    Parameters:
        signal (np.array): 1D array of data to filter.
        cutoff (float): The cutoff frequency (Hz) below which frequencies will be attenuated.
        fs (float): The sampling frequency (Hz).
        order (int): The order of the filter.
        
    Returns:
        np.array: The filtered signal.
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # Design Butterworth high-pass filter.
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    # Apply filter forward and backward to avoid phase shift.
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal


def remove_drift_and_plot(data, channel="COM", fs=100.0, cutoff=0.1):
    """
    Removes long-term drift from each axis (using a high-pass filter) and plots the original and filtered time series.
    
    Parameters:
        data (xarray.DataArray): DataArray with dimensions ("axis", "channel", "time").
        channel (str): The channel to analyze (default is "COM").
        fs (float): Sampling frequency in Hz.
        cutoff (float): High-pass filter cutoff frequency in Hz.
    """
    time = data.coords["time"].values
    axes = data.coords["axis"].values
    
    for axis_val in axes:
        # Check if the channel exists.
        available_channels = data.sel(axis=axis_val).coords["channel"].values
        if channel not in available_channels:
            print(f"Axis {axis_val}: Channel {channel} not found.")
            continue
        
        # Extract the time series.
        series_data = data.sel(axis=axis_val, channel=channel).values
        series_index = time if len(time) == len(series_data) else pd.RangeIndex(len(series_data))
        series = pd.Series(series_data, index=series_index)
        
        # Apply high-pass filter.
        filtered_data = high_pass_filter(series.values, cutoff=cutoff, fs=fs, order=4)
        
        # Plot original and filtered data.
        plt.figure(figsize=(10, 5))
        plt.plot(series.index, series, label="Original Data", color="blue", alpha=0.5)
        plt.plot(series.index, filtered_data, label="Filtered Data", color="red", linewidth=2)
        plt.xlabel("Time")
        plt.ylabel("Position")
        plt.title(f"Axis {axis_val} - Channel {channel}: High-Pass Filter (cutoff={cutoff}Hz)")
        plt.legend()
        plt.tight_layout()
        filename = f"trial_CoM/01/plots/Filtered_timeseries_axis_{axis_val}.png"
        #plt.savefig(filename, dpi=300)
        plt.show()
        print(f"Saved filtered plot for axis {axis_val} as {filename}")




# now we check for stationarity
"""
the following concept applies: if the drifts are way too large in amplitude the adf cannot catch them and it approves stationarity
therefore one hsould acutally think about the rolling mean and variance


"""
# stationarity 
def adf_test(data):
    """
    Performs the Augmented Dickey-Fuller (ADF) test for stationarity on the 
    concatenated dataset for each axis. It prints the ADF statistic and p-value
    for each axis and plots bar charts of these metrics.
    
    Parameters:
      concatenated_data (xarray.DataArray): The continuous time series data
        with dimensions "axis" and "time", obtained after concatenating cycles.
    """
    # Dictionary to store ADF test results for each axis.
    results = {}
    
    # Loop over each axis in the concatenated data.
    for axis_val in data.coords["axis"].values:
        # Extract the time series for the given axis.
        series = data.sel(axis=axis_val).values
        
        try:
            result = adfuller(series)
            adf_stat = result[0]
            p_value = result[1]
            print(f"Axis {axis_val}: ADF Statistic = {adf_stat:.4f}, p-value = {p_value:.4f}")
            results[axis_val] = {"adf_stat": adf_stat, "p_value": p_value}
        except Exception as e:
            print(f"Axis {axis_val}: Error performing ADF test: {e}")
            results[axis_val] = {"adf_stat": np.nan, "p_value": np.nan}
    
    # Prepare data for plotting: a bar chart for ADF statistics and one for p-values.
    axes_list = list(results.keys())
    adf_stats = [results[a]["adf_stat"] for a in axes_list]
    p_values = [results[a]["p_value"] for a in axes_list]

    return adf_stats, p_values




if __name__ == "__main__":

    #Plot the concatenated time series for each axis (x, y, z)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

    for axis in ['x', 'y', 'z']:
        data_axis = selected_marker.sel(axis=axis)
        if axis == 'x':
            ax1.plot(data_axis.time, data_axis, label="COM")
            ax1.set_title("X Axis")
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Position")
            ax1.legend()
        elif axis == 'y':
            ax2.plot(data_axis.time, data_axis, label="COM")
            ax2.set_title("Y Axis")
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("Position")
            ax2.legend()
        elif axis == 'z':
            ax3.plot(data_axis.time, data_axis, label="COM")
            ax3.set_title("Z Axis")
            ax3.set_xlabel("Time (s)")
            ax3.set_ylabel("Position")
            ax3.legend()

    plt.tight_layout()
    #plt.savefig("trial_CoM/01/plots/FWS_plots", dpi=300)
    plt.show()

    mean_variance(markers)

    
    rolling_mean_variance(markers)
    fs=100.0
    cutoff=0.1

    remove_drift_and_plot(markers,"COM", fs,cutoff)


#     adf_stats, p_values = adf_test(selected_marker)

# # Assume the order of axes is the same as in concatenated_data.coords["axis"].values
#     axes_list = list(markers.coords["axis"].values)

#     with open("trial_CoM/01/plots/adf_results.txt", "w") as f:
#         for i, axis_val in enumerate(axes_list):
#             stat = adf_stats[i]
#             pval = p_values[i]
#             f.write(f"Axis {axis_val}: ADF = {stat:.4f}, p-value = {pval:.4f}\n")
