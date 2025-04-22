
import numpy as np
import xarray as xr
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from statsmodels.tsa.stattools import adfuller
from scipy.signal import detrend,butter,filtfilt

trials = Path ("..data/01/PWS/markers.nc")
print(xr.open_dataarray(trials))

markers = xr.DataArray(trials).sel(channel="COM")