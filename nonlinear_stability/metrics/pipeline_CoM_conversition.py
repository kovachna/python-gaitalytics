# i need a pipeline to read and convert all the c3d data to mrkares, segemented adn unsegemented

#import all pacakges
from pathlib import Path
import gaitalytics
from pathlib import Path
from gaitalytics import api
import xarray as xr
import sys
import matplotlib.pyplot as plt
import argparse

#Folder contianing the privately owned branch for CoM calculation
package_dir = Path(r"C:\Users\Natascha\OneDrive - Hochschule Luzern\Thesis_Dokus\DART\data\pbt-analysis").resolve()
# append the path to the system path to load the modules
sys.path.append(str(package_dir))

# import the needed packages for CoM calculation
from src.utils.modelling.model import model_com_body
from src.utils.constants import BODY_SEGMENTS, SEGMENTS_WEIGHT, SEGMENTS_PROXI_DISTANCE, METRICS, TRIALS

