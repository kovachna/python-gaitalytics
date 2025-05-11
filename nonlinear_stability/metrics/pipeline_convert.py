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

def process_file(in_file: Path, out_file:Path, config_path:Path):
    """
    Load the raw C3D file, process it, and save the results to a new file.
    """
    # load configuration
    config = api.load_config(config_path)

    # load the raw C3D file
    trial = api.load_c3d_trial(in_file, config=config)

    # compute the CoM
    trial = model_com_body(trial)

    # export the trial to a new NOT segemented ndf file
    api.export_trial(trial,out_file)
    print(f"File without cycles saved to {out_file}")

def segment_file(in_file: Path, out_seg_file:Path, config_path:Path):
    """
    Load the raw C3D file, process it, and save the results to a new file.
    """
    # load configuration
    config = api.load_config(config_path)

    # load the raw C3D file
    trial = api.load_c3d_trial(in_file, config=config)

    # detect events (only check)
    event_table = api.detect_events(trial, config)
    try:
        api.check_events(event_table)
    except ValueError as e:
        print("Event‐check failed, there is a missmatch:", e)
    print(f"  → found {len(event_table)} events")

    # attach the events back onto the trial
    trial.events = event_table

    #4) segment your trial (contains both markers+COM & events)
    segments = api.segment_trial(trial)

    # export the trial to a new segmented ndf file
    api.export_trial(segments,out_seg_file)
    print(f"File with cycles saved to {out_seg_file}")


def main():

    parser = argparse.ArgumentParser(
        description="Batch compute CoM and segment gait cycles, with separate output trees."
    )
    parser.add_argument('input_root', type=Path, help="Root folder of raw C3D files.")
    parser.add_argument('processed_root', type=Path, help="Root folder to save processed trials.")
    parser.add_argument('segmented_root', type=Path, help="Root folder to save segmented trials.")
    parser.add_argument('--config yaml', type=Path, default=Path("../data/config.yaml"),
                        help="YAML config for Gait API.")
    args = parser.parse_args()

    # Gather all C3D files
    c3d_files = list(args.input_root.rglob("*.c3d"))
    if not c3d_files:
        print(f"No .c3d files found under {args.input_root}")
        return

    # Load config once
    config = api.load_config(args.config)

    for c3d_path in c3d_files:
        rel = c3d_path.relative_to(args.input_root)
        # Define output paths
        proc_path = args.processed_root / rel.with_suffix(".ndf")
        seg_path  = args.segmented_root  / rel.with_suffix("_segmented.ndf")

        # Ensure output folders exist
        proc_path.parent.mkdir(parents=True, exist_ok=True)
        seg_path.parent.mkdir(parents=True, exist_ok=True)

        # 1) Load raw C3D and compute CoM
        trial = api.load_c3d_trial(c3d_path, config)
        trial = model_com_body(trial)

        # 2) Export CoM-processed trial
        api.export_trial(trial, proc_path)
        print(f"Saved processed trial with CoM to: {proc_path}")

        # 3) Detect events & segment cycles on same trial
        events = api.detect_events(trial, config)
        try:
            api.check_events(events)
        except ValueError as e:
            print(f"Event check failed for {c3d_path}: {e}")
        trial.events = events
        segments = api.segment_trial(trial)

        # 4) Export segmented trials
        api.export_trial(segments, seg_path)
        print(f"Saved segmented trial to: {seg_path}\n")

if __name__ == '__main__':
    main()