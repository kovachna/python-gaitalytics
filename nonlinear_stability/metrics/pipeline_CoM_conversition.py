# i need a pipeline to read and convert all the c3d data to mrkares, segemented adn unsegemented


"""
batch_compute_and_segment.py

Object-oriented pipeline to batch process C3D files:
  - Compute markers (including CoM) for each trial and export unsegmented
  - Detect events and segment gait cycles, exporting segmented trials into a 'segmented' subfolder

Reads all .c3d files under `data_root`, preserves folder structure, and places output alongside originals.
"""
import sys
from pathlib import Path
import argparse
import gaitalytics
from gaitalytics import api
import xarray as xr
import matplotlib.pyplot as plt

# Add private branch for CoM calculation - becuse is private branch
package_dir_default = Path(r"C:\Users\Natascha\OneDrive - Hochschule Luzern\Thesis_Dokus\DART\data\pbt-analysis").resolve()


# import the needed packages for CoM calculation
from src.utils.modelling.model import model_com_body
from src.utils.constants import BODY_SEGMENTS, SEGMENTS_WEIGHT, SEGMENTS_PROXI_DISTANCE, METRICS, TRIALS

class TrialProcess:
    """
    Processes a single C3D trial:
      - Computes markers (CoM) and exports unsegmented trial
      - Detects events, segments gait cycles, and exports segmented trial
    """
    def __init__(self, c3d_path: Path, config, model_com_body):
        self.c3d_path = c3d_path
        self.config = config
        self.model_com_body = model_com_body

    def compute_and_export(self):
        """Load trial and compute CoM
        """
        trial = api.load_c3d_trial(self.c3d_path, self.config)
        trial = self.model_com_body(trial)

        # Export unsegmented trial (export_trial adds extension)
        out_base = self.c3d_path.with_suffix("")
        api.export_trial(trial, out_base)
        print(f"[UNSEG] {self.c3d_path} -> {out_base}.*")
        return trial

    def segment_and_export(self, trial):
        """Detect events
        """
        events = api.detect_events(trial, self.config)
        try:
            api.check_events(events)
        except ValueError as e:
            print(f"Event check failed for {self.c3d_path.name}: {e}. Check trials.")
        trial.events = events

        # Segment into gait cycles
        segments = api.segment_trial(trial)

        # Export into 'segmented' subfolder
        seg_folder = self.c3d_path.parent / 'segmented'
        seg_folder.mkdir(parents=True, exist_ok=True)
        seg_base = seg_folder / self.c3d_path.stem
        api.export_trial(segments, seg_base)
        print(f"[SEG]   {self.c3d_path} -> {seg_folder.name}/{seg_base.name}.*")


class BatchProcessor:
    """
    Recursively scans data for .c3d files and processes each via TrialProcess Class.
    """
    def __init__(self, data_root: Path, config_path: Path, package_dir: Path = None):
        self.data_root = data_root
        self.config = api.load_config(config_path)

        # Import the CoM model from private branch or default API
        pkg_dir = package_dir or package_dir_default
        if pkg_dir:
            sys.path.insert(0, str(pkg_dir))
            from src.utils.modelling.model import model_com_body as _mcb
        else:
            from gaitalytics import model_com_body as _mcb
        self.model_com_body = _mcb

    def run(self):
        files = list(self.data_root.rglob('*.c3d'))
        if not files:
            print(f"No .c3d files found under {self.data_root}")
            return

        for c3d in sorted(files):
            print(f"\nProcessing: {c3d}")
            proc = TrialProcess(c3d, self.config, self.model_com_body)
            trial = proc.compute_and_export()
            proc.segment_and_export(trial)


def main():
    parser = argparse.ArgumentParser(description="Batch compute markers and segment C3D trials.")
    parser.add_argument('--data-root',   type=Path, required=True,
                        help='Root directory containing .c3d files')
    parser.add_argument('--config',      type=Path, required=True,
                        help='YAML config for Gait API')
    parser.add_argument('--package-dir', type=Path, default=None,
                        help='Path to private pbt-analysis for CoM model')
    args = parser.parse_args()

    batch = BatchProcessor(
        data_root=args.data_root,
        config_path=args.config,
        package_dir=args.package_dir
    )
    batch.run()


if __name__ == '__main__':
    main()
