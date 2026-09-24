"""
telemetry/logger.py
====================
The episode logger — appends one row per simulation timestep to a CSV file.

HOW IT WORKS:
- At the start of each episode: open a new CSV file, write header
- Each simulation step: call logger.log_step(data_dict)
- At episode end: call logger.close(success, failure_stage)

WHY CSV?
--------
Simple, human-readable, works with pandas, numpy, and Excel.
For production you'd use HDF5 or MCAP, but CSV is perfect for Week 1
debugging — you can open it in any editor and immediately see what happened.
"""

import csv
import os
import time
from pathlib import Path
from telemetry.schema import ALL_COLUMNS, COL_SUCCESS, COL_FAILURE_STAGE, COL_EPISODE_DURATION


class EpisodeLogger:
    """
    Records all telemetry for a single episode to a CSV file.

    Usage:
        logger = EpisodeLogger(episode_id="ep_001", scenario_id="nominal",
                               output_dir="data/episodes/")
        for each step:
            logger.log_step({...})      # one dict per timestep
        logger.close(success=True)
    """

    def __init__(self, episode_id: str, scenario_id: str, output_dir: str):
        self.episode_id = episode_id
        self.scenario_id = scenario_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.filepath = self.output_dir / f"{episode_id}.csv"
        self.start_time = time.time()
        self.step_count = 0
        self._rows = []  # Buffer rows in memory, flush at end (faster I/O)

        print(f"[Logger] Episode {episode_id} | scenario={scenario_id} | file={self.filepath}")

    def log_step(self, data: dict):
        """
        Append one timestep's worth of data.

        `data` should contain all keys from schema.ALL_COLUMNS.
        Any missing keys are filled with None (NaN in pandas).
        """
        # Ensure all columns present; fill missing with None
        row = {col: data.get(col, None) for col in ALL_COLUMNS}
        self._rows.append(row)
        self.step_count += 1

    def close(self, success: bool, failure_stage: str = "none"):
        """
        Finalize the episode: fill outcome columns, write CSV to disk.

        Args:
            success:       True if the task completed successfully.
            failure_stage: Which task phase the failure occurred in
                           (e.g., "lift", "grasp"). "none" if success.
        """
        duration = time.time() - self.start_time

        # Fill outcome columns into all rows (same value repeated — allows
        # easy filtering: df[df['success']==False] in pandas)
        for row in self._rows:
            row[COL_SUCCESS] = success
            row[COL_FAILURE_STAGE] = failure_stage
            row[COL_EPISODE_DURATION] = round(duration, 4)

        # Write CSV
        with open(self.filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=ALL_COLUMNS)
            writer.writeheader()
            writer.writerows(self._rows)

        outcome = "✅ SUCCESS" if success else f"❌ FAILED at [{failure_stage}]"
        print(f"[Logger] {outcome} | steps={self.step_count} | "
              f"duration={duration:.2f}s | saved → {self.filepath}")

        return self.filepath
