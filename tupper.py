"""Read the 20-bin depth/Tupper table used by the CUDA preprocessing kernel."""

import csv
from pathlib import Path


def load_tupper_csv(path):
    path = Path(path)
    values = [None] * 20
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            index = int(row["bin_idx"])
            if not 0 <= index < 20 or values[index] is not None:
                raise ValueError("Invalid or repeated depth bin %s" % index)
            if float(row["depth_min"]) != index * 5.0 or float(row["depth_max"]) != (index + 1) * 5.0:
                raise ValueError("Expected 20 uniform bins over [0, 100]")
            raw = row["Tupper"].strip()
            # An empty bin has no measured upper bound. Use 1 conservatively.
            value = 1.0 if not raw else float(raw)
            if not 0.0 < value <= 1.0:
                raise ValueError("Tupper must be in (0, 1] at bin %d" % index)
            values[index] = value
    if any(value is None for value in values):
        raise ValueError("Tupper CSV must contain all 20 depth bins")
    return values
