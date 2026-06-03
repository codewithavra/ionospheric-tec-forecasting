"""
Reads raw per-satellite CSV files from the iisc*_TECU folders and
produces minute-wise maximum TEC values for each day folder.
"""
# Imports

import csv
import re
from datetime import datetime
from pathlib import Path

from src.configs.config import (
    BASE_DIR, OUTPUT_DIR,
    FOLDER_RANGE , SATELLITES
)

def discover_source_folders(base_dir: Path = BASE_DIR) -> list[tuple[int, Path]]:
    """Return sorted list of (day_number, folder_path) within FOLDER_RANGE."""
    lo, hi = FOLDER_RANGE
    pattern = re.compile(r"^iisc(\d{4})_TECU$")
    folders = []
    for p in base_dir.iterdir():
        if not p.is_dir():
            continue
        m = pattern.match(p.name)
        if not m:
            continue
        n = int(m.group(1))
        if lo <= n <= hi:
            folders.append((n, p))
    folders.sort(key=lambda x: x[0])
    return folders
 
 
def aggregate_minute_max(folder_path: Path) -> dict:
    """
    Read all G01_TEC.csv … G32_TEC.csv in *folder_path* and return a dict
    mapping ``datetime`` (second=0) → maximum TEC float observed that minute.
    """
    minute_max: dict = {}
    for sat_idx in SATELLITES:
        csv_path = folder_path / f"G{sat_idx:02d}_TEC.csv"
        if not csv_path.exists():
            continue
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                time_text = (row.get("Time (UTC)") or "").strip()
                tec_text  = (row.get("TEC (TECU)")  or "").strip()
                if not time_text or not tec_text:
                    continue
                try:
                    dt  = datetime.strptime(time_text, "%Y-%m-%d %H:%M:%S")
                    tec = float(tec_text)
                except ValueError:
                    continue
                minute_dt = dt.replace(second=0)
                prev = minute_max.get(minute_dt)
                if prev is None or tec > prev:
                    minute_max[minute_dt] = tec
    return minute_max
 
 
def write_sorted_csv(minute_max: dict, out_path: Path) -> None:
    """Write the minute_max dict to *out_path* in sorted order."""
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Time(UTC)", "TEC(TECU)"])
        for minute_dt in sorted(minute_max.keys()):
            writer.writerow([
                minute_dt.strftime("%d-%m-%y %H:%M"),
                f"{minute_max[minute_dt]:.6f}",
            ])
 
 
def build_sorted_dataset(
    base_dir: Path = BASE_DIR,
    output_dir: Path = OUTPUT_DIR,
) -> int:
    
    """
    Discover raw folders, aggregate minute-wise max TEC, write one sorted
    CSV per folder into *output_dir*.  Returns the number of files written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    folders = discover_source_folders(base_dir)
    written = 0
    for _, folder_path in folders:
        minute_max = aggregate_minute_max(folder_path)
        out_path   = output_dir / f"{folder_path.name}_sorted.csv"
        write_sorted_csv(minute_max, out_path)
        written += 1
    return written
