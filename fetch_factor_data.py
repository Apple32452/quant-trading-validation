"""Download three official Dartmouth Fama-French daily factor archives.

The SHA-256 pins are for the public 2026-09 CRSP release downloaded on
2026-11-12 (Eastern). The official URLs can change when the library updates.
If they change, STOP and record a new, explicitly versioned analysis rather
than silently mixing updated factor vintages with published study results.
Downloaded archives stay in ignored data/raw/ and are not redistributed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fetch_market_data import verified_download


FACTOR_SOURCE_ROOT = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"
)
ARCHIVES = {
    "ff3": {
        "upstream": "F-F_Research_Data_Factors_daily_CSV.zip",
        "local": "ff3_daily.zip",
        "sha256": "96c23b88e651cbfff3740a53f8225e59ee39893371e96385a1d8125ec2cd9c89",
    },
    "ff5": {
        "upstream": "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
        "local": "ff5_daily.zip",
        "sha256": "e58577223032aa45071a46686f7068900c371b516c9eb04e28e974c34264f84f",
    },
    "mom": {
        "upstream": "F-F_Momentum_Factor_daily_CSV.zip",
        "local": "momentum_daily.zip",
        "sha256": "e56d5bc845544e2993e77353b12c611cfce3149ef754f9b59fc67904ddadd369",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    for key, config in ARCHIVES.items():
        destination = args.output_dir / config["local"]
        digest = verified_download(
            destination,
            url=f"{FACTOR_SOURCE_ROOT}/{config['upstream']}",
            expected_sha256=config["sha256"],
        )
        print(f"{key}: {destination} | verified SHA-256 {digest}")
    print(
        "Source: official Kenneth French Data Library, 2026-09 CRSP release."
        " Factor revisions require a newly versioned study."
    )


if __name__ == "__main__":
    main()
