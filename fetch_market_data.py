"""Fetch a PINNED third-party ETF adjusted-close snapshot with SHA-256 checks.

The archive is a secondary mirror of yfinance adjusted closes. It is not an
official exchange feed; verify its upstream provenance before live use.
Downloaded data are deliberately kept outside Git under data/raw/.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import tempfile
from urllib.request import urlopen


SOURCE_REPO = "https://github.com/manisahni/marketdata"
SOURCE_REVISION = "65a079329e379e104495f459e12e76a8f995585a"
SOURCE_URL = (
    f"https://raw.githubusercontent.com/manisahni/marketdata/"
    f"{SOURCE_REVISION}/daily_adjclose.csv"
)
EXPECTED_SHA256 = "361efb287116fc8200582c15fcd315bcce2e218fa1e50b416bacc5acebff37b9"
DEFAULT_DESTINATION = Path("data/raw/daily_adjclose.csv")


def verified_download(
    destination: Path = DEFAULT_DESTINATION,
    *,
    url: str = SOURCE_URL,
    expected_sha256: str = EXPECTED_SHA256,
) -> str:
    """Atomically replace the destination only after a checksum match."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=destination.parent, prefix=".market-", delete=False
        ) as out:
            temporary_path = Path(out.name)
            digest = hashlib.sha256()
            with urlopen(url, timeout=50) as response:
                while block := response.read(1024 * 1024):
                    out.write(block)
                    digest.update(block)
            actual = digest.hexdigest()
        if actual.lower() != expected_sha256.lower():
            raise ValueError(
                f"Market-data checksum mismatch: {actual} != {expected_sha256}"
            )
        os.replace(temporary_path, destination)
        return actual
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_DESTINATION)
    args = parser.parse_args()
    digest = verified_download(args.output)
    print(f"Downloaded verified external adjusted-close snapshot: {args.output}")
    print(f"SHA-256: {digest}")
    print(f"Source: {SOURCE_REPO} at {SOURCE_REVISION}")
    print("Caution: third-party yfinance mirror; data are not independently audited.")


if __name__ == "__main__":
    main()
