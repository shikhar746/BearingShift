"""Download the Paderborn bearing archives (one .rar per bearing) with SHA-256 checks.

Default (no flags): only LIST the files and their sizes. Nothing is downloaded.
  python scripts/download_paderborn.py
Download after you have checked the size:
  python scripts/download_paderborn.py --yes
Extract into data/raw/paderborn/<bearing_id>/ (needs `unrar`, `unar`, `7z` or `bsdtar`):
  python scripts/download_paderborn.py --extract

Checksums: the dataset site publishes no checksums, so the first download of each file
records its SHA-256 in configs/paderborn_sha256.txt (commit that file). Every later
download is verified against it, so anyone reproducing gets byte-identical data.
Data licence: CC BY-NC 4.0 (see README for the citation).
"""
import argparse
import hashlib
import urllib.request
from pathlib import Path

import yaml

from bearingshift.data import load_bearing_table

CHECKSUM_FILE = Path("configs/paderborn_sha256.txt")


def read_checksums():
    sums = {}
    if CHECKSUM_FILE.exists():
        for line in CHECKSUM_FILE.read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                digest, name = line.split()
                sums[name] = digest
    return sums


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def remote_size(url):
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=30) as response:
        return int(response.headers.get("Content-Length", 0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--yes", action="store_true", help="actually download")
    parser.add_argument("--extract", action="store_true", help="extract downloaded archives")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))

    base_url = cfg["dataset"]["download_base_url"]
    archive_dir = Path(cfg["paths"]["archive_dir"])
    names = [f"{b}.rar" for b in load_bearing_table(cfg["dataset"]["bearing_table"])["bearing_id"]]

    if not args.yes and not args.extract:
        total = 0
        for name in names:
            size = remote_size(base_url + name)
            total += size
            print(f"{name:10s} {size / 1e6:9.1f} MB")
        print(f"{len(names)} files, total {total / 1e9:.2f} GB. Re-run with --yes to download.")
        return

    if args.yes:
        archive_dir.mkdir(parents=True, exist_ok=True)
        known = read_checksums()
        for name in names:
            target = archive_dir / name
            if not target.exists():
                print(f"Downloading {name} ...")
                part = target.with_suffix(".part")
                urllib.request.urlretrieve(base_url + name, part)
                part.rename(target)
            digest = sha256(target)
            if name in known and known[name] != digest:
                target.unlink()
                raise RuntimeError(f"Checksum mismatch for {name}: file deleted, please re-download")
            if name not in known:
                with open(CHECKSUM_FILE, "a") as f:
                    f.write(f"{digest}  {name}\n")
                print(f"Recorded new checksum for {name}")

    if args.extract:
        import rarfile
        raw_dir = Path(cfg["paths"]["raw_dir"])
        raw_dir.mkdir(parents=True, exist_ok=True)
        for name in names:
            print(f"Extracting {name} ...")
            rarfile.RarFile(archive_dir / name).extractall(raw_dir)


if __name__ == "__main__":
    main()
