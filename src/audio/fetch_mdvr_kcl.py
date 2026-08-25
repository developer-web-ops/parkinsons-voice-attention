"""Explicit acquisition of the MDVR-KCL Parkinson's voice dataset from Zenodo.

MDVR-KCL — "Mobile Device Voice Recordings at King's College London" — is a
CC-BY-4.0 corpus of 44.1 kHz mono WAV recordings from early and advanced
Parkinson's disease patients and healthy controls (Zenodo record 2867216).

Note on bit depth: the Zenodo record reports 16-bit, but every extracted WAV
inspected empirically is 24-bit PCM (see ``VERIFIED_AUDIO_FORMAT``). Both values
are recorded in the manifest — the source-reported value is preserved, not
overwritten — so the discrepancy is auditable rather than hidden.

This module NEVER downloads on import. Acquisition is an explicit action::

    python -m src.audio.fetch_mdvr_kcl --data-dir data_audio/mdvr_kcl

which (1) downloads the ~606 MB archive from Zenodo, (2) verifies it against the
MD5 published on the record, (3) extracts the WAV files, and (4) writes a
SHA-256 manifest (``manifest.sha256.json``) recording provenance for the archive
and every extracted file.

Raw audio and the archive are git-ignored and must never be committed; only the
generated manifest (metadata, not audio) is version-controlled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

# --- Authoritative dataset metadata -------------------------------------------
# Verified by the project owner against the Zenodo record on 2026-08-23.
# Source of record: https://zenodo.org/records/2867216
DATASET = {
    "title": (
        "Mobile Device Voice Recordings at King's College London (MDVR-KCL) "
        "from both early and advanced Parkinson's disease patients and healthy controls"
    ),
    "authors": ["Hagen Jaeger", "Dhaval Trivedi", "Michael Stadtschnitzer"],
    "doi": "10.5281/zenodo.2867216",
    "concept_doi": "10.5281/zenodo.2867215",
    "zenodo_url": "https://zenodo.org/records/2867216",
    "version": "v1",
    "publication_date": "2019-05-17",
    "license": "CC-BY-4.0",
    "license_url": "https://creativecommons.org/licenses/by/4.0/",
    "access": "open",
    # Source-reported format as stated on the Zenodo record. The empirically
    # verified format differs (24-bit, not 16-bit); see VERIFIED_AUDIO_FORMAT.
    # Both are preserved in the manifest — the reported value is never overwritten.
    "audio_format_reported": "WAV, 44.1 kHz, 16-bit",
}

# The single file published on the record. size_bytes and md5 are the values
# verified against the record; the download is refused unless the MD5 matches.
ARCHIVE = {
    "filename": "26_29_09_2017_KCL.zip",
    "size_bytes": 606144431,
    "md5": "98c51bdd2b092b93f8bb038dea4505fa",
    "md5_source": "Zenodo record 2867216 (published checksum)",
}

# Attribution required by CC-BY-4.0. This records only what the authoritative
# record supports; it does not assert the absence of any further terms.
ATTRIBUTION = {
    "required_by": "CC-BY-4.0",
    "citation": (
        "Jaeger, H., Trivedi, D., & Stadtschnitzer, M. (2019). Mobile Device Voice "
        "Recordings at King's College London (MDVR-KCL) from both early and advanced "
        "Parkinson's disease patients and healthy controls [Data set]. Zenodo. "
        "https://doi.org/10.5281/zenodo.2867216"
    ),
    "license_url": "https://creativecommons.org/licenses/by/4.0/",
    "license_legalcode": "https://creativecommons.org/licenses/by/4.0/legalcode",
}

# Empirically verified audio format. The Zenodo record reports 16-bit, but every
# extracted WAV file inspected has a 24-bit PCM `fmt ` chunk. This block records
# the measured truth WITHOUT overwriting the source-reported value in DATASET;
# both are surfaced in the manifest so the discrepancy is documented, not hidden.
VERIFIED_AUDIO_FORMAT = {
    "pcm_format_code": 1,  # WAVE_FORMAT_PCM
    "sample_rate_hz": 44100,
    "bit_depth": 24,
    "channels": 1,
    "encoding": "PCM, mono, 44.1 kHz, 24-bit",
    "method": (
        "Read the WAV 'fmt ' chunk headers of the extracted files with the Python "
        "stdlib (wave module + raw byte parse of audio_format/channels/sample_rate/"
        "bits_per_sample). All 73 files inspected agreed."
    ),
    "differs_from_reported": (
        "Zenodo record reports 16-bit; measured bit depth is 24-bit. The "
        "source-reported value is preserved in dataset.audio_format_reported and "
        "is intentionally not overwritten."
    ),
}

# Filename/convention anomalies found in the extracted corpus. These are valid
# recordings; the note is a hand-off to Phase 2B so subject-ID parsing stays
# leakage-safe. NOT an exhaustive data-quality audit.
KNOWN_ANOMALIES = [
    {
        "path": "26-29_09_2017_KCL/SpontaneousDialogue/HC/ID22hc_0_0_0.wav",
        "type": "filename_convention",
        "description": (
            "Missing underscore between subject ID and cohort: 'ID22hc' where every "
            "other file uses 'ID<NN>_<hc|pd>'. The audio itself is valid 24-bit PCM."
        ),
        "action_required": (
            "Phase 2B must parse subject IDs with a tolerant matcher (e.g. regex "
            r"r'ID(\d+)') rather than str.split('_')[0]. A naive split would mint a "
            "phantom subject 'ID22hc', splitting ID22's recordings across folds and "
            "leaking the subject between train and test in grouped cross-validation."
        ),
    },
]

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data_audio" / "mdvr_kcl"
MANIFEST_NAME = "manifest.sha256.json"
SCHEMA_VERSION = 2

_CHUNK = 1 << 20  # 1 MiB streaming chunk


def download_url() -> str:
    """Direct-download URL for the published archive."""
    return f"{DATASET['zenodo_url']}/files/{ARCHIVE['filename']}?download=1"


def _hash_file(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_of(path: Path) -> str:
    return _hash_file(path, "md5")


def sha256_of(path: Path) -> str:
    return _hash_file(path, "sha256")


def verify_md5(path: Path, expected: str = ARCHIVE["md5"]) -> str:
    """Raise unless ``path`` hashes to ``expected``. Returns the verified digest."""
    actual = md5_of(path)
    if actual != expected:
        raise ValueError(
            f"MD5 mismatch for {path.name}: got {actual}, expected {expected}. "
            "Archive is corrupt or not the verified MDVR-KCL release; refusing to proceed."
        )
    return actual


def download_archive(
    dest: Path,
    *,
    url: str | None = None,
    expected_size: int | None = ARCHIVE["size_bytes"],
    timeout: int = 60,
) -> Path:
    """Stream the archive to ``dest``. Only called from an explicit command."""
    url = url or download_url()
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}\n         -> {dest}")
    total = 0
    with urllib.request.urlopen(url, timeout=timeout) as resp, open(dest, "wb") as out:
        while True:
            chunk = resp.read(_CHUNK)
            if not chunk:
                break
            out.write(chunk)
            total += len(chunk)
            if expected_size:
                print(f"\r  {total:,} / {expected_size:,} bytes", end="", flush=True)
    print()
    if expected_size is not None and total != expected_size:
        raise ValueError(f"size mismatch for {dest.name}: got {total}, expected {expected_size}")
    return dest


def extract_archive(zip_path: Path, dest_dir: Path) -> Path:
    """Extract all members into ``dest_dir``, guarding against path traversal."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    root = dest_dir.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            target = (dest_dir / member).resolve()
            if root != target and root not in target.parents:
                raise ValueError(f"unsafe path in archive: {member!r}")
        zf.extractall(dest_dir)
    return dest_dir


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def manifest_skeleton() -> dict:
    """Verified, acquisition-independent metadata.

    Shared by the committed placeholder manifest and the fully-generated one so
    the recorded provenance can never drift from the code constants.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": dict(DATASET),
        # Audio format is recorded in two clearly-labelled forms: what the source
        # says, and what the bytes say. They disagree on bit depth (see below).
        "audio_format": {
            "source_reported": {
                "value": DATASET["audio_format_reported"],
                "source": "Zenodo record 2867216 metadata",
            },
            "empirically_verified": dict(VERIFIED_AUDIO_FORMAT),
        },
        "archive": {
            "filename": ARCHIVE["filename"],
            "size_bytes": ARCHIVE["size_bytes"],
            "md5": ARCHIVE["md5"],
            "md5_source": ARCHIVE["md5_source"],
            "sha256": None,
        },
        "attribution": dict(ATTRIBUTION),
        "acquisition": {
            "status": "pending",
            "acquired_at": None,
            "download_url": download_url(),
            "tool": "python -m src.audio.fetch_mdvr_kcl",
        },
        "known_anomalies": [dict(a) for a in KNOWN_ANOMALIES],
        "file_count": 0,
        "files": [],
        "notes": (
            "Per-file SHA-256 entries are generated by fetch_mdvr_kcl.py after a "
            "verified download + extraction. Raw audio and the archive itself are "
            "never committed to Git; only this metadata manifest is."
        ),
    }


def build_manifest(*, archive_path: Path, raw_dir: Path) -> dict:
    """Full manifest: skeleton + archive SHA-256 + one entry per extracted file."""
    m = manifest_skeleton()
    m["archive"]["sha256"] = sha256_of(archive_path)
    m["acquisition"]["status"] = "acquired"
    m["acquisition"]["acquired_at"] = _now_iso()
    files = [
        {
            "path": p.relative_to(raw_dir).as_posix(),
            "size_bytes": p.stat().st_size,
            "sha256": sha256_of(p),
        }
        for p in sorted(raw_dir.rglob("*"))
        if p.is_file()
    ]
    m["files"] = files
    m["file_count"] = len(files)
    return m


def write_manifest(manifest: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def acquire(
    data_dir: Path,
    *,
    keep_archive: bool = True,
    skip_download: bool = False,
    force: bool = False,
) -> Path:
    """Download (unless skipped), MD5-verify, extract, then write the manifest."""
    data_dir = Path(data_dir)
    raw_dir = data_dir / "raw"
    archive_path = data_dir / ARCHIVE["filename"]
    manifest_path = data_dir / MANIFEST_NAME

    if not skip_download:
        if archive_path.exists() and not force:
            print(f"Archive already present: {archive_path} (use --force to re-download)")
        else:
            download_archive(archive_path)
    if not archive_path.exists():
        raise FileNotFoundError(
            f"{archive_path} not found. Download it first, or drop --skip-download."
        )

    print("Verifying MD5 against the Zenodo-published checksum ...")
    verify_md5(archive_path)
    print("MD5 OK.")

    print(f"Extracting into {raw_dir} ...")
    extract_archive(archive_path, raw_dir)

    print("Computing SHA-256 manifest ...")
    manifest = build_manifest(archive_path=archive_path, raw_dir=raw_dir)
    write_manifest(manifest, manifest_path)
    print(f"Wrote {manifest_path} ({manifest['file_count']} files).")

    if not keep_archive:
        archive_path.unlink()
        print(f"Removed archive {archive_path}.")
    return manifest_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.audio.fetch_mdvr_kcl",
        description=(
            "Explicitly download, MD5-verify, extract and SHA-256-manifest the MDVR-KCL "
            "Parkinson's voice dataset. Does nothing unless run as a command."
        ),
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"target directory (default: {DEFAULT_DATA_DIR})",
    )
    p.add_argument(
        "--skip-download",
        action="store_true",
        help="verify + extract an archive already present in --data-dir (no network)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="re-download even if the archive already exists",
    )
    p.add_argument(
        "--no-keep-archive",
        dest="keep_archive",
        action="store_false",
        help="delete the ~606 MB archive after a successful extraction",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    acquire(
        args.data_dir,
        keep_archive=args.keep_archive,
        skip_download=args.skip_download,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
