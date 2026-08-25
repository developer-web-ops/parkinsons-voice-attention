"""Provenance and acquisition guards for the Phase 2A MDVR-KCL audio dataset.

These tests never touch the network and never require the ~606 MB dataset to be
present. They validate the committed provenance/manifest metadata and exercise
the acquisition helpers (hashing, manifest building, archive extraction) against
tiny local fixtures, so they stay green in CI where the raw audio is absent.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from src.audio import fetch_mdvr_kcl as fetch

DATA_DIR = Path(__file__).resolve().parents[1] / "data_audio" / "mdvr_kcl"
MANIFEST_PATH = DATA_DIR / "manifest.sha256.json"

# Values independently verified against the Zenodo record by the project owner.
VERIFIED_MD5 = "98c51bdd2b092b93f8bb038dea4505fa"
VERIFIED_SIZE = 606144431
ARCHIVE_NAME = "26_29_09_2017_KCL.zip"


@pytest.fixture
def manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


# --- verified constants -------------------------------------------------------


def test_verified_archive_constants_match_record():
    assert fetch.ARCHIVE["filename"] == ARCHIVE_NAME
    assert fetch.ARCHIVE["size_bytes"] == VERIFIED_SIZE
    assert fetch.ARCHIVE["md5"] == VERIFIED_MD5


def test_dataset_constants():
    d = fetch.DATASET
    assert d["doi"] == "10.5281/zenodo.2867216"
    assert d["concept_doi"] == "10.5281/zenodo.2867215"
    assert d["license"] == "CC-BY-4.0"
    # Source-reported bit depth is preserved verbatim (see audio_format tests for
    # the empirically-verified 24-bit value that supersedes it downstream).
    assert d["audio_format_reported"] == "WAV, 44.1 kHz, 16-bit"


def test_default_data_dir_points_at_namespace():
    assert fetch.DEFAULT_DATA_DIR == DATA_DIR


# --- committed manifest -------------------------------------------------------


def test_manifest_has_required_schema(manifest):
    for key in (
        "schema_version",
        "dataset",
        "audio_format",
        "archive",
        "attribution",
        "acquisition",
        "known_anomalies",
        "file_count",
        "files",
    ):
        assert key in manifest
    assert manifest["schema_version"] == fetch.SCHEMA_VERSION == 2
    assert manifest["archive"]["md5"] == VERIFIED_MD5
    assert manifest["archive"]["size_bytes"] == VERIFIED_SIZE
    assert manifest["dataset"]["license"] == "CC-BY-4.0"


def test_manifest_matches_module_constants(manifest):
    # committed provenance must never drift from the code source-of-truth
    assert manifest["dataset"] == dict(fetch.DATASET)
    assert manifest["attribution"] == dict(fetch.ATTRIBUTION)
    assert manifest["archive"]["filename"] == fetch.ARCHIVE["filename"]
    assert manifest["archive"]["md5"] == fetch.ARCHIVE["md5"]


def test_audio_format_records_reported_and_verified(manifest):
    """Both the source-reported (16-bit) and measured (24-bit) formats survive,
    clearly distinguished, with the reported value never overwritten."""
    fmt = manifest["audio_format"]

    # 1. Source-reported value preserved verbatim.
    assert fmt["source_reported"]["value"] == "WAV, 44.1 kHz, 16-bit"
    assert fmt["source_reported"]["value"] == fetch.DATASET["audio_format_reported"]
    assert "Zenodo" in fmt["source_reported"]["source"]

    # 2. Empirically-verified format matches the module constant exactly.
    verified = fmt["empirically_verified"]
    assert verified == dict(fetch.VERIFIED_AUDIO_FORMAT)
    assert verified["bit_depth"] == 24
    assert verified["sample_rate_hz"] == 44100
    assert verified["channels"] == 1
    assert verified["pcm_format_code"] == 1

    # 3. The two disagree on bit depth, and that is the whole point.
    assert "16-bit" in fmt["source_reported"]["value"]
    assert verified["bit_depth"] != 16


def test_id22_filename_anomaly_documented(manifest):
    """The ID22 missing-underscore filename is recorded with a Phase 2B hand-off
    so subject-grouped CV cannot silently leak the subject across folds."""
    anomalies = manifest["known_anomalies"]
    assert anomalies == [dict(a) for a in fetch.KNOWN_ANOMALIES]
    id22 = next(a for a in anomalies if "ID22hc" in a["path"])
    assert id22["path"] == "26-29_09_2017_KCL/SpontaneousDialogue/HC/ID22hc_0_0_0.wav"
    assert id22["type"] == "filename_convention"
    # the fix is a tolerant parser, not str.split("_")
    assert "split" in id22["action_required"]
    assert r"ID(\d+)" in id22["action_required"]


def test_manifest_status_is_internally_consistent(manifest):
    """Green whether or not the dataset has been fetched locally."""
    status = manifest["acquisition"]["status"]
    assert status in ("pending", "acquired")
    if status == "pending":
        assert manifest["files"] == []
        assert manifest["file_count"] == 0
        assert manifest["archive"]["sha256"] is None
    else:  # acquired
        assert manifest["file_count"] == len(manifest["files"]) > 0
        assert manifest["archive"]["sha256"]
        assert manifest["acquisition"]["acquired_at"]
        for entry in manifest["files"]:
            assert entry["path"]
            assert len(entry["sha256"]) == 64


def test_provenance_and_license_files_exist():
    prov = (DATA_DIR / "PROVENANCE.md").read_text(encoding="utf-8")
    assert "10.5281/zenodo.2867216" in prov
    assert "CC-BY-4.0" in prov
    lic = (DATA_DIR / "LICENSE.txt").read_text(encoding="utf-8")
    assert "CC-BY-4.0" in lic
    assert "creativecommons.org/licenses/by/4.0" in lic


# --- acquisition helpers (fixtures only, no network, no real audio) -----------


def test_hash_helpers_match_stdlib(tmp_path):
    blob = b"parkinsons-voice-audio-fixture\n" * 1000
    f = tmp_path / "sample.bin"
    f.write_bytes(blob)
    assert fetch.sha256_of(f) == hashlib.sha256(blob).hexdigest()
    assert fetch.md5_of(f) == hashlib.md5(blob).hexdigest()


def test_verify_md5_raises_on_mismatch(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"not the real archive")
    with pytest.raises(ValueError, match="MD5 mismatch"):
        fetch.verify_md5(f, VERIFIED_MD5)


def test_build_manifest_over_fixture(tmp_path):
    raw = tmp_path / "raw"
    (raw / "sub").mkdir(parents=True)
    (raw / "ID01_hc.wav").write_bytes(b"RIFFfake-wav-1")
    (raw / "sub" / "ID02_pd.wav").write_bytes(b"RIFFfake-wav-2")
    archive = tmp_path / ARCHIVE_NAME
    archive.write_bytes(b"pretend-zip-bytes")

    m = fetch.build_manifest(archive_path=archive, raw_dir=raw)

    assert m["acquisition"]["status"] == "acquired"
    assert m["acquisition"]["acquired_at"]
    assert m["file_count"] == 2
    assert {e["path"] for e in m["files"]} == {"ID01_hc.wav", "sub/ID02_pd.wav"}
    assert m["archive"]["sha256"] == fetch.sha256_of(archive)
    # metadata skeleton is preserved in the generated manifest
    assert m["dataset"] == dict(fetch.DATASET)
    assert m["audio_format"]["empirically_verified"]["bit_depth"] == 24
    assert m["known_anomalies"] == [dict(a) for a in fetch.KNOWN_ANOMALIES]


def test_extract_archive_extracts_and_blocks_traversal(tmp_path):
    good = tmp_path / "good.zip"
    with zipfile.ZipFile(good, "w") as zf:
        zf.writestr("a.wav", b"aaa")
        zf.writestr("nested/b.wav", b"bbb")
    out = tmp_path / "out"
    fetch.extract_archive(good, out)
    assert (out / "a.wav").read_bytes() == b"aaa"
    assert (out / "nested" / "b.wav").read_bytes() == b"bbb"

    evil = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil, "w") as zf:
        zf.writestr("../escape.wav", b"nope")
    with pytest.raises(ValueError, match="unsafe path"):
        fetch.extract_archive(evil, tmp_path / "out2")


def test_import_exposes_entry_points_without_running_them():
    # acquisition is opt-in: the callables exist but importing never invokes them
    assert callable(fetch.acquire)
    assert callable(fetch.download_archive)
    assert callable(fetch.main)
    url = fetch.download_url()
    assert url.startswith("https://zenodo.org/records/2867216/files/")
    assert ARCHIVE_NAME in url
