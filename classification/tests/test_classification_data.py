"""Partition identity, archive containment, pairing, and worker-independent augmentation tests."""

import hashlib
import io
import json
import random
import shutil
import tarfile
import tempfile
import unittest
import urllib.error
import zipfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from clvision.config import CORRUPTIONS, recipe
from clvision.data import (
    EpochBatches,
    ImageDataset,
    download,
    extract_archive,
    load_manifest,
    make_fixture,
    make_loader,
    relative_file,
    stratified_split,
    transform_image,
)

from cue_routing.io import canonical_hash


class ClassificationDataTests(unittest.TestCase):
    """Exercise data semantics without downloading or evaluating benchmark targets."""

    def setUp(self):
        """Isolate every fixture and its manifest from real experiment artifacts."""
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        """Remove only the temporary test directory."""
        self.temporary.cleanup()

    def test_stratified_split_fixed_and_disjoint(self):
        """Every class gets the same frozen validation count independent of training seed."""
        labels = np.repeat(np.arange(100), 500)
        train, validation = stratified_split(labels, 100, 50)
        self.assertEqual((len(train), len(validation)), (45000, 5000))
        self.assertFalse(set(train) & set(validation))
        self.assertEqual(np.bincount(labels[validation]).tolist(), [50] * 100)
        self.assertEqual((train, validation), stratified_split(labels, 100, 50))
        self.assertNotEqual(validation, stratified_split(labels, 100, 50, seed=1)[1])

    def test_manifest_detects_changed_array(self):
        """A byte change cannot silently keep the same dataset identity."""
        manifest = make_fixture(self.root)
        path = relative_file(self.root, manifest["arrays"]["train_images"]["path"])
        with path.open("r+b") as stream:
            stream.seek(-1, 2)
            stream.write(b"\x03")
        with self.assertRaises(ValueError):
            load_manifest(self.root / "manifests/cifar100.json")

    def test_manifest_detects_altered_split_and_bounds(self):
        """Recomputed JSON identities still cannot introduce overlaps or invalid indices."""
        for mutation in ("overlap", "bounds", "duplicate"):
            manifest = make_fixture(self.root / mutation)
            if mutation == "overlap":
                manifest["splits"]["validation"][0] = manifest["splits"]["train"][0]
            elif mutation == "bounds":
                manifest["splits"]["train"][0] = 900
            else:
                manifest["splits"]["test"][0] = manifest["splits"]["test"][1]
            manifest["dataset_id"] = canonical_hash(
                {key: value for key, value in manifest.items() if key not in ("dataset_id", "root")}
            )
            path = self.root / mutation / "manifests/cifar100.json"
            path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                load_manifest(path)

    def test_relocation_preserves_dataset_identity(self):
        """Roots are operational paths rather than part of the semantic fingerprint."""
        original = self.root / "original"
        manifest = make_fixture(original)
        moved = self.root / "moved"
        shutil.copytree(original, moved)
        loaded = load_manifest(moved / "manifests/cifar100.json", root_override=moved)
        self.assertEqual(loaded["dataset_id"], manifest["dataset_id"])
        self.assertEqual(ImageDataset(loaded, "test")[0][0].shape, (3, 32, 32))

    def test_epoch_batches_cover_every_sample_without_replacement(self):
        """Uneven final batches must not discard data or duplicate images."""
        first = list(EpochBatches(23, 8, 0, 1))
        self.assertEqual(sorted(sum(first, [])), list(range(23)))
        self.assertEqual([len(batch) for batch in first], [8, 8, 7])
        self.assertEqual(first, list(EpochBatches(23, 8, 0, 1)))
        self.assertNotEqual(first, list(EpochBatches(23, 8, 0, 2)))

    def test_augmentation_does_not_change_global_rng(self):
        """Worker transforms must never reset model or CUDA randomness."""
        random.seed(7)
        np.random.seed(7)
        torch.manual_seed(7)
        python_state, numpy_state, torch_state = (
            random.getstate(),
            np.random.get_state(),
            torch.get_rng_state(),
        )
        image = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
        first = transform_image(image, "cifar100", True, 823)
        second = transform_image(image, "cifar100", True, 823)
        torch.testing.assert_close(first, second)
        self.assertEqual(random.getstate(), python_state)
        self.assertTrue(np.array_equal(np.random.get_state()[1], numpy_state[1]))
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))

    def test_workers_do_not_change_batches(self):
        """The same seed/epoch must produce identical images with zero or two workers."""
        manifest = make_fixture(self.root)
        config = recipe(
            "cifar100", debug=True, device="cpu", precision="fp32", batch_size=4, workers=0
        )
        zero = list(make_loader(manifest, config, "train", epoch=3))
        two = list(make_loader(manifest, replace(config, workers=2), "train", epoch=3))
        self.assertEqual(len(zero), len(two))
        for left, right in zip(zero, two):
            torch.testing.assert_close(left[0], right[0])
            torch.testing.assert_close(left[1], right[1])

    def test_dominoes_components_and_counterfactual_alignment(self):
        """Only the digit panel changes inside a paired final vehicle example."""
        manifest = make_fixture(self.root, "dominoes")
        loaded = load_manifest(self.root / "manifests/dominoes.json")
        dataset = ImageDataset(loaded, "test")
        self.assertEqual(len(dataset), 24)
        for index in range(0, len(dataset), 2):
            first, target0 = dataset[index]
            second, target1 = dataset[index + 1]
            self.assertEqual(first.shape, (3, 64, 32))
            self.assertEqual(target0, target1)
            torch.testing.assert_close(first[:, 32:], second[:, 32:])
            self.assertFalse(torch.equal(first[:, :32], second[:, :32]))
        self.assertFalse(
            set(manifest["digit_splits"]["train"]) & set(manifest["digit_splits"]["validation"])
        )

    def test_native_tiny_fixture_and_final_label(self):
        """Larger-class execution must include label199 at native64px."""
        manifest = make_fixture(self.root, "tiny_imagenet")
        dataset = ImageDataset(manifest, "test")
        self.assertEqual(dataset[-1][0].shape, (3, 64, 64))
        self.assertEqual(dataset[-1][1], 199)

    def test_archive_rejects_traversal_symlinks_and_duplicate_names(self):
        """Malformed archives cannot overwrite outside files or hide duplicate records."""
        for kind in ("traversal", "link", "duplicate"):
            path = self.root / f"{kind}.zip"
            with zipfile.ZipFile(path, "w") as stream:
                if kind == "traversal":
                    stream.writestr("../escape", "bad")
                elif kind == "link":
                    info = zipfile.ZipInfo("link")
                    info.external_attr = 0o120777 << 16
                    stream.writestr(info, "outside")
                else:
                    stream.writestr("same", "one")
                    stream.writestr("same", "two")
            with self.assertRaises(ValueError):
                extract_archive(path, self.root / kind)
        path = self.root / "hardlink.tar"
        with tarfile.open(path, "w") as stream:
            info = tarfile.TarInfo("hardlink")
            info.type, info.linkname = tarfile.LNKTYPE, "../outside"
            stream.addfile(info)
        with self.assertRaises(ValueError):
            extract_archive(path, self.root / "hardlink")

    def test_safe_archive_is_atomic_and_never_overwrites(self):
        """Successful extraction remains contained and an existing dataset is immutable."""
        archive = self.root / "valid.zip"
        with zipfile.ZipFile(archive, "w") as stream:
            stream.writestr("dataset/file.txt", "verified")
        output = self.root / "extracted"
        extract_archive(archive, output)
        self.assertEqual((output / "dataset/file.txt").read_text(), "verified")
        with self.assertRaises(FileExistsError):
            extract_archive(archive, output)
        with self.assertRaises(ValueError):
            relative_file(output, "../outside")

    def test_corruption_grid_is_exact(self):
        """The four auxiliary archive corruptions cannot silently enter the primary average."""
        self.assertEqual(len(CORRUPTIONS), 15)
        self.assertNotIn("speckle_noise", CORRUPTIONS)
        self.assertNotIn("gaussian_blur", CORRUPTIONS)

    def test_download_resumes_checked_ranges_and_rejects_wrong_bytes(self):
        """A lost connection preserves progress, while the final release checksum remains mandatory."""
        expected = b"abcdef"
        digest = hashlib.md5(expected).hexdigest()
        destination = self.root / "archive.bin"

        class Response(io.BytesIO):
            """Supply controlled HTTP metadata independently of a real download server."""

            def __init__(self, data, status=200, headers=None):
                super().__init__(data)
                self.status, self.headers = status, headers or {}

        class Interrupted(Response):
            """Fail after returning a verified-position prefix, leaving an unfinished partial file."""

            def read(self, count=-1):
                if self.tell() >= 3:
                    raise urllib.error.URLError("connection interrupted")
                return super().read(3)

        requests = []
        responses = [Interrupted(expected), Response(b"def", 206, {"Content-Range": "bytes 3-5/6"})]

        def open_response(request, **kwargs):
            requests.append(request)
            return responses.pop(0)

        with patch("clvision.data.urllib.request.urlopen", side_effect=open_response):
            download("https://example.test/archive", destination, digest)
        self.assertEqual(destination.read_bytes(), expected)
        self.assertEqual(requests[1].get_header("Range"), "bytes=3-")
        self.assertFalse(destination.with_suffix(".bin.partial").exists())
        bad = self.root / "bad.bin"
        with patch("clvision.data.urllib.request.urlopen", return_value=Response(b"altered")):
            with self.assertRaises(ValueError):
                download("https://example.test/archive", bad, digest)
        self.assertFalse(bad.exists())

    def test_download_ignores_unsupported_resume_by_restarting_safely(self):
        """A server returning200 to a range request must not concatenate duplicate bytes."""
        destination = self.root / "archive.bin"
        destination.with_suffix(".bin.partial").write_bytes(b"abc")

        class Response(io.BytesIO):
            """Model an HTTP server without range support."""

            status, headers = 200, {}

        with patch("clvision.data.urllib.request.urlopen", return_value=Response(b"abcdef")):
            download(
                "https://example.test/archive", destination, hashlib.md5(b"abcdef").hexdigest()
            )
        self.assertEqual(destination.read_bytes(), b"abcdef")
