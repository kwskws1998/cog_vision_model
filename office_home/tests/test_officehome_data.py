"""Office-Home identity, complete-package, partition, archive, and sampler regressions."""

import io
import shutil
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import torch
from ohvision.config import CLASSES, DOMAINS, EXPECTED_COUNTS, ExperimentConfig
from ohvision.data import (
    build_splits,
    canonical_name,
    discover_images,
    download_officehome,
    extract_archive,
    load_manifest,
    make_fixture,
    prepare_manifest,
    training_loader,
    validate_manifest_config,
    verify_original_image_collection,
)
from ohvision.fingerprint import source_hash

from cue_routing.io import atomic_json, canonical_hash, read_json, seed_everything
from cue_routing.io import source_hash as core_source_hash

ROOT = Path(__file__).resolve().parents[2]


class OfficeHomeDataTests(unittest.TestCase):
    """Use a 65-class fixture while requiring exact real-package counts in production."""

    @classmethod
    def setUpClass(cls):
        """Build one immutable labeled fixture for independent read-only test cases."""
        cls.temporary = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.temporary.name)
        cls.path = make_fixture(cls.directory / "images")
        cls.manifest = load_manifest(cls.path)

    @classmethod
    def tearDownClass(cls):
        """Remove all artificial images after the class finishes."""
        cls.temporary.cleanup()

    def test_config_uses_all_65_classes_and_correct_domain_names(self):
        self.assertEqual(len(CLASSES), 65)
        self.assertEqual(len(set(CLASSES)), 65)
        self.assertEqual(sum(EXPECTED_COUNTS.values()), 15588)
        self.assertEqual(DOMAINS, ("Art", "Clipart", "Product", "Real_World"))
        for domain in DOMAINS:
            self.assertEqual(ExperimentConfig(target=domain).validate().target, domain)
        for updates in ({"target": "sketch"}, {"data_variant": "automatic"}, {"seed": True}):
            with self.assertRaises(ValueError):
                ExperimentConfig(**updates).validate()

    def test_split_is_order_invariant_disjoint_and_covers_every_class(self):
        names = list(self.manifest["images"])
        self.assertEqual(build_splits(names), build_splits(list(reversed(names))))
        for domain in DOMAINS:
            parts = self.manifest["splits"][domain]
            train, valid, test = (dict(parts[name]) for name in ("train", "crossval", "test"))
            self.assertFalse(set(train) & set(valid))
            self.assertEqual({**train, **valid}, test)
            self.assertEqual(set(train.values()), set(range(65)))
            self.assertEqual(set(valid.values()), set(range(65)))

    def test_real_preparation_rejects_incomplete_fixture(self):
        with self.assertRaisesRegex(ValueError, "Incomplete Office-Home"):
            prepare_manifest(self.directory / "images", self.directory / "must_not_exist.json")
        self.assertFalse((self.directory / "must_not_exist.json").exists())

    def test_rehashed_split_tampering_is_rejected(self):
        corrupted = read_json(self.path)
        parts = corrupted["splits"]["Art"]
        parts["train"][0], parts["crossval"][0] = parts["crossval"][0], parts["train"][0]
        content = {
            key: value for key, value in corrupted.items() if key not in ("data_root", "dataset_id")
        }
        corrupted["dataset_id"] = canonical_hash(content)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            atomic_json(path, corrupted)
            with self.assertRaisesRegex(ValueError, "prespecified partition"):
                load_manifest(path, verify_images=False)

    def test_image_variant_requires_explicit_matching_configuration(self):
        manifest = {"kind": "officehome", "variant": "resized_256_square"}
        with self.assertRaisesRegex(ValueError, "config requires original_resolution"):
            validate_manifest_config(manifest, ExperimentConfig())
        validate_manifest_config(manifest, ExperimentConfig(data_variant="resized_256_square"))

    def test_changed_imported_bytes_cannot_be_labeled_original(self):
        with self.assertRaisesRegex(ValueError, "verified original-resolution"):
            verify_original_image_collection({"Art/TV/a.jpg": {"sha256": "0" * 64}})

    def test_paths_reject_unknown_labels_and_traversal(self):
        self.assertEqual(
            canonical_name("real_world/train/Toothbrush/a.jpg"), "Real_World/ToothBrush/a.jpg"
        )
        for name in (
            "/Art/TV/a.jpg",
            "Art/../a.jpg",
            "Art/TV/../a.jpg",
            "Art/TV\\a.jpg",
            "Art/unknown/a.jpg",
            "Art/test/TV/a.jpg",
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                canonical_name(name)

    def test_duplicate_image_identity_across_train_val_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for split in ("train", "val"):
                folder = root / "art" / split / "TV"
                folder.mkdir(parents=True)
                (folder / "same.jpg").write_bytes(b"not opened before duplicate detection")
            with self.assertRaisesRegex(ValueError, "Duplicated image identity"):
                discover_images(root)

    def test_symlinked_class_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "Art").mkdir()
            (root / "Art" / "TV").symlink_to(
                self.directory / "images/Art/TV", target_is_directory=True
            )
            with self.assertRaisesRegex(ValueError, "Symlinked"):
                discover_images(root)

    def test_content_change_and_missing_image_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shutil.copytree(self.directory / "images", root / "images")
            manifest = read_json(self.path)
            manifest["data_root"] = str(root / "images")
            path = root / "manifest.json"
            atomic_json(path, manifest)
            name = next(iter(manifest["images"]))
            image = root / "images" / manifest["images"][name]["path"]
            image.write_bytes(image.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "content changed"):
                load_manifest(path)
            image.unlink()
            with self.assertRaises(FileNotFoundError):
                load_manifest(path)

    def test_loader_resume_worker_independence_and_source_exclusion(self):
        config = ExperimentConfig(
            device="cpu",
            precision="fp32",
            debug=True,
            image_size=64,
            pretrained=False,
            batch_per_domain=1,
            steps=3,
            eval_every=1,
            workers=0,
        )
        seed_everything(3)
        first = list(training_loader(self.manifest, config))
        resumed = list(training_loader(self.manifest, config, 1, 3))
        for actual, expected in zip(resumed, first[1:]):
            self.assertTrue(torch.equal(actual[0], expected[0]))
            self.assertTrue(torch.equal(actual[1], expected[1]))
        self.assertNotIn(
            config.target, training_loader(self.manifest, config).dataset.domain_indices
        )
        with self.assertRaisesRegex(ValueError, "stateless augmentation seed"):
            training_loader(self.manifest, config).dataset[0]
        parallel = ExperimentConfig.from_dict({**config.to_dict(), "workers": 2})
        for actual, expected in zip(training_loader(self.manifest, parallel), first):
            self.assertTrue(torch.equal(actual[0], expected[0]))
            self.assertTrue(torch.equal(actual[1], expected[1]))

    def test_augmentation_does_not_seed_cuda_or_advance_cpu_model_rng(self):
        config = ExperimentConfig(
            device="cpu", precision="fp32", debug=True, image_size=64, pretrained=False, workers=0
        )
        dataset = training_loader(self.manifest, config).dataset
        state = torch.get_rng_state().clone()
        with patch("torch.cuda.manual_seed_all") as cuda_seed:
            dataset[(0, 123)]
            cuda_seed.assert_not_called()
        self.assertTrue(torch.equal(torch.get_rng_state(), state))

    def test_extension_fingerprint_tracks_shared_and_extension_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "pyproject.toml").write_text('[project]\nname = "fixture"\n')
            (root / "src").mkdir()
            (root / "office_home").mkdir()
            core = root / "src" / "shared.py"
            extension = root / "office_home" / "extension.py"
            core.write_text("SETTING = 1\n")
            extension.write_text("SETTING = 1\n")
            core_before, extension_before = core_source_hash(root), source_hash(root)
            core.write_text("SETTING = 2\n")
            core_after, extension_after = core_source_hash(root), source_hash(root)
            self.assertNotEqual(core_before, core_after)
            self.assertNotEqual(extension_before, extension_after)
            extension.write_text("SETTING = 2\n")
            self.assertEqual(core_source_hash(root), core_after)
            self.assertNotEqual(source_hash(root), extension_after)


class OfficeHomeArchiveTests(unittest.TestCase):
    """Reject unsafe archives and corrupted caches before publishing dataset directories."""

    def test_zip_traversal_and_tar_links_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "bad.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("../outside.jpg", b"x")
            with self.assertRaises(ValueError):
                extract_archive(archive, root / "images")
            with tarfile.open(root / "bad.tgz", "w:gz") as handle:
                member = tarfile.TarInfo("Art/TV/link.jpg")
                member.type = tarfile.SYMTYPE
                member.linkname = "/tmp/outside"
                handle.addfile(member)
            with self.assertRaisesRegex(ValueError, "Links"):
                extract_archive(root / "bad.tgz", root / "images")
            self.assertFalse((root / "images").exists())

    def test_nested_zip_import_and_duplicate_tar_members(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "valid.zip"
            with zipfile.ZipFile(path, "w") as handle:
                for domain in DOMAINS:
                    handle.writestr(f"OfficeHomeDataset_10072016/{domain}/TV/a.jpg", b"x")
            extract_archive(path, root / "images")
            self.assertEqual(len(discover_images(root / "images")), 4)
            duplicate = root / "duplicate.tgz"
            with tarfile.open(duplicate, "w:gz") as handle:
                for _ in range(2):
                    member = tarfile.TarInfo("Art/TV/a.jpg")
                    member.size = 1
                    handle.addfile(member, io.BytesIO(b"x"))
            with self.assertRaisesRegex(ValueError, "duplicate file member"):
                extract_archive(duplicate, root / "invalid")
            self.assertFalse((root / "invalid").exists())

    def test_full_cached_download_checks_every_domain_without_network(self):
        from cue_routing.io import file_hash

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources = {}
            for domain in DOMAINS:
                path = root / f"{domain}.tgz"
                path.write_bytes(domain.encode())
                sources[domain] = {"url": "https://example.invalid/file", "sha256": file_hash(path)}
            atomic_json(root / "sources.json", {"original_domain_archives": sources})
            with patch("requests.get") as network:
                archives, _ = download_officehome(root, root / "sources.json")
                network.assert_not_called()
            self.assertEqual(len(archives), 4)

    def test_four_archive_import_and_existing_destination_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archives = []
            for domain in DOMAINS:
                path = root / f"{domain}.tgz"
                with tarfile.open(path, "w:gz") as handle:
                    member = tarfile.TarInfo(f"{domain}/TV/a.jpg")
                    member.size = 1
                    handle.addfile(member, io.BytesIO(b"x"))
                archives.append(path)
            extract_archive(archives, root / "images")
            self.assertEqual(len(discover_images(root / "images")), 4)
            with self.assertRaises(FileExistsError):
                extract_archive(archives, root / "images")

    def test_corrupt_cached_archive_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "Art.tgz").write_bytes(b"corrupt cache")
            sources = {
                domain: {"url": "https://example.invalid/file", "sha256": "0" * 64}
                for domain in DOMAINS
            }
            atomic_json(root / "sources.json", {"original_domain_archives": sources})
            with patch("requests.get") as network:
                with self.assertRaisesRegex(ValueError, "cached Office-Home archive hash"):
                    download_officehome(root, root / "sources.json")
                network.assert_not_called()
            self.assertEqual((root / "Art.tgz").read_bytes(), b"corrupt cache")
