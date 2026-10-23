"""Check Conda installation isolation, requirement selection, and runtime verification."""

import importlib.util
import json
import subprocess
import tempfile
import tomllib
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("environment_bootstrap", ROOT / "bootstrap.py")
BOOTSTRAP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BOOTSTRAP)


def runtime_report(prefix, device="cuda", development=False):
    """Represent a runtime response without creating or installing an environment."""
    dependencies = BOOTSTRAP.dependency_pins(ROOT, development)
    tag = "+cu128" if device == "cuda" else "+cpu"
    dependencies.update({"torch": "2.8.0" + tag, "torchvision": "0.23.0" + tag})
    return {
        "prefix": str(prefix),
        "python": "3.11.17",
        "torch": dependencies["torch"],
        "torchvision": dependencies["torchvision"],
        "cuda_runtime": "12.8" if device == "cuda" else None,
        "cuda_available": device == "cuda",
        "arch_list": ["sm_90", "sm_120"] if device == "cuda" else [],
        "device_name": "NVIDIA GeForce RTX 5090" if device == "cuda" else None,
        "device_capability": [12, 0] if device == "cuda" else None,
        "kernel_checked": device == "cuda",
        "kernel_test_passed": device == "cuda",
        "dependencies": dependencies,
    }


class EnvironmentSetupTests(unittest.TestCase):
    """Exercise installer subprocess boundaries without installing any packages."""

    def exercise_setup(self, device="cuda", development=False, verify_only=False):
        """Run setup against a temporary existing prefix and capture every external command."""
        with tempfile.TemporaryDirectory() as temporary:
            prefix = Path(temporary).resolve() / "conda-env"
            (prefix / "conda-meta").mkdir(parents=True)
            (prefix / "conda-meta" / "history").touch()
            report = runtime_report(prefix, device, development)
            calls = []

            def run(command, **kwargs):
                """Return simulated Conda and interpreter output and record invocation isolation."""
                calls.append((command, kwargs))
                if command[1:] == ["info", "--base"]:
                    output = str(Path(temporary) / "base")
                elif BOOTSTRAP.RUNTIME_PROBE in command:
                    output = json.dumps(report)
                else:
                    output = ""
                return subprocess.CompletedProcess(command, 0, stdout=output)

            with (
                patch.object(BOOTSTRAP.platform, "system", return_value="Linux"),
                patch.object(BOOTSTRAP.subprocess, "run", side_effect=run),
                patch.object(BOOTSTRAP, "write_verification") as write,
                patch.dict(
                    BOOTSTRAP.os.environ, {"PYTHONPATH": "outside", "PYTHONHOME": "outside"}
                ),
            ):
                actual = BOOTSTRAP.install_environment(
                    ROOT, prefix, device, "/conda", development, verify_only
                )
            write.assert_called_once()
            self.assertEqual(actual["verification_only"], verify_only)
            self.assertEqual(actual["requirements_file"], f"requirements-{device}.txt")
            for command, kwargs in calls:
                self.assertTrue(kwargs["check"])
                self.assertNotIn("PYTHONPATH", kwargs["env"])
                self.assertNotIn("PYTHONHOME", kwargs["env"])
                self.assertEqual(kwargs["env"]["PYTHONNOUSERSITE"], "1")
                self.assertIn(command[0], ("/conda", str(prefix / "bin/python")))
            return [command for command, _ in calls]

    def test_cuda_requirements_and_dev_are_installed_in_one_resolution(self):
        calls = self.exercise_setup(development=True)
        installations = [command for command in calls if command[1:4] == ["-m", "pip", "install"]]
        self.assertEqual(len(installations), 2)
        self.assertEqual(
            installations[0][4:],
            ["-r", str(ROOT / "requirements-cuda.txt"), "-r", str(ROOT / "requirements-dev.txt")],
        )
        self.assertEqual(installations[1][4:], ["--no-deps", "-e", str(ROOT)])

    def test_cpu_requirements_without_optional_dev(self):
        calls = self.exercise_setup(device="cpu")
        installations = [command for command in calls if command[1:4] == ["-m", "pip", "install"]]
        self.assertEqual(installations[0][4:], ["-r", str(ROOT / "requirements-cpu.txt")])
        self.assertFalse(any(str(ROOT / "requirements-dev.txt") in command for command in calls))

    def test_verify_only_does_not_install_or_create(self):
        calls = self.exercise_setup(verify_only=True)
        self.assertFalse(any("install" in command or "create" in command for command in calls))
        self.assertTrue(any(command[1:] == ["-m", "pip", "check"] for command in calls))

    def test_verify_only_missing_prefix_fails_before_any_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            prefix = Path(temporary) / "missing"
            with (
                patch.object(BOOTSTRAP.platform, "system", return_value="Linux"),
                patch.object(
                    BOOTSTRAP.subprocess,
                    "run",
                    return_value=subprocess.CompletedProcess([], 0, stdout="/conda-base"),
                ) as run,
            ):
                with self.assertRaisesRegex(ValueError, "existing Conda"):
                    BOOTSTRAP.install_environment(ROOT, prefix, "cpu", "/conda", verify_only=True)
            self.assertEqual(run.call_count, 1)
            self.assertFalse(prefix.exists())

    def test_conda_base_is_rejected_before_installation(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary) / "base"
            (base / "conda-meta").mkdir(parents=True)
            (base / "conda-meta" / "history").touch()
            with (
                patch.object(BOOTSTRAP.platform, "system", return_value="Linux"),
                patch.object(
                    BOOTSTRAP.subprocess,
                    "run",
                    return_value=subprocess.CompletedProcess([], 0, stdout=str(base)),
                ) as run,
            ):
                with self.assertRaisesRegex(ValueError, "base environment"):
                    BOOTSTRAP.install_environment(ROOT, base, "cpu", "/conda")
            self.assertEqual(run.call_count, 1)

    def test_project_metadata_uses_requirements_as_dependency_source(self):
        metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
        self.assertNotIn("dependencies", metadata["project"])
        self.assertIn("dependencies", metadata["project"]["dynamic"])
        self.assertEqual(
            metadata["tool"]["setuptools"]["dynamic"]["dependencies"]["file"],
            ["requirements.txt"],
        )


class RuntimeVerificationTests(unittest.TestCase):
    """Reject mismatched wheels and dependencies even if pip reports no dependency conflict."""

    def test_rejects_cpu_wheel_when_cuda_is_requested(self):
        prefix = ROOT / ".conda"
        report = runtime_report(prefix)
        report["dependencies"]["torch"] = "2.8.0+cpu"
        with self.assertRaisesRegex(RuntimeError, "Expected torch"):
            BOOTSTRAP.validate_runtime(report, prefix, "cuda", "Linux", ROOT)

    def test_rejects_incorrect_runtime_and_imported_version(self):
        prefix = ROOT / ".conda"
        for field, value in (("cuda_runtime", "12.6"), ("torchvision", "0.22.0+cu128")):
            with self.subTest(field=field):
                report = runtime_report(prefix)
                report[field] = value
                with self.assertRaises(RuntimeError):
                    BOOTSTRAP.validate_runtime(report, prefix, "cuda", "Linux", ROOT)

    def test_rejects_incorrect_python_prefix_and_common_dependency(self):
        prefix = ROOT / ".conda"
        for field, value in (("python", "3.12.0"), ("prefix", "/unrelated")):
            with self.subTest(field=field):
                report = runtime_report(prefix)
                report[field] = value
                with self.assertRaises(RuntimeError):
                    BOOTSTRAP.validate_runtime(report, prefix, "cuda", "Linux", ROOT)
        report = runtime_report(prefix)
        report["dependencies"]["numpy"] = "1.26.4"
        with self.assertRaisesRegex(RuntimeError, "Expected numpy"):
            BOOTSTRAP.validate_runtime(report, prefix, "cuda", "Linux", ROOT)

    def test_windows_cpu_requires_build_tags_and_macos_does_not(self):
        windows, _ = BOOTSTRAP.wheel_specification("cpu", "Windows")
        macos, _ = BOOTSTRAP.wheel_specification("cpu", "Darwin")
        self.assertEqual(windows, ["torch==2.8.0+cpu", "torchvision==0.23.0+cpu"])
        self.assertEqual(macos, ["torch==2.8.0", "torchvision==0.23.0"])
        with self.assertRaises(ValueError):
            BOOTSTRAP.wheel_specification("cuda", "Windows")

    def test_cuda_provisioning_without_a_gpu_does_not_claim_kernel_success(self):
        prefix = ROOT / ".conda"
        report = runtime_report(prefix)
        report.update(
            cuda_available=False,
            arch_list=[],
            device_name=None,
            device_capability=None,
            kernel_checked=False,
            kernel_test_passed=False,
        )
        BOOTSTRAP.validate_runtime(report, prefix, "cuda", "Linux", ROOT)
        report["kernel_test_passed"] = True
        with self.assertRaisesRegex(RuntimeError, "cannot be verified"):
            BOOTSTRAP.validate_runtime(report, prefix, "cuda", "Linux", ROOT)

    def test_available_gpu_requires_successful_kernel_check(self):
        prefix = ROOT / ".conda"
        for field in ("kernel_checked", "kernel_test_passed"):
            with self.subTest(field=field):
                report = runtime_report(prefix)
                report[field] = False
                with self.assertRaisesRegex(RuntimeError, "CUDA kernel verification failed"):
                    BOOTSTRAP.validate_runtime(report, prefix, "cuda", "Linux", ROOT)


class CudaProbeTests(unittest.TestCase):
    """Simulate CUDA routing with CPU tensors to test kernels, gradients, and failure handling."""

    def simulated_torch(self):
        """Execute actual CPU tensor/autograd operations while recording requested CUDA calls."""
        convolution = torch.nn.Conv2d(3, 4, 3, padding=1)
        wrapped = Mock(wraps=convolution)
        wrapped.to.return_value = wrapped
        cuda = Mock()
        cuda.is_available.return_value = True
        cuda.get_arch_list.return_value = ["sm_90", "sm_120"]
        cuda.current_device.return_value = 0
        cuda.get_device_name.return_value = "NVIDIA GeForce RTX 5090"
        cuda.get_device_capability.return_value = (12, 0)
        simulated = types.SimpleNamespace(
            cuda=cuda,
            ones=Mock(
                side_effect=lambda shape, device, requires_grad: torch.ones(
                    shape, device="cpu", requires_grad=requires_grad
                )
            ),
            nn=types.SimpleNamespace(Conv2d=Mock(return_value=wrapped)),
            isfinite=torch.isfinite,
        )
        return simulated, wrapped

    def test_cpu_environment_never_calls_cuda(self):
        simulated = Mock()
        result = BOOTSTRAP.probe_cuda_runtime(simulated, "cpu")
        self.assertFalse(result["cuda_available"])
        self.assertFalse(result["kernel_checked"])
        self.assertFalse(result["kernel_test_passed"])
        self.assertEqual(simulated.mock_calls, [])

    def test_no_gpu_skips_kernels_explicitly(self):
        simulated, _ = self.simulated_torch()
        simulated.cuda.is_available.return_value = False
        simulated.cuda.get_arch_list.return_value = []
        result = BOOTSTRAP.probe_cuda_runtime(simulated, "cuda")
        self.assertFalse(result["kernel_checked"])
        self.assertFalse(result["kernel_test_passed"])
        self.assertIn("unverified", result["kernel_skip_reason"])
        simulated.ones.assert_not_called()
        simulated.cuda.current_device.assert_not_called()
        simulated.cuda.synchronize.assert_not_called()

    def test_available_gpu_executes_convolution_backward_and_synchronizes(self):
        simulated, convolution = self.simulated_torch()
        result = BOOTSTRAP.probe_cuda_runtime(simulated, "cuda")
        self.assertTrue(result["kernel_checked"])
        self.assertTrue(result["kernel_test_passed"])
        self.assertEqual(result["device_capability"], [12, 0])
        self.assertEqual(result["arch_list"], ["sm_90", "sm_120"])
        simulated.ones.assert_called_once_with((2, 3, 8, 8), device="cuda:0", requires_grad=True)
        convolution.to.assert_called_once_with("cuda:0")
        self.assertTrue(all(parameter.grad is not None for parameter in convolution.parameters()))
        simulated.cuda.synchronize.assert_called_once_with(0)

    def test_unsupported_kernel_error_includes_device_and_architectures(self):
        simulated, _ = self.simulated_torch()
        simulated.cuda.get_arch_list.return_value = ["sm_90"]
        simulated.ones.side_effect = RuntimeError("CUDA error: no kernel image is available")
        result = BOOTSTRAP.probe_cuda_runtime(simulated, "cuda")
        self.assertTrue(result["kernel_checked"])
        self.assertFalse(result["kernel_test_passed"])
        self.assertIn("no kernel image", result["kernel_test_error"])
        prefix = ROOT / ".conda"
        report = {**runtime_report(prefix), **result}
        with self.assertRaisesRegex(RuntimeError, "RTX 5090.*sm_90.*no kernel image"):
            BOOTSTRAP.validate_runtime(report, prefix, "cuda", "Linux", ROOT)

    def test_asynchronous_cuda_failure_is_not_reported_as_success(self):
        simulated, _ = self.simulated_torch()
        simulated.cuda.synchronize.side_effect = RuntimeError("asynchronous CUDA launch failed")
        result = BOOTSTRAP.probe_cuda_runtime(simulated, "cuda")
        self.assertTrue(result["kernel_checked"])
        self.assertFalse(result["kernel_test_passed"])
        self.assertIn("asynchronous", result["kernel_test_error"])


if __name__ == "__main__":
    unittest.main()
