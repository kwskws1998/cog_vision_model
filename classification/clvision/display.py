"""Trainer-style step displays and append-only telemetry independent of experiment recipes."""

import json
import math
import sys
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm


@dataclass(frozen=True)
class DisplayOptions:
    """Control presentation without changing scientific configurations or checkpoint identities."""

    progress: str = "bar"
    logging_steps: int = 200
    progress_interval: float = 1.0

    def validate(self):
        """Reject unusable refresh intervals and logging frequencies before starting a job."""
        if self.progress not in ("bar", "plain", "off"):
            raise ValueError("progress must be bar, plain, or off")
        if type(self.logging_steps) is not int or self.logging_steps < 1:
            raise ValueError("logging_steps must be a positive integer")
        if (
            type(self.progress_interval) not in (int, float)
            or not math.isfinite(self.progress_interval)
            or self.progress_interval <= 0
        ):
            raise ValueError("progress_interval must be finite and positive")
        return self


OPTIONS = ContextVar("classification_display_options", default=DisplayOptions())


@contextmanager
def display_options(options):
    """Keep display overrides scoped to one invocation, including exceptional exits."""
    token = OPTIONS.set(options.validate())
    try:
        yield
    finally:
        OPTIONS.reset(token)


def add_display_arguments(parser):
    """Expose the same logging options in every classification training entrypoint."""
    parser.add_argument("--progress", choices=("bar", "plain", "off"), default="bar")
    parser.add_argument("--logging-steps", type=int, default=200)
    parser.add_argument("--progress-interval", type=float, default=1.0)


def options_from_arguments(arguments):
    """Validate presentation inputs separately from source-bound training settings."""
    return DisplayOptions(
        **{
            key: getattr(arguments, key, getattr(DisplayOptions(), key))
            for key in ("progress", "logging_steps", "progress_interval")
        }
    ).validate()


def write(message):
    """Print one permanent line without overwriting active training or evaluation bars."""
    tqdm.write(str(message), file=sys.stdout)
    sys.stdout.flush()


class BatchProgress:
    """Display accepted work with resumable counters and a separately labeled step ETA."""

    def __init__(self, description, total, initial=0, enabled=True, position=0, leave=True):
        self.description, self.total, self.completed = description, total, initial
        self.initial, self.enabled, self.position, self.leave = initial, enabled, position, leave
        self.options = OPTIONS.get()
        self.bar, self.postfix = None, {}
        self.started, self.last_emit = time.monotonic(), -math.inf

    def __enter__(self):
        """Render the saved counter immediately and start timing only new work."""
        if not 0 <= self.initial <= self.total:
            raise ValueError("Progress counters must be within the declared step budget")
        if self.enabled and self.options.progress == "bar":
            self.bar = tqdm(
                total=self.total,
                initial=self.initial,
                desc=self.description,
                file=sys.stdout,
                unit="step",
                mininterval=self.options.progress_interval,
                miniters=1,
                dynamic_ncols=True,
                ascii=True,
                position=self.position,
                leave=self.leave,
            )
        self.emit_plain(force=True)
        return self

    def update(self, completed, **postfix):
        """Count only accepted batches and throttle rendering without any device synchronization."""
        if not self.completed <= completed <= self.total:
            raise ValueError("Progress must be monotonic and cannot exceed its declared total")
        delta, self.completed = completed - self.completed, completed
        self.postfix.update(postfix)
        if self.bar is not None:
            self.bar.set_postfix(self.postfix, refresh=False)
            self.bar.update(delta)
        self.emit_plain()

    def stage(self, name):
        """Keep validation/checkpoint activity visible while optimizer-step progress is paused."""
        self.postfix["stage"] = name
        if self.bar is not None:
            self.bar.set_postfix(self.postfix)
        self.emit_plain(force=True)

    def emit_plain(self, force=False):
        """Write pipe-friendly snapshots with no carriage returns or cursor-control sequences."""
        now = time.monotonic()
        if not self.enabled or self.options.progress != "plain":
            return
        if not force and now - self.last_emit < self.options.progress_interval:
            return
        self.last_emit = now
        elapsed, accepted = now - self.started, self.completed - self.initial
        rate = accepted / elapsed if elapsed > 0 and accepted else None
        remaining = (self.total - self.completed) / rate if rate else None
        percent = 100 * self.completed / self.total if self.total else 0
        metadata = " ".join(f"{key}={value}" for key, value in self.postfix.items())
        write(
            f"[{self.description}] {percent:.1f}% {self.completed}/{self.total} "
            f"elapsed={tqdm.format_interval(elapsed)} "
            f"step_remaining~={tqdm.format_interval(remaining) if remaining is not None else '?'} "
            f"steps/s={rate:.2f} {metadata}"
            if rate
            else f"[{self.description}] {percent:.1f}% {self.completed}/{self.total} "
            f"elapsed={tqdm.format_interval(elapsed)} step_remaining~=? steps/s=? {metadata}"
        )

    def __exit__(self, exception_type, exception, traceback):
        """Close failed or interrupted bars at their actual count rather than displaying 100%."""
        status = (
            "failed" if exception_type else "complete" if self.completed == self.total else "paused"
        )
        if exception_type and issubclass(exception_type, KeyboardInterrupt):
            status = "interrupted"
        self.postfix["status"] = status
        if self.bar is not None:
            self.bar.set_postfix(self.postfix, refresh=False)
            self.bar.close()
        self.emit_plain(force=True)


class TrainingLog:
    """Log weighted step windows and epochs while retaining explicit resume-session provenance."""

    def __init__(self, config, run_dir, total_steps, completed_steps):
        self.config, self.run_dir = config, Path(run_dir)
        self.total_steps, self.completed = total_steps, completed_steps
        self.session = uuid.uuid4().hex
        self.window, self.examples, self.window_steps = {}, 0, 0
        self.last_epoch, self.last_lr = 0.0, config.learning_rate
        self.meter = BatchProgress(
            f"Train {config.dataset}/{config.method}/s{config.seed}", total_steps, completed_steps
        )

    def record(self, event, **values):
        """Append telemetry, keeping attempted work from older sessions distinct after rollback."""
        row = {
            "event": event,
            "session_id": self.session,
            "utc": datetime.now(timezone.utc).isoformat(),
            "step": self.completed,
            "total_steps": self.total_steps,
            **values,
        }
        with (self.run_dir / "training_log.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")

    def __enter__(self):
        """Start one process session at its checkpoint's accepted optimizer-step count."""
        self.record("session_start", resumed_from_step=self.completed, config=self.config.to_dict())
        self.meter.__enter__()
        return self

    def step(self, step, epoch, batch, batches, examples, metrics, learning_rate):
        """Average logged loss by examples and refresh a run-wide optimizer-step progress bar."""
        self.completed = step
        self.examples += examples
        self.window_steps += 1
        for key in ("loss", "train_accuracy"):
            self.window[key] = self.window.get(key, 0.0) + metrics[key] * examples
        fraction = epoch - 1 + batch / batches
        self.last_epoch, self.last_lr = fraction, learning_rate
        self.meter.update(
            step,
            epoch=f"{fraction:.2f}/{self.config.epochs}",
            loss=f"{self.window['loss'] / self.examples:.4f}",
            lr=f"{learning_rate:.3g}",
            stage="train",
        )
        if step % OPTIONS.get().logging_steps == 0:
            self.flush(fraction, learning_rate)

    def flush(self, epoch, learning_rate):
        """Record each remaining window once, including a short final or interrupted window."""
        if not self.examples:
            return
        values = {key: value / self.examples for key, value in self.window.items()}
        self.record(
            "train",
            epoch=epoch,
            learning_rate=learning_rate,
            window_steps=self.window_steps,
            examples=self.examples,
            **values,
        )
        write(
            f"[train] step={self.completed}/{self.total_steps} epoch={epoch:.3f} "
            f"loss={values['loss']:.4f} accuracy={values['train_accuracy']:.4f} lr={learning_rate:.6g}"
        )
        self.window, self.examples, self.window_steps = {}, 0, 0

    def __exit__(self, exception_type, exception, traceback):
        """Persist failure/interruption without counting rejected updates or claiming completion."""
        self.flush(self.last_epoch, self.last_lr)
        self.meter.__exit__(exception_type, exception, traceback)
        status = (
            "failed"
            if exception_type
            else "complete"
            if self.completed == self.total_steps
            else "paused"
        )
        if exception_type and issubclass(exception_type, KeyboardInterrupt):
            status = "interrupted"
        self.record("session_end", status=status, error=str(exception) if exception else None)
