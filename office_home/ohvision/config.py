"""Explicit Office-Home identities with the same validated optimizer and model recipe."""

from dataclasses import dataclass, replace

from cue_routing.config import METHODS
from cue_routing.config import ExperimentConfig as SharedConfig

DOMAINS = ("Art", "Clipart", "Product", "Real_World")
CLASSES = tuple(
    sorted(
        (
            "Alarm_Clock",
            "Backpack",
            "Batteries",
            "Bed",
            "Bike",
            "Bottle",
            "Bucket",
            "Calculator",
            "Calendar",
            "Candles",
            "Chair",
            "Clipboards",
            "Computer",
            "Couch",
            "Curtains",
            "Desk_Lamp",
            "Drill",
            "Eraser",
            "Exit_Sign",
            "Fan",
            "File_Cabinet",
            "Flipflops",
            "Flowers",
            "Folder",
            "Fork",
            "Glasses",
            "Hammer",
            "Helmet",
            "Kettle",
            "Keyboard",
            "Knives",
            "Lamp_Shade",
            "Laptop",
            "Marker",
            "Monitor",
            "Mop",
            "Mouse",
            "Mug",
            "Notebook",
            "Oven",
            "Pan",
            "Paper_Clip",
            "Pen",
            "Pencil",
            "Postit_Notes",
            "Printer",
            "Push_Pin",
            "Radio",
            "Refrigerator",
            "Ruler",
            "Scissors",
            "Screwdriver",
            "Shelf",
            "Sink",
            "Sneakers",
            "Soda",
            "Speaker",
            "Spoon",
            "Table",
            "Telephone",
            "ToothBrush",
            "Toys",
            "Trash_Can",
            "TV",
            "Webcam",
        )
    )
)
EXPECTED_COUNTS = {"Art": 2427, "Clipart": 4365, "Product": 4439, "Real_World": 4357}
SPLIT_POLICY = {
    "algorithm": "sha256-class-stratified-v1",
    "split_seed": 0,
    "validation_numerator": 1,
    "validation_denominator": 5,
    "validation_rounding": "floor_per_class_with_minimum_one",
    "target_partition": "complete_held_out_domain",
}


@dataclass(frozen=True)
class ExperimentConfig(SharedConfig):
    """Validate Office-Home targets while preserving every shared optimization constraint."""

    target: str = "Art"
    data_variant: str = "original_resolution"

    def validate(self):
        """Delegate common checks through a PACS-valid proxy without changing this recipe."""
        if self.target not in DOMAINS:
            raise ValueError("Unknown Office-Home target domain")
        if self.data_variant not in ("original_resolution", "resized_256_square"):
            raise ValueError("Unknown Office-Home image variant")
        SharedConfig.validate(replace(self, target="sketch"))
        return self


__all__ = ["CLASSES", "DOMAINS", "EXPECTED_COUNTS", "ExperimentConfig", "METHODS", "SPLIT_POLICY"]
