"""Stable names for the selectable small-image ResNet-18 implementations."""

ACTIVATIONS = ("postact", "preact")


def backbone_description(activation):
    """Describe the actual implementation without implying a reproduced benchmark score."""
    descriptions = {
        "postact": "torchvision post-activation small-image ResNet-18",
        "preact": "Zipf reference pre-activation small-image ResNet-18 (108e27a)",
    }
    if activation not in ACTIVATIONS:
        raise ValueError("activation must be postact or preact")
    return descriptions[activation]


def backbone_reference_note(activation):
    """Keep published-score comparisons accurate for the selected backbone."""
    backbone_description(activation)
    if activation == "preact":
        return (
            "The pre-activation block and stem follow the pinned Zipf author implementation; "
            "this alone does not reproduce its complete training or evaluation protocol."
        )
    return (
        "Our post-activation backbone differs from the pinned author pre-activation implementation."
    )
