"""Check a trained model against this project before using it.

Models trained with training-synth come as a package: weights plus `model_card.json` (classes in order,
image size, scores, dataset fingerprint). A model whose classes differ from the catalogue would report
wrong part names, so the class list is checked when the models are loaded.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from screwsynth.common.io import read_json, read_yaml

log = logging.getLogger("screwsynth.measure")


def find_card(weights: str | Path) -> Path | None:
    """model_card.json next to the weights (package/best.pt), in the run's package folder, or in the
    package above an export (package/exports/ncnn_fp32/best_ncnn_model)."""
    w = Path(weights)
    for c in (
        w.parent / "model_card.json",
        w.parent.parent / "package" / "model_card.json",
        w.parent.parent.parent / "model_card.json",
    ):
        if c.is_file():
            return c
    return None


def check_classes(
    weights: str | Path, names: Mapping[int, str], classes_yaml: str | Path | None = None
) -> tuple[list[str], list[str]]:
    """(errors, warnings) about the model's classes.

    Errors: the model card and the weights disagree (a mixed-up package).
    Warnings: classes the catalogue does not know (their parts cannot be sorted), or catalogue classes
    the model never learned (they will never be found).
    """
    errors: list[str] = []
    warnings: list[str] = []
    model_names = [names[i] for i in sorted(names)]
    card_path = find_card(weights)
    if card_path is not None:
        card: dict[str, Any] = read_json(card_path)
        if card.get("classes") and list(card["classes"]) != model_names:
            errors.append(f"{card_path} lists different classes than the weights {weights}")
    if classes_yaml and Path(classes_yaml).is_file():
        expected = set((read_yaml(Path(classes_yaml)) or {}).get("names", {}).values())
        unknown = sorted(set(model_names) - expected)
        missing = sorted(expected - set(model_names))
        if unknown:
            warnings.append(
                f"the model knows classes the catalogue does not: {', '.join(unknown[:10])}"
            )
        if missing:
            warnings.append(
                f"{len(missing)} catalogue class(es) were not trained and will never be found: "
                + ", ".join(missing[:10])
            )
    return errors, warnings
