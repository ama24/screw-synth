"""Sim-to-real report: realism statistics (tool 06), real vs synthetic evaluation (tool 08), the
top error sources by class / size / finish / session, and suggested config changes."""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2

from screwsynth.common.io import read_json, write_json, write_text


def realism(run_dir: Path, real_dataset: Path, out_html: Path, n: int = 20) -> dict[str, Any]:
    """Tool 06 compare: synthetic processed scenes vs the real images of the dataset."""
    from screwsynth.postprocess import compare

    items = []
    for d in sorted((run_dir / "processed").iterdir())[:n]:
        meta = read_json(d / "meta.json")
        img = cv2.cvtColor(cv2.imread(str(d / meta["image"])), cv2.COLOR_BGR2RGB)
        items.append((img, cv2.imread(str(d / "instance.png"), cv2.IMREAD_UNCHANGED)))
    real = []
    for r in list(csv.DictReader((real_dataset / "sources.csv").open(encoding="utf-8")))[:n]:
        img = cv2.imread(r["full_res_image"])
        if img is not None:
            real.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not items or not real:
        return {
            "suggestions": ["no images to compare (render a run and build the real dataset first)"]
        }
    return compare.report(items, real, out_html)


def error_sources(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    """Worst classes (mAP), worst sizes / families / finishes / sessions (length exact rate)."""
    det = evaluation.get("detection", {})
    per_class = det.get("per_class_map50_95_mask", {})
    worst_classes = sorted(per_class.items(), key=lambda kv: kv[1])[:8]
    errors = evaluation.get("length", {}).get("errors", [])
    finish_of: dict[tuple[str, int], str] = {}
    session_of: dict[str, str] = {}
    for e in errors:
        sd = e.get("scene_dir")
        if sd and sd not in session_of:
            meta = read_json(Path(sd) / "meta.json")
            session_of[sd] = meta.get("session", Path(sd).parent.parent.name)
            for o in meta.get("objects", []):
                finish_of[(sd, o["instance_id"])] = o.get("finish", "unknown")

    def rate(key) -> list[tuple[str, float, int]]:
        groups: dict[str, list] = defaultdict(list)
        for e in errors:
            if e.get("pred") is None:
                continue
            groups[str(key(e))].append(e["pred_snapped"] == e["true_mm"])
        return sorted(((g, sum(v) / len(v), len(v)) for g, v in groups.items()), key=lambda x: x[1])

    return {
        "worst_classes_map": worst_classes,
        "length_by_size": rate(lambda e: e["size"]),
        "length_by_family": rate(lambda e: e["family"]),
        "length_by_finish": rate(
            lambda e: finish_of.get((e["scene_dir"], e["instance_id"]), "unknown")
        ),
        "length_by_session": rate(lambda e: session_of.get(e["scene_dir"], "?")),
        "missed": sum(1 for e in errors if not e.get("detected")),
        "wrong_class": sum(
            1 for e in errors if e.get("detected") and e.get("class") != e.get("true_class")
        ),
    }


def suggestions(
    src: Mapping[str, Any],
    realism_summary: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    synthetic: Mapping[str, Any] | None,
) -> list[str]:
    out = list(realism_summary.get("suggestions", []))
    real_map = evaluation.get("detection", {}).get("map50_95_mask")
    syn_map = (synthetic or {}).get("detection", {}).get("map50_95_mask")
    if real_map is not None and syn_map is not None and syn_map - real_map > 0.15:
        out.append(
            f"large sim-to-real gap in mAP ({syn_map:.2f} synthetic vs {real_map:.2f} real): fix the realism "
            "items above first; then consider fine-tuning on real data (training-synth: start from your synthetic run, train on the real set)"
        )
    for name, ap in src.get("worst_classes_map", [])[:3]:
        if ap < 0.5:
            out.append(
                f"{name}: real mAP {ap:.2f} — render more of it (scenegen class balance) or check its finishes"
            )
    for fin, rate, n in src.get("length_by_finish", []):
        if n >= 5 and rate < 0.8:
            out.append(
                f"lengths of {fin} parts are often wrong ({rate:.0%} exact, n={n}): check that finish's material in scenegen.materials (too bright/dark?)"
            )
    if src.get("missed", 0):
        out.append(
            f"{src['missed']} labelled screws were not detected: lower measure.infer.conf or add similar layouts"
        )
    if src.get("wrong_class", 0):
        out.append(
            f"{src['wrong_class']} screws got the wrong class: see the size confusion in the evaluation report"
        )
    return out or ["no obvious problems found"]


def markdown(
    evaluation: Mapping[str, Any],
    synthetic: Mapping[str, Any] | None,
    src: Mapping[str, Any],
    sugg: Sequence[str],
    html: Path | None,
) -> str:
    from screwsynth.measure.evaluate import evaluation_markdown

    lines = [
        evaluation_markdown(evaluation.get("name", "real"), evaluation, synthetic),
        "## Top error sources",
        "",
    ]
    if src["worst_classes_map"]:
        lines.append(
            "Lowest real mAP: " + ", ".join(f"{c} {v:.2f}" for c, v in src["worst_classes_map"])
        )
    for key, title in (
        ("length_by_size", "size"),
        ("length_by_family", "family"),
        ("length_by_finish", "finish"),
        ("length_by_session", "session"),
    ):
        if src[key]:
            lines.append(
                f"Length exact rate by {title}: "
                + ", ".join(f"{g} {r:.0%} ({n})" for g, r, n in src[key])
            )
    lines += ["", "## Suggested changes", ""] + [f"- {s}" for s in sugg]
    if html is not None:
        lines += ["", f"Image statistics (real vs rendered): {html}"]
    return "\n".join(lines) + "\n"


def write(
    out_dir: Path,
    evaluation: dict,
    synthetic: dict | None,
    realism_summary: dict,
    html: Path | None,
) -> Path:
    src = error_sources(evaluation)
    sugg = suggestions(src, realism_summary, evaluation, synthetic)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        out_dir / "sim_to_real.json",
        {
            "evaluation": evaluation,
            "synthetic": synthetic,
            "realism": realism_summary,
            "error_sources": src,
            "suggestions": sugg,
        },
    )
    path = out_dir / "sim_to_real.md"
    write_text(path, markdown(evaluation, synthetic, src, sugg, html))
    return path
