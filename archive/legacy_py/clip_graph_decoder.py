"""Leak-safe, clip-level structural decoder for CUHK-X QA rows.

The decoder uses only training QA answers during ``fit`` and groups questions
from the same clip during prediction.  It is intended as a fast baseline and
as a parent for sensor-specialist overrides; it does not inspect media.
"""

from __future__ import annotations

import itertools
import math
import re
from collections import Counter, defaultdict

import pandas as pd

LABELS = "ABCD"
MODALITIES = {"depth", "depth_color", "ir", "thermal", "imu", "skeleton", "radar"}


def norm(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def clip_key(path: object) -> str:
    parts = [p for p in str(path).replace("\\", "/").split("/") if p]
    if len(parts) >= 2:
        modality, stem = parts[-2].lower(), parts[-1].split(".")[0].lower()
        if modality in MODALITIES and stem in MODALITIES:
            parts = parts[:-2]
    return "/".join(parts)


def atoms(value: object) -> tuple[str, ...]:
    return tuple(norm(piece) for piece in str(value).split(",") if piece.strip())


class ClipGraphDecoder:
    def __init__(self, smoothing: float = 8.0, support_weight: float = 1.0):
        self.smoothing = smoothing
        self.support_weight = support_weight
        self.option_ok = Counter()
        self.option_n = Counter()
        self.slice_ok = Counter()
        self.slice_n = Counter()
        self.set_memory = defaultdict(Counter)
        self.multi_sizes = defaultdict(Counter)
        self.before = Counter()

    @staticmethod
    def options(row: pd.Series) -> list[str]:
        return [str(row[label]) for label in LABELS]

    def fit(self, frame: pd.DataFrame) -> "ClipGraphDecoder":
        for _, row in frame.iterrows():
            answer = str(row["answer"]).strip().upper()
            source, category = row["source"], row["category"]
            values = self.options(row)
            option_set = tuple(sorted(map(norm, values)))
            slice_key = (source, category)

            if category != "sequence":
                for label, value in zip(LABELS, values):
                    key = (source, category, norm(value))
                    self.option_n[key] += 1
                    self.slice_n[slice_key] += 1
                    if label in answer:
                        self.option_ok[key] += 1
                        self.slice_ok[slice_key] += 1

            if category == "multi":
                self.multi_sizes[slice_key][len(answer)] += 1
            elif category == "sequence" and set(answer) == set(LABELS):
                ordered = tuple(norm(row[label]) for label in answer)
                for i, left in enumerate(ordered):
                    for right in ordered[i + 1:]:
                        self.before[(left, right)] += 1
            else:
                for label in answer:
                    if label in LABELS:
                        self.set_memory[(source, category, option_set)][norm(row[label])] += 1
        return self

    def _logit(self, row: pd.Series, value: str) -> float:
        slice_key = (row["source"], row["category"])
        base = self.slice_ok[slice_key] / max(1, self.slice_n[slice_key])
        key = (row["source"], row["category"], norm(value))
        p = (self.option_ok[key] + self.smoothing * base) / (self.option_n[key] + self.smoothing)
        p = min(max(p, 1e-6), 1 - 1e-6)
        return math.log(p / (1 - p))

    def _sequence(self, row: pd.Series) -> str:
        best = None
        for labels in itertools.permutations(LABELS):
            text = [norm(row[label]) for label in labels]
            score = 0.0
            for i, left in enumerate(text):
                for right in text[i + 1:]:
                    a, b = self.before[(left, right)], self.before[(right, left)]
                    score += math.log((a + 1.5) / (a + b + 3.0))
            candidate = (score, "".join(labels))
            if best is None or candidate > best:
                best = candidate
        return best[1]

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        grouped = defaultdict(list)
        for index, row in frame.iterrows():
            grouped[clip_key(row["path"])].append((index, row))

        output = {}
        for group in grouped.values():
            support = Counter()
            for _, row in group:
                if row["source"] == "HAU" and row["category"] in {"single", "multi", "combination", "sequence"}:
                    values = self.options(row)
                    if row["category"] == "combination":
                        for value in values:
                            support.update(atoms(value))
                    else:
                        support.update(map(norm, values))

            for _, row in group:
                category = row["category"]
                if category == "sequence":
                    output[row["qa_id"]] = self._sequence(row)
                    continue

                values = self.options(row)
                option_set = tuple(sorted(map(norm, values)))
                memory = self.set_memory[(row["source"], category, option_set)]
                scores = []
                for value in values:
                    own = Counter(atoms(value) if category == "combination" else [norm(value)])
                    external = sum(max(0, support[a] - own[a]) for a in own)
                    scores.append(self._logit(row, value) + math.log1p(memory[norm(value)]) + self.support_weight * external)

                if category == "multi":
                    size_counts = self.multi_sizes[(row["source"], category)]
                    size = max(size_counts, key=lambda n: (size_counts[n], -n)) if size_counts else 1
                    selected = sorted(range(4), key=lambda i: (-scores[i], i))[:size]
                    prediction = "".join(LABELS[i] for i in sorted(selected))
                else:
                    prediction = LABELS[max(range(4), key=lambda i: (scores[i], -i))]
                output[row["qa_id"]] = prediction

        return pd.DataFrame({"qa_id": frame["qa_id"], "prediction": frame["qa_id"].map(output)})
