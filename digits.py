"""Trusted data and release evaluator for the bundled classifiers."""

from dataclasses import dataclass
from hashlib import sha256
import json
from operator import mul
from pathlib import Path
import random
import struct


DATA = Path(__file__).resolve().parent / "digits-data"
WIDTHS = (32, 16)
INT8 = (8, 8)
FP16 = (16, 16)
FALLBACKS = ((8, 16), (16, 8), FP16)
MIN_BYTES = sum(WIDTHS)
FORMATS = {"fp32": (32, 32), "fp16": FP16, "bf16": ("bf16", "bf16"), "int8": INT8}


@dataclass(frozen=True)
class Digit:
    source: str
    row: int
    pixels: bytes
    label: int

    @property
    def sample_id(self):
        return f"{self.source}:{self.row}"


@dataclass(frozen=True)
class Fixture:
    training: list[Digit]
    calibration: list[Digit]
    development: list[Digit]
    holdout: list[Digit]
    fingerprints: dict[str, str]


def validate_contract(request):
    budget = request.get("budget_bytes")
    loss = request.get("max_drop_bps")
    fixture = request.get("fixture")
    if fixture not in ("clean", "injected_overlap", "synthetic"):
        raise ValueError("Choose Optical Digits, injected overlap, or the synthetic precision fixture.")
    max_budget = 192 if fixture == "synthetic" else 96
    if type(budget) is not int or not 1 <= budget <= max_budget:
        raise ValueError(f"Budget must be an integer from 1 to {max_budget} bytes.")
    if type(loss) is not int or not 0 <= loss <= 1000:
        raise ValueError("Maximum accuracy drop must be 0 to 1000 basis points.")
    return budget, loss, fixture


def read_digits(raw, name):
    samples = []
    for number, line in enumerate(raw.splitlines(), 1):
        row = list(map(int, line.split(b",")))
        if len(row) != 65 or not 0 <= row[-1] <= 9 or any(
            not 0 <= value <= 16 for value in row[:-1]
        ):
            raise ValueError(f"Invalid digit at {name}:{number}")
        samples.append(Digit(name, number, bytes(row[:-1]), row[-1]))
    return samples


def load_fixture(kind):
    if kind == "synthetic":
        raw = (DATA / "synthetic.csv").read_bytes()
        partitions = {name: [] for name in (b"calibration", b"development", b"holdout")}
        for line in raw.splitlines():
            name, row = line.split(b",", 1)
            if name not in partitions:
                raise ValueError("Unknown synthetic partition")
            partitions[name].append(row)
        calibration, development, holdout = [
            read_digits(b"\n".join(rows), f"synthetic.csv/{name.decode()}")
            for name, rows in partitions.items()
        ]
        return Fixture([], calibration, development, holdout,
                       {"dataset_sha256": sha256(raw).hexdigest()})
    train = (DATA / "optdigits.tra").read_bytes()
    test = (DATA / "optdigits.tes").read_bytes()
    training_file = read_digits(train, "optdigits.tra")
    holdout = read_digits(test, "optdigits.tes")
    if len(training_file) != 3823 or len(holdout) != 1797:
        raise ValueError("Unexpected Optical Digits split size")
    random.Random(42).shuffle(training_file)
    calibration = training_file[640:960]
    if kind == "injected_overlap":
        calibration = [*calibration, holdout[0]]
    return Fixture(
        training_file[960:], calibration, training_file[:640], holdout,
        {"training_file_sha256": sha256(train).hexdigest(),
         "holdout_file_sha256": sha256(test).hexdigest()},
    )


def audit_split(data):
    partitions = {
        "training": data.training, "calibration": data.calibration,
        "development": data.development, "holdout": data.holdout,
    }
    seen = {}
    conflicts = []
    for partition, samples in partitions.items():
        for sample in samples:
            prior_partition, prior_sample = seen.setdefault(sample.pixels, (partition, sample))
            if prior_partition != partition:
                conflicts.append({
                    "first_split": prior_partition,
                    "first_sample": prior_sample.sample_id,
                    "second_split": partition,
                    "second_sample": sample.sample_id,
                    "pixels_sha256": sha256(sample.pixels).hexdigest(),
                })
    return {
        "status": "overlap" if conflicts else "clean",
        "method": "Exact pixel identity across partitions, independent of labels.",
        "counts": {name: len(samples) for name, samples in partitions.items()},
        "conflict_count": len(conflicts),
        "conflicts": conflicts[:10],
        **data.fingerprints,
    }


def linear(inputs, weights, biases):
    return [sum(map(mul, row, inputs)) + bias for row, bias in zip(weights, biases)]


def relu(values):
    return [max(0.0, value) for value in values]


def round_bf16(value):
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    rounded = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000
    return struct.unpack("<f", struct.pack("<I", rounded))[0]


def round_activations(values, bits, clip_limit):
    if bits in (16, 32):
        code = "e" if bits == 16 else "f"
        codec = struct.Struct(f"<{len(values)}{code}")
        return codec.unpack(codec.pack(*values))
    if bits == "bf16":
        return [round_bf16(value) for value in values]
    step = clip_limit / 127.0
    return [round(min(value, clip_limit) / step) * step for value in values]


def calibration_sweeps(weights, biases, samples):
    layers = ([], [])
    for digit in samples:
        first = relu(linear([pixel / 16.0 for pixel in digit.pixels], weights[0], biases[0]))
        second = relu(linear(first, weights[1], biases[1]))
        layers[0].extend(first)
        layers[1].extend(second)
    sweeps = []
    for values in layers:
        upper = max(values)
        sweeps.append((values, [upper * fraction / 100 for fraction in range(65, 101)]))
    return sweeps


def shift_horizontal(pixels, offset):
    return bytes(
        pixels[row * 8 + column - offset] if 0 <= column - offset < 8 else 0
        for row in range(8) for column in range(8)
    )


def make_slices(samples):
    return {
        "clean": samples,
        "shift_left": [
            Digit(item.source, item.row, shift_horizontal(item.pixels, -1), item.label)
            for item in samples
        ],
        "shift_right": [
            Digit(item.source, item.row, shift_horizontal(item.pixels, 1), item.label)
            for item in samples
        ],
    }


def predict(weights, biases, digit, bits, limits):
    values = [pixel / 16.0 for pixel in digit.pixels]
    for layer in range(len(WIDTHS)):
        values = relu(linear(values, weights[layer], biases[layer]))
        values = round_activations(values, bits[layer], limits[layer])
    logits = linear(values, weights[-1], biases[-1])
    return max(range(len(logits)), key=logits.__getitem__)


def encoded_bytes(bits):
    return sum(width * (16 if precision == "bf16" else precision) // 8
               for width, precision in zip(WIDTHS, bits))


def profile(weights, biases, slices, limits, bits):
    return {
        name: {
            "total": len(samples),
            "correct": sum(predict(weights, biases, digit, bits, limits) == digit.label
                           for digit in samples),
        }
        for name, samples in slices.items()
    }


def passes_loss(candidate, baseline, max_drop_bps):
    return all(
        (baseline[name]["correct"] - score["correct"]) * 10_000
        <= max_drop_bps * score["total"]
        for name, score in candidate.items()
    )


def synthetic_model():
    # Compare two sensor channels after adding a common-mode signal. The
    # projections share the same true decision margin at different magnitudes.
    first = [[float(column == unit % 2) + 24.0 * (column == 2)
              for column in range(64)] for unit in range(32)]
    second = [[0.5 if column in (unit, unit + 16) else 0.0
               for column in range(32)] for unit in range(16)]
    output = [[0.125 if column % 2 == channel else 0.0
               for column in range(16)] for channel in range(2)]
    return {"weights": [first, second, output],
            "biases": [[4.0 + (unit // 2) % 4 for unit in range(32)],
                       [0.0] * 16, [0.0] * 2]}


def evaluate_synthetic(data, budget, loss, source, run_sweeps):
    model = synthetic_model()
    weights, biases = model["weights"], model["biases"]
    rows, timing = run_sweeps(source, calibration_sweeps(weights, biases, data.calibration))
    limits = [min(sweep, key=lambda row: row["mse"])["threshold"] for sweep in rows]
    development = {"clean": data.development}
    dev_scores = {name: profile(weights, biases, development, limits, bits)
                  for name, bits in FORMATS.items()}
    sizes = {name: encoded_bytes(bits) for name, bits in FORMATS.items()}
    attempts = []
    selected = None
    for name in ("int8", "fp16", "bf16"):
        if sizes[name] > budget:
            outcome = "over_budget"
        elif not passes_loss(dev_scores[name], dev_scores["fp32"], loss):
            outcome = "accuracy_failed"
        else:
            outcome = "selected"
            selected = name
        attempts.append({"format": name, "bits": FORMATS[name], "encoded_bytes": sizes[name],
                         "development": dev_scores[name], "result": outcome})
        if selected:
            break

    # Freeze selection before looking at holdout. The comparison is diagnostic.
    holdout = {"clean": [], "low_margin": []}
    for sample in data.holdout:
        group = "low_margin" if abs(sample.pixels[0] - sample.pixels[1]) == 1 else "clean"
        holdout[group].append(sample)
    holdout_scores = {name: profile(weights, biases, holdout, limits, bits)
                      for name, bits in FORMATS.items()}
    passed = selected is not None and passes_loss(holdout_scores[selected], holdout_scores["fp32"], loss)
    result = {
        "decision": "approve" if passed else "block",
        "reason": "holdout_passed" if passed else (
            "holdout_accuracy_failed" if selected else "no_candidate_passed_development"),
        "model_sha256": sha256(json.dumps(model, sort_keys=True).encode()).hexdigest(),
        "thresholds": limits,
        "verified_threshold_count": sum(map(len, rows)),
        "reference_verified": True,
        "baseline_format": "fp32",
        "baseline_encoded_bytes": sizes["fp32"],
        "selected_format": selected,
        "selected_bits": FORMATS[selected] if selected else None,
        "attempts": attempts,
        "precision_comparison": {
            name: {"encoded_bytes": sizes[name], "development": dev_scores[name],
                   "holdout": holdout_scores[name]}
            for name in FORMATS
        },
        "timing_ms": timing,
        "source": source,
    }
    if selected:
        result["holdout"] = {"fp32": holdout_scores["fp32"], "candidate": holdout_scores[selected]}
    return result


def evaluate_release(data, budget, loss, source, run_sweeps, fixture="clean"):
    if fixture == "synthetic":
        return evaluate_synthetic(data, budget, loss, source, run_sweeps)
    raw_model = (DATA / "frozen-run.json").read_bytes()
    model = json.loads(raw_model)
    weights, biases = model["weights"], model["biases"]
    if model["summary"]["development_source_sha256"] != data.fingerprints["training_file_sha256"]:
        raise ValueError("Frozen model and training data fingerprints differ")
    sweeps = calibration_sweeps(weights, biases, data.calibration)
    rows, timing = run_sweeps(source, sweeps)
    limits = [min(sweep, key=lambda candidate: candidate["mse"])["threshold"] for sweep in rows]
    development = make_slices(data.development)
    baseline = profile(weights, biases, development, limits, FP16)
    attempts = []
    selected = None
    for bits in (INT8, *FALLBACKS):
        size = encoded_bytes(bits)
        if size > budget:
            attempts.append({"bits": bits, "encoded_bytes": size, "result": "over_budget"})
            continue
        metrics = baseline if bits == FP16 else profile(weights, biases, development, limits, bits)
        passed = passes_loss(metrics, baseline, loss)
        attempts.append({"bits": bits, "encoded_bytes": size, "development": metrics,
                         "result": "selected" if passed else "accuracy_failed"})
        if passed:
            selected = bits
            break

    result = {
        "decision": "block",
        "reason": "no_candidate_passed_development",
        "model_sha256": sha256(raw_model).hexdigest(),
        "thresholds": limits,
        "verified_threshold_count": sum(map(len, rows)),
        "reference_verified": True,
        "fp16_encoded_bytes": encoded_bytes(FP16),
        "development_fp16": baseline,
        "attempts": attempts,
        "selected_bits": selected,
        "timing_ms": timing,
        "source": source,
    }
    if selected is None:
        return result
    holdout = make_slices(data.holdout)
    baseline = profile(weights, biases, holdout, limits, FP16)
    candidate = baseline if selected == FP16 else profile(weights, biases, holdout, limits, selected)
    passed = passes_loss(candidate, baseline, loss)
    return {
        **result,
        "decision": "approve" if passed else "block",
        "reason": "holdout_passed" if passed else "holdout_accuracy_failed",
        "holdout": {"fp16": baseline, "candidate": candidate},
    }
