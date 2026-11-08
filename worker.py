"""Compile and verify a restricted Python calibration kernel inside a worker."""

import ast
import hashlib
import json
import math
import os
from pathlib import Path
import random
import socket
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory
import time

try:
    import resource
except ImportError:  # pragma: no cover - Windows local preview
    resource = None

from digits import MIN_BYTES, audit_split, evaluate_release, load_fixture, validate_contract


EXAMPLE = """def score(value, threshold):
    step = threshold / 127.0
    clipped = clip(value, threshold)
    level = round_even(clipped / step)
    delta = value - level * step
    return delta * delta
"""

OPERATORS = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}
FUNCTIONS = {"clip": 2, "round_even": 1}


def expression(node, names):
    if isinstance(node, ast.Name) and node.id in names:
        return names[node.id]
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        if abs(node.value) <= 1_000_000:
            return repr(float(node.value))
    if isinstance(node, ast.UnaryOp) and type(node.op) in (ast.UAdd, ast.USub):
        sign = "-" if isinstance(node.op, ast.USub) else "+"
        return f"({sign}{expression(node.operand, names)})"
    if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
        left = expression(node.left, names)
        right = expression(node.right, names)
        return f"({left} {OPERATORS[type(node.op)]} {right})"
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        function = node.func.id
        if len(node.args) == FUNCTIONS.get(function) and not node.keywords:
            arguments = ", ".join(expression(argument, names) for argument in node.args)
            return f"{function}({arguments})"
    raise ValueError(f"Unsupported expression: {type(node).__name__}")


def translate(source):
    if not isinstance(source, str) or len(source) > 2_048:
        raise ValueError("Expected at most 2048 characters of Python source")
    module = ast.parse(source)
    if sum(1 for _ in ast.walk(module)) > 150:
        raise ValueError("Kernel exceeds the syntax budget")
    if len(module.body) != 1 or not isinstance(module.body[0], ast.FunctionDef):
        raise ValueError("Expected exactly one function: score(value, threshold)")
    function = module.body[0]
    args = function.args
    if (
        function.name != "score"
        or function.decorator_list or function.returns or getattr(function, "type_params", ())
        or args.posonlyargs or args.kwonlyargs or args.vararg or args.kwarg or args.defaults
        or [argument.arg for argument in args.args] != ["value", "threshold"]
        or any(argument.annotation for argument in args.args)
    ):
        raise ValueError("Expected signature: score(value, threshold)")
    if not isinstance(function.body[-1], ast.Return) or len(function.body) > 13:
        raise ValueError("Expected up to 12 assignments followed by a return")
    names = {"value": "value", "threshold": "threshold"}
    lines = []
    for statement in function.body[:-1]:
        if (
            not isinstance(statement, ast.Assign) or len(statement.targets) != 1
            or not isinstance(statement.targets[0], ast.Name)
            or statement.targets[0].id in names or statement.targets[0].id in FUNCTIONS
        ):
            raise ValueError("Assignments must bind fresh local variables")
        local = f"local_{len(lines)}"
        lines.append(f"    [[maybe_unused]] const double {local} = {expression(statement.value, names)};")
        names[statement.targets[0].id] = local
    lines.append(f"    return {expression(function.body[-1].value, names)};")
    return "\n".join(lines)


HARNESS = r"""
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

double clip(double value, double limit) {
    return std::min(std::max(value, -limit), limit);
}

double round_even(double value) { return std::nearbyint(value); }

double score(double value, double threshold) {
// KERNEL
}

int main() {
    std::size_t count{};
    if (!(std::cin >> count) || count == 0 || count > 32768) return 1;
    std::vector<double> samples(count);
    for (auto& value : samples) {
        if (!(std::cin >> value) || !std::isfinite(value)) return 1;
    }
    std::cout << std::setprecision(17);
    double threshold{};
    while (std::cin >> threshold) {
        const auto start = std::chrono::steady_clock::now();
        double error = 0;
        std::size_t clipped = 0;
        for (const auto value : samples) {
            const auto squared_error = score(value, threshold);
            if (!std::isfinite(squared_error) || squared_error < 0) return 2;
            error += squared_error;
            clipped += std::abs(value) > threshold;
        }
        if (!std::isfinite(error)) return 2;
        const auto elapsed = std::chrono::steady_clock::now() - start;
        const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed);
        std::cout << error / static_cast<double>(count) << ' ' << clipped << ' '
                  << ns.count() << '\n';
    }
}
"""


def reference(samples, threshold):
    step = threshold / 127.0
    total_error = 0.0
    clipped_count = 0
    for value in samples:
        clipped_value = min(max(value, -threshold), threshold)
        delta = value - round(clipped_value / step) * step
        total_error += delta * delta
        clipped_count += abs(value) > threshold
    return total_error / len(samples), clipped_count


def fp16_mse(samples):
    binary16 = struct.Struct("<e")
    total_error = 0.0
    for value in samples:
        reconstructed, = binary16.unpack(binary16.pack(value))
        delta = value - reconstructed
        total_error += delta * delta
    return total_error / len(samples)


def run_sweeps(source, sweeps):
    program = HARNESS.replace("// KERNEL", translate(source))
    with TemporaryDirectory() as directory:
        path = Path(directory)
        (path / "kernel.cpp").write_text(program, encoding="utf-8")
        executable = path / ("kernel.exe" if sys.platform == "win32" else "kernel")
        started = time.perf_counter()
        subprocess.run(
            ["g++", "-std=c++20", "-O3", "-ffp-contract=off", "-Wall", "-Wextra",
             "-Wconversion", "-Werror", "kernel.cpp", "-o", str(executable)],
            cwd=path, check=True, capture_output=True, timeout=15,
        )
        compile_ms = (time.perf_counter() - started) * 1_000
        results = [verified_sweep(executable, samples, thresholds) for samples, thresholds in sweeps]
    return [result[0] for result in results], {
        "compile": compile_ms,
        "native_sweep": sum(result[1] for result in results) / 1_000_000,
        "python_reference_sweep": sum(result[2] for result in results) / 1_000_000,
    }


def verified_sweep(executable, samples, thresholds):
    payload = "\n".join(str(value) for value in [len(samples), *samples, *thresholds])
    execution = subprocess.run(
        [str(executable)], input=payload, check=True, capture_output=True,
        text=True, timeout=4,
    )
    rows = []
    reference_ns = 0
    native_ns = 0
    for threshold, line in zip(thresholds, execution.stdout.splitlines(), strict=True):
        error, clipped, elapsed = line.split()
        started = time.perf_counter_ns()
        expected_error, expected_clipped = reference(samples, threshold)
        reference_ns += time.perf_counter_ns() - started
        if not math.isclose(float(error), expected_error, rel_tol=1e-12, abs_tol=1e-14):
            raise ValueError(f"Generated kernel disagrees with reference at threshold {threshold}")
        if int(clipped) != expected_clipped:
            raise ValueError(f"Clipping count disagrees with reference at threshold {threshold}")
        native_ns += int(elapsed)
        rows.append({
            "threshold": threshold,
            "mse": float(error),
            "saturation_pct": 100 * int(clipped) / len(samples),
        })
    return rows, native_ns, reference_ns


def calibrate(source, max_saturation):
    if type(max_saturation) not in (float, int) or not 0 <= max_saturation <= 100:
        raise ValueError("Saturation limit must be a percentage from 0 to 100")
    rng = random.Random(42)
    samples = [rng.gauss(0, 1) for _ in range(32_736)]
    samples.extend(rng.uniform(-8, 8) for _ in range(32))
    thresholds = [1.5 + index * 0.25 for index in range(19)]
    (rows,), timing_ms = run_sweeps(source, [(samples, thresholds)])
    eligible = [row for row in rows if row["saturation_pct"] <= max_saturation]
    return {
        "sample_count": len(samples),
        "dataset": "Deterministic synthetic activations, seed 42",
        "limit_pct": max_saturation,
        "recommendation": min(eligible, key=lambda row: row["mse"], default=None),
        "format_comparison": {
            "fp16": {"mse": fp16_mse(samples), "value_bytes": 2 * len(samples)},
            "int8": {"value_bytes": len(samples)},
        },
        "rows": rows,
        "reference_verified": True,
        "timing_ms": timing_ms,
        "source": source,
    }


def containment_network():
    """Prove that the worker cannot open an outbound TCP connection."""
    try:
        with socket.create_connection(("1.1.1.1", 80), timeout=2):
            return {"verdict": "unexpectedly_allowed"}
    except OSError as error:
        return {
            "verdict": "blocked",
            "error_type": type(error).__name__,
            "evidence": str(error)[:160],
        }


def containment_filesystem():
    """Prove that /tmp works while the read-only root rejects a host write."""
    writable_path = Path("/tmp/quantgate-containment.txt")
    blocked_path = Path("/quantgate-containment.txt")
    writable_path.write_text("probe", encoding="ascii")
    try:
        blocked_path.write_text("probe", encoding="ascii")
    except OSError as error:
        return {
            "verdict": "blocked",
            "tmpfs_write": writable_path.exists(),
            "error_type": type(error).__name__,
            "evidence": str(error)[:160],
        }
    return {"verdict": "unexpectedly_allowed", "tmpfs_write": writable_path.exists()}


def containment_disk():
    """Prove that the writable tmpfs has a finite quota."""
    path = Path("/tmp/quantgate-disk-probe.bin")
    written = 0
    try:
        with path.open("wb") as stream:
            chunk = b"q" * (1024 * 1024)
            for _ in range(48):
                stream.write(chunk)
                stream.flush()
                written += len(chunk)
    except OSError as error:
        return {
            "verdict": "blocked",
            "written_bytes": written,
            "error_type": type(error).__name__,
            "evidence": str(error)[:160],
        }
    finally:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    return {"verdict": "unexpectedly_allowed", "written_bytes": written}


def containment_processes():
    """Prove the process quota without exhausting the gVisor worker."""
    quota = None
    configured_quota = os.environ.get("QUANTGATE_PIDS_LIMIT")
    if configured_quota and configured_quota.isdigit():
        quota = int(configured_quota)
    for quota_path in ("/sys/fs/cgroup/pids.max", "/sys/fs/cgroup/pids/pids.max"):
        if quota is not None:
            break
        try:
            raw_quota = Path(quota_path).read_text(encoding="ascii").strip()
        except OSError:
            continue
        if raw_quota.isdigit():
            quota = int(raw_quota)
            break

    if not hasattr(os, "fork"):
        return {
            "verdict": "blocked",
            "children_started": 0,
            "configured_pids_limit": quota,
            "probe_limit": 2,
            "error_type": "UnsupportedPlatform",
            "evidence": "fork is unavailable on this worker platform",
        }

    # A small RLIMIT_NPROC probe gives us a deterministic EAGAIN boundary.
    # Trying to fill the Docker pids cgroup directly can make runsc terminate
    # the whole worker before Python can report the denial.
    probe_limit = 2
    children = []
    try:
        if resource is not None:
            resource.setrlimit(resource.RLIMIT_NPROC, (probe_limit, probe_limit))
        for _ in range(probe_limit + 1):
            child = os.fork()
            if child == 0:
                time.sleep(10)
                os._exit(0)
            children.append(child)
    except OSError as error:
        for child in children:
            try:
                os.kill(child, 9)
            except OSError:
                pass
        for child in children:
            try:
                os.waitpid(child, 0)
            except OSError:
                pass
        return {
            "verdict": "blocked",
            "children_started": len(children),
            "configured_pids_limit": quota,
            "probe_limit": probe_limit,
            "error_type": type(error).__name__,
            "evidence": str(error)[:160],
        }
    for child in children:
        try:
            os.kill(child, 9)
        except OSError:
            pass
        try:
            os.waitpid(child, 0)
        except OSError:
            pass
    return {
        "verdict": "unexpectedly_allowed",
        "children_started": len(children),
        "configured_pids_limit": quota,
        "probe_limit": probe_limit,
    }


def package_probe(request):
    """Exercise representative malicious-package behaviors in the worker."""
    package = request.get("package", "fixture-malware")
    version = request.get("version", "0.0.0")
    manifest_hash = hashlib.sha256(f"{package}@{version}".encode()).hexdigest()
    events = []
    try:
        Path("/run/secrets/aws").read_text(encoding="utf-8")
        events.append({"type": "secret_read", "verdict": "unexpectedly_allowed"})
    except OSError as error:
        events.append({"type": "secret_read", "verdict": "blocked", "evidence": str(error)[:120]})
    try:
        with socket.create_connection(("1.1.1.1", 80), timeout=2):
            events.append({"type": "network_egress", "verdict": "unexpectedly_allowed"})
    except OSError as error:
        events.append({"type": "network_egress", "verdict": "blocked", "evidence": str(error)[:120]})
    try:
        Path("/package-probe-write").write_text("blocked", encoding="ascii")
        events.append({"type": "root_write", "verdict": "unexpectedly_allowed"})
    except OSError as error:
        events.append({"type": "root_write", "verdict": "blocked", "evidence": str(error)[:120]})
    verdict = "DO NOT INSTALL" if all(event["verdict"] == "blocked" for event in events) else "REVIEW"
    return {"package": package, "version": version, "manifest_sha256": manifest_hash, "events": events, "verdict": verdict}


def release_job(request, preflight=False):
    budget, loss, fixture = validate_contract(request)
    data = load_fixture(fixture)
    audit = audit_split(data)
    result = {"split": audit, "budget_bytes": budget, "max_drop_bps": loss, "fixture": fixture}
    if audit["conflicts"]:
        return {**result, "decision": "block", "reason": "split_overlap"}
    if budget < MIN_BYTES:
        return {**result, "decision": "block", "reason": "budget_below_minimum",
                "minimum_bytes": MIN_BYTES}
    if preflight:
        return {**result, "decision": "continue"}
    source = request.get("source")
    return {**result, **evaluate_release(data, budget, loss, source, run_sweeps, fixture)}


def main():
    mode = sys.argv[1:]
    if mode == ["containment"]:
        print("containment-probe-ready", flush=True)
        while True:
            time.sleep(1)
    if mode == ["containment-network"]:
        print(json.dumps(containment_network()), flush=True)
        return
    if mode == ["containment-filesystem"]:
        print(json.dumps(containment_filesystem()), flush=True)
        return
    if mode == ["containment-disk"]:
        print(json.dumps(containment_disk()), flush=True)
        return
    if mode == ["containment-processes"]:
        print(json.dumps(containment_processes()), flush=True)
        return
    if mode == ["package-probe"]:
        request = json.loads(sys.stdin.buffer.read(16_385))
        print(json.dumps(package_probe(request)), flush=True)
        return
    try:
        request = json.loads(sys.stdin.buffer.read(16_385))
        if not isinstance(request, dict):
            raise ValueError("Expected a job object")
        mode = sys.argv[1:]
        if mode in (["release-preflight"], ["release"]):
            result = release_job(request, preflight=mode == ["release-preflight"])
        elif not mode:
            result = calibrate(request.get("source"), request.get("max_saturation"))
        else:
            raise ValueError("Unknown worker operation")
    except (ValueError, SyntaxError, RecursionError) as error:
        print(json.dumps({"error": str(error)[:240], "repairable": True}))
    except (OSError, subprocess.SubprocessError):
        print(json.dumps({"error": "Native compilation or execution failed."}))
    else:
        print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
