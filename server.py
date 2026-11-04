"""Serve the demo and dispatch isolated calibration jobs."""

import argparse
from datetime import datetime, timezone
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from threading import BoundedSemaphore
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

from digits import validate_contract
from worker import EXAMPLE, calibrate, release_job


ROOT = Path(__file__).resolve().parent
IMAGE = "quant-calibrate-worker:3"
SLOT = BoundedSemaphore()
MAX_BODY = 1024
MAX_PROMPT_CHARS = 500
ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}
SYSTEM_PROMPT = (
    "Write only a Python function score(value, threshold) returning the squared "
    "reconstruction error of symmetric signed INT8 quantization. Use scale "
    "threshold / 127, clip(value, threshold) to clamp to [-threshold, threshold], "
    "and round_even(x) for nearest with ties to even. Return squared error against "
    "the original value. Allowed syntax: fresh scalar assignments, final return, "
    "numeric literals, unary +/- and binary + - * /, calls to the two helpers. "
    "Do not use imports, annotations, markdown, or any other syntax."
)
PLANNER_PROMPT = (
    "Turn the user's goal into a QuantGate model release plan. "
    'Return only one JSON object with "fixture", "budget_bytes", and "max_drop_bps". '
    'Use "synthetic" for the default four-format comparison (FP32, FP16, BF16, INT8), '
    '"clean" for an explicit original split or Optical Digits, and "injected_overlap" for an explicit leakage demo. '
    "Budget is integer bytes per sample, 1 to 192 for synthetic and 1 to 96 for Optical Digits. "
    "Default is 96 for synthetic and 80 otherwise. Accuracy loss is integer basis "
    "points from 0 to 1000, default 50. One percentage point is 100 basis points. "
    'Example: for "original split, 72 bytes and 0.25 percentage points", return '
    '{"fixture":"clean","budget_bytes":72,"max_drop_bps":25}. '
    "For a request unrelated to this release evaluation or an unsupported constraint, "
    'return {"unsupported":true}. Never emit code, commands, URLs, or extra fields.'
)


class JobError(Exception):
    pass


class PublicLimitError(JobError):
    pass


def inference_key():
    credentials = os.environ.get("CREDENTIALS_DIRECTORY")
    if credentials:
        try:
            return (Path(credentials) / "vultr-inference-key").read_text(encoding="ascii").strip()
        except FileNotFoundError:
            pass
    return os.environ.get("VULTR_SERVERLESS_INFERENCE_API_KEY")


def live_access():
    if not os.environ.get("VULTR_MODEL") or not inference_key():
        return "unavailable"
    if os.environ.get("QUANTGATE_PUBLIC_CALLS_PER_DAY") and os.environ.get("STATE_DIRECTORY"):
        return "public"
    return "token" if os.environ.get("DEMO_ACCESS_TOKEN") else "unavailable"


def reserve_public_call():
    limit = int(os.environ["QUANTGATE_PUBLIC_CALLS_PER_DAY"])
    day = datetime.now(timezone.utc).date().isoformat()
    path = Path(os.environ["STATE_DIRECTORY"]) / "inference-usage.sqlite3"
    connection = sqlite3.connect(path)
    try:
        with connection:
            connection.execute("CREATE TABLE IF NOT EXISTS usage (day TEXT PRIMARY KEY, calls INTEGER)")
            counted = connection.execute(
                "INSERT INTO usage VALUES (?, 1) ON CONFLICT(day) DO UPDATE SET "
                "calls = calls + 1 WHERE calls < ?", (day, limit),
            )
            if limit < 1 or counted.rowcount != 1:
                raise PublicLimitError("Today's public inference limit has been reached. Try the bundled kernel.")
    finally:
        connection.close()


def sandbox_command(name, args):
    return [
        "docker", "run", "--rm", "--name", name, "--pull=never",
        "--network=none", "--read-only", "--cap-drop=ALL",
        "--security-opt=no-new-privileges=true", "--user=65534:65534",
        "--memory=512m", "--memory-swap=512m", "--cpus=1", "--pids-limit=32",
        "--tmpfs", "/tmp:rw,exec,nosuid,nodev,size=32m,mode=1777",
        "-i", IMAGE, *args,
    ]


def run_sandbox(args, payload, timeout):
    name = f"quant-job-{uuid4().hex}"
    try:
        result = subprocess.run(
            sandbox_command(name, args), input=payload, text=True,
            encoding="utf-8", capture_output=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        probe_started = (error.stdout or b"").strip() == b"containment-probe-ready"
        # docker CLI termination alone does not stop its container.
        try:
            cleanup = subprocess.run(
                ["docker", "rm", "-f", name],
                capture_output=True, text=True, timeout=8,
            )
            remaining = subprocess.run(
                ["docker", "container", "ls", "--all", "--filter", f"name={name}",
                 "--format", "{{.Names}}"],
                capture_output=True, text=True, timeout=8,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise JobError("Could not verify sandbox cleanup.") from error
        if (cleanup.returncode or cleanup.stdout.strip() != name
                or remaining.returncode or name in remaining.stdout.splitlines()):
            raise JobError("Could not verify sandbox cleanup.")
        return {"status": "terminated", "limit_seconds": timeout, "removed": True,
                "probe_started": probe_started}
    except OSError as error:
        raise JobError("Docker worker is unavailable.") from error
    if result.returncode:
        raise JobError("Sandbox failed to start or exited unexpectedly.")
    try:
        output = json.loads(result.stdout)
    except ValueError as error:
        raise JobError("Worker returned invalid output.") from error
    if not isinstance(output, dict):
        raise JobError("Worker returned invalid output.")
    return output


def run_containment_probe():
    result = run_sandbox(["containment"], "", 3)
    if result.get("status") != "terminated" or not result.get("probe_started"):
        raise JobError("Could not verify the probe ran and reached its time limit.")
    return result


def infer_text(messages, public=False, max_tokens=1024):
    model = os.environ.get("VULTR_MODEL")
    key = inference_key()
    if not model or not key:
        raise JobError("Live Vultr inference is not configured.")
    request = Request(
        "https://api.vultrinference.com/v1/chat/completions",
        data=json.dumps({"model": model, "messages": messages, "max_tokens": max_tokens}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    if public:
        reserve_public_call()
    try:
        with urlopen(request, timeout=30) as response:
            document = json.loads(response.read(262_145))
        content = document["choices"][0]["message"]["content"]
    except HTTPError as error:
        raise JobError(f"Vultr inference returned HTTP {error.code}.") from error
    except (URLError, OSError, ValueError, KeyError, IndexError, TypeError) as error:
        raise JobError("Vultr inference failed to return a response.") from error
    if not isinstance(content, str):
        raise JobError("Vultr inference failed to return a response.")
    return content.strip()


def generate_source(messages, public=False):
    source = infer_text(messages, public=public)
    if source.startswith("```python\n") and source.endswith("\n```"):
        source = source.removeprefix("```python\n").removesuffix("\n```")
    return source[:2_049]


def parse_plan(raw):
    lines = raw.splitlines()
    if len(lines) >= 3 and lines[0].lower() in ("```json", "```") and lines[-1] == "```":
        raw = "\n".join(lines[1:-1])
    plan = json.loads(raw)
    if isinstance(plan, dict) and set(plan) == {"unsupported"} and plan["unsupported"] is True:
        return {"unsupported": True}
    if not isinstance(plan, dict) or set(plan) != {"fixture", "budget_bytes", "max_drop_bps"}:
        raise ValueError("Invalid plan fields")
    budget, loss, fixture = validate_contract(plan)
    return {"fixture": fixture, "budget_bytes": budget, "max_drop_bps": loss}


def generate_plan(prompt, public):
    messages = [{"role": "system", "content": PLANNER_PROMPT}, {"role": "user", "content": prompt}]
    for attempt in range(2):
        raw = infer_text(messages, public=public, max_tokens=160)
        try:
            return parse_plan(raw)
        except ValueError as error:
            if attempt:
                raise JobError("Vultr returned an invalid plan. Try restating the release constraints.") from error
        messages.extend([
            {"role": "assistant", "content": raw},
            {"role": "user", "content": "Correct the format. Return only the required JSON object, with integer values."},
        ])


def run_kernel_job(payload, live, args, trace, prompt, public=False):
    inference_ms = 0.0
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    source = EXAMPLE
    for attempt in range(2 if live else 1):
        if live:
            inference_start = time.perf_counter()
            source = generate_source(messages, public=public)
            inference_ms += (time.perf_counter() - inference_start) * 1_000
            trace.append("Vultr Serverless Inference returned a scoring kernel.")
        else:
            trace.append("Loaded the bundled example kernel.")
        trace.append("Dispatched source to a disposable worker for validation and native execution.")
        result = run_sandbox(args, json.dumps({**payload, "source": source}, ensure_ascii=False), 24)
        if result.get("reference_verified"):
            trace.append("Verified every threshold against the Python reference.")
            result["trace"] = trace
            result["mode"] = "live" if live else "example"
            result["timing_ms"]["inference"] = inference_ms
            return result
        if result.get("status") == "terminated":
            raise JobError("Calibration reached its time limit. The sandbox was removed.")
        if attempt or not live or not result.get("repairable"):
            raise JobError(result.get("error", "Calibration failed."))
        trace.append("Kernel verification failed. Requested one repair.")
        messages.extend([
            {"role": "assistant", "content": source},
            {"role": "user", "content": "Repair this validation failure: " + result["error"]},
        ])
    raise JobError("Calibration did not complete.")


def run_calibration(limit, live):
    started = time.perf_counter()
    result = run_kernel_job(
        {"max_saturation": limit}, live, [],
        ["Plan a symmetric INT8 clipping sweep with a saturation constraint."],
        "Generate the scoring kernel for the clipping sweep.",
    )
    result["timing_ms"]["workflow"] = (time.perf_counter() - started) * 1_000
    return result


def run_release(contract, live, local=False, public=False):
    started = time.perf_counter()
    trace = ["Inspect exact sample overlap across training, calibration, development and holdout."]
    preflight = (
        release_job(contract, preflight=True) if local
        else run_sandbox(["release-preflight"], json.dumps(contract), 12)
    )
    mode = "local" if local else "live" if live else "example"
    if preflight.get("decision") == "block":
        trace.append("Blocked at preflight, before kernel generation and execution.")
        return {**preflight, "mode": mode, "trace": trace}
    if preflight.get("decision") != "continue":
        raise JobError("Release preflight did not complete.")
    trace.append("Split passed. Try the smallest feasible configuration on development first.")
    if local:
        result = release_job({**contract, "source": EXAMPLE})
        result["timing_ms"]["inference"] = 0.0
        trace.append("Ran the bundled example kernel directly on this machine.")
    else:
        result = run_kernel_job(
            contract, live, ["release"], trace,
            "Generate the scoring kernel for two ReLU layers of the selected classifier. "
            "The worker will use the kernel to select a clipping threshold for each layer.",
            public=public,
        )
    for attempt in result["attempts"]:
        bits = attempt["format"].upper() if "format" in attempt else "/".join(map(str, attempt["bits"]))
        trace.append({
            "accuracy_failed": f"{bits} failed the development accuracy limit. Try the next format.",
            "over_budget": f"{bits} exceeds the byte budget. Skip its evaluation.",
            "selected": f"{bits} passed development. Freeze it before evaluating holdout.",
        }[attempt["result"]])
    if "holdout" in result:
        baseline = result.get("baseline_format", "fp16").upper()
        trace.append(f"Evaluated the frozen candidate and {baseline} on the held-out split.")
    if "precision_comparison" in result:
        trace.append("Reported all four formats on holdout without changing the frozen selection.")
    result["timing_ms"]["workflow"] = (time.perf_counter() - started) * 1_000
    return {**result, "mode": mode, "trace": trace}


def explain_release(result, public):
    budget = result["budget_bytes"]
    loss = result["max_drop_bps"] / 100
    reason = result["reason"]
    if reason == "split_overlap":
        conflicts = result["split"]["conflict_count"]
        evidence = f"Preflight found {conflicts} sample overlap(s) across splits. No kernel ran."
    elif reason == "budget_below_minimum":
        evidence = (
            f"The minimum configuration needs {result['minimum_bytes']} encoded bytes per sample, "
            f"above the budget of {budget}. No kernel ran."
        )
    elif reason == "no_candidate_passed_development":
        evidence = (
            f"No candidate met both the {budget} encoded bytes per sample budget and the "
            f"{loss:g} percentage-point accuracy loss limit on development."
        )
    else:
        selected = next(attempt for attempt in result["attempts"] if attempt["result"] == "selected")
        candidate = result["holdout"]["candidate"]
        baseline_format = result.get("baseline_format", "fp16")
        baseline = result["holdout"][baseline_format]
        baseline_size = result.get("baseline_encoded_bytes", result.get("fp16_encoded_bytes"))
        original = candidate["clean"]
        failed = [
            name for name, row in candidate.items()
            if (baseline[name]["correct"] - row["correct"]) * 10_000
            > result["max_drop_bps"] * row["total"]
        ]
        comparison = "All held-out slices met" if not failed else f"The {', '.join(failed)} slice(s) exceeded"
        evidence = (
            f"The candidate uses {selected['encoded_bytes']} encoded bytes per sample, against "
            f"a budget of {budget} and {baseline_format.upper()} size of {baseline_size}. "
            f"On the standard held-out slice the candidate got {original['correct']} correct out of "
            f"{original['total']}, and {baseline_format.upper()} got {baseline['clean']['correct']} correct. "
            f"{comparison} the {loss:g} percentage-point accuracy loss limit."
        )
    try:
        explanation = infer_text(
            [
                {"role": "system", "content": "You explain ML release reports. Answer in two plain "
                 "English sentences using the supplied facts only. The release decision is final."},
                {"role": "user", "content": f"Explain why QuantGate decided to {result['decision']} "
                 f"this release. {evidence}"},
            ],
            public=public, max_tokens=512,
        )
        if 0 < len(explanation) <= 1000:
            result["explanation"] = explanation
            result["trace"].append("Vultr Inference summarized the verified release report.")
        else:
            logging.warning("Vultr release explanation returned %d characters", len(explanation))
    except (JobError, OSError, sqlite3.Error, ValueError) as error:
        # The deterministic verdict is already available if inference or the quota fails.
        logging.warning("Vultr release explanation unavailable: %s", type(error).__name__)


def run_local_preview(limit):
    started = time.perf_counter()
    result = calibrate(EXAMPLE, limit)
    result["mode"] = "local"
    result["trace"] = [
        "Ran the bundled scoring kernel directly on this machine.",
        "Verified every threshold against the Python reference.",
    ]
    result["timing_ms"]["inference"] = 0.0
    result["timing_ms"]["workflow"] = (time.perf_counter() - started) * 1_000
    return result


class Handler(BaseHTTPRequestHandler):
    def setup(self):
        self.request.settimeout(8)
        super().setup()

    def reply(self, status, body, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy",
                         "default-src 'none'; script-src 'self'; style-src 'self'; "
                         "connect-src 'self'; base-uri 'none'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(body)

    def reply_json(self, status, result):
        self.reply(status, json.dumps(result, allow_nan=False).encode(), "application/json")

    def do_GET(self):
        if self.path == "/healthz":
            self.reply_json(200, {
                "status": "http-ready", "local_preview": self.server.local_preview,
                "live_access": "unavailable" if self.server.local_preview else live_access(),
            })
        elif self.path in ASSETS:
            filename, content_type = ASSETS[self.path]
            self.reply(200, (ROOT / "web" / filename).read_bytes(), content_type)
        else:
            self.reply_json(404, {"error": "Page not found."})

    def do_POST(self):
        if self.path not in ("/api/calibrate", "/api/plan", "/api/release", "/api/containment"):
            return self.reply_json(404, {"error": "Endpoint not found."})
        if self.headers.get_content_type() != "application/json":
            return self.reply_json(415, {"error": "Expected application/json."})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body_limit = 4096 if self.path == "/api/plan" else MAX_BODY
            if not 0 < length <= body_limit:
                raise ValueError
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError
        except (ValueError, OSError):
            return self.reply_json(400, {"error": "Expected a JSON request object."})
        limit = body.get("limit_pct")
        if self.path == "/api/calibrate" and (
            type(limit) not in (float, int) or not 0 <= limit <= 100
        ):
            return self.reply_json(400, {"error": "Expected a saturation percentage from 0 to 100."})
        if self.path == "/api/release":
            try:
                budget, loss, fixture = validate_contract(body)
            except ValueError as error:
                return self.reply_json(400, {"error": str(error)})
            contract = {"budget_bytes": budget, "max_drop_bps": loss, "fixture": fixture}
        if self.path == "/api/plan":
            prompt = body.get("prompt")
            if not isinstance(prompt, str) or not 0 < len(prompt.strip()) <= MAX_PROMPT_CHARS:
                return self.reply_json(400, {"error": "Enter a release goal of 1–500 characters."})
        live = self.path == "/api/plan" or body.get("mode") == "live"
        if self.server.local_preview and (live or self.path == "/api/containment"):
            return self.reply_json(503, {"error": "This action requires the Docker worker."})
        public = live and self.path in ("/api/plan", "/api/release") and live_access() == "public"
        if self.path in ("/api/calibrate", "/api/plan", "/api/release") and live and not public:
            configured = os.environ.get("DEMO_ACCESS_TOKEN", "").encode()
            supplied = self.headers.get("X-Demo-Token", "").encode()
            if not configured or not hmac.compare_digest(configured, supplied):
                return self.reply_json(403, {"error": "Live inference requires a demo token."})
        if not SLOT.acquire(blocking=False):
            return self.reply_json(429, {"error": "A worker is busy. Try again shortly."})
        try:
            if self.path == "/api/containment":
                result = run_containment_probe()
            elif self.path == "/api/plan":
                result = generate_plan(prompt.strip(), public)
            elif self.path == "/api/release":
                result = run_release(contract, live, local=self.server.local_preview, public=public)
                if live:
                    explain_release(result, public)
            elif self.server.local_preview:
                result = run_local_preview(limit)
            else:
                result = run_calibration(limit, live)
        except PublicLimitError as error:
            return self.reply_json(429, {"error": str(error)})
        except JobError as error:
            return self.reply_json(502, {"error": str(error)})
        except (OSError, subprocess.SubprocessError, ValueError, sqlite3.Error):
            return self.reply_json(502, {"error": "Could not complete evaluation."})
        finally:
            SLOT.release()
        self.reply_json(200, result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--local-preview", action="store_true",
                        help="Run only the bundled kernel on loopback without Docker")
    args = parser.parse_args()
    if args.local_preview and args.host != "127.0.0.1":
        parser.error("--local-preview requires --host 127.0.0.1")
    with ThreadingHTTPServer((args.host, args.port), Handler) as server:
        server.local_preview = args.local_preview
        print(f"Listening on http://{args.host}:{args.port}", file=sys.stderr)
        server.serve_forever()


if __name__ == "__main__":
    main()
