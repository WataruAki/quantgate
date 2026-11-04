"""Exercise the deployment with the same worker path used by the web app."""

import argparse
from http.client import HTTPConnection
import json
import os
from threading import Thread

from server import Handler, JobError, ThreadingHTTPServer, run_calibration, run_containment_probe, run_sandbox
from worker import EXAMPLE


def require(condition, message):
    if not condition:
        raise JobError(message)


def post(port, route, body, token=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["X-Demo-Token"] = token
    connection = HTTPConnection("127.0.0.1", port, timeout=150)
    try:
        connection.request("POST", route, json.dumps(body), headers)
        response = connection.getresponse()
        return response.status, json.loads(response.read())
    finally:
        connection.close()


def verify_release(port, live, local):
    contract = {"fixture": "clean", "budget_bytes": 80, "max_drop_bps": 50}
    mode = "live" if live else "example"
    token = os.environ.get("DEMO_ACCESS_TOKEN") if live else None
    if live and not token:
        raise JobError("Set DEMO_ACCESS_TOKEN for the live HTTP smoke test.")
    cases = (
        ("split overlap", {"fixture": "injected_overlap"}, "block", "split_overlap"),
        ("undersized budget", {"budget_bytes": 47}, "block", "budget_below_minimum"),
        ("INT8 approval", {}, "approve", "holdout_passed"),
        ("strict holdout rejection", {"max_drop_bps": 0}, "block", "holdout_accuracy_failed"),
    )
    for name, changes, decision, reason in cases:
        # A single live generation job exercises the inference path; other cases use the bundled kernel.
        request = {**contract, **changes,
                   "mode": "example" if reason == "holdout_accuracy_failed" else mode}
        status, result = post(port, "/api/release", request, token)
        require(status == 200 and result.get("decision") == decision and result.get("reason") == reason,
                f"{name}: unexpected response {status} ({result.get('error', result.get('reason'))})")
        if reason == "split_overlap":
            require(result["split"]["conflict_count"] == 1 and "source" not in result,
                    "Preflight did not stop on the injected duplicate.")
        if reason == "holdout_passed":
            require(result.get("reference_verified") and result.get("verified_threshold_count") == 72
                    and result.get("selected_bits") == [8, 8]
                    and result["holdout"]["candidate"]["clean"]["correct"] == 1719,
                    "Approved INT8 result differed from the frozen experiment.")
            print(json.dumps({"mode": result["mode"], "thresholds": result["thresholds"],
                              "holdout": result["holdout"], "timing_ms": result["timing_ms"]}))
        if reason == "holdout_accuracy_failed":
            require(result.get("selected_bits") == [8, 16]
                    and [row["result"] for row in result["attempts"]] == ["accuracy_failed", "selected"],
                    "Strict policy did not retry using mixed precision.")
        print(f"PASS: HTTP release {name} ({result['mode']}).")
    status, result = post(port, "/api/release", {
        "fixture": "synthetic", "budget_bytes": 96, "max_drop_bps": 50,
    })
    require(status == 200 and result.get("reason") == "holdout_accuracy_failed"
            and result.get("selected_format") == "int8"
            and result.get("verified_threshold_count") == 72
            and result["split"]["conflict_count"] == 0,
            "Synthetic contract did not freeze INT8 and block it on holdout.")
    comparison = result["precision_comparison"]
    require(set(comparison) == {"fp32", "fp16", "bf16", "int8"},
            "The synthetic report did not compare all four formats.")
    for name, size, low_margin_correct in (
        ("fp32", 192, 128), ("fp16", 96, 128), ("bf16", 96, 112), ("int8", 48, 76),
    ):
        scores = comparison[name]
        require(scores["encoded_bytes"] == size
                and scores["development"]["clean"] == {"correct": 192, "total": 192}
                and scores["holdout"]["clean"] == {"correct": 128, "total": 128}
                and scores["holdout"]["low_margin"] == {"correct": low_margin_correct, "total": 128},
                f"Unexpected synthetic activation size or slice accuracy for {name}.")
    print("PASS: HTTP synthetic four-format comparison and frozen-candidate rejection.")
    if not local:
        status, result = post(port, "/api/containment", {})
        require(status == 200 and result.get("status") == "terminated"
                and result.get("probe_started") and result.get("removed"),
                "Timeout containment could not be verified.")
        print("PASS: HTTP containment worker started, timed out and was removed.")


def verify_release_http(live, local):
    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        server.local_preview = local
        thread = Thread(target=server.serve_forever)
        thread.start()
        try:
            verify_release(server.server_port, live, local)
        finally:
            server.shutdown()
            thread.join()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Also call Vultr Inference")
    parser.add_argument("--release", action="store_true", help="Test release and containment through HTTP")
    parser.add_argument("--local-preview", action="store_true", help="Run the release tests without Docker")
    args = parser.parse_args()
    if args.local_preview and (not args.release or args.live):
        parser.error("--local-preview requires --release and the bundled kernel")
    try:
        if args.release:
            verify_release_http(args.live, args.local_preview)
            return
        calibration = run_calibration(1, live=False)
        if len(calibration["rows"]) != 19 or calibration["recommendation"] is None:
            raise JobError("Unexpected result from the bundled calibration.")
        print("PASS: bundled kernel compiled in Docker and all 19 rows matched Python.")

        forbidden_source = "import os\n" + EXAMPLE
        invalid = run_sandbox(
            [], json.dumps({"source": forbidden_source, "max_saturation": 1}), 24,
        )
        if not invalid.get("repairable") or invalid.get("reference_verified"):
            raise JobError("The worker did not reject a kernel with an import.")
        print("PASS: worker rejected Python outside the allowed subset.")

        containment = run_containment_probe()
        print("PASS: worker signaled readiness, exceeded its timeout, and was removed.")

        if args.live:
            generated = run_calibration(1, live=True)
            print("PASS: Vultr generated a kernel that compiled and matched the reference.")
            print(json.dumps({
                "mode": generated["mode"],
                "recommendation": generated["recommendation"],
                "timing_ms": generated["timing_ms"],
            }, indent=2))
        print(json.dumps({"containment": containment}))
    except JobError as error:
        parser.exit(1, f"FAIL: {error}\n")


if __name__ == "__main__":
    main()
