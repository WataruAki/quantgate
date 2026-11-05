# QuantGate

QuantGate is an MLOps release gate for the Vultr Agent Rush Blast Radius Zero track. It checks a classifier's activation storage and accuracy against a release contract. It rejects exact pixel overlaps between splits before sending any request to Vultr Inference, generates a constrained Python scoring kernel, and compiles it to native C++ in Docker to calibrate two layers. The trusted evaluator tries INT8 first on development and tries the next format when the evidence requires it. It freezes the first qualifying candidate before comparing it with FP16 on the held-out split.

The report records sample counts, fingerprints, attempted formats, holdout results and the final decision. Each verdict carries an independent attestation with contract, split, report and containment policy SHA-256 fingerprints. The planner blocks common prompt-injection markers such as requests to skip holdout checks or fake a passing result. The injected overlap demo duplicates one original holdout sample into calibration. The audit detects exact matches and does not cover near-duplicates, group leakage or temporal leakage. Activation rounding is simulated. Storage counts encoded values and excludes scales and process RAM.

The digits fixture uses **Optical Recognition of Handwritten Digits** by E. Alpaydin and C. Kaynak (1998), UCI Machine Learning Repository, [DOI 10.24432/C50P49](https://doi.org/10.24432/C50P49), under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The original files are included. The bundled checkpoint was trained on 2,863 training rows, with 320 calibration rows, 640 development rows and 1,797 writer-separated holdout rows. Left and right shift slices are derived by translating each original image one cell and filling the exposed column with zeroes.

The synthetic comparison uses one fixed two-channel sensor classifier with analytically assigned weights. Run `python3 generate_synthetic.py` to recreate `digits-data/synthetic.csv` from seed 1138. Each record contains its partition, 64 integer inputs and a label. The file contains 128 calibration samples, 192 development samples and 256 holdout samples. Half of the holdout has a one-unit margin between the two sensor readings. The report fingerprints the CSV and compares FP32, FP16, BF16 and INT8 activation rounding on the same samples. All four share the same classifier and arithmetic. The byte count measures the encoded activation values for the two layers.

Use **Synthetic sensors** with 96 bytes/sample and 0.50 percentage points of permitted accuracy loss. The gate freezes the smallest qualifying candidate on development. On holdout, the report compares all four formats against FP32 for diagnosis. The frozen candidate alone determines the release verdict. In the bundled run, INT8 passes development and is blocked on the low-margin group.

## Run on a Vultr Cloud Compute VM

The target host is Linux with Python 3.11+ and Docker Engine. The worker image provides GCC and Python. Build it before starting the web process.

```sh
docker build -t quant-calibrate-worker:3 .
python3 verify_vultr.py --release
python3 server.py
```

The web server listens on `127.0.0.1:8080`. For public deployment, route traffic through a TLS reverse proxy with request rate limits and a response timeout of at least 150 seconds. Run the app with a dedicated service account and access to its Docker daemon. Check the public URL from an unauthenticated browser. The bundled kernel is available without a token and still uses a disposable worker.

To enable live generation, configure these environment variables in the web process. Use a supported chat model from the inference account.

| Variable | Value |
| --- | --- |
| `VULTR_SERVERLESS_INFERENCE_API_KEY` | Vultr inference credential held by the orchestrator |
| `VULTR_MODEL` | Model ID |
| `DEMO_ACCESS_TOKEN` | Separate random token for the live demo |

Private live mode uses a separate demo token. The public demo uses a host-encrypted systemd credential named `vultr-inference-key`. It enables unauthenticated live release evaluation with a shared limit of 40 provider calls per UTC day, including repairs. The limit is recorded in SQLite in the service's state directory. The bundled kernel remains available through the release API. The server sends a bounded prompt to the Vultr chat endpoint and can request one repair when the returned kernel fails validation or disagrees with the reference.

With live inference enabled, enter a release goal of up to 500 characters in **Prompt** and select **Run evaluation**. Vultr proposes a split, byte budget and maximum accuracy loss. QuantGate checks the split and budget before the coding request, then shows the interpreted contract, generated source and verified release result. Vultr explains the measured decision in a separate inference call. Planning, generation and explanation use the shared inference quota. If the explanation call fails, the measured decision remains available.

Run `python3 verify_vultr.py --release --live` on the VM after setting the environment variables. The smoke test calls the web endpoints on loopback and covers the injected overlap, an undersized budget, an Inference-generated INT8 approval, a strict holdout rejection, and timeout cleanup. It prints timings and results without printing credentials.

## Execution boundary

The orchestrator sends JSON over standard input to a disposable Docker container. It passes no host volumes, inference credentials, or Docker socket into the worker. The container runs as UID 65534 with networking disabled, a read-only root, a bounded temporary filesystem, dropped capabilities, no new privileges, and limits on memory, CPU and process count. A worker compiles and runs the kernel inside that same container. On a timeout the orchestrator removes the named container and checks the removal response.

The accepted Python subset contains a scalar scoring function with fresh local assignments, arithmetic, and two numerical helpers. The worker maps identifiers to generated C++ names and owns the data buffers and iteration. The generated function receives two doubles. Every candidate is checked against the independent Python reference. The recorded timings separate inference, compilation, native sweep, Python reference sweep, and total workflow. The sweep timing is a single measurement and can vary between runs.

Expand **Containment probe** below the prompt and select **Run containment checks**. The same worker path tests outbound network access, a write outside the writable `/tmp` tmpfs, the tmpfs quota, the PID quota and an indefinite process. The report shows the observed kernel errors, then confirms the worker exceeded the three-second deadline and was removed. Standard Docker containers share a kernel with the host.

## Local preview

On Windows or Linux with Python 3.11+ and GCC available on PATH:

```sh
python server.py --local-preview
```

Open `http://127.0.0.1:8080` and select **Run bundled example**. Local preview compiles the synthetic comparison's bundled kernel on the host and verifies the result. Live inference and the containment probe require the Docker deployment.

In live mode, ask for the Optical Digits original split with an 80 byte budget and 0.50 percentage points of permitted accuracy loss to see an INT8 approval. Set a 47 byte budget to see the minimum size gate. Set zero allowed loss to test the fallback to mixed precision on development and its rejection on the shifted holdout slice. Ask for the injected overlap to exercise the split gate.
