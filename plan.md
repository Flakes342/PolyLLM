## PHASE 1 — Single Rust inference server (CPU) for one GGUF model

Goal: get a working Rust inference container serving /generate on CPU and streaming text.
Choose first model (M1). Download GGUF into models/m1.gguf (locally).
Implement Rust wrapper (use llama.cpp binary or llm/candle crate + actix-web or axum):
Endpoint: POST /generate body {prompt, max_tokens, temp}.
Stream output back (SSE or chunked JSON).
/health and /metrics (Prometheus).
Include run_llama.sh that calls the inference binary with model path.
Dockerize: docker build -t polyllm-model-m1:latest inference-rust/.

Test locally:

    docker run --rm -v $(pwd)/models:/models -p 8101:8101 polyllm-model-m1:latest
    curl -X POST http://localhost:8101/generate -d '{"prompt":"Hello","max_tokens":32}' -H 'Content-Type: application/json'


Record basic CPU latencies for short/medium prompts.
Deliverable: one Rust container running model1 on CPU streaming responses.

## PHASE 2 — FastAPI router (baseline) + Redis caching + rate limiting

Goal: simple orchestrator that forwards requests and caches results.
Create router/ FastAPI app:
Endpoint POST /generate accepts {prompt, max_tokens, temp, prefer_backend: "cpu"|"gpu"|"auto"}.
Hash prompt+params → check Redis cache; return cached if present.
Simple model registry JSON (env var) mapping m1→http://model1:8101, etc.
Forward request to selected model endpoint and stream response back to client.
Add Redis token-bucket rate limiter per API key/IP.
Dockerize router. Add to docker-compose.yml with Redis.
Test end-to-end locally: router → model1. Verify cache hit/miss and rate limiting.
Deliverable: Router forwarding working with cache + rate-limiting.

## PHASE 3 — Spin up remaining 4 model containers

Goal: uniform model service images for all 5 GGUF models (CPU mode).
Repeat PHASE1 for models m2..m5: quantize/download, place at models/m2.gguf … models/m5.gguf. Build 5 containers (or same image with different MODEL_PATH env). Use ports 8101..8105.
Update docker-compose.yml to run all 5 model services + router + redis.
Smoke test: send the same prompt to each model endpoint; log responses and latency.
Deliverable: 5 model containers running locally on CPU.

## PHASE 4 — Router intelligence (data collection + training + integration) (4 days)

Goal: build a learned router (LightGBM) rather than static rules.
Create prompt dataset (≈3–5k prompts):
Collect from public prompt lists (coding prompts, math problems, creative tasks), plus synthetic variations.

Collect responses:
Script that sends each prompt to all 5 models (using router or direct model endpoints) and logs: prompt, model_id, response_text, latency, token_count.
Auto-score responses (quick practical approach):
Use embedding similarity: compute embedding of each model response and pick the one closest to a high-quality reference (if you have a reference model) OR use a cheap automatic grader: run a small higher-quality LLM (e.g., a hosted eval model) to score responses (short list only to save tokens).

Create label = best_model per prompt using score - λ * latency_penalty.

Feature engineering: tokens_in_prompt, has_code_flag, prompt_length, embedding PCA dims (10 dims).
Train LightGBM classifier → output router_model.txt.
Integrate trained router into FastAPI: load LightGBM model, featurize incoming prompt, predict model_id + confidence.
Add fallback cascade: if confidence < threshold or model times out, call bigger fallback (e.g., m5 or GPU model).
Deliverable: Trained router loaded in service that picks models intelligently and logs decisions.

## PHASE 5 — Monitoring & Metrics

Goal: show the exact metrics panel in the demo.

Add Prometheus metrics:
Router: requests_total, model_selected_count{model}, cache_hits, rate_limited_count, request_latency_seconds_histogram.
Models: requests_total, gen_latency_seconds_histogram, tokens_generated_total.
Add GPU metrics (later) via nvidia-smi exporter or Node exporter + DCGM for k8s.
Launch Prometheus & Grafana (docker-compose) and import a dashboard JSON with panels:
Model selection distribution
p50/p95/p99 per-model latency
tokens/sec per model
Redis cache hit ratio
CPU & GPU utilization (host-level)
Ensure logs include request_id, prompt_hash, model_id, latency, tokens.
Deliverable: Grafana dashboard reachable locally showing live metrics.

## PHASE 6 — UI (React) that toggles CPU/GPU and shows details

Goal: interactive web UI for the demo experience.

Build simple React app:
Input area for prompt.
Toggle: Prefer Backend with options: cpu, gpu, auto (router).
Submit button; show streaming tokens (connect to router /generate via SSE/WebSocket).
After response completes show:
Chosen model name and confidence.
Router rationale (features + probability).
Metrics snapshot: latency, model tokens/sec, cache hit (pull from router /last_request_stats).
Buttons: “Re-run on GPU”, “Re-run on CPU”.
Host UI in Docker as ui service (or serve with npm start during demo).
Deliverable: UI working locally with toggle + display.

## PHASE 7 — Benchmarking scripts & reports

Goal: reproducible benchmarks to display in demo.
Implement benchmarks/load_test.py:
Accepts workload type: short/medium/long, concurrency, total requests.
Rotates prompts from dataset and hits router /generate.
Records per-request latency, token counts, model selected, and CPU/GPU metrics via Prometheus API.
Run baseline local tests for CPU model performance and log results: p50/p95/p99 latency, tok/s. Save CSV & generate simple plots.
For GPU, run same tests after deploying to GPU VM (next phase) to produce comparison graphs.
Deliverable: benchmark CSVs + plots (latency vs concurrency, tokens/sec per model).

## PHASE 8 — Prepare cloud model storage & GPU support

Goal: make the same containers run on AWS GPU instance.
Upload model GGUF files to S3 (do not commit to repo). Example: s3://your-bucket/models/m1.gguf.
Add startup script in model container to aws s3 cp s3://... /models/m1.gguf on startup if file missing (or use init container approach).
Prepare GPU-enabled Rust inference binary (Candle or llama.cpp built with CUDA). Test locally if possible. Ensure run_llama.sh uses GPU flags when USE_GPU=true.
Prepare Docker image tag polyllm/inference:gpu-ready and push to registry (ECR or Docker Hub).
Deliverable: GPU-capable image & S3 model storage + startup fetch logic.

## PHASE 9 — Quick AWS GPU demo deployment

Goal: spin up a single GPU instance with the same Docker images and demonstrate GPU path.
Launch an EC2 instance (e.g., g4dn.xlarge or g5.xlarge) with an AMI that has NVIDIA drivers & Docker + NVIDIA Container Toolkit installed. (Use AWS docs or community AMI.)
SSH into instance, attach an EBS volume for /models, aws s3 cp model files onto it, or rely on container startup script to pull from S3.
Run GPU container:

    docker run --gpus '"device=0"' -v /models:/models -p 8103:8103 polyllm/inference:gpu-ready \
      -e MODEL_PATH=/models/m3.gguf -e USE_GPU=true

Start router on a small CPU EC2 and update registry so router can call GPU host. Or run router on same instance for demo (less ideal but faster).
Test a GPU-heavy prompt, measure latency/tok/s. Compare with CPU run from step 6.
Deliverable: deployed GPU container responding to router requests; measured GPU performance CSV.

## PHASE 10 — Integrate metrics & UI GPU/CPU toggle

Goal: make the UI show the deploy selection and metrics clearly.
Add UI dropdown to choose Prefer Backend (cpu/gpu/auto) and pass in request body to router. Router respects prefer_backend and selects a model that is running on desired backend (use model metadata or a requires_gpu flag).
UI shows the model chosen and the underlying host (e.g., model3@gpu-1 or model2@cpu-node-1).
UI also shows benchmark charts (precomputed plots and live stats pulled via Prometheus queries).
Deliverable: UI final touches for demo.

## PHASE 11 — Final polish, warm-up & demo recording

Goal: prepare and record the demo video.
Warm-up: ensure models are loaded (run a warmup prompt to avoid cold start latency).
Run a short benchmark to show recent numbers in Grafana (p95 latency and tokens/sec).

Demo script (2–3 min):
Show UI with prompt; toggle cpu → submit → show result + chosen model + router rationale + metrics panel.
Toggle gpu → submit same prompt → show result + new model + metrics.
Show Grafana dashboard with benchmarks & resource graphs.
Show a log entry in router (request_id, model, latency, cache_hit).
Record with screen recorder and voiceover (explain what is happening, include the architecture diagram).
Deliverable: demo video file and a demo_notes.md with exact commands to reproduce the demo.

Benchmark methodology & results summary.
Known limitations & next steps.
Push code to GitHub, tag a release, attach demo video and benchmark CSV.
