### Phase 1 – Foundations
Project setupCreate GitHub repo, setup issues & milestones.Decide 5 models (e.g., Mistral-7B, LLaMA-7B, LLaMA-13B, CodeLLaMA, TinyLLaMA).Record licenses + usage notes in docs/models.md.Note target hardware (CPU/GPU, RAM).Environment & infra basicsInstall Rust toolchain, Python (3.10+), Docker, Redis, Postgres.Setup Prometheus + Grafana stack (docker-compose).Create base repo structure:
    inference-rust/
    router/
    monitoring/
    docs/

### Phase 2 – Models & Inference
Quantize modelsDownload 5 model weights legally.Convert to GGUF (int4/int8).Store checksums + metadata in Postgres.Rust inference serviceBuild Rust server (per model) with llama.cpp/candle binding.Expose /generate (REST or gRPC) and /health.Add streaming response support.Add Prometheus metrics export (/metrics).Containerize service with Dockerfile.Bring up all 5 modelsRun each inference service in Docker.Verify each responds to test prompt.Add liveness/readiness probes.
### Phase 3 – Router Service
Router scaffolding (FastAPI)Endpoint /generate (accepts prompt).Calls chosen model (for now round-robin).Returns streaming response to client.Redis integration. Add request cache (prompt+params key, TTL=5min).Add rate limiting (token bucket per IP/API key).Database schemaPostgres tables:models (id, version, quant, path, perf stats).requests (id, timestamp, user, model_id, latency, tokens).Store request metadata on every call.
### Phase 4 – Intelligent Router
Data collectionBuild script to send same prompts → all 5 models.Log outputs + latency + tokens/sec.Labeling & trainingScore outputs (LLM grader or heuristic).Train LightGBM router (features: token count, prompt length, code keywords).Save model artifact.Integrate routerReplace round-robin with trained model.Add fallback logic: if chosen model fails, cascade to larger model.Log router confidence + decision reason.
### Phase 5 – Monitoring & Observability
Prometheus metricsRouter: request_count, cache_hit_rate, rate_limited_count.Each model: latency p50/p95, tokens/sec, errors.Grafana dashboardsPanels: per-model throughput, latency, router decision distribution, cache hit ratio.Alert rules: high latency, low cache hit, high error rate.TracingAdd OpenTelemetry: request_id flows router → model → back.Log JSON structured entries to Loki or stdout.
### Phase 6 – Benchmarking
Load test setupWrite Locust or custom async load tester. Run short/medium/long prompts with concurrency 1–32.Benchmark reportCollect latency p50/p95/p99, tokens/sec, throughput per model. Compare router accuracy (% prompts routed to optimal).Write results in benchmarks/report.md.
### Phase 7 – Frontend & UX
Web UI React + Tailwind chat interface.Dropdown: pick specific model OR router decides.Show which model was used + router confidence.Streaming response displayed in chat bubbles.Public API Endpoints: /generate, /health, /metrics.Add API key auth.
### Phase 8 – Hardening & Deployment
Error handlingGraceful fallbacks if model crashes.Retry with exponential backoff.Circuit breaker on repeated failures.Deployment setupDocker Compose for local demo.Helm/Kubernetes manifests for multi-node deployment.Horizontal Pod Autoscaler (CPU/GPU utilization).SecurityTLS termination via Nginx/Ingress.PII handling: hash prompts in logs.JWT-based auth for users.
### Phase 9 – Delivery
Docs & packaging README with setup steps.Architecture diagram.API docs (OpenAPI schema).Grafana JSON dashboards committed.DemoRecord 2-min demo video (router in action + Grafana).Publish benchmark results.Write short blog post / portfolio summary.
