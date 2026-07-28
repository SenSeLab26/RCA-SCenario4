"""Order Backend - the replicated service whose node we terminate.

Each replica models a service with a *bounded* amount of concurrency: only
WORKER_SLOTS requests are processed at once, each taking SERVICE_TIME_MS.
Everything beyond that waits in a queue.

That bound is the whole reason this scenario produces a latency spike. If a
replica could serve unlimited concurrent work, losing one of three replicas
would simply mean fewer requests served - not slower ones. With a bounded
worker pool, the two survivors must absorb the full offered load with two
thirds of the capacity, so queue wait time climbs and OpenTelemetry records a
brownout.

A replica also refuses readiness for WARMUP_SECONDS after it starts, which
models a real cold start (cache fill, JIT warm-up, connection pools). This is
what makes "restabilization" take measurably longer than "pod scheduled".
"""

import json
import os
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

POD_NAME = os.environ.get("POD_NAME", socket.gethostname())
NODE_NAME = os.environ.get("NODE_NAME", "unknown")
PORT = int(os.environ.get("PORT", "8000"))
WORKER_SLOTS = int(os.environ.get("WORKER_SLOTS", "2"))
SERVICE_TIME_MS = float(os.environ.get("SERVICE_TIME_MS", "100"))
WARMUP_SECONDS = float(os.environ.get("WARMUP_SECONDS", "15"))
OTLP_ENDPOINT = os.environ.get("OTLP_ENDPOINT", "http://jaeger:4317")

# --- Telemetry pipeline (same shape as Scenarios 1-3, different service name) ---
resource = Resource(
    attributes={
        "service.name": "order-backend",
        "k8s.pod.name": POD_NAME,
        "k8s.node.name": NODE_NAME,
    }
)
provider = TracerProvider(resource=resource)
provider.add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter(endpoint=OTLP_ENDPOINT, insecure=True))
)
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)
propagator = TraceContextTextMapPropagator()

# The capacity bound. Total throughput per replica is roughly
# WORKER_SLOTS / SERVICE_TIME_MS -> with the defaults, 2 / 0.1s = 20 req/s.
slots = threading.Semaphore(WORKER_SLOTS)
STARTED_AT = time.time()


def is_warm():
    """False until the replica has finished its simulated cold start."""
    return (time.time() - STARTED_AT) >= WARMUP_SECONDS


class BacklogServer(ThreadingHTTPServer):
    """ThreadingHTTPServer with a listen backlog big enough for this load.

    socketserver defaults request_queue_size to 5. Every request here arrives on
    a fresh TCP connection (the gateway sends `Connection: close` so Kubernetes
    can rebalance each one), so at tens of connections per second that backlog
    overflows, the kernel drops SYNs, and TCP retransmission turns an ordinary
    request into a multi-second one. That shows up as random timeouts even when
    the service is perfectly healthy - noise that would drown the real signal.
    """

    request_queue_size = 512
    daemon_threads = True


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        """Silence per-request logging - at 30+ req/s it is pure noise."""

    def _respond(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        # Probes must never queue behind application work, so handle them first.
        if self.path.startswith("/healthz"):
            warm = is_warm()
            self._respond(
                200 if warm else 503,
                {"ready": warm, "pod": POD_NAME, "node": NODE_NAME},
            )
            return

        if self.path.startswith("/livez"):
            self._respond(200, {"alive": True, "pod": POD_NAME})
            return

        if not self.path.startswith("/order"):
            self._respond(404, {"error": "not found"})
            return

        self._handle_order()

    def _handle_order(self):
        # Continue the trace the API gateway started, so Jaeger shows one trace
        # spanning gateway -> backend rather than two disconnected spans.
        ctx = propagator.extract(carrier={k.lower(): v for k, v in self.headers.items()})

        with tracer.start_as_current_span("order-backend.process", context=ctx) as span:
            span.set_attribute("k8s.pod.name", POD_NAME)
            span.set_attribute("k8s.node.name", NODE_NAME)
            span.set_attribute("backend.worker_slots", WORKER_SLOTS)

            # Time spent waiting for a free worker slot IS the brownout signal.
            queued_at = time.time()
            slots.acquire()
            queue_wait = time.time() - queued_at
            try:
                time.sleep(SERVICE_TIME_MS / 1000.0)
            finally:
                slots.release()

            span.set_attribute("backend.queue_wait_ms", round(queue_wait * 1000, 2))
            span.set_attribute("backend.service_time_ms", SERVICE_TIME_MS)

            self._respond(
                200,
                {
                    "pod": POD_NAME,
                    "node": NODE_NAME,
                    "queue_wait_ms": round(queue_wait * 1000, 2),
                    "service_time_ms": SERVICE_TIME_MS,
                },
            )


if __name__ == "__main__":
    print(
        f"order-backend up | pod={POD_NAME} node={NODE_NAME} "
        f"slots={WORKER_SLOTS} service_time={SERVICE_TIME_MS}ms "
        f"warmup={WARMUP_SECONDS}s -> capacity ~{WORKER_SLOTS / (SERVICE_TIME_MS / 1000):.0f} req/s",
        flush=True,
    )
    BacklogServer(("0.0.0.0", PORT), Handler).serve_forever()
