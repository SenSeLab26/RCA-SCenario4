"""API Gateway - the front door that the load generator hammers.

The gateway is deliberately *not* a bottleneck: it has no concurrency limit of
its own, it just forwards each incoming request to the `order-backend`
Kubernetes Service and records the result as a span. All the latency we measure
therefore belongs to the backend replicas.

Three details matter for this scenario:

1. `Connection: close` on the upstream call. Kubernetes load balances at the
   connection level (kube-proxy / iptables), so a pooled keep-alive connection
   would stay pinned to one backend pod. Closing each connection means every
   request gets a fresh balancing decision - that is what lets us observe
   traffic being *rerouted* away from a dead replica.
2. The Service name is resolved to its ClusterIP *once*, at startup, and every
   request then talks to that IP directly. Resolving per request meant a DNS
   lookup per request, and Kubernetes DNS runs over UDP with `timeout:5` in
   /etc/resolv.conf - so each dropped DNS packet cost exactly five seconds and
   registered as a failed request. That produced a steady drizzle of fake
   "errors" while the cluster was completely healthy. The ClusterIP is stable
   for the life of the Service, and kube-proxy still balances every new
   connection to it, so nothing about the rerouting behaviour is lost.
3. The pod and node that served each request are recorded as span attributes.
   Counting distinct pods per second is how we later prove the load balancer
   went 3 replicas -> 2 -> 3 again.
"""

import json
import os
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, urlunparse

import requests
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.trace.status import Status, StatusCode

POD_NAME = os.environ.get("POD_NAME", socket.gethostname())
NODE_NAME = os.environ.get("NODE_NAME", "unknown")
PORT = int(os.environ.get("PORT", "8080"))
BACKEND_URL = os.environ.get("BACKEND_URL", "http://order-backend:8000/order")
BACKEND_TIMEOUT = float(os.environ.get("BACKEND_TIMEOUT", "2.0"))
OTLP_ENDPOINT = os.environ.get("OTLP_ENDPOINT", "http://jaeger:4317")

resource = Resource(
    attributes={
        "service.name": "api-gateway",
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


class UpstreamAddress:
    """Resolves the backend Service name to its ClusterIP once and caches it.

    See note 2 in the module docstring: a DNS lookup per request turns every
    dropped UDP packet into a five-second stall. Resolving once removes that
    entire failure class. We re-resolve only if a request fails, which covers
    the (rare) case of the Service being recreated with a new ClusterIP.
    """

    def __init__(self, url):
        self.original = url
        parsed = urlparse(url)
        self.hostname = parsed.hostname
        self.port = parsed.port or 80
        self.parsed = parsed
        self.lock = threading.Lock()
        self.url = None

    def resolve(self, attempts=30):
        """Blocking resolve with retries - CoreDNS may not be ready at startup."""
        for attempt in range(attempts):
            try:
                address = socket.gethostbyname(self.hostname)
            except socket.gaierror as exc:
                if attempt == attempts - 1:
                    raise
                print(f"waiting for DNS to resolve {self.hostname} ({exc})", flush=True)
                time.sleep(2)
                continue

            netloc = f"{address}:{self.port}"
            with self.lock:
                self.url = urlunparse(self.parsed._replace(netloc=netloc))
            print(f"resolved {self.hostname} -> {address} (cached for the process lifetime)",
                  flush=True)
            return self.url

    def current(self):
        with self.lock:
            return self.url

    def invalidate(self):
        """Called after an upstream failure, in case the ClusterIP moved."""
        try:
            self.resolve(attempts=1)
        except socket.gaierror:
            pass


upstream = UpstreamAddress(BACKEND_URL)


class BacklogServer(ThreadingHTTPServer):
    """See the matching note in order_backend.py: the stdlib default listen
    backlog of 5 drops SYNs under this connection rate, which manufactures
    fake timeouts that have nothing to do with the fault we are studying."""

    request_queue_size = 512
    daemon_threads = True


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        """Silence per-request logging."""

    def _respond(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/healthz"):
            self._respond(200, {"ready": True, "pod": POD_NAME})
            return

        if not self.path.startswith("/order"):
            self._respond(404, {"error": "not found"})
            return

        self._handle_order()

    def _handle_order(self):
        with tracer.start_as_current_span("api-gateway.order") as span:
            span.set_attribute("http.method", "GET")
            span.set_attribute("http.route", "/order")
            span.set_attribute("gateway.pod", POD_NAME)

            headers = {"Connection": "close", "Host": upstream.hostname}
            propagator.inject(carrier=headers)

            started = time.time()
            try:
                response = requests.get(
                    upstream.current(), timeout=BACKEND_TIMEOUT, headers=headers
                )
                upstream_ms = (time.time() - started) * 1000
                response.raise_for_status()
                served_by = response.json()

                # These two attributes are the load-balancing evidence.
                span.set_attribute("backend.pod", served_by.get("pod", "unknown"))
                span.set_attribute("backend.node", served_by.get("node", "unknown"))
                span.set_attribute("backend.queue_wait_ms", served_by.get("queue_wait_ms", 0))
                span.set_attribute("http.status_code", response.status_code)
                span.set_attribute("upstream.duration_ms", round(upstream_ms, 2))

                self._respond(200, {"status": "ok", "upstream": served_by})

            except requests.exceptions.RequestException as exc:
                # A dead replica is still a Service endpoint until Kubernetes
                # notices, so these are the requests that get black-holed.
                upstream_ms = (time.time() - started) * 1000
                error_msg = f"Upstream failure calling order-backend: {exc}"
                span.set_attribute("upstream.duration_ms", round(upstream_ms, 2))
                span.add_event("exception", {"exception.message": error_msg})
                span.set_status(Status(StatusCode.ERROR, error_msg))
                if isinstance(exc, requests.exceptions.ConnectionError):
                    upstream.invalidate()
                self._respond(503, {"status": "error", "detail": str(exc)})


if __name__ == "__main__":
    upstream.resolve()
    print(
        f"api-gateway up | pod={POD_NAME} node={NODE_NAME} "
        f"-> {upstream.current()} (timeout {BACKEND_TIMEOUT}s)",
        flush=True,
    )
    BacklogServer(("0.0.0.0", PORT), Handler).serve_forever()
