# One image serves both roles (api-gateway and order-backend); the Kubernetes
# manifests choose which script to run via the container command.
FROM python:3.12-slim

# Pinned to match requirements.txt so container and host agree on OTel versions.
RUN pip install --no-cache-dir \
    opentelemetry-api==1.42.1 \
    opentelemetry-sdk==1.42.1 \
    opentelemetry-exporter-otlp==1.42.1 \
    requests==2.34.2

WORKDIR /app
COPY order_backend.py api_gateway.py ./

# Unbuffered output so `kubectl logs` shows lines as they happen.
ENV PYTHONUNBUFFERED=1

CMD ["python", "order_backend.py"]
