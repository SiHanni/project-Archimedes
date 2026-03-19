"""Phase 7 — OpenTelemetry hook (no-op unless endpoint set)."""

from __future__ import annotations

import logging

from app.config import Settings

log = logging.getLogger(__name__)


def maybe_init_otel(settings: Settings) -> None:
    ep = (settings.otel_exporter_otlp_endpoint or "").strip()
    if not ep:
        return
    try:
        # Lazy import so base install stays light
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create({"service.name": "archimedes-worker"})
        provider = TracerProvider(resource=resource)
        processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=ep, insecure=True))
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)
        log.info("OTLP exporter enabled: %s", ep)
    except ImportError:
        log.warning("OTEL packages not installed; pip install opentelemetry-exporter-otlp")
    except Exception as e:  # noqa: BLE001
        log.warning("OTEL init failed: %s", e)
