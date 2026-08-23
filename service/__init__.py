"""FastAPI service: thin routes, budget tracker, rate limiter, log redaction.

See DESIGN.md §9 and IMPLEMENTATION.md §1 for the module map: thin routes
(`service.app`), the clock-injected budget tracker (`service.budget`), the
rotating-salt rate limiter (`service.rate_limit`), privacy-compliant
exchange logging (`service.exchange_log`), the read-only starter cache
(`service.starter_cache`) and the chart permalink store
(`service.chart_store`).

`service.main` is the composition root: it reads the environment, builds the
real dependencies and serves the app via the ASGI factory
`create_service_app` (backing the `api` docker-compose service). Importing
`service.*` never loads torch/docling/fitz (issue #125).
"""
