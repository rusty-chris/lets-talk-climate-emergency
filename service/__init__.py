"""FastAPI service: thin routes, budget tracker, rate limiter, log redaction.

See DESIGN.md §9 and IMPLEMENTATION.md §1 for the module map. Modules land
here in later issues (#14, #15, #22); this package currently exists so the
scaffolding's import contract (issue #1) is enforced from day one.

`service.main` provides the stub FastAPI app (`/health`) that backs the
`api` docker-compose service until the real service lands.
"""
