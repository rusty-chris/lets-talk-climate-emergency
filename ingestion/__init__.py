"""Ingestion pipeline: manifest validation, licensing gate, fetch, parse, chunk.

See DESIGN.md §2 and IMPLEMENTATION.md §1 for the module map. Modules land
here in later issues (#5-#9); this package currently exists so the
scaffolding's import contract (issue #1) is enforced from day one.
"""
