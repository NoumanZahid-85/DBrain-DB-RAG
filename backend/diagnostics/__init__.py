"""Diagnostic inspectors for live PostgreSQL, MySQL, and MongoDB instances.

Each inspector connects to a running database (read-only credentials) and
returns a structured ``Metrics`` dict that the RCA chain fuses with doc-RAG
retrieval to produce a pinpoint root-cause diagnosis.

Public surface:
    from backend.diagnostics.pg_inspector    import PgInspector
    from backend.diagnostics.mysql_inspector import MysqlInspector
    from backend.diagnostics.mongo_inspector import MongoInspector
    from backend.diagnostics.dispatcher     import DiagnosticDispatcher
"""
