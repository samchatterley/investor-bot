# ADR-002: File-Based State vs. SQLite

**Date:** April 2026
**Status:** Accepted

## Context

InvestorBot needs persistent state for several concerns: the current set of open positions, a history of trade decisions and their outcomes, run metadata, and an audit log linking each Claude recommendation to the downstream execution decision. From v1.0 through v1.5 this state was stored as a collection of JSON files on disk, with `fcntl` advisory locks used to serialise concurrent writers.

Several problems emerged with this approach over time:

- `fcntl` locking is POSIX-only. The codebase became non-portable to Windows and some container environments where `fcntl` semantics differ.
- JSON files have no transaction semantics. A process crash mid-write leaves a partially written file that may be syntactically invalid or internally inconsistent. In one real incident during v1.4 operation, a spurious duplicate scheduler run wrote a new `trade_history.json` that overwrote the previous file entirely, silently dropping several completed trade records from the audit log.
- As the decisions dashboard grew in scope, cross-cutting queries (e.g. join all recommendations with their execution outcomes for a given date range) required loading multiple JSON files into memory and joining them in Python, which was slow and brittle.
- File proliferation made backup and restore procedures cumbersome — a consistent snapshot required quiescing all writers and copying several files atomically.

## Decision

Starting in v1.6, all persistent state is stored in a single SQLite database at `logs/investorbot.db`. SQLite provides ACID transactions (eliminating the overwrite race that caused the v1.4 incident), a relational schema that supports JOIN queries across the `audit`, `decisions`, and `runs` tables, and a single-file deployment model that is simpler to back up and restore than a directory of JSON files.

The migration from the old JSON layout to the SQLite schema runs automatically on first startup after upgrade. No manual migration step is required. The JSON files are left in place but are no longer read or written by any component after migration.

## Consequences

**Positive:**
- ACID transactions prevent partial writes and overwrite races. The class of incident that occurred in v1.4 is no longer possible.
- JOIN queries across audit, decisions, and runs tables are now first-class, enabling the decisions dashboard to be implemented entirely in SQL rather than in-memory Python joins.
- Single-file state (`logs/investorbot.db`) is easy to back up atomically using SQLite's online backup API or a simple file copy while the database is idle.
- Portable across platforms — SQLite has no dependency on POSIX `fcntl`.

**Negative:**
- SQLite's write concurrency is limited to one writer at a time. For the current single-host, scheduler-driven architecture this is not a bottleneck, but horizontal scaling to multiple concurrent writers would require migration to a client-server database (PostgreSQL, etc.).
- The schema must be versioned and migration scripts maintained as the data model evolves.
- Developers and operators need basic SQL literacy to inspect and debug state, whereas JSON files were human-readable with any text editor.
