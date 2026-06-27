"""Inject a bounded request timeout into alpaca-py clients.

alpaca-py (0.43.x) builds a plain `requests.Session` with NO timeout, never sets one on its
requests, and its client `__init__` exposes no timeout option — so any broker/data call can block
indefinitely on a hung socket. Because the scheduler runs jobs sequentially, one hung Alpaca call
freezes the ENTIRE scheduler (the same failure mode that hit the Anthropic call → 1.124).

`with_request_timeout` patches a client's underlying session so every request defaults to a bounded
(connect, read) timeout. On a hang the session then raises `requests.Timeout`, which propagates to
the run-level handler → the run aborts cleanly and the scheduler frees up for the next job.
"""

from __future__ import annotations

# (connect, read) seconds. Alpaca calls normally return in <1s; a 30s read is generous headroom
# while still bounding a true hang, and 10s fast-fails a dead connect.
_DEFAULT_TIMEOUT: tuple[float, float] = (10.0, 30.0)


def with_request_timeout[T](client: T, timeout: tuple[float, float] = _DEFAULT_TIMEOUT) -> T:
    """Patch an alpaca-py client's session so every request carries a default timeout.

    Returns the client unchanged if it exposes no patchable `_session` — so a future alpaca-py
    refactor degrades gracefully instead of crashing. The guard test asserts the patch actually
    takes effect on the pinned version, so this never silently becomes a no-op in production.
    """
    session = getattr(client, "_session", None)
    if session is None or not hasattr(session, "request"):
        return client

    _orig_request = session.request

    def _request_with_timeout(method, url, **kwargs):
        kwargs.setdefault("timeout", timeout)
        return _orig_request(method, url, **kwargs)

    session.request = _request_with_timeout
    return client
