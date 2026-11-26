"""Access control for the public HTTP server: optional API keys and per-caller rate limits.

Only tool calls are metered; discovery and listing requests are free. Configuration is by
environment variable, and everything is off by default (local use needs none of it):

* ``PLOTTER_MCP_RATE_PER_MIN``: tool calls a caller may make per minute (token-bucket refill);
  ``0`` disables rate limiting.
* ``PLOTTER_MCP_RATE_BURST``: bucket size, i.e. how many calls may arrive at once
  (default: the per-minute rate).
* ``PLOTTER_MCP_API_KEYS_SHA256``: comma-separated SHA-256 hex digests of API keys. A caller
  presenting a listed key (``Authorization: Bearer <key>`` or ``X-API-Key: <key>``) is
  identified by the key and gets ``PLOTTER_MCP_KEY_RATE_PER_MIN`` (default 0 = unlimited).
  An unknown key is rejected rather than silently downgraded.
* ``PLOTTER_MCP_TRUSTED_PROXY_HOPS``: how many proxies in front of the server append to
  ``X-Forwarded-For`` (Cloud Run: 1). The caller's address is taken that many entries from
  the right, so a forged header cannot change it. ``0`` uses the socket peer.

State is in memory, so limits are per server instance.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import threading
import time
from dataclasses import dataclass

_MAX_TRACKED = 20_000
_SALT = secrets.token_bytes(16)  # log tags are not linkable across restarts


@dataclass
class Decision:
    status: int  # 200 allow, 401 bad key, 429 rate limited
    caller_tag: str
    retry_after: int = 0


class _Buckets:
    def __init__(self, per_min: float, burst: float):
        self.rate = per_min / 60.0
        self.burst = burst
        self._state: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def take(self, caller: str) -> int:
        """Spend one token; return 0 if allowed, else seconds until a token is available."""
        now = time.monotonic()
        with self._lock:
            tokens, stamp = self._state.get(caller, (self.burst, now))
            tokens = min(self.burst, tokens + (now - stamp) * self.rate)
            if tokens >= 1:
                self._state[caller] = (tokens - 1, now)
                allowed = 0
            else:
                self._state[caller] = (tokens, now)
                allowed = max(1, int((1 - tokens) / self.rate + 0.999))
            if len(self._state) > _MAX_TRACKED:  # drop callers whose buckets are full again
                full = [k for k, (t, s) in self._state.items()
                        if t + (now - s) * self.rate >= self.burst]
                for k in full:
                    del self._state[k]
            return allowed


class Access:
    def __init__(self, per_min=0.0, burst=None, key_digests=(), key_per_min=0.0, proxy_hops=0):
        self.anon = _Buckets(per_min, burst or per_min) if per_min > 0 else None
        self.keyed = _Buckets(key_per_min, key_per_min) if key_per_min > 0 else None
        self.keys = {d.strip().lower() for d in key_digests if d.strip()}
        self.proxy_hops = proxy_hops

    @classmethod
    def from_env(cls, env=os.environ) -> "Access":
        burst = env.get("PLOTTER_MCP_RATE_BURST")
        return cls(per_min=float(env.get("PLOTTER_MCP_RATE_PER_MIN", "0")),
                   burst=float(burst) if burst else None,
                   key_digests=env.get("PLOTTER_MCP_API_KEYS_SHA256", "").split(","),
                   key_per_min=float(env.get("PLOTTER_MCP_KEY_RATE_PER_MIN", "0")),
                   proxy_hops=int(env.get("PLOTTER_MCP_TRUSTED_PROXY_HOPS", "0")))

    def client_ip(self, headers: dict, peer) -> str:
        if self.proxy_hops > 0:
            hops = [h.strip() for h in headers.get("x-forwarded-for", "").split(",") if h.strip()]
            if len(hops) >= self.proxy_hops:
                return hops[-self.proxy_hops]
        return peer[0] if peer else "unknown"

    @staticmethod
    def _presented_key(headers: dict) -> str | None:
        auth = headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            return auth[7:].strip() or None
        return headers.get("x-api-key", "").strip() or None

    def check(self, headers: dict, peer, costly: bool) -> Decision:
        key = self._presented_key(headers)
        if key is not None:
            digest = hashlib.sha256(key.encode()).hexdigest()
            if not any(hmac.compare_digest(digest, d) for d in self.keys):
                return Decision(401, "key:invalid")
            caller, buckets = "key:" + digest, self.keyed
        else:
            caller, buckets = "ip:" + self.client_ip(headers, peer), self.anon
        tag = caller.split(":")[0] + ":" + hashlib.sha256(_SALT + caller.encode()).hexdigest()[:10]
        if costly and buckets is not None:
            wait = buckets.take(caller)
            if wait:
                return Decision(429, tag, wait)
        return Decision(200, tag)
