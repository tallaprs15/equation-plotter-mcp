"""Access control: per-caller buckets, forged X-Forwarded-For, API keys."""

import hashlib

from equation_plotter.ratelimit import Access


def test_rate_limit_per_caller_and_forged_forwarding():
    acc = Access(per_min=2, proxy_hops=1)
    h = {"x-forwarded-for": "6.6.6.6, 1.2.3.4"}  # forged entry, then the proxy-added address
    assert acc.check(h, None, costly=True).status == 200
    assert acc.check({"x-forwarded-for": "7.7.7.7, 1.2.3.4"}, None, costly=True).status == 200
    d = acc.check(h, None, costly=True)
    assert d.status == 429 and d.retry_after >= 1
    assert acc.check(h, None, costly=False).status == 200  # discovery is never metered
    assert acc.check({"x-forwarded-for": "5.5.5.5"}, None, costly=True).status == 200


def test_api_keys():
    acc = Access(per_min=1, key_digests=[hashlib.sha256(b"s3cret").hexdigest()])
    assert acc.check({"authorization": "Bearer wrong"}, ("1.1.1.1", 1), True).status == 401
    for _ in range(5):  # keyed callers are unlimited by default
        assert acc.check({"x-api-key": "s3cret"}, ("1.1.1.1", 1), True).status == 200
