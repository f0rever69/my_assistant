"""Bing web search with a hard wall-clock budget and bounded fallback."""

from __future__ import annotations

import base64
import json
import logging
import math
import queue
import threading
import time
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx
from bs4 import BeautifulSoup
from langchain.tools import tool
from my_assistant.community.search_time_range import (
    DDGS_TIMELIMIT_BY_TIME_RANGE,
    SearchTimeRange,
)
from my_assistant.config import get_app_config

logger = logging.getLogger(__name__)

_BING_SEARCH_URL = "https://www.bing.com/search"
_DEFAULT_TOTAL_TIMEOUT_SECONDS = 20.0
_DEFAULT_PROVIDER_TIMEOUT_SECONDS = 8.0
_DEFAULT_MAX_RETRIES = 1
_MAX_RESULTS = 10
_MIN_ATTEMPT_SECONDS = 0.001
_ALLOWED_PROVIDERS = {"bing", "duckduckgo"}


class SearchProviderTimeout(TimeoutError):
    pass


class SearchProviderError(RuntimeError):
    pass


def _positive_float(value: object, default: float) -> float:
    if isinstance(value, bool):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) and number > 0 else default


def _bounded_int(value: object, default: int, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        return min(maximum, max(minimum, int(value)))
    except (TypeError, ValueError):
        return default


def _tool_settings(requested_max_results: int = 5) -> dict[str, Any]:
    config = get_app_config().get_tool_config("web_search")
    extra = (getattr(config, "model_extra", None) or {}) if config is not None else {}
    requested_max_results = _bounded_int(
        requested_max_results, 5, minimum=1, maximum=_MAX_RESULTS
    )
    fallback = str(extra.get("fallback_provider", "duckduckgo")).strip().lower()
    if fallback not in _ALLOWED_PROVIDERS:
        fallback = "duckduckgo"
    return {
        # A configured value is the deployment default and therefore wins over
        # the generated tool-schema default.  Explicit callers can still use a
        # different value when no deployment override is present.
        "max_results": _bounded_int(
            extra.get("max_results", requested_max_results),
            requested_max_results,
            minimum=1,
            maximum=_MAX_RESULTS,
        ),
        "total_timeout_seconds": _positive_float(
            extra.get("total_timeout_seconds"), _DEFAULT_TOTAL_TIMEOUT_SECONDS
        ),
        "provider_timeout_seconds": _positive_float(
            extra.get("provider_timeout_seconds"), _DEFAULT_PROVIDER_TIMEOUT_SECONDS
        ),
        # This is the number of retries after the first attempt for each
        # provider.  The total deadline remains the hard upper bound.
        "max_retries": _bounded_int(
            extra.get("max_retries", _DEFAULT_MAX_RETRIES),
            _DEFAULT_MAX_RETRIES,
            minimum=0,
            maximum=3,
        ),
        "fallback_provider": fallback,
        "market": str(extra.get("market", "en-US")),
        "safesearch": str(extra.get("safesearch", "moderate")).lower(),
    }


def _call_with_timeout(call, timeout: float):
    """Bound waiting even when a synchronous provider ignores its own timeout."""
    result_queue: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

    def run() -> None:
        try:
            result_queue.put((True, call()))
        except Exception as exc:  # noqa: BLE001 - provider boundary must surface any error
            result_queue.put((False, exc))

    thread = threading.Thread(target=run, name="web-search-provider", daemon=True)
    thread.start()
    thread.join(timeout=max(0.0, timeout))
    if thread.is_alive():
        raise SearchProviderTimeout("Provider exceeded the remaining search deadline")
    succeeded, value = result_queue.get_nowait()
    if not succeeded:
        raise value
    return value


def _unwrap_bing_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.hostname not in {"bing.com", "www.bing.com"} or not parsed.path.startswith("/ck/a"):
        return url
    encoded = parse_qs(parsed.query).get("u", [""])[0]
    if not encoded.startswith("a1"):
        return url
    try:
        payload = encoded[2:]
        payload += "=" * (-len(payload) % 4)
        decoded = base64.urlsafe_b64decode(payload).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return url
    decoded_url = urlparse(decoded)
    return decoded if decoded_url.scheme in {"http", "https"} and decoded_url.netloc else url


def _parse_bing_results(html: str, max_results: int) -> list[dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    results: list[dict[str, str]] = []
    for item in soup.select("li.b_algo"):
        title_link = item.select_one("h2 a[href]")
        if title_link is None:
            continue
        title = title_link.get_text(" ", strip=True)
        url = _unwrap_bing_url(str(title_link.get("href", "")))
        if not title or urlparse(url).scheme not in {"http", "https"}:
            continue
        snippet = item.select_one(".b_caption p") or item.select_one(".b_caption")
        results.append(
            {
                "title": title,
                "url": url,
                "content": snippet.get_text(" ", strip=True) if snippet else "",
                "source_engine": "bing",
            }
        )
        if len(results) >= max_results:
            break
    return results


def _bing_time_filter(time_range: SearchTimeRange | None) -> str | None:
    if time_range is None:
        return None
    limit = DDGS_TIMELIMIT_BY_TIME_RANGE[time_range]
    if limit == "y":
        today = int(time.time() // 86400)
        interval = f"ez5_{today - 365}_{today}"
    else:
        interval = "ez" + {"d": "1", "w": "2", "m": "3"}[limit]
    return f'ex1:"{interval}"'


def _search_bing(
    query: str,
    *,
    max_results: int,
    timeout: float,
    market: str,
    safesearch: str,
    time_range: SearchTimeRange | None,
) -> list[dict[str, str]]:
    params: dict[str, str | int] = {
        "q": query,
        "setmkt": market,
        "adlt": safesearch if safesearch in {"strict", "moderate", "off"} else "moderate",
        "count": max_results,
    }
    time_filter = _bing_time_filter(time_range)
    if time_filter:
        params["filters"] = time_filter

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml",
    }
    with httpx.Client(timeout=httpx.Timeout(timeout), follow_redirects=True, headers=headers) as client:
        response = client.get(_BING_SEARCH_URL, params=params)
    response.raise_for_status()
    html = response.text
    lowered = html.lower()
    if "b_captcha" in lowered or "verify you are human" in lowered or "unusual traffic" in lowered:
        raise SearchProviderError("Bing returned an anti-bot challenge")
    return _parse_bing_results(html, max_results)


def _search_duckduckgo(
    query: str,
    *,
    max_results: int,
    timeout: float,
    market: str,
    safesearch: str,
    time_range: SearchTimeRange | None,
) -> list[dict[str, str]]:
    from ddgs import DDGS

    language, _, country = market.partition("-")
    region = f"{country.lower()}-{language.lower()}" if country else "wt-wt"
    with DDGS(timeout=max(1, math.ceil(timeout))) as client:
        results = client.text(
            query,
            backend="duckduckgo",
            max_results=max_results,
            region=region,
            safesearch=safesearch,
            **({"timelimit": DDGS_TIMELIMIT_BY_TIME_RANGE[time_range]} if time_range else {}),
        )
    return [
        {
            "title": item.get("title", ""),
            "url": item.get("href", item.get("link", "")),
            "content": item.get("body", item.get("snippet", "")),
            "source_engine": "duckduckgo",
        }
        for item in results or []
    ]


def _search_provider(provider: str, query: str, settings: dict[str, Any], time_range):
    search = _search_bing if provider == "bing" else _search_duckduckgo
    return search(
        query,
        max_results=settings["max_results"],
        timeout=settings["provider_timeout_seconds"],
        market=settings["market"],
        safesearch=settings["safesearch"],
        time_range=time_range,
    )


def _is_timeout_error(exc: Exception) -> bool:
    return isinstance(exc, (httpx.TimeoutException, TimeoutError)) or "timeout" in type(exc).__name__.lower() or "timed out" in str(exc).lower()


def _is_no_results_error(exc: Exception) -> bool:
    """Recognize provider-specific empty-result exceptions as a valid response."""
    return "no results found" in str(exc).strip().lower()


def _search_with_fallback(
    query: str,
    settings: dict[str, Any],
    time_range: SearchTimeRange | None,
) -> tuple[list[dict[str, str]], dict[str, Any] | None, list[str]]:
    providers = ["bing"]
    if settings["fallback_provider"] != "bing":
        providers.append(settings["fallback_provider"])

    deadline = time.monotonic() + settings["total_timeout_seconds"]
    attempted: list[str] = []
    failures: list[str] = []
    saw_empty_response = False

    for provider in providers:
        for attempt in range(settings["max_retries"] + 1):
            remaining = deadline - time.monotonic()
            # Do not start a retry with only scheduling noise left in the
            # deadline; it cannot produce a useful response and would create
            # an unnecessary provider thread.
            if remaining <= _MIN_ATTEMPT_SECONDS:
                failures.append("timeout")
                break

            attempted.append(provider)
            attempt_timeout = min(settings["provider_timeout_seconds"], remaining)
            settings_for_call = dict(settings)
            settings_for_call["provider_timeout_seconds"] = attempt_timeout
            try:
                results = _call_with_timeout(
                    lambda provider=provider, settings_for_call=settings_for_call: _search_provider(
                        provider, query, settings_for_call, time_range
                    ),
                    timeout=attempt_timeout,
                )
            except Exception as exc:  # noqa: BLE001 - classify provider failures
                if _is_no_results_error(exc):
                    saw_empty_response = True
                    break
                logger.warning(
                    "Web search provider %s attempt %s failed (%s)",
                    provider,
                    attempt + 1,
                    type(exc).__name__,
                )
                failures.append("timeout" if _is_timeout_error(exc) else "provider_error")
                # A retry gets the same hard deadline; if there is no time left
                # the next loop iteration records timeout and exits.
                continue
            if results:
                return results, None, attempted
            # An empty, successful provider response is a real no-results signal.
            # Give the alternate provider a chance before returning that signal.
            saw_empty_response = True
            break

    if failures:
        timed_out = "timeout" in failures
        return [], {
            "error": (
                "Web search provider timed out before returning results."
                if timed_out
                else "Web search providers failed before returning results."
            ),
            "error_type": "timeout" if timed_out else "provider_unavailable",
            "query": query,
            "providers_tried": attempted,
        }, attempted
    if saw_empty_response:
        return [], {
            "error": "No results found",
            "error_type": "no_results",
            "query": query,
            "providers_tried": attempted,
        }, attempted
    return [], {
        "error": "Web search deadline expired before a provider could run.",
        "error_type": "timeout",
        "query": query,
        "providers_tried": attempted,
    }, attempted


@tool("web_search", parse_docstring=True)
def web_search_tool(
    query: str,
    max_results: int = 5,
    time_range: SearchTimeRange | None = None,
) -> str:
    """Search the web with Bing, with a bounded DuckDuckGo fallback.

    Args:
        query: Search keywords describing what you want to find.
        max_results: Maximum number of results to return. Default is 5.
        time_range: Optional relative publication/update window.
    """
    settings = _tool_settings(max_results)
    results, error, attempted = _search_with_fallback(query, settings, time_range)
    if error is not None:
        return json.dumps(error, ensure_ascii=False)
    return json.dumps(
        {
            "query": query,
            "total_results": len(results),
            "results": results,
            "providers_tried": attempted,
        },
        indent=2,
        ensure_ascii=False,
    )
