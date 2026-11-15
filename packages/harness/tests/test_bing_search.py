from __future__ import annotations

import json
import time
from types import SimpleNamespace

from my_assistant.client import MyAssistantClient
from my_assistant.community.bing_search import tools as search_tools
from my_assistant.runtime.step_limit import format_step_limit_error
from my_assistant.tui.runtime import translate
from my_assistant.tui.view_state import AssistantError, RunEnded


def _settings(**overrides):
    settings = {
        "max_results": 5,
        "total_timeout_seconds": 1.0,
        "provider_timeout_seconds": 0.5,
        "max_retries": 1,
        "fallback_provider": "duckduckgo",
        "market": "en-US",
        "safesearch": "moderate",
    }
    settings.update(overrides)
    return settings


def test_retries_provider_then_fails_over(monkeypatch):
    calls = {"bing": 0, "duckduckgo": 0}

    def search(provider, *_args, **_kwargs):
        calls[provider] += 1
        if provider == "bing":
            raise search_tools.SearchProviderError("provider unavailable")
        return [{"title": "result", "source_engine": provider}]

    monkeypatch.setattr(search_tools, "_search_provider", search)

    results, error, attempted = search_tools._search_with_fallback(
        "query", _settings(), None
    )

    assert error is None
    assert results[0]["source_engine"] == "duckduckgo"
    assert calls == {"bing": 2, "duckduckgo": 1}
    assert attempted == ["bing", "bing", "duckduckgo"]


def test_zero_retries_still_fails_over_to_fallback(monkeypatch):
    attempted_calls = []

    def search(provider, *_args, **_kwargs):
        attempted_calls.append(provider)
        if provider == "bing":
            raise search_tools.SearchProviderError("unavailable")
        return [{"title": "result", "source_engine": provider}]

    monkeypatch.setattr(search_tools, "_search_provider", search)

    results, error, attempted = search_tools._search_with_fallback(
        "query", _settings(max_retries=0), None
    )

    assert error is None
    assert results
    assert attempted_calls == ["bing", "duckduckgo"]
    assert attempted == attempted_calls


def test_empty_results_are_distinct_from_provider_failures(monkeypatch):
    monkeypatch.setattr(search_tools, "_search_provider", lambda *_args, **_kwargs: [])

    results, error, attempted = search_tools._search_with_fallback(
        "query", _settings(), None
    )

    assert results == []
    assert error["error_type"] == "no_results"
    assert attempted == ["bing", "duckduckgo"]
    assert error["providers_tried"] == attempted


def test_provider_no_results_exception_is_not_retried(monkeypatch):
    calls = []

    def search(provider, *_args, **_kwargs):
        calls.append(provider)
        raise search_tools.SearchProviderError("No results found")

    monkeypatch.setattr(search_tools, "_search_provider", search)

    _results, error, attempted = search_tools._search_with_fallback(
        "query", _settings(), None
    )

    assert error["error_type"] == "no_results"
    assert calls == ["bing", "duckduckgo"]
    assert attempted == calls


def test_provider_timeout_returns_within_total_deadline(monkeypatch):
    def slow_search(*_args, **_kwargs):
        time.sleep(0.3)
        return []

    monkeypatch.setattr(search_tools, "_search_provider", slow_search)
    started = time.monotonic()

    _results, error, attempted = search_tools._search_with_fallback(
        "query",
        _settings(total_timeout_seconds=0.05, provider_timeout_seconds=0.5),
        None,
    )
    elapsed = time.monotonic() - started

    assert error["error_type"] == "timeout"
    assert attempted and attempted[0] == "bing"
    assert len(attempted) <= 2
    assert elapsed < 0.2


def test_provider_failure_is_not_reported_as_no_results(monkeypatch):
    def search(provider, *_args, **_kwargs):
        if provider == "bing":
            raise search_tools.SearchProviderError("HTTP 503")
        return []

    monkeypatch.setattr(search_tools, "_search_provider", search)

    _results, error, _attempted = search_tools._search_with_fallback(
        "query", _settings(max_retries=0), None
    )

    assert error["error_type"] == "provider_unavailable"


def test_tool_output_reports_all_attempted_providers(monkeypatch):
    monkeypatch.setattr(
        search_tools,
        "get_app_config",
        lambda: SimpleNamespace(
            get_tool_config=lambda _name: SimpleNamespace(model_extra={})
        ),
    )

    def search(provider, *_args, **_kwargs):
        if provider == "bing":
            raise search_tools.SearchProviderError("HTTP 503")
        return [{"title": "fallback", "source_engine": provider}]

    monkeypatch.setattr(search_tools, "_search_provider", search)
    payload = json.loads(search_tools.web_search_tool.invoke({"query": "query"}))

    assert payload["providers_tried"] == ["bing", "bing", "duckduckgo"]
    assert payload["results"][0]["source_engine"] == "duckduckgo"


def test_configured_max_results_overrides_tool_schema_default(monkeypatch):
    monkeypatch.setattr(
        search_tools,
        "get_app_config",
        lambda: SimpleNamespace(
            get_tool_config=lambda _name: SimpleNamespace(
                model_extra={"max_results": 7}
            )
        ),
    )

    assert search_tools._tool_settings(5)["max_results"] == 7


def test_bing_parser_extracts_title_url_and_snippet():
    results = search_tools._parse_bing_results(
        """
        <ol>
          <li class="b_algo">
            <h2><a href="https://example.com/article">Example article</a></h2>
            <div class="b_caption"><p>Useful summary text.</p></div>
          </li>
        </ol>
        """,
        5,
    )

    assert results == [
        {
            "title": "Example article",
            "url": "https://example.com/article",
            "content": "Useful summary text.",
            "source_engine": "bing",
        }
    ]


def test_step_limit_is_rendered_as_partial_failure_in_tui_and_chat():
    explanation = format_step_limit_error(12)
    actions = translate(
        SimpleNamespace(type="end", data={"error": explanation, "error_type": "step_limit"})
    )

    assert isinstance(actions[0], AssistantError)
    assert isinstance(actions[1], RunEnded)
    assert "step limit (12)" in explanation
    assert "intermediate messages and tool results" in explanation

    client = MyAssistantClient.__new__(MyAssistantClient)
    client.stream = lambda *_args, **_kwargs: iter(
        [
            SimpleNamespace(
                type="messages-tuple",
                data={"type": "ai", "id": "partial", "content": "Partial answer."},
            ),
            SimpleNamespace(
                type="end",
                data={"error": explanation, "error_type": "step_limit"},
            ),
        ]
    )

    response = client.chat("request")
    assert response.startswith("Partial answer.")
    assert explanation in response

