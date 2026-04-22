#!/usr/bin/env python3
"""
benchmark_models.py — Model benchmarking framework for Atwater.

Measures JSON reliability, tool call accuracy, response latency, and
multi-turn coherence for local models served by LM Studio.

Usage:
    python tools/benchmark_models.py --model qwen3-8b --url http://localhost:1234/v1
    python tools/benchmark_models.py --model qwen3-4b --url http://localhost:1234/v1 --suite json,latency
    python tools/benchmark_models.py --model qwen3-8b --url http://localhost:1234/v1 --output report.md

NOTE: This is the FRAMEWORK. It requires a running LM Studio server with the
target model loaded. Without LM Studio, the benchmarks will fail with
connection errors. The structure and test logic are complete.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    """Result from a single benchmark test."""
    name: str
    model: str
    score: float
    details: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0


@dataclass
class SuiteResult:
    """Aggregate results from a full benchmark suite."""
    model: str
    url: str
    results: list[BenchmarkResult] = field(default_factory=list)
    total_duration: float = 0.0


# ---------------------------------------------------------------------------
# LM Studio client (minimal, no external deps beyond stdlib + openai)
# ---------------------------------------------------------------------------


class LMStudioClient:
    """
    Minimal OpenAI-compatible client for LM Studio.

    Uses the `openai` package if available, falls back to raw HTTP requests.
    """

    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = None
        try:
            from openai import OpenAI
            self._client = OpenAI(base_url=self.base_url, api_key="lm-studio")
        except ImportError:
            pass

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        response_format: dict | None = None,
        tools: list[dict] | None = None,
    ) -> dict[str, Any]:
        """
        Send a chat completion request and return the parsed response.

        Returns a dict with keys: content, finish_reason, usage, latency_ms.
        """
        if self._client is None:
            raise RuntimeError(
                "openai package not installed. Run: pip install openai"
            )

        start = time.perf_counter()

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        if tools is not None:
            kwargs["tools"] = tools

        response = self._client.chat.completions.create(**kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000

        choice = response.choices[0]
        return {
            "content": choice.message.content or "",
            "finish_reason": choice.finish_reason,
            "tool_calls": (
                [tc.model_dump() for tc in choice.message.tool_calls]
                if choice.message.tool_calls
                else []
            ),
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            },
            "latency_ms": elapsed_ms,
        }


# ---------------------------------------------------------------------------
# Benchmark Suite
# ---------------------------------------------------------------------------

# Sample JSON schemas for testing structured output
GRADER_SCHEMA = {
    "type": "object",
    "properties": {
        "overall_score": {"type": "number", "minimum": 0, "maximum": 1},
        "dimensions": {
            "type": "object",
            "properties": {
                "originality": {
                    "type": "object",
                    "properties": {
                        "score": {"type": "number"},
                        "reasoning": {"type": "string"},
                    },
                    "required": ["score", "reasoning"],
                },
                "quality": {
                    "type": "object",
                    "properties": {
                        "score": {"type": "number"},
                        "reasoning": {"type": "string"},
                    },
                    "required": ["score", "reasoning"],
                },
            },
            "required": ["originality", "quality"],
        },
        "novel_finding": {"type": ["string", "null"]},
        "suggest_knowledge_write": {"type": "boolean"},
    },
    "required": ["overall_score", "dimensions", "suggest_knowledge_write"],
}

DIRECTOR_SCHEMA = {
    "type": "object",
    "properties": {
        "proposed_hypothesis": {
            "type": "object",
            "properties": {
                "background": {"type": "string"},
                "layout": {"type": "string"},
                "reasoning": {"type": "string"},
            },
            "required": ["background", "layout", "reasoning"],
        },
    },
    "required": ["proposed_hypothesis"],
}

# Sample tool definitions for testing tool calling
SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "knowledge_read",
            "description": "Search the knowledge base for relevant entries",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "tier": {
                        "type": "string",
                        "enum": ["rule", "pattern", "observation"],
                        "description": "Filter by tier",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Max results",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "state_write",
            "description": "Write a value to shared state",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "value": {},
                },
                "required": ["key", "value"],
            },
        },
    },
]

# Sample tool call test cases: (prompt, expected_tool, expected_args_subset)
TOOL_CALL_EXAMPLES = [
    (
        "Search the knowledge base for font performance data",
        "knowledge_read",
        {"query": "font performance"},
    ),
    (
        "Look up rules about typography in the knowledge base",
        "knowledge_read",
        {"query": "typography", "tier": "rule"},
    ),
    (
        "Save the current hypothesis to shared state with key 'proposed_hypothesis'",
        "state_write",
        {"key": "proposed_hypothesis"},
    ),
    (
        "Find observations about dark backgrounds",
        "knowledge_read",
        {"query": "dark backgrounds", "tier": "observation"},
    ),
    (
        "Write the workflow state to shared state",
        "state_write",
        {"key": "workflow_state"},
    ),
]


class BenchmarkSuite:
    """
    Model benchmarking suite for Atwater agent roles.

    Tests structured output reliability, tool calling accuracy, response
    latency, and multi-turn coherence.
    """

    def __init__(self, client: LMStudioClient) -> None:
        self.client = client

    # ------------------------------------------------------------------
    # Test: JSON reliability
    # ------------------------------------------------------------------

    def test_json_reliability(
        self,
        schema: dict,
        n: int = 100,
        schema_name: str = "grader",
    ) -> BenchmarkResult:
        """
        Test how reliably the model produces valid JSON matching a schema.

        Sends n requests with response_format set to the schema and counts
        successful parses.

        Args:
            schema: JSON Schema to enforce.
            n: Number of test iterations.
            schema_name: Human name for the schema (for reporting).

        Returns:
            BenchmarkResult with score = parse_success_rate (0.0 to 1.0).
        """
        successes = 0
        failures = 0
        errors: list[str] = []
        latencies: list[float] = []

        prompts = [
            "Score this creative output: A minimalist dark background with sans-serif headline.",
            "Evaluate this design: Gradient overlay on product hero shot with large typography.",
            "Rate this creative: Split layout with lifestyle photography and serif body text.",
            "Assess quality: Grid layout with 4 product angles and monospace captions.",
            "Judge this output: Full-bleed textured background with floating CTA button.",
        ]

        start = time.perf_counter()

        for i in range(n):
            prompt = prompts[i % len(prompts)]
            try:
                resp = self.client.chat(
                    messages=[
                        {"role": "system", "content": f"You are a creative grader. Respond with JSON matching the {schema_name} schema."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {"name": schema_name, "schema": schema},
                    },
                )
                # Try parsing the response as JSON
                parsed = json.loads(resp["content"])
                # Validate required keys exist
                required = schema.get("required", [])
                if all(k in parsed for k in required):
                    successes += 1
                else:
                    failures += 1
                    missing = [k for k in required if k not in parsed]
                    errors.append(f"iter {i}: missing keys {missing}")
                latencies.append(resp["latency_ms"])
            except json.JSONDecodeError as e:
                failures += 1
                errors.append(f"iter {i}: JSON parse error: {e}")
            except Exception as e:
                failures += 1
                errors.append(f"iter {i}: {type(e).__name__}: {e}")

        duration = time.perf_counter() - start
        rate = successes / n if n > 0 else 0.0

        return BenchmarkResult(
            name=f"json_reliability_{schema_name}",
            model=self.client.model,
            score=rate,
            details={
                "successes": successes,
                "failures": failures,
                "total": n,
                "mean_latency_ms": statistics.mean(latencies) if latencies else 0,
                "p95_latency_ms": (
                    sorted(latencies)[int(len(latencies) * 0.95)]
                    if latencies
                    else 0
                ),
            },
            errors=errors[:10],  # cap error log
            duration_seconds=duration,
        )

    # ------------------------------------------------------------------
    # Test: Tool call accuracy
    # ------------------------------------------------------------------

    def test_tool_call_accuracy(
        self,
        examples: list[tuple[str, str, dict]] | None = None,
        n: int = 50,
    ) -> BenchmarkResult:
        """
        Test tool calling accuracy.

        For each example, checks that the model:
        1. Calls the correct tool (by name)
        2. Includes the expected argument subset

        Args:
            examples: List of (prompt, expected_tool_name, expected_args_subset).
            n: Total iterations (cycles through examples).

        Returns:
            BenchmarkResult with score = accuracy (0.0 to 1.0).
        """
        if examples is None:
            examples = TOOL_CALL_EXAMPLES

        correct = 0
        total = 0
        errors: list[str] = []

        start = time.perf_counter()

        for i in range(n):
            prompt, expected_tool, expected_args = examples[i % len(examples)]
            total += 1

            try:
                resp = self.client.chat(
                    messages=[
                        {"role": "system", "content": "You are an Atwater agent. Use the available tools to accomplish the task."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    tools=SAMPLE_TOOLS,
                )

                tool_calls = resp.get("tool_calls", [])
                if not tool_calls:
                    errors.append(f"iter {i}: no tool call made")
                    continue

                tc = tool_calls[0]
                fn_name = tc.get("function", {}).get("name", "")
                fn_args_raw = tc.get("function", {}).get("arguments", "{}")

                try:
                    fn_args = json.loads(fn_args_raw) if isinstance(fn_args_raw, str) else fn_args_raw
                except json.JSONDecodeError:
                    fn_args = {}

                # Check tool name
                if fn_name != expected_tool:
                    errors.append(
                        f"iter {i}: expected tool '{expected_tool}', got '{fn_name}'"
                    )
                    continue

                # Check argument subset
                args_match = all(
                    k in fn_args for k in expected_args
                )
                if args_match:
                    correct += 1
                else:
                    missing = [k for k in expected_args if k not in fn_args]
                    errors.append(f"iter {i}: missing args {missing}")

            except Exception as e:
                errors.append(f"iter {i}: {type(e).__name__}: {e}")

        duration = time.perf_counter() - start
        accuracy = correct / total if total > 0 else 0.0

        return BenchmarkResult(
            name="tool_call_accuracy",
            model=self.client.model,
            score=accuracy,
            details={"correct": correct, "total": total},
            errors=errors[:10],
            duration_seconds=duration,
        )

    # ------------------------------------------------------------------
    # Test: Response latency
    # ------------------------------------------------------------------

    def test_response_latency(
        self,
        prompt: str = "Score this creative: dark background, hero layout, sans-serif headline.",
        n: int = 20,
    ) -> BenchmarkResult:
        """
        Measure response latency statistics.

        Args:
            prompt: Fixed prompt for consistent measurement.
            n: Number of iterations.

        Returns:
            BenchmarkResult with details containing mean, p50, p95, p99.
        """
        latencies: list[float] = []
        errors: list[str] = []

        start = time.perf_counter()

        for i in range(n):
            try:
                resp = self.client.chat(
                    messages=[
                        {"role": "system", "content": "You are a creative grader. Respond with a brief JSON score."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=256,
                )
                latencies.append(resp["latency_ms"])
            except Exception as e:
                errors.append(f"iter {i}: {type(e).__name__}: {e}")

        duration = time.perf_counter() - start

        if not latencies:
            return BenchmarkResult(
                name="response_latency",
                model=self.client.model,
                score=0.0,
                details={"error": "no successful responses"},
                errors=errors,
                duration_seconds=duration,
            )

        sorted_lat = sorted(latencies)
        n_lat = len(sorted_lat)

        stats = {
            "mean_ms": statistics.mean(sorted_lat),
            "p50_ms": sorted_lat[n_lat // 2],
            "p95_ms": sorted_lat[int(n_lat * 0.95)],
            "p99_ms": sorted_lat[int(n_lat * 0.99)],
            "min_ms": sorted_lat[0],
            "max_ms": sorted_lat[-1],
            "stdev_ms": statistics.stdev(sorted_lat) if n_lat > 1 else 0,
            "samples": n_lat,
        }

        # Score: inverse of mean latency (lower is better), normalised
        # against a 5-second baseline
        score = max(0.0, min(1.0, 1.0 - (stats["mean_ms"] / 5000.0)))

        return BenchmarkResult(
            name="response_latency",
            model=self.client.model,
            score=score,
            details=stats,
            errors=errors,
            duration_seconds=duration,
        )

    # ------------------------------------------------------------------
    # Test: Multi-turn coherence
    # ------------------------------------------------------------------

    def test_multi_turn_coherence(
        self,
        turns: int = 10,
        n: int = 5,
    ) -> BenchmarkResult:
        """
        Test multi-turn conversation coherence.

        Runs n conversations of `turns` messages each, checking that the
        model maintains context and doesn't contradict itself.

        Args:
            turns: Number of turns per conversation.
            n: Number of conversations to run.

        Returns:
            BenchmarkResult with score = coherence rate (0.0 to 1.0).
        """
        coherent_conversations = 0
        errors: list[str] = []

        system_msg = {
            "role": "system",
            "content": (
                "You are an Atwater Director agent. You are selecting parameter "
                "combinations for creative production. Remember all previous "
                "decisions. If asked about a previous choice, you must be "
                "consistent with what you said before. Respond in JSON with "
                'a "decision" key and a "reasoning" key.'
            ),
        }

        turn_prompts = [
            "Choose a background style for our next creative. Options: dark, gradient, minimal, textured.",
            "What background did you just choose? Repeat it exactly.",
            "Now choose a layout. Options: hero, split, grid, asymmetric.",
            "What layout did you just choose? And what background did you choose earlier?",
            "Rate your confidence in this combination from 0-10.",
            "What was your confidence rating?",
            "Should we try a different combination? Why or why not?",
            "Summarise all your decisions so far.",
            "If the grader gives a low score, what would you change first?",
            "Final summary: list every decision you made in this conversation.",
        ]

        start = time.perf_counter()

        for conv_idx in range(n):
            messages = [system_msg]
            conversation_coherent = True

            for turn_idx in range(min(turns, len(turn_prompts))):
                messages.append({"role": "user", "content": turn_prompts[turn_idx]})

                try:
                    resp = self.client.chat(
                        messages=messages,
                        temperature=0.1,
                        max_tokens=512,
                    )
                    content = resp["content"]
                    messages.append({"role": "assistant", "content": content})

                    # Basic coherence check: recall questions should mention
                    # the same terms as earlier answers
                    if turn_idx in (1, 3, 5):  # recall turns
                        prev_answer = messages[-3]["content"]  # the answer being recalled
                        # Very simple check: at least one word from prev answer appears
                        prev_words = set(prev_answer.lower().split())
                        curr_words = set(content.lower().split())
                        overlap = prev_words & curr_words - {"the", "a", "an", "is", "was", "and", "or", "i", "my"}
                        if len(overlap) < 2:
                            conversation_coherent = False
                            errors.append(
                                f"conv {conv_idx} turn {turn_idx}: "
                                f"poor recall (overlap={len(overlap)} words)"
                            )

                except Exception as e:
                    conversation_coherent = False
                    errors.append(f"conv {conv_idx} turn {turn_idx}: {e}")
                    break

            if conversation_coherent:
                coherent_conversations += 1

        duration = time.perf_counter() - start
        score = coherent_conversations / n if n > 0 else 0.0

        return BenchmarkResult(
            name="multi_turn_coherence",
            model=self.client.model,
            score=score,
            details={
                "coherent_conversations": coherent_conversations,
                "total_conversations": n,
                "turns_per_conversation": turns,
            },
            errors=errors[:10],
            duration_seconds=duration,
        )


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_report(suite_result: SuiteResult) -> str:
    """
    Generate a markdown report from benchmark results.

    Args:
        suite_result: The aggregate results from a benchmark run.

    Returns:
        Formatted markdown string.
    """
    lines = [
        f"# Atwater Model Benchmark Report",
        f"",
        f"**Model:** {suite_result.model}",
        f"**Server:** {suite_result.url}",
        f"**Total Duration:** {suite_result.total_duration:.1f}s",
        f"",
        f"---",
        f"",
        f"## Summary",
        f"",
        f"| Test | Score | Duration |",
        f"|------|-------|----------|",
    ]

    for r in suite_result.results:
        score_str = f"{r.score:.1%}" if r.score <= 1.0 else f"{r.score:.2f}"
        lines.append(f"| {r.name} | {score_str} | {r.duration_seconds:.1f}s |")

    lines.append("")

    for r in suite_result.results:
        lines.append(f"## {r.name}")
        lines.append("")
        lines.append(f"**Score:** {r.score:.4f}")
        lines.append("")

        if r.details:
            lines.append("### Details")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(r.details, indent=2))
            lines.append("```")
            lines.append("")

        if r.errors:
            lines.append(f"### Errors ({len(r.errors)} shown)")
            lines.append("")
            for err in r.errors:
                lines.append(f"- {err}")
            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark LM Studio models for Atwater agent roles.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name as loaded in LM Studio.",
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:1234/v1",
        help="LM Studio base URL.",
    )
    parser.add_argument(
        "--suite",
        type=str,
        default="json,tools,latency,coherence",
        help="Comma-separated list of tests to run: json,tools,latency,coherence",
    )
    parser.add_argument(
        "--json-n",
        type=int,
        default=100,
        help="Number of iterations for JSON reliability test.",
    )
    parser.add_argument(
        "--tools-n",
        type=int,
        default=50,
        help="Number of iterations for tool call accuracy test.",
    )
    parser.add_argument(
        "--latency-n",
        type=int,
        default=20,
        help="Number of iterations for latency test.",
    )
    parser.add_argument(
        "--coherence-turns",
        type=int,
        default=10,
        help="Turns per conversation for coherence test.",
    )
    parser.add_argument(
        "--coherence-n",
        type=int,
        default=5,
        help="Number of conversations for coherence test.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write markdown report (prints to stdout if omitted).",
    )

    args = parser.parse_args(argv)
    tests = {t.strip() for t in args.suite.split(",")}

    client = LMStudioClient(base_url=args.url, model=args.model)
    suite = BenchmarkSuite(client)

    suite_result = SuiteResult(model=args.model, url=args.url)
    total_start = time.perf_counter()

    if "json" in tests:
        print(f"Running JSON reliability test (n={args.json_n})...", file=sys.stderr)
        result = suite.test_json_reliability(
            schema=GRADER_SCHEMA, n=args.json_n, schema_name="grader"
        )
        suite_result.results.append(result)
        print(f"  → Score: {result.score:.1%}", file=sys.stderr)

        # Also test Director schema
        result2 = suite.test_json_reliability(
            schema=DIRECTOR_SCHEMA, n=args.json_n // 2, schema_name="director"
        )
        suite_result.results.append(result2)
        print(f"  → Director: {result2.score:.1%}", file=sys.stderr)

    if "tools" in tests:
        print(f"Running tool call accuracy test (n={args.tools_n})...", file=sys.stderr)
        result = suite.test_tool_call_accuracy(n=args.tools_n)
        suite_result.results.append(result)
        print(f"  → Score: {result.score:.1%}", file=sys.stderr)

    if "latency" in tests:
        print(f"Running latency test (n={args.latency_n})...", file=sys.stderr)
        result = suite.test_response_latency(n=args.latency_n)
        suite_result.results.append(result)
        print(
            f"  → Mean: {result.details.get('mean_ms', 0):.0f}ms, "
            f"P95: {result.details.get('p95_ms', 0):.0f}ms",
            file=sys.stderr,
        )

    if "coherence" in tests:
        print(
            f"Running multi-turn coherence test "
            f"(turns={args.coherence_turns}, n={args.coherence_n})...",
            file=sys.stderr,
        )
        result = suite.test_multi_turn_coherence(
            turns=args.coherence_turns, n=args.coherence_n
        )
        suite_result.results.append(result)
        print(f"  → Score: {result.score:.1%}", file=sys.stderr)

    suite_result.total_duration = time.perf_counter() - total_start

    report = generate_report(suite_result)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\nReport written to {args.output}", file=sys.stderr)
    else:
        print(report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
