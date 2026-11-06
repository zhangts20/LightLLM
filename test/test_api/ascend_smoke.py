"""Exercise a running text server, including paged prefixes and concurrent decode.

Example: python test/test_api/ascend_smoke.py --base-url http://127.0.0.1:8910
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json

import requests


def generate(base_url, prompt):
    response = requests.post(
        base_url + "/generate",
        json={"inputs": prompt, "parameters": {"do_sample": False, "max_new_tokens": 16, "return_details": True}},
        timeout=120,
    )
    response.raise_for_status()
    result = response.json()
    assert result["count_output_tokens"] > 0, result
    assert result["tokens"], result
    return result


def token_ids(result):
    return [token["id"] for token in result["tokens"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8910")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    short = "The capital of France is"
    first = generate(base, short)
    repeated = generate(base, short)
    assert token_ids(first) == token_ids(repeated), "Repeated greedy generation changed"
    # More than two 128-token pages, and a non-page-aligned tail.
    prefix = "The quick brown fox jumps over the lazy dog. " * 32
    long_prompt = prefix + "Continue the story:"
    long_first = generate(base, long_prompt)
    long_second = generate(base, long_prompt)
    assert token_ids(long_first) == token_ids(long_second), "Cached-prefix output changed"
    assert long_second["tokens"][0].get("prompt_cache_len", 0) >= 128, "No paged prefix cache hit"
    prompts = [short, "One plus one equals", prefix + "Summarize the story:"]
    with ThreadPoolExecutor(max_workers=3) as executor:
        concurrent = list(executor.map(lambda prompt: generate(base, prompt), prompts))
    assert token_ids(concurrent[0]) == token_ids(first), "Batched greedy generation changed"
    with requests.post(
        base + "/generate_stream",
        json={"inputs": short, "parameters": {"do_sample": False, "max_new_tokens": 16}},
        timeout=120,
        stream=True,
    ) as response:
        response.raise_for_status()
        events = [
            json.loads(line[5:].strip())
            for line in response.iter_lines(decode_unicode=True)
            if line.startswith("data:") and line[5:].strip() != "[DONE]"
        ]
    assert events, "No streaming events"
    assert events[-1]["finished"], "Stream did not finish"
    assert [event["token"]["id"] for event in events] == token_ids(first), "Stream output changed"
    print(
        json.dumps(
            {
                "status": "passed",
                "requests": 8,
                "short_output": first["generated_text"],
                "long_prompt_tokens": long_second["prompt_tokens"],
                "cached_prefix_tokens": long_second["tokens"][0].get("prompt_cache_len", 0),
                "stream_events": len(events),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
