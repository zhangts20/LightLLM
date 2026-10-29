"""Exercise text, image, multi-image and cache requests with generated fixtures."""
import argparse
import json
from pathlib import Path
import tempfile
import urllib.error
import urllib.request

from PIL import Image


def request_json(base_url, path, payload=None, timeout=120):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(base_url + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        message = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {message[:1000]}") from error


def check(base_url, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fixtures = {}
    for name, color, size in [
        ("red", "red", (256, 256)),
        ("blue", "blue", (320, 240)),
        ("green_large", "green", (1600, 900)),
    ]:
        path = output_dir / (name + ".png")
        Image.new("RGB", size, color=color).save(path)
        fixtures[name] = path.resolve().as_uri()

    model = request_json(base_url, "/v1/models")["data"][0]["id"]
    cases = [
        ("text", [], "Reply with the single word OK.", "ok"),
        ("red", ["red"], "What is the dominant color? Answer with one English word.", "red"),
        ("two_images", ["red", "blue"], "What is the dominant color in the second image? One English word.", "blue"),
        ("repeat_red", ["red"], "What is the dominant color? Answer with one English word.", "red"),
        ("large_image", ["green_large"], "What is the dominant color? Answer with one English word.", "green"),
    ]
    records = []
    for name, images, prompt, expected in cases:
        content = [{"type": "image_url", "image_url": {"url": fixtures[image]}} for image in images]
        content.append({"type": "text", "text": prompt})
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": 64,
            "stream": False,
        }
        try:
            response = request_json(base_url, "/v1/chat/completions", payload)
            choice = response["choices"][0]
            text = choice["message"].get("content") or ""
            record = {
                "case": name,
                "expected": expected,
                "text": text,
                "finish_reason": choice.get("finish_reason"),
                "usage": response.get("usage"),
                "passed": expected in text.lower(),
            }
        except Exception as error:
            record = {"case": name, "passed": False, "error": str(error)}
        records.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    result = {
        "model": model,
        "base_url": base_url,
        "generated_fixtures_only": True,
        "passed": all(record["passed"] for record in records),
        "cases": records,
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    if args.output_dir:
        result = check(args.base_url, args.output_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="lightllm_multimodal_smoke_") as directory:
            result = check(args.base_url, directory)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
