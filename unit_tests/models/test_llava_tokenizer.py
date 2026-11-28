from types import SimpleNamespace

import pytest

from lightllm.models.llava.model import LlavaTokenizer


class Tokenizer:
    bos_token_id = 1

    def __call__(self, text, add_special_tokens=True):
        tokens = [ord(character) + 2 for character in text]
        return SimpleNamespace(input_ids=([self.bos_token_id] if add_special_tokens else []) + tokens)


@pytest.mark.parametrize("add_special_tokens", [False, True])
def test_llava_accepts_serving_tokenizer_kwargs_and_preserves_image_order(add_special_tokens):
    tokenizer = LlavaTokenizer(Tokenizer(), {"text_config": {}, "vision_config": {"patch_size": 14, "image_size": 336}})
    images = [SimpleNamespace(token_id=10000, token_num=576), SimpleNamespace(token_id=20000, token_num=576)]
    result = tokenizer.encode(
        "a<image>b<image>c", SimpleNamespace(images=images), add_special_tokens=add_special_tokens
    )
    expected = ([1] if add_special_tokens else []) + [ord("a") + 2]
    expected += list(range(10000, 10576)) + [ord("b") + 2]
    expected += list(range(20000, 20576)) + [ord("c") + 2]
    assert result == expected


def test_llava_text_only_accepts_serving_tokenizer_kwargs():
    tokenizer = LlavaTokenizer(Tokenizer(), {"text_config": {}, "vision_config": {"patch_size": 14, "image_size": 336}})
    assert tokenizer.encode("abc", add_special_tokens=False) == [ord(letter) + 2 for letter in "abc"]
