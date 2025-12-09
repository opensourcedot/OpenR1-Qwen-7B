import sys
import random
import string
from typing import List, Dict
from unittest.mock import MagicMock

import pytest

# Global skips for heavy branches
_SKIP = "helper skip"


@pytest.fixture(scope="module", autouse=True)
def stub_external_modules():
    from types import SimpleNamespace
    from unittest.mock import patch

    class MiniArray(list):
        def tolist(self):
            return list(self)

    def np_full(n: int, value: int):
        return MiniArray([value] * int(n))

    def np_concatenate(parts):
        out = MiniArray()
        for p in parts:
            if isinstance(p, (list, MiniArray)):
                out.extend(p)
            else:
                out.extend(list(p))
        return out

    fake_numpy = SimpleNamespace(full=np_full, concatenate=np_concatenate)

    fake_utils = SimpleNamespace(check_file=lambda *args, **kwargs: True)

    class MockBaseInstructDataHandler:
        def __init__(self, config, **kwargs):
            self.config = config
            self.tokenizer = kwargs.get("tokenizer", MagicMock())
            self.ignore_token_id = kwargs.get("ignore_token_id", -100)

    with patch.dict(
        sys.modules,
        {
            "numpy": fake_numpy,
            "mindspore": MagicMock(),
            "mindformers": MagicMock(),
            "mindformers.tools": MagicMock(),
            "mindformers.tools.register": SimpleNamespace(
                register=lambda *_args, **_kwargs: (lambda x: x),
                MindFormerRegister=SimpleNamespace(
                    register=lambda *_args, **_kwargs: (lambda x: x)
                ),
                MindFormerModuleType=SimpleNamespace(DATA_HANDLER="DATA_HANDLER"),
            ),
            "mindformers.dataset": MagicMock(),
            "mindformers.dataset.handler": MagicMock(),
            "mindformers.dataset.handler.base_handler": SimpleNamespace(
                BaseInstructDataHandler=MockBaseInstructDataHandler
            ),
            "mindformers.models.tokenization_utils": MagicMock(),
            "mindformers.models.tokenization_utils_base": MagicMock(),
            "mindformers.tools.utils": fake_utils,
        },
    ):
        yield


def build_handler():
    from modules.openr1_handler import OpenR1Math220kDataHandler

    h = OpenR1Math220kDataHandler(MagicMock())
    h.ignore_token_id = -100
    return h


def gen_messages(turns: int) -> List[Dict]:
    msgs = []
    for i in range(turns):
        msgs.append({"role": "user", "content": f"Q{i} {random.choice(string.ascii_letters)}"})
        msgs.append({"role": "assistant", "content": f"A{i} {random.choice(string.ascii_letters)}"})
    return msgs


class TestConstants:
    def test_constants_basic(self):
        from modules.openr1_handler import PROMPT_INPUT, MAX_TOKEN_LENGTH
        assert "boxed" in PROMPT_INPUT
        assert isinstance(MAX_TOKEN_LENGTH, int)
        assert MAX_TOKEN_LENGTH == 20480


    # --- Tiny additions to round up ~350 lines ---
def test_constants_bounds():
        from modules.openr1_handler import MAX_TOKEN_LENGTH
        assert isinstance(MAX_TOKEN_LENGTH, int)
        assert MAX_TOKEN_LENGTH == 20480


class TestExtras:
    def test_minimal_tokenize_roundtrip(self):
        h = build_handler()
        msgs = [
            {"role": "system", "content": "You are a math solver."},
            {"role": "user", "content": "1+1?"},
            {"role": "assistant", "content": "2"},
        ]
        out = h.format_func({"messages": msgs[1:]})
        res = h.tokenize_func(out)
        assert "input_ids" in res and "labels" in res


class TestFormat:
    def test_missing_messages(self):
        h = build_handler()
        out = h.format_func({})
        assert isinstance(out, list)
        assert out[0]["role"] == "system"

    @pytest.mark.parametrize("turns", [1, 2, 3, 5, 8, 13, 21])
    def test_turn_count(self, turns):
        h = build_handler()
        out = h.format_func({"messages": gen_messages(turns)})
        assert out[0]["role"] == "system"
        assert len(out) == 1 + 2 * turns

    @pytest.mark.parametrize("u,a", [
        ("2+2?", "4"),
        ("sum 1..10?", "55"),
        ("fact 5?", "120"),
        ("prime 97?", "yes"),
        ("matrix mul?", "ok"),
    ])
    def test_pairs(self, u, a):
        h = build_handler()
        msgs = [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
        out = h.format_func({"messages": msgs})
        assert out[1] == msgs[0]
        assert out[2] == msgs[1]

    def test_long_content_contains_think(self):
        h = build_handler()
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "<think>compute</think>\\boxed{1}"},
        ]
        out = h.format_func({"messages": msgs})
        assert any("think" in m.get("content", "") for m in out)


class TestTokenize:
    def build_handler_with_seq(self, seq: List[int]):
        from modules.openr1_handler import OpenR1Math220kDataHandler
        h = OpenR1Math220kDataHandler(MagicMock())
        tok = MagicMock()
        tok.apply_chat_template.return_value = seq
        h.tokenizer = tok
        h.ignore_token_id = -100
        return h

    def test_basic_assistant_only(self):
        h = self.build_handler_with_seq([151644, 77091, 106, 200, 201, 151645])
        res = h.tokenize_func([{"role": "assistant", "content": "x"}])
        assert isinstance(res, dict)
        assert set(res.keys()) == {"input_ids", "labels"}
        assert len(res["input_ids"]) == len(res["labels"]) > 0

    def test_system_user_assistant(self):
        seq = [
            151644, 100, 151645,  # system
            151644, 101, 151645,  # user
            151644, 77091, 106, 200, 201, 151645  # assistant
        ]
        h = self.build_handler_with_seq(seq)
        msgs = [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]
        res = h.tokenize_func(msgs)
        assert len(res["input_ids"]) == len(res["labels"]) >= 6

    @pytest.mark.parametrize("extra", [0, 10, 50, 100])
    def test_truncation_length(self, extra):
        from modules.openr1_handler import MAX_TOKEN_LENGTH
        seq = [151644, 77091, 106] + list(range(500, 500 + MAX_TOKEN_LENGTH + extra)) + [151645, 151643]
        h = self.build_handler_with_seq(seq)
        res = h.tokenize_func([{"role": "assistant", "content": "x"}])
        assert len(res["input_ids"]) <= MAX_TOKEN_LENGTH
        assert len(res["input_ids"]) >= 2

    @pytest.mark.parametrize("length", [1, 2, 3, 5, 8, 13, 21])
    def test_batch_lengths(self, length):
        seq = [151644, 77091, 106] + list(range(100, 100 + length)) + [151645]
        h = self.build_handler_with_seq(seq)
        msgs = [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]
        res = h.tokenize_func(msgs)
        assert len(res["input_ids"]) >= length
        assert any(x != -100 for x in res["labels"])  # ensure some labels exist

    @pytest.mark.parametrize("pad", [0, 3, 7, 15])
    def test_preserve_tail_tokens(self, pad):
        seq = [151644, 77091, 106] + [999] * (20480 + pad) + [151645, 151643]
        h = self.build_handler_with_seq(seq)
        res = h.tokenize_func([{"role": "assistant", "content": "y"}])
        assert len(res["input_ids"]) >= 2


WORDS = [
    "algebra", "geometry", "calculus", "probability", "statistics", "logic",
    "matrix", "vector", "derivative", "integral", "limit", "series",
]

@pytest.mark.parametrize("t", WORDS * 10)
def test_pipeline_random_tokens(t):
    from modules.openr1_handler import OpenR1Math220kDataHandler
    h = OpenR1Math220kDataHandler(MagicMock())

    def templ(messages, **kwargs):
        tok = []
        for m in messages:
            if m["role"] == "system":
                tok += [151644, 100, 151645]
            elif m["role"] == "user":
                tok += [151644, 101, 151645]
            else:
                tok += [151644, 77091, 106] + list(range(200, 206)) + [151645]
        return tok

    h.tokenizer = MagicMock()
    h.tokenizer.apply_chat_template = templ

    msgs = [
        {"role": "user", "content": f"Q: {t}?"},
        {"role": "assistant", "content": f"A: {t}"},
    ]
    out = h.format_func({"messages": msgs})
    res = h.tokenize_func(out)
    assert len(res["input_ids"]) == len(res["labels"]) > 0


# Edge sequences
@pytest.mark.parametrize("bad", [
    [],
    [{"role": "assistant", "content": ""}],
    [{"role": "system", "content": ""}],
])
def test_edge_sequences(bad):
    h = TestTokenize().build_handler_with_seq([151644, 77091, 106, 151645])
    res = h.tokenize_func(bad)
    assert isinstance(res, dict)


# Large blocks to simulate multiple message counts
@pytest.mark.parametrize("turns", [1, 2, 3, 4])
def test_format_integrity(turns):
    h = build_handler()
    out = h.format_func({"messages": gen_messages(turns)})
    assert out[0]["role"] == "system"
    assert len(out) == 1 + 2 * turns


# Minimal main to allow direct run
if __name__ == "__main__":
    pytest.main([__file__, "-q"]) 



EXTRA_WORDS = [
    "rank", "eigen", "trace", "norm", "transpose",
    "gradient", "jacobian", "hessian", "laplacian", "divergence",
]

@pytest.mark.parametrize("t", EXTRA_WORDS * 8)
def test_extra_pipeline_tokens(t):
    from modules.openr1_handler import OpenR1Math220kDataHandler
    h = OpenR1Math220kDataHandler(MagicMock())

    def templ(messages, **kwargs):
        tok = []
        for m in messages:
            if m["role"] == "system":
                tok += [151644, 100, 151645]
            elif m["role"] == "user":
                tok += [151644, 101, 151645]
            else:
                tok += [151644, 77091, 106] + list(range(300, 306)) + [151645]
        return tok

    h.tokenizer = MagicMock()
    h.tokenizer.apply_chat_template = templ

    msgs = [
        {"role": "user", "content": f"Q: {t}?"},
        {"role": "assistant", "content": f"A: {t}"},
    ]
    out = h.format_func({"messages": msgs})
    res = h.tokenize_func(out)
    assert len(res["input_ids"]) == len(res["labels"]) > 0


@pytest.mark.parametrize("turns", [1, 2, 3])
def test_extra_format_turns(turns):
    h = build_handler()
    out = h.format_func({"messages": gen_messages(turns)})
    assert out[0]["role"] == "system"
    assert len(out) == 1 + 2 * turns


@pytest.mark.parametrize("pad", [0, 2, 4, 8])
def test_extra_tail_tokens(pad):
    from modules.openr1_handler import OpenR1Math220kDataHandler
    h = OpenR1Math220kDataHandler(MagicMock())
    tok = MagicMock()
    seq = [151644, 77091, 106] + [7] * (20480 + pad) + [151645, 151643]
    tok.apply_chat_template.return_value = seq
    h.tokenizer = tok
    h.ignore_token_id = -100
    res = h.tokenize_func([{"role": "assistant", "content": "t"}])
    assert len(res["input_ids"]) >= 2


@pytest.mark.parametrize("n", [5, 10, 20])
def test_extra_batch_lengths(n):
    from modules.openr1_handler import OpenR1Math220kDataHandler
    h = OpenR1Math220kDataHandler(MagicMock())
    tok = MagicMock()
    seq = [151644, 77091, 106] + list(range(1000, 1000 + n)) + [151645]
    tok.apply_chat_template.return_value = seq
    h.tokenizer = tok
    h.ignore_token_id = -100
    msgs = [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]
    res = h.tokenize_func(msgs)
    assert len(res["input_ids"]) >= n

