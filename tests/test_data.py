"""Unit tests for distill.data: message validation + chat rendering (offline)."""
from distill.data import render_chat, valid_messages


class FakeTokenizer:
    """Renders messages as '<role>content' lines; accepts the apply_chat_template kwargs."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        assert tokenize is False
        assert add_generation_prompt is False        # full conversation = training target
        return "\n".join(f"<{m['role']}>{m['content']}" for m in messages)


class TestValidMessages:
    def test_user_then_assistant_is_valid(self):
        msgs = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
        assert valid_messages(msgs) is True

    def test_missing_assistant_invalid(self):
        assert valid_messages([{"role": "user", "content": "q"}]) is False

    def test_empty_assistant_invalid(self):
        msgs = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "   "}]
        assert valid_messages(msgs) is False

    def test_non_list_invalid(self):
        assert valid_messages(None) is False
        assert valid_messages("nope") is False


class TestRenderChat:
    def test_includes_both_turns_verbatim(self):
        tok = FakeTokenizer()
        msgs = [{"role": "user", "content": "Solve x+2=5"},
                {"role": "assistant", "content": "Reasoning... x=3"}]
        text = render_chat(msgs, tok)
        assert "Solve x+2=5" in text
        assert "Reasoning... x=3" in text
