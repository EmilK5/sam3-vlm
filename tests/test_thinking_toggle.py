"""
Tests for the roi_* config fields and the per-call qwen3-vl thinking toggle (7.4).

The toggle keeps a single model and turns "thinking" off for the structured
policy-loop decision via an extra chat.completions arg. No network: a capture
client records the create() kwargs.
"""

import dataclasses
import types

from config import Config, thinking_call_kwargs
from agent import policy_vlm


def test_roi_config_defaults_match_actions_getattr():
    cfg = Config()
    assert cfg.roi_margin == 0.10
    assert cfg.roi_min_size == 32.0
    assert cfg.roi_max_depth == 2
    assert 0.0 < cfg.roi_dup_iou <= 1.0
    assert cfg.policy_enable_thinking is False


def test_thinking_call_kwargs_toggle():
    assert thinking_call_kwargs(True) == {}                 # thinking on -> no extra arg
    off = thinking_call_kwargs(False)
    assert off["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False


class _CaptureClient:
    """Records the kwargs passed to chat.completions.create and returns a legal action."""

    def __init__(self):
        self.kwargs = None
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.kwargs = kwargs
        msg = types.SimpleNamespace(content='{"action": "tile", "args": {"conf": 0.3}}')
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


_MESSAGES = [{"role": "user", "content": "x"}]


def test_policy_request_disables_thinking_by_default():
    client = _CaptureClient()
    policy_vlm._request(client, Config(), _MESSAGES)     # policy_enable_thinking=False
    assert client.kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False


def test_policy_request_keeps_thinking_when_enabled():
    cfg = dataclasses.replace(Config(), policy_enable_thinking=True)
    client = _CaptureClient()
    policy_vlm._request(client, cfg, _MESSAGES)
    assert "extra_body" not in client.kwargs               # model default: thinking on
