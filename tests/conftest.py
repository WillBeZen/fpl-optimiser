"""Shared test fixtures.

Tests never touch the network (docs/claude.md, invariant 5). This guard turns any
attempt into an immediate, clearly-worded failure instead of a silent hang on a
slow or Cloudflare-blocked site — e.g. if the opponent_stats adapter ever falls
back from a mocked source to a live one.
"""
import socket

import pytest


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise RuntimeError("tests must not touch the network — mock the fetch instead")

    monkeypatch.setattr(socket, "getaddrinfo", refuse)        # DNS lookups
    monkeypatch.setattr(socket.socket, "connect", refuse)     # raw connections
