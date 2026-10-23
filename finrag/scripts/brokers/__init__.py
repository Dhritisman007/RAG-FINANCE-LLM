"""
Broker adapters: get_adapter("csv" | "cas" | "upstox" | "dhan" | "groww", **kwargs) and list_adapters().
See base.py for the contract and README "Portfolio sources" for setup per broker.
"""
from __future__ import annotations

from importlib import import_module

from .base import (AuthError, AuthExpired, AuthStatus, BrokerAdapter, BrokerError, BrokerUnavailable,  # noqa: F401
                   CredentialsMissing, Holding, IncorrectPassword, MalformedResponse, Price, RateLimited,
                   UnsupportedOperation)

_REGISTRY = {
    "csv": ("brokers.csv_adapter", "CsvAdapter"),
    "cas": ("brokers.cas_adapter", "CasAdapter"),
    "upstox": ("brokers.upstox_adapter", "UpstoxAdapter"),
    "dhan": ("brokers.dhan_adapter", "DhanAdapter"),
    "groww": ("brokers.groww_adapter", "GrowwAdapter"),
}


def list_adapters() -> list[str]:
    return list(_REGISTRY)


def get_adapter(name: str, **kwargs) -> BrokerAdapter:
    if name not in _REGISTRY:
        raise ValueError(f"unknown portfolio source {name!r} — one of {', '.join(_REGISTRY)}")
    module, cls = _REGISTRY[name]
    return getattr(import_module(module), cls)(**kwargs)
