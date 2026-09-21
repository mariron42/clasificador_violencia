"""Use the standard authenticated Hugging Face backend and TLS verification."""
from __future__ import annotations
import os

def configure_hf_backend() -> None:
    """Retain the historical call interface without disabling TLS verification."""
    if os.environ.get("HF_HUB_DISABLE_SSL_VERIFY", "0") == "1":
        raise ValueError("TLS verification must stay enabled. Configure your trusted CA bundle instead.")
