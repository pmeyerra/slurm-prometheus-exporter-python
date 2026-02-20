"""Tests for JWT expiration validation in the Slurm REST API client."""

import base64
import json
import time

import pytest

from slurm_prometheus_exporter.slurmrestapi import client


def _make_jwt(payload: dict, header: dict | None = None) -> str:
    """Build a minimal unsigned JWT string from a payload dict."""
    header = header or {"alg": "HS256", "typ": "JWT"}
    h = base64.urlsafe_b64encode(json.dumps(header).encode()).rstrip(b"=").decode()
    p = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()
    return f"{h}.{p}.fakesignature"


# ---------------------------------------------------------------------------
# validate_jwt_not_expired
# ---------------------------------------------------------------------------


def test_validate_jwt_valid_token_does_not_raise():
    """Valid JWT with future expiry passes validation."""
    token = _make_jwt({"exp": int(time.time()) + 3600})
    client.validate_jwt_not_expired(token)


def test_validate_jwt_expired_token_raises():
    """Expired JWT raises ExpiredTokenError."""
    token = _make_jwt({"exp": int(time.time()) - 60})
    with pytest.raises(client.ExpiredTokenError, match="expired"):
        client.validate_jwt_not_expired(token)


def test_validate_jwt_token_at_exact_expiry_raises():
    """JWT whose exp equals current time is treated as expired."""
    token = _make_jwt({"exp": int(time.time())})
    with pytest.raises(client.ExpiredTokenError):
        client.validate_jwt_not_expired(token)


def test_validate_jwt_non_jwt_string_is_skipped():
    """Non-JWT token (no dots) is silently accepted to support other token formats."""
    client.validate_jwt_not_expired("not-a-jwt-token")


def test_validate_jwt_two_part_token_is_skipped():
    """Token with only two segments is not a valid JWT and is skipped."""
    client.validate_jwt_not_expired("header.payload")


def test_validate_jwt_no_exp_claim_is_skipped():
    """JWT without an exp claim is accepted since expiry is optional in the spec."""
    token = _make_jwt({"sub": "user"})
    client.validate_jwt_not_expired(token)


def test_validate_jwt_invalid_base64_payload_is_skipped():
    """Malformed base64 payload is silently skipped rather than crashing."""
    client.validate_jwt_not_expired("header.!!!invalid!!!.signature")


# ---------------------------------------------------------------------------
# SlurmRestApiClient integration
# ---------------------------------------------------------------------------


def test_client_init_raises_on_expired_token(tmp_path):
    """Client refuses to initialize when the token file contains an expired JWT."""
    token = _make_jwt({"exp": int(time.time()) - 60})
    token_file = tmp_path / "token"
    token_file.write_text(token)

    with pytest.raises(client.ExpiredTokenError):
        client.SlurmRestApiClient(
            base_url="http://localhost:6820",
            token_file=str(token_file),
        )


def test_client_init_succeeds_with_valid_token(tmp_path):
    """Client initializes successfully when the token file contains a valid JWT."""
    token = _make_jwt({"exp": int(time.time()) + 3600})
    token_file = tmp_path / "token"
    token_file.write_text(token)

    api_client = client.SlurmRestApiClient(
        base_url="http://localhost:6820",
        token_file=str(token_file),
    )
    assert api_client is not None
