"""Tests for security engine — encryption, authentication, authorization."""
import base64
import os
from datetime import timedelta

import pytest

from src.security.engine import (
    AuthenticationError,
    AuthorizationError,
    EncryptionError,
    SecurityEngine,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    """Create a fresh SecurityEngine for each test."""
    return SecurityEngine(master_key=b"test-master-key-32-bytes-long!!!")


# ---------------------------------------------------------------------------
# Encryption tests
# ---------------------------------------------------------------------------


class TestEncryption:
    """Tests for AES-GCM encryption/decryption."""

    def test_encrypt_decrypt_roundtrip(self, engine):
        """Encrypted data can be decrypted back to original."""
        plaintext = b"sensitive financial data: account=12345, balance=1000000"
        ciphertext = engine.encrypt(plaintext)
        assert ciphertext != plaintext
        decrypted = engine.decrypt(ciphertext)
        assert decrypted == plaintext

    def test_encrypt_produces_different_ciphertexts(self, engine):
        """Same plaintext produces different ciphertexts (nonce uniqueness)."""
        plaintext = b"same data"
        ct1 = engine.encrypt(plaintext)
        ct2 = engine.encrypt(plaintext)
        assert ct1 != ct2

    def test_encrypt_string_input(self, engine):
        """Encrypt accepts string input and returns bytes."""
        plaintext = "string data to encrypt"
        ciphertext = engine.encrypt(plaintext)
        assert isinstance(ciphertext, bytes)
        decrypted = engine.decrypt(ciphertext)
        assert decrypted == plaintext.encode()

    def test_decrypt_tampered_data_raises(self, engine):
        """Tampered ciphertext fails authentication."""
        plaintext = b"original data"
        ciphertext = bytearray(engine.encrypt(plaintext))
        # Flip a bit in the ciphertext
        ciphertext[-1] ^= 0xFF
        with pytest.raises(EncryptionError):
            engine.decrypt(bytes(ciphertext))

    def test_decrypt_with_wrong_key_raises(self):
        """Decryption with wrong key fails."""
        engine1 = SecurityEngine(master_key=b"key-one-32-bytes-long!!!!!!!!!!!")
        engine2 = SecurityEngine(master_key=b"key-two-32-bytes-long!!!!!!!!!!!")
        ciphertext = engine1.encrypt(b"secret")
        with pytest.raises(EncryptionError):
            engine2.decrypt(ciphertext)

    def test_encrypt_empty_data(self, engine):
        """Encrypting empty data works."""
        ciphertext = engine.encrypt(b"")
        decrypted = engine.decrypt(ciphertext)
        assert decrypted == b""

    def test_encrypt_large_data(self, engine):
        """Encrypting large data works."""
        plaintext = os.urandom(1024 * 1024)  # 1MB
        ciphertext = engine.encrypt(plaintext)
        decrypted = engine.decrypt(ciphertext)
        assert decrypted == plaintext


# ---------------------------------------------------------------------------
# Key derivation tests
# ---------------------------------------------------------------------------


class TestKeyDerivation:
    """Tests for PBKDF2 key derivation."""

    def test_derive_key_deterministic_with_salt(self, engine):
        """Same password + salt produces same key."""
        key1 = engine.derive_key("password123", salt=b"fixed-salt")
        key2 = engine.derive_key("password123", salt=b"fixed-salt")
        assert key1 == key2

    def test_derive_key_different_passwords(self, engine):
        """Different passwords produce different keys."""
        key1 = engine.derive_key("password1", salt=b"salt")
        key2 = engine.derive_key("password2", salt=b"salt")
        assert key1 != key2

    def test_derive_key_different_salts(self, engine):
        """Different salts produce different keys."""
        key1 = engine.derive_key("password", salt=b"salt1")
        key2 = engine.derive_key("password", salt=b"salt2")
        assert key1 != key2

    def test_derive_key_length(self, engine):
        """Derived key has expected length."""
        key = engine.derive_key("password", salt=b"salt")
        assert len(key) == 32  # AES-256


# ---------------------------------------------------------------------------
# Authentication tests
# ---------------------------------------------------------------------------


class TestAuthentication:
    """Tests for password hashing and verification."""

    def test_hash_password_returns_hash(self, engine):
        """Password hashing returns a non-empty string."""
        hashed = engine.hash_password("my_secure_password")
        assert isinstance(hashed, str)
        assert len(hashed) > 0

    def test_hash_password_different_each_time(self, engine):
        """Same password produces different hashes (salt uniqueness)."""
        hash1 = engine.hash_password("password")
        hash2 = engine.hash_password("password")
        assert hash1 != hash2

    def test_verify_correct_password(self, engine):
        """Correct password verifies successfully."""
        hashed = engine.hash_password("correct_password")
        assert engine.verify_password("correct_password", hashed) is True

    def test_verify_wrong_password(self, engine):
        """Wrong password fails verification."""
        hashed = engine.hash_password("correct_password")
        assert engine.verify_password("wrong_password", hashed) is False

    def test_verify_password_timing_safe(self, engine):
        """Password verification is timing-safe (no early exit)."""
        hashed = engine.hash_password("password")
        # Both should take similar time — we can't easily test timing,
        # but we can verify it doesn't raise
        assert engine.verify_password("wrong", hashed) is False


# ---------------------------------------------------------------------------
# Token-based authentication tests
# ---------------------------------------------------------------------------


class TestTokenAuthentication:
    """Tests for JWT-like token generation and verification."""

    def test_generate_token(self, engine):
        """Token generation returns a string."""
        token = engine.generate_token(user_id="user123", roles=["admin"])
        assert isinstance(token, str)
        assert len(token) > 0

    def test_verify_valid_token(self, engine):
        """Valid token verifies and returns claims."""
        token = engine.generate_token(user_id="user123", roles=["admin"])
        claims = engine.verify_token(token)
        assert claims["user_id"] == "user123"
        assert claims["roles"] == ["admin"]

    def test_verify_expired_token_raises(self, engine):
        """Expired token raises AuthenticationError."""
        token = engine.generate_token(
            user_id="user123",
            roles=["admin"],
            expires_in=timedelta(seconds=-1),  # Already expired
        )
        with pytest.raises(AuthenticationError):
            engine.verify_token(token)

    def test_verify_tampered_token_raises(self, engine):
        """Tampered token raises AuthenticationError."""
        token = engine.generate_token(user_id="user123", roles=["admin"])
        # Tamper with the token
        tampered = token[:-5] + "XXXXX"
        with pytest.raises(AuthenticationError):
            engine.verify_token(tampered)

    def test_verify_invalid_token_raises(self, engine):
        """Completely invalid token raises AuthenticationError."""
        with pytest.raises(AuthenticationError):
            engine.verify_token("not.a.valid.token")


# ---------------------------------------------------------------------------
# Authorization tests
# ---------------------------------------------------------------------------


class TestAuthorization:
    """Tests for role-based access control."""

    def test_check_permission_granted(self, engine):
        """User with correct role is authorized."""
        token = engine.generate_token(user_id="user123", roles=["admin"])
        # Should not raise
        engine.authorize(token, required_role="admin")

    def test_check_permission_denied(self, engine):
        """User without correct role is denied."""
        token = engine.generate_token(user_id="user123", roles=["user"])
        with pytest.raises(AuthorizationError):
            engine.authorize(token, required_role="admin")

    def test_check_any_role_granted(self, engine):
        """User with any of the required roles is authorized."""
        token = engine.generate_token(user_id="user123", roles=["editor"])
        # Should not raise
        engine.authorize(token, required_role=["admin", "editor"])

    def test_check_any_role_denied(self, engine):
        """User with none of the required roles is denied."""
        token = engine.generate_token(user_id="user123", roles=["viewer"])
        with pytest.raises(AuthorizationError):
            engine.authorize(token, required_role=["admin", "editor"])

    def test_authorize_expired_token_raises(self, engine):
        """Expired token cannot authorize."""
        token = engine.generate_token(
            user_id="user123",
            roles=["admin"],
            expires_in=timedelta(seconds=-1),
        )
        with pytest.raises(AuthenticationError):
            engine.authorize(token, required_role="admin")


# ---------------------------------------------------------------------------
# HMAC tests
# ---------------------------------------------------------------------------


class TestHMAC:
    """Tests for HMAC signing and verification."""

    def test_sign_and_verify(self, engine):
        """Signed data verifies correctly."""
        data = b"transaction: buy 100 AAPL @ 150.00"
        signature = engine.sign(data)
        assert engine.verify_signature(data, signature) is True

    def test_verify_wrong_data_fails(self, engine):
        """Signature verification fails for different data."""
        data = b"original data"
        signature = engine.sign(data)
        assert engine.verify_signature(b"different data", signature) is False

    def test_verify_wrong_signature_fails(self, engine):
        """Signature verification fails for wrong signature."""
        data = b"some data"
        engine.sign(data)
        wrong_sig = base64.urlsafe_b64encode(b"wrong" * 10).decode()
        assert engine.verify_signature(data, wrong_sig) is False


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestIntegration:
    """Integration tests combining multiple security features."""

    def test_full_flow_encrypt_authorize(self, engine):
        """Full flow: create token, authorize, encrypt/decrypt data."""
        # Create admin token
        token = engine.generate_token(user_id="admin_user", roles=["admin"])
        # Authorize
        engine.authorize(token, required_role="admin")
        # Encrypt sensitive data
        sensitive = b"account=1234; ssn=123-45-6789"
        ciphertext = engine.encrypt(sensitive)
        # Decrypt
        plaintext = engine.decrypt(ciphertext)
        assert plaintext == sensitive

    def test_password_hash_and_token_flow(self, engine):
        """Full auth flow: hash password, verify, generate token."""
        # Hash password
        hashed = engine.hash_password("user_password")
        # Verify
        assert engine.verify_password("user_password", hashed) is True
        # Generate token
        token = engine.generate_token(user_id="user1", roles=["user"])
        # Authorize
        engine.authorize(token, required_role="user")

    def test_multiple_users_different_permissions(self, engine):
        """Different users have different access levels."""
        admin_token = engine.generate_token(user_id="admin", roles=["admin"])
        user_token = engine.generate_token(user_id="user", roles=["user"])

        # Admin can access admin resources
        engine.authorize(admin_token, required_role="admin")
        # User cannot access admin resources
        with pytest.raises(AuthorizationError):
            engine.authorize(user_token, required_role="admin")
        # User can access user resources
        engine.authorize(user_token, required_role="user")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case tests."""

    def test_encrypt_binary_data(self, engine):
        """Encrypting binary data works."""
        data = bytes(range(256))
        ciphertext = engine.encrypt(data)
        assert engine.decrypt(ciphertext) == data

    def test_unicode_string_encryption(self, engine):
        """Encrypting unicode strings works."""
        data = "日本語テスト 🚀 émojis"
        ciphertext = engine.encrypt(data)
        assert engine.decrypt(ciphertext) == data.encode("utf-8")

    def test_token_with_special_characters_in_user_id(self, engine):
        """Token handles special characters in user_id."""
        token = engine.generate_token(user_id="user@domain.com", roles=["user"])
        claims = engine.verify_token(token)
        assert claims["user_id"] == "user@domain.com"

    def test_engine_with_custom_key(self):
        """Engine works with a custom master key."""
        custom_key = b"my-custom-32-byte-master-key!!!!"
        engine = SecurityEngine(master_key=custom_key)
        ct = engine.encrypt(b"test")
        assert engine.decrypt(ct) == b"test"
