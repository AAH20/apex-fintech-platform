"""Security engine for encryption, authentication, and authorization.

Provides:
- AES-256-GCM encryption/decryption
- PBKDF2 key derivation
- Password hashing (PBKDF2-HMAC-SHA256)
- JWT-like token generation/verification
- Role-based access control (RBAC)
- HMAC signing/verification

References:
    CFA Institute — Cybersecurity for Investment Management
    NIST SP 800-132 — Recommendation for Key Derivation
    NIST SP 800-63B — Digital Identity Guidelines
    RFC 7519 — JSON Web Token (JWT)
    RFC 2104 — HMAC
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from datetime import timedelta
from typing import Any

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class EncryptionError(Exception):
    """Raised when encryption or decryption fails."""


class AuthenticationError(Exception):
    """Raised when authentication fails."""


class AuthorizationError(Exception):
    """Raised when authorization fails."""


# ---------------------------------------------------------------------------
# Security Engine
# ---------------------------------------------------------------------------


class SecurityEngine:
    """Unified security engine for encryption, authentication, and authorization.

    Uses AES-256-GCM for authenticated encryption, PBKDF2-HMAC-SHA256 for
    key derivation and password hashing, and HMAC-SHA256 for token signing.
    """

    # Constants
    _AES_KEY_SIZE = 32  # 256 bits
    _GCM_NONCE_SIZE = 12  # 96 bits (recommended for GCM)
    _PBKDF2_ITERATIONS = 600_000  # OWASP 2023 recommendation
    _SALT_SIZE = 32
    _TOKEN_SECRET_SIZE = 32

    def __init__(self, master_key: bytes | None = None) -> None:
        """Initialize the security engine.

        Args:
            master_key: 32-byte master key for encryption. If None, a random
                key is generated (suitable for testing only).
        """
        if master_key is None:
            master_key = os.urandom(self._AES_KEY_SIZE)
        if len(master_key) != self._AES_KEY_SIZE:
            raise ValueError(
                f"master_key must be exactly {self._AES_KEY_SIZE} bytes, "
                f"got {len(master_key)}"
            )
        self._master_key = master_key
        self._aesgcm = AESGCM(master_key)

    # -----------------------------------------------------------------------
    # Encryption / Decryption (AES-256-GCM)
    # -----------------------------------------------------------------------

    def encrypt(self, plaintext: bytes | str) -> bytes:
        """Encrypt data using AES-256-GCM.

        Args:
            plaintext: Data to encrypt (bytes or str).

        Returns:
            Serialized ciphertext: nonce (12 bytes) || ciphertext || tag (16 bytes).

        Raises:
            EncryptionError: If encryption fails.
        """
        try:
            if isinstance(plaintext, str):
                plaintext = plaintext.encode("utf-8")
            nonce = os.urandom(self._GCM_NONCE_SIZE)
            ciphertext = self._aesgcm.encrypt(nonce, plaintext, None)
            return nonce + ciphertext
        except Exception as e:
            raise EncryptionError(f"Encryption failed: {e}") from e

    def decrypt(self, ciphertext: bytes) -> bytes:
        """Decrypt data encrypted with AES-256-GCM.

        Args:
            ciphertext: Serialized ciphertext from encrypt().

        Returns:
            Decrypted plaintext bytes.

        Raises:
            EncryptionError: If decryption or authentication fails.
        """
        try:
            if len(ciphertext) < self._GCM_NONCE_SIZE + 16:
                raise ValueError("Ciphertext too short")
            nonce = ciphertext[: self._GCM_NONCE_SIZE]
            encrypted = ciphertext[self._GCM_NONCE_SIZE :]
            return self._aesgcm.decrypt(nonce, encrypted, None)
        except EncryptionError:
            raise
        except Exception as e:
            raise EncryptionError(f"Decryption failed: {e}") from e

    # -----------------------------------------------------------------------
    # Key Derivation (PBKDF2-HMAC-SHA256)
    # -----------------------------------------------------------------------

    def derive_key(self, password: str, salt: bytes) -> bytes:
        """Derive a 256-bit key from a password using PBKDF2-HMAC-SHA256.

        Args:
            password: The password to derive from.
            salt: Cryptographic salt (should be unique per key).

        Returns:
            32-byte derived key.
        """
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=self._AES_KEY_SIZE,
            salt=salt,
            iterations=self._PBKDF2_ITERATIONS,
        )
        return kdf.derive(password.encode("utf-8"))

    # -----------------------------------------------------------------------
    # Password Hashing (PBKDF2-HMAC-SHA256)
    # -----------------------------------------------------------------------

    def hash_password(self, password: str) -> str:
        """Hash a password using PBKDF2-HMAC-SHA256.

        Args:
            password: The password to hash.

        Returns:
            Encoded hash string: algorithm$iterations$salt$hash (base64).
        """
        salt = os.urandom(self._SALT_SIZE)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=self._AES_KEY_SIZE,
            salt=salt,
            iterations=self._PBKDF2_ITERATIONS,
        )
        hash_bytes = kdf.derive(password.encode("utf-8"))
        salt_b64 = base64.b64encode(salt).decode("ascii")
        hash_b64 = base64.b64encode(hash_bytes).decode("ascii")
        return f"pbkdf2_sha256${self._PBKDF2_ITERATIONS}${salt_b64}${hash_b64}"

    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify a password against its hash.

        Args:
            password: The password to verify.
            hashed: The hash string from hash_password().

        Returns:
            True if the password matches, False otherwise.
        """
        try:
            parts = hashed.split("$")
            if len(parts) != 4 or parts[0] != "pbkdf2_sha256":
                return False
            iterations = int(parts[1])
            salt = base64.b64decode(parts[2])
            expected_hash = base64.b64decode(parts[3])

            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=self._AES_KEY_SIZE,
                salt=salt,
                iterations=iterations,
            )
            kdf.verify(password.encode("utf-8"), expected_hash)
            return True
        except Exception:
            return False

    # -----------------------------------------------------------------------
    # Token Generation / Verification (JWT-like with HMAC-SHA256)
    # -----------------------------------------------------------------------

    def generate_token(
        self,
        user_id: str,
        roles: list[str],
        expires_in: timedelta = timedelta(hours=1),
        additional_claims: dict[str, Any] | None = None,
    ) -> str:
        """Generate a signed token (JWT-like).

        Args:
            user_id: Unique identifier for the user.
            roles: List of role strings.
            expires_in: Token lifetime.
            additional_claims: Optional additional claims.

        Returns:
            Signed token string: header.payload.signature (base64url).
        """
        now = time.time()
        exp = now + expires_in.total_seconds()

        header = {"alg": "HS256", "typ": "JWT"}
        payload = {
            "user_id": user_id,
            "roles": roles,
            "iat": now,
            "exp": exp,
        }
        if additional_claims:
            payload.update(additional_claims)

        header_b64 = self._b64url_encode(json.dumps(header, separators=(",", ":")))
        payload_b64 = self._b64url_encode(json.dumps(payload, separators=(",", ":")))
        signing_input = f"{header_b64}.{payload_b64}"
        signature = self._hmac_sign(signing_input.encode("utf-8"))
        sig_b64 = self._b64url_encode(signature)

        return f"{signing_input}.{sig_b64}"

    def verify_token(self, token: str) -> dict[str, Any]:
        """Verify a token and return its claims.

        Args:
            token: The token string from generate_token().

        Returns:
            Dictionary of claims.

        Raises:
            AuthenticationError: If the token is invalid, tampered, or expired.
        """
        try:
            parts = token.split(".")
            if len(parts) != 3:
                raise AuthenticationError("Invalid token format")

            header_b64, payload_b64, sig_b64 = parts
            signing_input = f"{header_b64}.{payload_b64}"

            # Verify signature
            expected_sig = self._hmac_sign(signing_input.encode("utf-8"))
            provided_sig = self._b64url_decode(sig_b64)
            if not hmac.compare_digest(expected_sig, provided_sig):
                raise AuthenticationError("Invalid token signature")

            # Decode payload
            payload_json = self._b64url_decode(payload_b64)
            claims = json.loads(payload_json)

            # Check expiration
            exp = claims.get("exp")
            if exp is None or time.time() > exp:
                raise AuthenticationError("Token has expired")

            return claims
        except AuthenticationError:
            raise
        except Exception as e:
            raise AuthenticationError(f"Token verification failed: {e}") from e

    # -----------------------------------------------------------------------
    # Authorization (Role-Based Access Control)
    # -----------------------------------------------------------------------

    def authorize(self, token: str, required_role: str | list[str]) -> dict[str, Any]:
        """Authorize a token against required role(s).

        Args:
            token: The token to authorize.
            required_role: A single role string or list of role strings.

        Returns:
            The token claims if authorized.

        Raises:
            AuthenticationError: If the token is invalid.
            AuthorizationError: If the user lacks the required role.
        """
        claims = self.verify_token(token)
        user_roles = set(claims.get("roles", []))

        if isinstance(required_role, str):
            required = {required_role}
        else:
            required = set(required_role)

        if not user_roles & required:
            raise AuthorizationError(
                f"Access denied: requires one of {required}, user has {user_roles}"
            )

        return claims

    # -----------------------------------------------------------------------
    # HMAC Signing / Verification
    # -----------------------------------------------------------------------

    def sign(self, data: bytes | str) -> str:
        """Sign data using HMAC-SHA256.

        Args:
            data: Data to sign.

        Returns:
            Base64-encoded signature string.
        """
        if isinstance(data, str):
            data = data.encode("utf-8")
        signature = self._hmac_sign(data)
        return base64.urlsafe_b64encode(signature).decode("ascii")

    def verify_signature(self, data: bytes | str, signature: str) -> bool:
        """Verify an HMAC-SHA256 signature.

        Args:
            data: The original data.
            signature: The signature string from sign().

        Returns:
            True if the signature is valid, False otherwise.
        """
        if isinstance(data, str):
            data = data.encode("utf-8")
        try:
            expected = self._hmac_sign(data)
            provided = base64.urlsafe_b64decode(signature)
            return hmac.compare_digest(expected, provided)
        except Exception:
            return False

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _hmac_sign(self, data: bytes) -> bytes:
        """Sign data with HMAC-SHA256 using the master key."""
        return hmac.new(self._master_key, data, hashlib.sha256).digest()

    @staticmethod
    def _b64url_encode(data: bytes | str) -> str:
        """Base64url encode without padding."""
        if isinstance(data, str):
            data = data.encode("utf-8")
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")

    @staticmethod
    def _b64url_decode(data: str) -> bytes:
        """Base64url decode, adding padding if needed."""
        padding = 4 - len(data) % 4
        if padding != 4:
            data += "=" * padding
        return base64.urlsafe_b64decode(data)
