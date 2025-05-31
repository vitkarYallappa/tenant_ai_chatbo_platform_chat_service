"""
Encryption and cryptographic utilities.

This module provides secure encryption, hashing, and cryptographic
utilities for the Chat Service including PII protection and data security.
"""

import hashlib
import hmac
import secrets
import base64
from typing import Optional, Dict, Any, Union, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.hazmat.backends import default_backend
import jwt

import structlog
from src.utils.logger import get_logger
from src.config.settings import get_settings

# Initialize logger
logger = get_logger(__name__)


@dataclass
class EncryptedData:
    """Container for encrypted data with metadata."""
    ciphertext: str
    algorithm: str
    key_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None


class EncryptionError(Exception):
    """Custom exception for encryption-related errors."""
    pass


class EncryptionManager:
    """
    Centralized encryption and decryption management.

    Provides symmetric and asymmetric encryption, hashing,
    and key management capabilities.
    """

    def __init__(self, master_key: Optional[str] = None):
        """
        Initialize encryption manager.

        Args:
            master_key: Optional master key for encryption
        """
        self.settings = get_settings()
        self._master_key = master_key or self._derive_master_key()
        self._fernet = Fernet(self._master_key.encode() if isinstance(self._master_key, str) else self._master_key)

        # Key rotation tracking
        self._key_versions: Dict[str, bytes] = {}
        self._current_key_version = "v1"

        logger.info("Encryption manager initialized")

    def _derive_master_key(self) -> bytes:
        """
        Derive master key from settings.

        Returns:
            Derived master key bytes
        """
        # Use JWT secret as base for key derivation
        password = self.settings.JWT_SECRET_KEY.encode()
        salt = b"chat_service_encryption_salt"  # In production, use a random salt

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )

        key = base64.urlsafe_b64encode(kdf.derive(password))
        return key

    def encrypt_text(
            self,
            plaintext: str,
            key_version: Optional[str] = None
    ) -> EncryptedData:
        """
        Encrypt text data.

        Args:
            plaintext: Text to encrypt
            key_version: Optional key version to use

        Returns:
            EncryptedData object with encrypted content

        Raises:
            EncryptionError: If encryption fails
        """
        try:
            if not plaintext:
                raise EncryptionError("Cannot encrypt empty text")

            # Use current key version if not specified
            version = key_version or self._current_key_version

            # Encrypt the data
            ciphertext = self._fernet.encrypt(plaintext.encode('utf-8'))
            encoded_ciphertext = base64.urlsafe_b64encode(ciphertext).decode('utf-8')

            return EncryptedData(
                ciphertext=encoded_ciphertext,
                algorithm="fernet",
                key_id=version,
                timestamp=datetime.now(timezone.utc),
                metadata={"original_length": len(plaintext)}
            )

        except Exception as e:
            logger.error("Text encryption failed", error=str(e))
            raise EncryptionError(f"Encryption failed: {str(e)}")

    def decrypt_text(self, encrypted_data: EncryptedData) -> str:
        """
        Decrypt text data.

        Args:
            encrypted_data: EncryptedData object to decrypt

        Returns:
            Decrypted plaintext string

        Raises:
            EncryptionError: If decryption fails
        """
        try:
            if encrypted_data.algorithm != "fernet":
                raise EncryptionError(f"Unsupported algorithm: {encrypted_data.algorithm}")

            # Decode and decrypt
            ciphertext = base64.urlsafe_b64decode(encrypted_data.ciphertext.encode('utf-8'))
            plaintext_bytes = self._fernet.decrypt(ciphertext)

            return plaintext_bytes.decode('utf-8')

        except Exception as e:
            logger.error("Text decryption failed", error=str(e))
            raise EncryptionError(f"Decryption failed: {str(e)}")

    def encrypt_dict(self, data: Dict[str, Any]) -> EncryptedData:
        """
        Encrypt dictionary data.

        Args:
            data: Dictionary to encrypt

        Returns:
            EncryptedData object with encrypted content
        """
        import json

        try:
            json_string = json.dumps(data, sort_keys=True, separators=(',', ':'))
            return self.encrypt_text(json_string)

        except Exception as e:
            logger.error("Dictionary encryption failed", error=str(e))
            raise EncryptionError(f"Dictionary encryption failed: {str(e)}")

    def decrypt_dict(self, encrypted_data: EncryptedData) -> Dict[str, Any]:
        """
        Decrypt dictionary data.

        Args:
            encrypted_data: EncryptedData object to decrypt

        Returns:
            Decrypted dictionary
        """
        import json

        try:
            json_string = self.decrypt_text(encrypted_data)
            return json.loads(json_string)

        except Exception as e:
            logger.error("Dictionary decryption failed", error=str(e))
            raise EncryptionError(f"Dictionary decryption failed: {str(e)}")


class HashManager:
    """
    Secure hashing utilities for passwords and data integrity.
    """

    @staticmethod
    def hash_password(password: str, salt: Optional[bytes] = None) -> Tuple[str, str]:
        """
        Hash a password using Scrypt.

        Args:
            password: Password to hash
            salt: Optional salt bytes

        Returns:
            Tuple of (hashed_password, salt)
        """
        if not password:
            raise ValueError("Password cannot be empty")

        # Generate salt if not provided
        if salt is None:
            salt = secrets.token_bytes(32)

        # Use Scrypt for password hashing
        kdf = Scrypt(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            n=2 ** 14,  # CPU/memory cost
            r=8,  # Block size
            p=1,  # Parallelization
            backend=default_backend()
        )

        hashed = kdf.derive(password.encode('utf-8'))

        # Return base64 encoded hash and salt
        return (
            base64.urlsafe_b64encode(hashed).decode('utf-8'),
            base64.urlsafe_b64encode(salt).decode('utf-8')
        )

    @staticmethod
    def verify_password(password: str, hashed_password: str, salt: str) -> bool:
        """
        Verify a password against its hash.

        Args:
            password: Password to verify
            hashed_password: Base64 encoded hash
            salt: Base64 encoded salt

        Returns:
            True if password matches, False otherwise
        """
        try:
            # Decode salt and hash
            salt_bytes = base64.urlsafe_b64decode(salt.encode('utf-8'))
            expected_hash = base64.urlsafe_b64decode(hashed_password.encode('utf-8'))

            # Hash the provided password
            kdf = Scrypt(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt_bytes,
                n=2 ** 14,
                r=8,
                p=1,
                backend=default_backend()
            )

            # Verify using constant-time comparison
            kdf.verify(password.encode('utf-8'), expected_hash)
            return True

        except Exception:
            return False

    @staticmethod
    def sha256_hash(data: Union[str, bytes]) -> str:
        """
        Generate SHA256 hash of data.

        Args:
            data: Data to hash

        Returns:
            Hexadecimal hash string
        """
        if isinstance(data, str):
            data = data.encode('utf-8')

        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def hmac_signature(data: str, secret: str) -> str:
        """
        Generate HMAC signature for data.

        Args:
            data: Data to sign
            secret: Secret key for signing

        Returns:
            HMAC signature
        """
        return hmac.new(
            secret.encode('utf-8'),
            data.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()


class PIIProtector:
    """
    PII (Personally Identifiable Information) protection utilities.
    """

    def __init__(self, encryption_manager: EncryptionManager):
        """
        Initialize PII protector.

        Args:
            encryption_manager: EncryptionManager instance
        """
        self.encryption_manager = encryption_manager

        # Common PII patterns
        self.pii_patterns = {
            'email': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            'phone': r'\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}',
            'ssn': r'\b\d{3}-?\d{2}-?\d{4}\b',
            'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
        }

    def mask_pii(self, text: str, mask_char: str = "*") -> str:
        """
        Mask PII in text with mask characters.

        Args:
            text: Text containing potential PII
            mask_char: Character to use for masking

        Returns:
            Text with PII masked
        """
        import re

        masked_text = text

        for pii_type, pattern in self.pii_patterns.items():
            def mask_match(match):
                matched_text = match.group(0)
                if pii_type == 'email':
                    # Mask email: keep first char and domain
                    parts = matched_text.split('@')
                    if len(parts) == 2:
                        username = parts[0]
                        if len(username) > 1:
                            masked_username = username[0] + mask_char * (len(username) - 1)
                        else:
                            masked_username = mask_char
                        return f"{masked_username}@{parts[1]}"
                else:
                    # Mask other PII types: keep first and last char
                    if len(matched_text) <= 2:
                        return mask_char * len(matched_text)
                    else:
                        return matched_text[0] + mask_char * (len(matched_text) - 2) + matched_text[-1]

                return matched_text

            masked_text = re.sub(pattern, mask_match, masked_text)

        return masked_text

    def detect_pii(self, text: str) -> Dict[str, list]:
        """
        Detect PII in text.

        Args:
            text: Text to analyze

        Returns:
            Dictionary mapping PII types to found instances
        """
        import re

        detected_pii = {}

        for pii_type, pattern in self.pii_patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                detected_pii[pii_type] = matches

        return detected_pii

    def encrypt_pii_fields(self, data: Dict[str, Any], pii_fields: list) -> Dict[str, Any]:
        """
        Encrypt specified PII fields in a dictionary.

        Args:
            data: Data dictionary
            pii_fields: List of field names containing PII

        Returns:
            Dictionary with PII fields encrypted
        """
        encrypted_data = data.copy()

        for field in pii_fields:
            if field in encrypted_data and encrypted_data[field]:
                encrypted_field = self.encryption_manager.encrypt_text(str(encrypted_data[field]))
                encrypted_data[field] = {
                    'encrypted': True,
                    'data': encrypted_field.ciphertext,
                    'algorithm': encrypted_field.algorithm,
                    'key_id': encrypted_field.key_id
                }

        return encrypted_data


class TokenManager:
    """
    JWT token management utilities.
    """

    def __init__(self):
        """Initialize token manager."""
        self.settings = get_settings()

    def create_access_token(
            self,
            data: Dict[str, Any],
            expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create JWT access token.

        Args:
            data: Data to encode in token
            expires_delta: Optional custom expiration time

        Returns:
            JWT token string
        """
        to_encode = data.copy()

        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(
                minutes=self.settings.JWT_EXPIRE_MINUTES
            )

        to_encode.update({
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "iss": "chat-service"
        })

        encoded_jwt = jwt.encode(
            to_encode,
            self.settings.JWT_SECRET_KEY,
            algorithm=self.settings.JWT_ALGORITHM
        )

        return encoded_jwt

    def verify_token(self, token: str) -> Dict[str, Any]:
        """
        Verify and decode JWT token.

        Args:
            token: JWT token to verify

        Returns:
            Decoded token data

        Raises:
            jwt.InvalidTokenError: If token is invalid
        """
        try:
            payload = jwt.decode(
                token,
                self.settings.JWT_SECRET_KEY,
                algorithms=[self.settings.JWT_ALGORITHM]
            )
            return payload

        except jwt.ExpiredSignatureError:
            raise jwt.InvalidTokenError("Token has expired")
        except jwt.InvalidTokenError as e:
            raise jwt.InvalidTokenError(f"Token validation failed: {str(e)}")

    def create_api_key(self, tenant_id: str, user_id: str) -> str:
        """
        Create API key for tenant/user.

        Args:
            tenant_id: Tenant identifier
            user_id: User identifier

        Returns:
            Generated API key
        """
        # Format: cb_{env}_{32_hex_chars}
        environment = self.settings.ENVIRONMENT.value[:4]
        random_part = secrets.token_hex(16)  # 32 hex characters

        api_key = f"cb_{environment}_{random_part}"

        logger.info(
            "API key created",
            tenant_id=tenant_id,
            user_id=user_id,
            key_prefix=api_key[:16]
        )

        return api_key

    def hash_api_key(self, api_key: str) -> str:
        """
        Hash API key for storage.

        Args:
            api_key: API key to hash

        Returns:
            Hashed API key
        """
        return HashManager.sha256_hash(api_key)


# Global instances
encryption_manager = EncryptionManager()
hash_manager = HashManager()
pii_protector = PIIProtector(encryption_manager)
token_manager = TokenManager()


def encrypt_sensitive_data(data: str) -> EncryptedData:
    """
    Convenience function to encrypt sensitive data.

    Args:
        data: Data to encrypt

    Returns:
        EncryptedData object
    """
    return encryption_manager.encrypt_text(data)


def decrypt_sensitive_data(encrypted_data: EncryptedData) -> str:
    """
    Convenience function to decrypt sensitive data.

    Args:
        encrypted_data: EncryptedData to decrypt

    Returns:
        Decrypted data string
    """
    return encryption_manager.decrypt_text(encrypted_data)


def generate_secure_token(length: int = 32) -> str:
    """
    Generate a cryptographically secure random token.

    Args:
        length: Token length in bytes

    Returns:
        URL-safe base64 encoded token
    """
    return secrets.token_urlsafe(length)


def constant_time_compare(a: str, b: str) -> bool:
    """
    Compare two strings in constant time to prevent timing attacks.

    Args:
        a: First string
        b: Second string

    Returns:
        True if strings are equal, False otherwise
    """
    return secrets.compare_digest(a.encode('utf-8'), b.encode('utf-8'))


# Export commonly used functions and classes
__all__ = [
    'EncryptionManager',
    'HashManager',
    'PIIProtector',
    'TokenManager',
    'EncryptedData',
    'EncryptionError',
    'encryption_manager',
    'hash_manager',
    'pii_protector',
    'token_manager',
    'encrypt_sensitive_data',
    'decrypt_sensitive_data',
    'generate_secure_token',
    'constant_time_compare',
]