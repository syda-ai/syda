"""
Security utilities for API key encryption and masking
"""
from cryptography.fernet import Fernet
from config import settings
import base64
import hashlib


def _get_fernet_key() -> bytes:
    """
    Generate Fernet key from encryption key in settings
    Fernet requires a 32-byte base64-encoded key
    """
    # Hash the encryption key to ensure it's the right length
    key = hashlib.sha256(settings.encryption_key.encode()).digest()
    return base64.urlsafe_b64encode(key)


def encrypt_api_key(api_key: str) -> str:
    """
    Encrypt an API key for secure storage
    
    Args:
        api_key: Plain text API key
        
    Returns:
        Encrypted API key as string
    """
    fernet = Fernet(_get_fernet_key())
    encrypted = fernet.encrypt(api_key.encode())
    return encrypted.decode()


def decrypt_api_key(encrypted_key: str) -> str:
    """
    Decrypt an API key
    
    Args:
        encrypted_key: Encrypted API key
        
    Returns:
        Plain text API key
    """
    fernet = Fernet(_get_fernet_key())
    decrypted = fernet.decrypt(encrypted_key.encode())
    return decrypted.decode()


def mask_api_key(api_key: str, show_chars: int = 4) -> str:
    """
    Mask an API key for display purposes
    
    Args:
        api_key: API key to mask
        show_chars: Number of characters to show at start and end
        
    Returns:
        Masked API key (e.g., "sk-a****xyz9")
    """
    if not api_key or len(api_key) <= show_chars * 2:
        return "****"
    
    start = api_key[:show_chars]
    end = api_key[-show_chars:]
    middle_length = len(api_key) - (show_chars * 2)
    
    return f"{start}{'*' * min(middle_length, 20)}{end}"

