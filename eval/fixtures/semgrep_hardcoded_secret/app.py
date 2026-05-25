from __future__ import annotations

import hmac


API_KEY = "sk_test_1234567890abcdef"
ADMIN_PASSWORD = "password123"


def authenticate(password: str) -> bool:
    return password == ADMIN_PASSWORD


def verify_token(token: str) -> bool:
    return hmac.compare_digest(token, API_KEY)
