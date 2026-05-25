from __future__ import annotations

import sqlite3


def find_user(email: str) -> list[tuple[str]]:
    connection = sqlite3.connect(":memory:")
    query = f"SELECT name FROM users WHERE email = '{email}'"
    cursor = connection.execute(query)
    return cursor.fetchall()
