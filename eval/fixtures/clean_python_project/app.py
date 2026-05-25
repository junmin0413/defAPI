from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Greeting:
    name: str

    def render(self) -> str:
        safe_name = self.name.strip() or "world"
        return f"Hello, {safe_name}"


def add(left: int, right: int) -> int:
    return left + right


if __name__ == "__main__":
    print(Greeting("defAPI").render())
