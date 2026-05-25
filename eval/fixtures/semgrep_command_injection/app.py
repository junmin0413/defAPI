from __future__ import annotations

import subprocess


def ping_host(hostname: str) -> str:
    command = f"ping -c 1 {hostname}"
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout


if __name__ == "__main__":
    print(ping_host("127.0.0.1"))
