# Eval Fixtures

These projects are intentionally small and are used by eval runners to check scan quality and latency.

- `clean_python_project`: baseline fixture without intentional vulnerable code.
- `semgrep_hardcoded_secret`: Python source with hardcoded secret patterns.
- `semgrep_command_injection`: Python source with unsafe shell command execution.
- `semgrep_sql_injection`: Python source with unsafe SQL string interpolation.
- `trivy_vulnerable_requirements`: dependency manifest with old package versions.

The vulnerable fixtures are not application templates. Keep them minimal so scanner output is easy to reason about.
