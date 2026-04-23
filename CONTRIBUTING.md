# Contributing

Thanks for your interest in contributing.

## How to contribute

1. Fork the repository.
2. Create a feature branch from `main`.
3. Make your changes with clear commit messages.
4. Run basic checks before opening a pull request.
5. Open a pull request with a short summary and test notes.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

## Pull request checklist

- Code is focused and minimal.
- README/docs are updated if behavior changed.
- Commands in docs still run.
- No unrelated file changes.

## Reporting issues

Please include:

- Environment details (OS, Python version, PyTorch version)
- Exact command used
- Full traceback or error message
- Steps to reproduce
