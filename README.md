## Installation

This repository provides two dependency files:

- `requirements.txt`: minimal runtime dependencies
- `requirements-lock.txt`: full frozen environment used for thesis runs (maximum reproducibility)

### Minimal setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
