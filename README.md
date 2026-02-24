# SERS Data Analysis for Salmonella Detection

This toolkit is developed for the analysis of Surface-Enhanced Raman Spectroscopy (SERS) data within the [NSF SENS-D](https://sensd.ai) program. The project focuses on bridging sensor physics with computational intelligence for rapid Salmonella detection.

This specific repository hosts the research and implementation conducted by Jiahe (LJ) Li at the University of Missouri, under the supervision of Dr. Derek Anderson.

---

## Installation

This project is packaged via standard `pyproject.toml` and requires Python 3.12+.

### Developers

Clone the repo and install in editable mode with dev dependencies (testing, linting, notebooks):

```bash
# Clone
git clone https://github.com/LJXXXD/sensd-sers-analysis.git
cd sensd-sers-analysis

# Via uv (recommended)
uv sync --extra dev
uv run pre-commit install   # optional

# Or via pip
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install   # optional
```

---

## Contact

**Jiahe (LJ) Li** — j.li@missouri.edu — University of Missouri
