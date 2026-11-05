#!/usr/bin/env bash
# Rename the package (doc 012 §7.1, decided 2026-11-05):
#   dmipy_jax                    -> compore   (the platform: kernels, fitting, simulation, oracles, SBI/UQ)
#   dmipy_jax.prism + io.fixel + workflows.refine_peaks_flow
#                                -> lafix     (the product: Laplace fixel posterior refine, DIPY plug-in)
# One repository, two distributions (uv workspace: root = compore, packages/lafix = lafix).
# Historical logs (docs/decisions/*, CHANGELOG.md) are NOT rewritten: they cite the old names on purpose.
# Run from the repository root on a clean tree:  bash scripts/rename_to_compore_lafix.sh [--dry-run]
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
DRY=${1:-}
[ -n "$(git status --porcelain)" ] && { echo "working tree not clean"; exit 1; }
[ -d dmipy_jax ] || { echo "dmipy_jax/ not found (already renamed?)"; exit 1; }

# files to rewrite: tracked text files except the historical logs and this script
FILES=$(git ls-files '*.py' '*.md' '*.toml' '*.yml' '*.yaml' '*.cff' '*.ipynb' '*.spec' '*.sh' '*.txt' '*.cfg' '*.ini' '*.rst' \
        | grep -v -E '^(docs/decisions/|CHANGELOG\.md|scripts/rename_to_compore_lafix\.sh)' | xargs grep -l -E 'dmipy_jax|dmipy-jax' || true)
echo "files to rewrite: $(echo "$FILES" | grep -c . )   occurrences: $(echo "$FILES" | xargs grep -o -E 'dmipy_jax|dmipy-jax' | wc -l)"
if [ "$DRY" = "--dry-run" ]; then exit 0; fi

# 1. move the platform
git mv dmipy_jax compore

# 2. carve out the product
mkdir -p packages/lafix/lafix
git mv compore/prism/prism_jax.py           packages/lafix/lafix/prism_jax.py
git mv compore/prism/prism_uncertainty.py   packages/lafix/lafix/prism_uncertainty.py
git mv compore/prism/dipy_refine.py         packages/lafix/lafix/dipy_refine.py
git mv compore/prism/tracking.py            packages/lafix/lafix/tracking.py
git mv compore/prism/__init__.py            packages/lafix/lafix/__init__.py
git mv compore/io/fixel.py                  packages/lafix/lafix/fixel.py
git mv compore/workflows/refine_peaks_flow.py packages/lafix/lafix/workflow.py
[ -z "$(ls -A compore/prism 2>/dev/null | grep -v __pycache__)" ] && rm -rf compore/prism
[ -z "$(ls -A compore/workflows 2>/dev/null | grep -v -E '__pycache__|__init__.py')" ] && git rm -rq compore/workflows 2>/dev/null && rm -rf compore/workflows || true
git mv tests/validation/test_prism_jax.py         tests/validation/test_lafix_prism.py
git mv tests/validation/test_prism_uncertainty.py tests/validation/test_lafix_uncertainty.py
git mv tests/validation/test_prism_exchange.py    tests/validation/test_lafix_exchange.py

# 3. rewrite imports and names (specific paths first, then the package)
FILES=$(git ls-files '*.py' '*.md' '*.toml' '*.yml' '*.yaml' '*.cff' '*.ipynb' '*.spec' '*.sh' '*.txt' '*.cfg' '*.ini' '*.rst' \
        | grep -v -E '^(docs/decisions/|CHANGELOG\.md|scripts/rename_to_compore_lafix\.sh)' | xargs grep -l -E 'dmipy_jax|dmipy-jax' || true)
echo "$FILES" | xargs sed -i -E \
  -e 's/dmipy_jax\.prism\.prism_jax/lafix.prism_jax/g' \
  -e 's/dmipy_jax\.prism\.prism_uncertainty/lafix.prism_uncertainty/g' \
  -e 's/dmipy_jax\.prism\.dipy_refine/lafix.dipy_refine/g' \
  -e 's/dmipy_jax\.prism\.tracking/lafix.tracking/g' \
  -e 's/from dmipy_jax\.prism import/from lafix import/g' \
  -e 's/dmipy_jax\.prism/lafix/g' \
  -e 's/dmipy_jax\.io\.fixel/lafix.fixel/g' \
  -e 's/dmipy_jax\.workflows\.refine_peaks_flow/lafix.workflow/g' \
  -e 's/dmipy_jax\/prism/packages\/lafix\/lafix/g' \
  -e 's/dmipy_jax/compore/g' \
  -e 's/dmipy-jax/compore/g'

# 4. packaging: root = compore, packages/lafix = lafix (uv workspace)
python3 - <<'PY'
import re, pathlib
p = pathlib.Path("pyproject.toml"); s = p.read_text()
s = s.replace('name = "sbi4dwi"', 'name = "compore"')
s = s.replace('description = "Simulation-Based Inference for Diffusion-Weighted Imaging"',
              'description = "Compartments and pores: differentiable diffusion-MRI microstructure physics, Monte Carlo oracles and simulation-based inference in JAX"')
s = s.replace('"sbi4dwi[sbi,sim,algebra,inverse,nn,viz,data,baselines]",', '"compore[sbi,sim,algebra,inverse,nn,viz,data,baselines]",\n  "lafix",')
s = s.replace('# Everything above (the development environment)', '# The product (Laplace fixel refine), developed in packages/lafix\nlafix = ["lafix"]\n# Everything above (the development environment)')
s = re.sub(r'\[project.scripts\]\ndipy_refine_peaks = "[^"]+"\n', '[project.scripts]\n', s)
s = s.replace('[tool.uv.sources]\n', '[tool.uv.workspace]\nmembers = ["packages/lafix"]\n\n[tool.uv.sources]\nlafix = { workspace = true }\n')
p.write_text(s)
pathlib.Path("packages/lafix/pyproject.toml").write_text('''[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "lafix"
version = "0.1.0"
description = "Laplace fixel posteriors: refine any DIPY peaks with a differentiable multi-fixel fit and a calibrated per-fixel uncertainty"
readme = "README.md"
license = { text = "MIT" }
requires-python = ">=3.12"
authors = [{ name = "Morgan Hough" }]
dependencies = [
  "compore",
  "jax",
  "numpy",
  "scipy",
  "optax>=0.2.6",
  "dipy>=1.12.1",
  "nibabel",
]

[project.scripts]
dipy_refine_peaks = "lafix.workflow:main"
lafix = "lafix.workflow:main"

[tool.uv.sources]
compore = { workspace = true }

[tool.hatch.build.targets.wheel]
packages = ["lafix"]
''')
pathlib.Path("packages/lafix/README.md").write_text('''# lafix

Laplace fixel posteriors for diffusion MRI. `lafix` takes any DIPY
`PeaksAndMetrics` (CSD, MSMT-CSD, FORCE), refines it with a differentiable
K-fixel stick + zeppelin + CSF/GM fit under a Rician likelihood, and returns
refined peaks plus a per-fixel Laplace posterior (orientation σ, intra-axonal
fraction σ) with an in-vivo calibration constant from 44 HCP scan–rescan
subjects. Whole brain in about two minutes on CPU.

```bash
uv add lafix           # pulls compore (the physics) and dipy
dipy_refine_peaks dwi.nii.gz dwi.bval dwi.bvec mask.nii.gz peaks.pam5 --out_dir out/
```

Python: `from lafix import dipy_refine as dr; pam_ref, post, fit = dr.refine_peaks(data, gtab, mask, pam)`.

Results and validation: the `compore` repository, `docs/decisions/008-*.md`.
''')
PY

# 5. lazy package init: drop 'prism' from compore.__all__
sed -i -E "s/'signal_models', 'prism'\]/'signal_models']/" compore/__init__.py

# 6. Sphinx: document both packages
python3 - <<'PY'
import pathlib
p = pathlib.Path("docs/conf.py"); s = p.read_text()
s = s.replace("""    {
        'path': '../compore',""", """    {
        'path': '../packages/lafix/lafix',
        'destination': 'reference/lafix',
        'separate_modules': False,
        'max_depth': 2,
    },
    {
        'path': '../compore',""")
p.write_text(s)
PY

uv lock && uv sync --all-extras
echo
echo "renamed. Now: JAX_PLATFORMS=cpu uv run pytest -q tests/validation compore/tests/test_oracle.py && uv run dipy_refine_peaks --help | head -3"
echo "then commit, and rename the GitHub repository and this directory to compore yourself."
