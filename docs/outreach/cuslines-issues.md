# Draft issues for dipy/GPUStreamlines (cuslines)

Status: drafts, 2026-11-07, for the user to send (two separate issues).
Evidence in `tests/validation/test_doc014_tools.py` (GPU-gated tests) and
doc 014 §4.3. Checked against master b1a7812 (2026-06-03) and the PyPI
release cuslines 2.2.3 (CUDA backend, GB10). No existing issue mentions
either (`gh search issues --repo dipy/GPUStreamlines`).

---

## Issue 1 (data-corrupting)

**Title:** `generate_sft` returns the last chunk's streamlines once per chunk when seeds exceed `chunk_size`

`GenericTracker.generate_sft` (`cuslines/generic_tracker.py`) loops over
seed chunks, calls `self.seed_propagator.propagate(chunk)` and appends
`self.seed_propagator.as_generator()` to a list, then builds one
`ArraySequence` from all generators after the loop. `as_generator`
(`cuda_python/cu_propagate_seeds.py`) is lazy: it reads
`self.slines[ii]`, `self.sline_lens[ii]` and `self.nSlines[ii]` when
iterated, and `propagate` overwrites those for the next chunk. So every
generator yields the streamlines of the **last** chunk; the earlier chunks
are lost and the last one is repeated.

**Reproduction** (ProbDirectionGetter, CUDA, 35 376 seeds in voxel space,
one GPU):

| call | streamlines |
|---|---|
| all seeds, `chunk_size=40000` (1 chunk) | 91 712 |
| all seeds, `chunk_size=25000` (2 chunks) | 55 008 |
| last 10 376 seeds alone | 27 504 (= 55 008 / 2) |
| all seeds, `chunk_size=10000` (4 chunks) | 54 536 |
| last 5 376 seeds alone | 13 634 (= 54 536 / 4) |

A 12³ single-fibre phantom with 512 seeds gives the same pattern
(`test_cuslines_generate_sft_is_chunk_invariant_upstream`, marked xfail).
The default `chunk_size` is 100 000, so any whole-brain run with more
seeds than that is affected silently; the output looks like a valid
tractogram (lengths, mask coverage) but covers only the last chunk's
seeds. On the DiSCo phantom this took the connectome correlation with
ground truth from 0.87 to 0.55.

**Fix:** materialise each chunk before the next `propagate`, e.g.
`generators.append(list(self.seed_propagator.as_generator()))` or use
`as_array_sequence()` per chunk and concatenate; `generate_trx` already
consumes per chunk and is not affected. Happy to open the PR with the
phantom test.

---

## Issue 2 (half-voxel bias)

**Title:** Stop map is sampled half a voxel off the data grid (texture fetch without the texel-centre offset)

`check_point_d` (`cuslines/cuda_c/tracking_helpers.cu`) reads the stopping
map with

```c
float val = tex3D<float>(*metric_map, (float) point.z, (float) point.y, (float) point.x);
```

and the texture is created with `cudaFilterModeLinear` and
`normalizedCoords = 0` (`cuslines/cuda_python/cutils.py`). With unnormalised
coordinates CUDA centres texel `i` at `i + 0.5`, so this fetch returns the
map interpolated at `point - 0.5` in voxel units. `trilinear_interp_d` for
the PMF/ODF data uses `floor(point)` with the usual
`[-0.5, DIM - 0.5)` bounds, i.e. DIPY's convention where voxel `i` spans
`[i - 0.5, i + 0.5)`. The two are inconsistent by half a voxel, and the
seeds and output streamlines follow the data convention.

**Reproduction** (ProbDirectionGetter, single-fibre PMF along z on a 12³
grid, stop map = 1 on voxels z = 3..7, seed at (5, 5, 5), identity
affine): DIPY's `BinaryStoppingCriterion` stops the streamline inside
`[2.5, 7.5)`; cuslines returns points spanning z = 3.02 … 7.97, i.e. the
boundary is at `[3, 8)`.

**Effect:** streamlines leave the stopping region half a voxel late on
the +x/+y/+z faces and half a voxel early on the −faces, so endpoints
are biased by half a voxel relative to DIPY's `LocalTracking`. For an
ROI connectome the effect is small (DiSCo 1, 16 ROIs: r 0.864 against
ground truth with the raw endpoints vs 0.877 for DIPY's probabilistic
tracker on the same PMF), but it is systematic and will matter for
thin stopping masks (e.g. a one-voxel GM ribbon as in ACT-style
tracking).

**Fix:** add the texel-centre offset in the fetch,

```c
float val = tex3D<float>(*metric_map, (float) point.z + 0.5f, (float) point.y + 0.5f, (float) point.x + 0.5f);
```

(or switch the texture to `cudaFilterModePoint` and keep the +0.5, which
also matches DIPY's nearest-neighbour `BinaryStoppingCriterion` exactly).
The same fetch pattern should be checked in the Metal / WebGPU / numba
backends. Happy to open a PR with the one-line change and the phantom
test if that is wanted.
