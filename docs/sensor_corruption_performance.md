# Sensor corruption performance

The glass-blur implementation preserves the original output and NumPy random
stream while moving its sequential pixel loop out of Python. Random offsets are
still drawn by `numpy.random.randint` in the original iteration order. The loop
uses cached Numba compilation when Numba is installed, and a Python fallback
otherwise. Install a Numba version compatible with the environment's Python and
NumPy to enable acceleration; it is optional rather than a change to the
project's older pinned dependencies.

The original tuple assignment uses NumPy views: writing the first pixel also
changes the second right-hand-side view. The optimized loop deliberately retains
that behavior; changing it to a conventional pixel swap would change benchmark
images. Gaussian filtering, severity settings, image size, and returned dtype
are unchanged. Random offsets occupy at most about 12 MiB for a 512 × 512 RGB
image, and are released after each invocation.

Local measurements with NumPy 1.26.4, Numba 0.67.0, and 512 × 512 RGB images:

| Severity | Original seconds | Optimized seconds |
|---|---:|---:|
| 1 | 4.706 | 0.035 |
| 2 | 4.704 | 0.034 |
| 3 | 4.739 | 0.033 |
| 4 | 3.178 | 0.032 |
| 5 | 3.132 | 0.035 |
| 6 | 3.191 | 0.041 |
| 7 | 3.154 | 0.034 |
| 8 | 3.158 | 0.036 |
| 9 | 1.599 | 0.040 |
| 10 | 1.563 | 0.041 |

These single-image timings exclude first-use compilation. For every row, the
complete floating-point output and the following NumPy RNG state were exactly
equal to the original loop. They are not end-to-end rollout speedups.

The accompanying fog fix retains the original 256 × 256 fractal for images up
to that size and chooses a covering power-of-two map for larger images. This
fix was already used in the evaluation environment before the glass-blur
optimization; upstream previously failed when broadcasting fog onto 512-pixel
frames.

Run the lightweight tests without robot assets or an EGL context:

```bash
python -m unittest discover -s tests -p test_glass_blur.py
```

Tests cover all ten severities with three seeds on rectangular RGB images,
pixel equality, full random-state equality, the Python fallback, and fog behavior
for small and large images. On shared machines, set `NUMBA_CACHE_DIR` to an
appropriate persistent cache directory before running simulators.
