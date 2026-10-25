"""Accelerate the original glass-blur pixel loop without changing its output."""

import numpy as np

try:
    from numba import njit
except ImportError:
    njit = None


def _apply_offsets(image, offsets, delta):
    for iteration in range(offsets.shape[0]):
        for row in range(offsets.shape[1]):
            h = image.shape[0] - delta - row
            for column in range(offsets.shape[2]):
                w = image.shape[1] - delta - column
                dx, dy = offsets[iteration, row, column]
                # The upstream tuple assignment holds views into the image.
                # Its first write changes the second RHS view: both pixels end
                # up with the target's value. Preserve that actual behavior,
                # rather than replacing it with a mathematically different swap.
                for channel in range(image.shape[2]):
                    image[h, w, channel] = image[h + dy, w + dx, channel]


if njit is not None:
    _apply_offsets = njit(cache=True)(_apply_offsets)


def shuffle_pixels(image, delta, iterations):
    """Use NumPy's caller RNG, including the original order/count of draws."""
    shape = (iterations, max(0, image.shape[0] - 2 * delta),
             max(0, image.shape[1] - 2 * delta), 2)
    offsets = np.random.randint(-delta, delta, size=shape)
    _apply_offsets(image, offsets, delta)
