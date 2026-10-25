"""Run with ``python -m unittest discover -s tests -p test_glass_blur.py``.

Load only the image functions so these checks need no robot assets or renderer.
"""

import ast
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
from skimage.filters import gaussian


ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / "libero/libero/envs"
spec = importlib.util.spec_from_file_location(
    "libero.libero.envs.glass_blur_shuffle", ENV / "glass_blur_shuffle.py"
)
shuffle = importlib.util.module_from_spec(spec)
# Keep the production module identity in Numba's on-disk cache without loading
# the full simulator package during these image-only tests.
sys.modules[spec.name] = shuffle
spec.loader.exec_module(shuffle)
tree = ast.parse((ENV / "env_wrapper.py").read_text())
function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "glass_blur")
scope = {"np": np, "gaussian": gaussian, "shuffle_pixels": shuffle.shuffle_pixels}
exec(compile(ast.Module(body=[function], type_ignores=[]), str(ENV / "env_wrapper.py"), "exec"), scope)
glass_blur = scope["glass_blur"]


def original_shuffle(image, delta, iterations):
    for _ in range(iterations):
        for h in range(image.shape[0] - delta, delta, -1):
            for w in range(image.shape[1] - delta, delta, -1):
                dx, dy = np.random.randint(-delta, delta, size=(2,))
                h_prime, w_prime = h + dy, w + dx
                image[h, w], image[h_prime, w_prime] = image[h_prime, w_prime], image[h, w]


def original_glass_blur(image, severity):
    prior = scope["shuffle_pixels"]
    try:
        scope["shuffle_pixels"] = original_shuffle
        return glass_blur(image, severity)
    finally:
        scope["shuffle_pixels"] = prior


class GlassBlurEquivalence(unittest.TestCase):
    def test_all_severities_preserve_pixels_and_rng(self):
        image = np.random.RandomState(991).randint(0, 256, (37, 53, 3), dtype=np.uint8)
        for severity in range(1, 11):
            for seed in (0, 17, 123):
                with self.subTest(severity=severity, seed=seed):
                    np.random.seed(seed)
                    expected = original_glass_blur(image, severity)
                    expected_state = np.random.get_state()
                    np.random.seed(seed)
                    actual = glass_blur(image, severity)
                    actual_state = np.random.get_state()
                    np.testing.assert_array_equal(actual, expected)
                    self.assertEqual(actual_state[0], expected_state[0])
                    np.testing.assert_array_equal(actual_state[1], expected_state[1])
                    self.assertEqual(actual_state[2:], expected_state[2:])

    def test_python_fallback_preserves_pixel_assignment(self):
        apply = getattr(shuffle._apply_offsets, "py_func", shuffle._apply_offsets)
        image = np.arange(21 * 27 * 3, dtype=np.uint8).reshape(21, 27, 3)
        np.random.seed(71)
        expected = image.copy()
        original_shuffle(expected, 3, 2)
        expected_state = np.random.get_state()
        np.random.seed(71)
        offsets = np.random.randint(-3, 3, size=(2, 15, 21, 2))
        actual = image.copy()
        apply(actual, offsets, 3)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(np.random.get_state()[1], expected_state[1])
        self.assertEqual(np.random.get_state()[2:], expected_state[2:])

    def test_fog_preserves_small_images_and_supports_large_frames(self):
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in ("plasma_fractal", "fog")]
        fog_scope = {"np": np}
        exec(compile(ast.Module(body=functions, type_ignores=[]), "fog", "exec"), fog_scope)
        image = np.random.RandomState(8).randint(0, 256, (224, 224, 3), dtype=np.uint8)
        np.random.seed(123)
        expected = image.astype(np.float64) / 255.0
        maximum = expected.max()
        expected += 0.5 * fog_scope["plasma_fractal"](mapsize=256, wibbledecay=3)[:224, :224, None]
        expected = np.clip(expected * maximum / (maximum + 0.5), 0, 1) * 255
        np.random.seed(123)
        np.testing.assert_array_equal(fog_scope["fog"](image, 1), expected)
        large = np.full((384, 512, 3), 127, dtype=np.uint8)
        actual = fog_scope["fog"](large, 1)
        self.assertEqual(actual.shape, large.shape)
        self.assertTrue(np.isfinite(actual).all())


if __name__ == "__main__":
    unittest.main()
