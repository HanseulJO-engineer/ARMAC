import unittest
import tempfile
from pathlib import Path
from zipfile import ZipFile
import cv2
import numpy as np
from autofocus.metrics import calculate_focus, BY_KEY
from autofocus.dataset import open_dataset, discover_sources, grayscale
from autofocus.search import find_coarse_sharpest, estimate_fine_position
from autofocus.analysis import AnalysisOptions, analyze_dataset, Cancelled, rank_correlation, ranks, top_indices
from autofocus.export import export_results
import csv
import json


def write_image(path, image):
    ok, buf = cv2.imencode(".png", image)
    assert ok
    path.write_bytes(buf.tobytes())


class FocusTests(unittest.TestCase):
    def test_normalized_variance_known_intensities(self):
        a = np.tile([0., 100., 200.], (3, 1))
        self.assertAlmostEqual(calculate_focus(a, ["normalized_variance"])["normalized_variance"], 200/3)

    def test_all_measures_on_constant_and_black_images(self):
        for value in (0, 72):
            scores = calculate_focus(np.full((12, 13), value, np.uint8), tuple(BY_KEY))
            self.assertTrue(all(abs(s) < 1e-10 for s in scores.values()), scores)

    def test_sharp_texture_beats_blurred_image(self):
        rng = np.random.default_rng(13)
        sharp = rng.integers(0, 256, (128, 128), dtype=np.uint8)
        blur = cv2.GaussianBlur(sharp, (0, 0), 3)
        keys = ["normalized_variance", "tenengrad", "laplacian_variance", "squared_gradient", "brenner", "wavelet1", "wavelet2", "modified_laplacian", "gaussian_derivative", "tenengrad_variance"]
        a, b = calculate_focus(sharp, keys), calculate_focus(blur, keys)
        for key in keys:
            self.assertGreater(a[key], b[key], key)

    def test_16bit_intensity_uses_fixed_scale(self):
        image = np.full((5,5), 32768, np.uint16)
        self.assertAlmostEqual(grayscale(image)[0,0], 32768 * 255 / 65535)

    def test_local_trap_is_corrected(self):
        x = 10 + np.arange(80)*.01
        y = 5*np.exp(-((np.arange(80)-15)/6)**2) + 9*np.exp(-((np.arange(80)-65)/6)**2)
        result = find_coarse_sharpest(x, y, start_fraction=.15)
        self.assertEqual(result.index, 65)
        self.assertEqual(result.local_index, 15)
        self.assertTrue(result.corrected)
        for strategy in ("global", "multistart"):
            self.assertEqual(find_coarse_sharpest(x, y, strategy).index, 65)

    def test_quadratic_recovers_substep_vertex(self):
        x = 10.26 + np.arange(9)*.005
        true_peak = 10.2813
        y = 8 - 500*(x-true_peak)**2
        coarse = int(np.argmax(y))
        result = estimate_fine_position(x, y, coarse, 5)
        self.assertAlmostEqual(result.distance_mm, true_peak, places=10)
        self.assertAlmostEqual(result.r_squared, 1, places=10)

    def test_fine_position_is_guarded(self):
        x = np.arange(7, dtype=float)
        self.assertIsNone(estimate_fine_position(x, np.arange(7), 6).distance_mm)
        self.assertIsNone(estimate_fine_position(x, np.ones(7), 3).distance_mm)
        self.assertIsNone(estimate_fine_position(x, (x-3)**2, 3).distance_mm)

    def test_rank_ties_and_undefined_correlations(self):
        np.testing.assert_equal(ranks(np.array([2.,1.,2.,4.])), [1.5,0,1.5,3])
        corr = rank_correlation({"a": np.arange(4), "b": -np.arange(4), "c": np.ones(4)})
        self.assertAlmostEqual(corr[0,1], -1)
        self.assertTrue(np.isnan(corr[0,2]))

    def test_top_three_preserve_original_order_for_ties(self):
        self.assertEqual(top_indices([1., 5., 2., 5., 3.]), [1, 3, 4])


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def make_set(self):
        for name in ("img10.png", "img2.png", "img1.png"):
            write_image(self.path / name, np.arange(256, dtype=np.uint8).reshape(16,16))
        (self.path / "set.WDs.yml").write_text("%YAML:1.0\nWD: [10.26, 10.265, 10.27, 10.275]\n")

    def test_natural_order_and_metadata_offset(self):
        self.make_set()
        dataset = open_dataset(self.path)
        self.assertEqual([e.name for e in dataset.images], ["img1.png", "img2.png", "img10.png"])
        self.assertEqual(dataset.images[0].distance_mm, 10.26)
        self.assertEqual(open_dataset(self.path, metadata_offset=1).images[0].distance_mm, 10.265)
        self.assertEqual(len(dataset.warnings), 2)
        manual = open_dataset(self.path, start_mm=10, step_um=10, use_metadata=False)
        self.assertEqual(manual.images[-1].distance_mm, 10.02)

    def test_zip_and_directory_match(self):
        self.make_set()
        archive = self.path / "set.zip"
        with ZipFile(archive, "w") as z:
            for file in self.path.iterdir():
                if file != archive:
                    z.write(file, "set/"+file.name)
        directory, zipped = open_dataset(self.path), open_dataset(archive)
        a = analyze_dataset(directory, AnalysisOptions(("normalized_variance",)))
        b = analyze_dataset(zipped, AnalysisOptions(("normalized_variance",)))
        np.testing.assert_equal(a.scores["normalized_variance"], b.scores["normalized_variance"])
        self.assertIn("::", zipped.display_path(zipped.images[0]))

    def test_bad_image_does_not_shift_physical_positions(self):
        self.make_set()
        (self.path / "img2.png").write_bytes(b"not an image")
        result = analyze_dataset(open_dataset(self.path), AnalysisOptions(("normalized_variance",)))
        self.assertEqual([e.name for e in result.entries], ["img1.png", "img10.png"])
        self.assertEqual(result.entries[1].distance_mm, 10.27)
        self.assertTrue(any("img2.png" in w for w in result.warnings))

    def test_cancel_and_export_reproducibility(self):
        self.make_set()
        data = open_dataset(self.path)
        with self.assertRaises(Cancelled):
            analyze_dataset(data, AnalysisOptions(), cancelled=lambda: True)
        result = analyze_dataset(data, AnalysisOptions(("normalized_variance", "tenengrad")))
        report = export_results([result, result], self.path / "export")
        self.assertTrue(report.exists())
        self.assertIn("data:image/png;base64,", report.read_text())
        self.assertTrue((report.parent / "scanning_speed_comparison.png").exists())
        with (report.parent / "top_three.csv").open(encoding="utf-8-sig") as f:
            top = list(csv.DictReader(f))
        self.assertEqual(len(top), 12)
        self.assertEqual(top[0]["filename"], "img1.png")
        self.assertEqual(top[0]["image_sequence_1_based"], "1")
        sub = next(p for p in report.parent.iterdir() if p.is_dir())
        summary = json.loads((sub / "summary.json").read_text())
        self.assertEqual(summary["images_analyzed"], 3)
        self.assertEqual(summary["settings"]["fit_window"], 5)
        self.assertEqual(len(summary["methods"]["normalized_variance"]["top_three"]), 3)
        self.assertGreater(summary["methods"]["normalized_variance"]["statistics"]["median_scoring_ms"], 0)
        with (sub / "scores.csv").open(encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 3)
        self.assertEqual(float(rows[0]["working_distance_mm"]), 10.26)
        self.assertTrue(Path(rows[0]["absolute_source_path"]).is_absolute())

    def test_invalid_roi_is_not_silently_skipped(self):
        self.make_set()
        with self.assertRaises(ValueError):
            analyze_dataset(open_dataset(self.path), AnalysisOptions(roi=(.9,0,.5,1)))


if __name__ == "__main__":
    unittest.main()
