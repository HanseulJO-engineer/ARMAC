"""Headless Qt integration checks: threading, selection, paths, and plots."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import tempfile
import time
import sys
import unittest
from pathlib import Path
import cv2
import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest
from autofocus.gui import FocusWindow


class DesktopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.callback_errors = []
        self.old_excepthook = sys.excepthook
        sys.excepthook = lambda kind, value, trace: self.callback_errors.append(value)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(4)
        sharp = rng.integers(0, 256, (64,64), dtype=np.uint8)
        for speed in (1,5):
            folder = self.root / f"pollen.spd{speed}"
            folder.mkdir()
            for i in range(7):
                sigma = abs(i-3)*.7
                image = cv2.GaussianBlur(sharp, (0,0), sigma) if sigma else sharp
                ok, data = cv2.imencode(".png", image)
                self.assertTrue(ok)
                (folder / f"image{i+1}.png").write_bytes(data.tobytes())
        self.window = FocusWindow(self.root)
        self.window.show()
        self.app.processEvents()

    def tearDown(self):
        self.window.close()
        self.app.processEvents()
        self.temp.cleanup()
        sys.excepthook = self.old_excepthook
        self.assertEqual(self.callback_errors, [], "Unhandled Qt callback exception")

    def test_end_to_end_desktop_flow(self):
        window = self.window
        self.assertEqual(window.sources.count(), 2)
        self.assertEqual(window.files_table.rowCount(), 7)
        self.assertTrue(Path(window.file_path.text()).is_file())
        window.checks["gaussian_derivative"].click()
        self.assertIn("GAUSSIAN", window.formula_panel.title.text())
        self.assertIsNotNone(window.formula_panel.equation.pixmap())
        window.select_methods(True)
        window.start_analysis(True)
        self.assertTrue(window.busy_panel.isVisible())
        self.assertFalse(window.run_button.isEnabled())
        deadline = time.monotonic() + 15
        while window.worker.isRunning() and time.monotonic() < deadline:
            QTest.qWait(30)
        self.app.processEvents()
        self.assertFalse(window.worker.isRunning(), "analysis thread timed out")
        self.assertEqual(len(window.results), 2)
        self.assertEqual(window.result.entries[window.result.methods["normalized_variance"].search.index].sequence, 3)
        self.assertTrue(window.export_button.isEnabled())
        self.assertEqual(window.progress.value(), 100)
        self.assertFalse(window.busy_panel.isVisible())
        self.assertEqual(len(window.top_cards), 3)
        self.assertEqual(window.top_cards[0].index, 3)
        self.assertIn("image4.png", window.top_cards[0].filename.text())
        self.assertIn("#4 / 7", window.top_cards[0].position.text())
        window.top_cards[1].clicked.emit(window.top_cards[1].index)
        self.assertEqual(window.current_entry.sequence, window.result.entries[window.top_cards[1].index].sequence)
        for index in (1,2,4):
            window.tabs.setCurrentIndex(index)
            self.app.processEvents()
        self.assertIsNotNone(window.detail_plot.canvas)
        self.assertIsNotNone(window.compare_plot.canvas)
        self.assertIsNotNone(window.batch_plot.canvas)
        self.assertEqual(window.batch_table.rowCount(), 26)
        window.compare_view.setCurrentIndex(1)
        window.tabs.setCurrentIndex(1)
        self.assertEqual(len(window.compare_plot.canvas.figure.axes), 3)
        window.method_combo.setCurrentIndex(window.method_combo.findData("laplacian_variance"))
        self.assertIn("LAPLACIAN", window.formula_panel.title.text())
        self.assertIsNotNone(window.formula_panel.equation.pixmap())
        for key in window.result.scores:
            window.method_combo.setCurrentIndex(window.method_combo.findData(key))
            self.assertIsNotNone(window.formula_panel.equation.pixmap())
        window.tabs.setCurrentIndex(3)
        window.files_table.setCurrentCell(1, 0)
        self.assertEqual(window.current_entry.sequence, 1)
        self.assertEqual(window.tabs.currentIndex(), 0)
        window.set_roi((.15,.15,.7,.7))
        self.assertEqual(window.options().roi, (.15,.15,.7,.7))
        window.edges.setChecked(True)
        self.assertIsNotNone(window.viewer.pixmap)
        window.copy_path()
        self.assertEqual(self.app.clipboard().text(), window.file_path.text())


if __name__ == "__main__":
    unittest.main()
