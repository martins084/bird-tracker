import unittest

import cv2
import numpy as np

from tracker import BirdDetector, BirdTracker, DiffDetector, _boxes_from_mask


class FalsePositiveTests(unittest.TestCase):
    def test_widespread_motion_is_rejected(self):
        mask = np.zeros((240, 320), np.uint8)
        mask[::2, :] = 255
        self.assertEqual(_boxes_from_mask(mask, 12, 20000), [])

    def test_large_fragmented_object_rejected_small_bird_kept(self):
        mask = np.zeros((240, 320), np.uint8)
        for y in range(30, 100, 12):
            for x in range(30, 100, 12):
                mask[y:y + 8, x:x + 8] = 255
        mask[180:188, 250:258] = 255
        self.assertEqual(_boxes_from_mask(mask, 12, 20000), [(250, 180, 8, 8)])

    def test_intermittent_noise_never_confirms(self):
        tracker = BirdTracker()
        for frame in range(20):
            self.assertEqual(tracker.update([(20, 20, 8, 8)] if frame % 2 else [], frame), [])
        self.assertEqual(tracker.total_confirmed, 0)

    def test_bird_confirms_and_survives_occlusion_without_recounting(self):
        tracker = BirdTracker()
        for frame in range(5):
            live = tracker.update([(20 + frame * 3, 20, 8, 8)], frame)
        self.assertEqual(len(live), 1)
        bird_id = live[0].id
        self.assertEqual(tracker.update([], 5), [])
        self.assertEqual(tracker.update([(38, 20, 8, 8)], 6)[0].id, bird_id)
        self.assertEqual(tracker.total_confirmed, 1)

    def test_large_size_jump_does_not_reuse_track(self):
        tracker = BirdTracker(min_hits=1)
        first = tracker.update([(50, 50, 8, 8)], 0)[0].id
        second = tracker.update([(35, 35, 40, 40)], 1)[0].id
        self.assertNotEqual(first, second)

    def test_warmup_then_small_moving_bird(self):
        for detector in (BirdDetector(), DiffDetector()):
            with self.subTest(detector=type(detector).__name__):
                background = np.full((240, 320, 3), 180, np.uint8)
                for _ in range(25):
                    self.assertEqual(detector.detect(background)[0], [])
                tracker = BirdTracker()
                for frame in range(15):
                    image = background.copy()
                    cv2.circle(image, (40 + frame * 7, 100), 5, (20, 20, 20), -1)
                    tracker.update(detector.detect(image)[0], frame)
                self.assertGreaterEqual(tracker.total_confirmed, 1)


if __name__ == '__main__':
    unittest.main()
