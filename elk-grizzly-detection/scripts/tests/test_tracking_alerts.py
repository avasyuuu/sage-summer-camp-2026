"""State-machine tests for track confirmation and per-object SMS alerts."""

import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from alerts import (  # noqa: E402
    AlertConfig,
    AlertConfigurationError,
    AlertManager,
    DeliveryResult,
    SMSAlertSender,
    load_alert_config,
)
from tracker import AnimalTrackRegistry  # noqa: E402


class FakeCrop:
    size = 1

    def copy(self):
        return self


class FakeFrame:
    shape = (100, 100, 3)

    def __getitem__(self, key):
        return FakeCrop()


FRAME = FakeFrame()


def detection(track_id=None, confidence=0.8, box=(10, 10, 50, 50), label="bear"):
    return {
        "track_id": track_id,
        "label": label,
        "confidence": confidence,
        "box": box,
    }


def classify_as_dangerous(track):
    track.apply_classification(
        {
            "species": "Ursus arctos",
            "common_name": "grizzly bear",
            "score": 0.95,
        },
        {"hazard": "dangerous", "danger_score": 9},
    )


class FakeSender:
    def __init__(self, succeeds=True):
        self.succeeds = succeeds
        self.calls = []

    def send(self, body):
        self.calls.append(body)
        if self.succeeds:
            return [DeliveryResult("+15551232832", "SM-test", "queued")]
        return [DeliveryResult("+15551232832", error="test failure")]


class TrackRegistryTests(unittest.TestCase):
    def test_confirms_on_third_consecutive_frame(self):
        registry = AnimalTrackRegistry(confirmation_frames=3)

        self.assertEqual(registry.update([detection(1)], FRAME, "one.jpg"), [])
        self.assertEqual(registry.update([detection(1)], FRAME, "two.jpg"), [])
        confirmed = registry.update([detection(1)], FRAME, "three.jpg")

        self.assertEqual([track.track_id for track in confirmed], [1])
        self.assertTrue(registry.tracks[1].confirmed)
        self.assertEqual(registry.tracks[1].consecutive_seen, 3)

    def test_miss_resets_unconfirmed_streak(self):
        registry = AnimalTrackRegistry(confirmation_frames=3)
        registry.update([detection(1)], FRAME, "one.jpg")
        registry.update([detection(1)], FRAME, "two.jpg")
        registry.update([], FRAME, "miss.jpg")
        registry.update([detection(1)], FRAME, "four.jpg")

        track = registry.tracks[1]
        self.assertFalse(track.confirmed)
        self.assertEqual(track.consecutive_seen, 1)

    def test_sparse_motion_and_label_changes_keep_one_identity(self):
        registry = AnimalTrackRegistry(confirmation_frames=3)
        first = detection(box=(0, 40, 20, 70), label="bear")
        second = detection(box=(25, 40, 48, 70), label="cow")
        third = detection(box=(57, 39, 82, 70), label="bear")

        registry.update([first], FRAME, "one.jpg")
        registry.update([second], FRAME, "two.jpg")
        confirmed = registry.update([third], FRAME, "three.jpg")

        self.assertEqual(first["track_id"], second["track_id"])
        self.assertEqual(second["track_id"], third["track_id"])
        self.assertEqual([track.track_id for track in confirmed], [1])

    def test_two_moving_animals_remain_separate(self):
        registry = AnimalTrackRegistry(confirmation_frames=3)
        frames = [
            [
                detection(box=(5, 10, 25, 35)),
                detection(box=(70, 55, 95, 85)),
            ],
            [
                detection(box=(12, 10, 32, 35)),
                detection(box=(62, 55, 87, 85)),
            ],
            [
                detection(box=(20, 10, 40, 35)),
                detection(box=(53, 55, 78, 85)),
            ],
        ]

        for index, detections in enumerate(frames, 1):
            confirmed = registry.update(
                detections,
                FRAME,
                f"frame{index}.jpg",
            )

        self.assertEqual(len(confirmed), 2)
        self.assertEqual(
            [detection["track_id"] for detection in frames[0]],
            [1, 2],
        )
        self.assertEqual(
            [detection["track_id"] for detection in frames[2]],
            [1, 2],
        )

    def test_different_fixed_camera_scenes_do_not_match(self):
        registry = AnimalTrackRegistry(confirmation_frames=3)
        vertical = np.tile(np.arange(100, dtype=np.uint8)[:, None], (1, 100))
        horizontal = vertical.T
        first_scene = np.repeat(vertical[:, :, None], 3, axis=2)
        second_scene = np.repeat(horizontal[:, :, None], 3, axis=2)
        first = detection()
        second = detection()

        registry.update([first], first_scene, "camera-a.jpg")
        registry.update([second], second_scene, "camera-b.jpg")

        self.assertNotEqual(first["track_id"], second["track_id"])
        self.assertEqual(registry.tracks[first["track_id"]].consecutive_seen, 0)
        self.assertEqual(registry.tracks[second["track_id"]].consecutive_seen, 1)

    def test_multiple_tracks_confirm_independently(self):
        registry = AnimalTrackRegistry(confirmation_frames=3)
        registry.update([detection(1)], FRAME, "one.jpg")
        registry.update([detection(1), detection(2)], FRAME, "two.jpg")
        first = registry.update([detection(1), detection(2)], FRAME, "three.jpg")
        second = registry.update([detection(1), detection(2)], FRAME, "four.jpg")

        self.assertEqual([track.track_id for track in first], [1])
        self.assertEqual([track.track_id for track in second], [2])

    def test_track_is_removed_on_configured_missed_frame(self):
        registry = AnimalTrackRegistry(
            confirmation_frames=1,
            max_missed_frames=3,
        )
        seen = detection()
        registry.update([seen], FRAME, "seen.jpg")
        track_id = seen["track_id"]
        registry.update([], FRAME, "miss1.jpg")
        registry.update([], FRAME, "miss2.jpg")
        self.assertIn(track_id, registry.tracks)

        registry.update([], FRAME, "miss3.jpg")
        self.assertNotIn(track_id, registry.tracks)

    def test_reappearing_retired_id_is_a_new_track(self):
        registry = AnimalTrackRegistry(
            confirmation_frames=1,
            max_missed_frames=1,
        )
        first = registry.update([detection(3)], FRAME, "first.jpg")[0]
        first.alerted = True
        registry.update([], FRAME, "empty.jpg")

        second = registry.update([detection(3)], FRAME, "return.jpg")[0]
        self.assertIsNot(first, second)
        self.assertFalse(second.alerted)


class TrackAlertTests(unittest.TestCase):
    def _confirmed_track(self):
        registry = AnimalTrackRegistry(confirmation_frames=1)
        track = registry.update([detection(1)], FRAME, "bear.jpg")[0]
        classify_as_dangerous(track)
        return track

    def test_successful_delivery_marks_track_alerted(self):
        track = self._confirmed_track()
        sender = FakeSender()
        manager = AlertManager(sender=sender)

        first = manager.handle_tracks("bear.jpg", [track])
        second = manager.handle_tracks("bear2.jpg", [track])

        self.assertEqual(first[0].status, "sent")
        self.assertEqual(second, [])
        self.assertTrue(track.alerted)
        self.assertEqual(len(sender.calls), 1)

    def test_failed_delivery_remains_eligible_for_retry(self):
        track = self._confirmed_track()
        sender = FakeSender(succeeds=False)
        manager = AlertManager(sender=sender)

        first = manager.handle_tracks("bear.jpg", [track])
        second = manager.handle_tracks("bear2.jpg", [track])

        self.assertEqual(first[0].status, "failed")
        self.assertEqual(second[0].status, "failed")
        self.assertFalse(track.alerted)
        self.assertEqual(len(sender.calls), 2)

    def test_safe_track_does_not_alert(self):
        track = self._confirmed_track()
        track.hazard = "safe"
        track.danger_score = 3
        sender = FakeSender()

        results = AlertManager(sender=sender).handle_tracks(
            "elk.jpg", [track]
        )

        self.assertEqual(results, [])
        self.assertEqual(sender.calls, [])

    def test_phone_number_is_masked_in_logs(self):
        track = self._confirmed_track()
        output = io.StringIO()

        with redirect_stdout(output):
            AlertManager(sender=FakeSender()).handle_tracks("bear.jpg", [track])

        log = output.getvalue()
        self.assertNotIn("+15551232832", log)
        self.assertIn("***2832", log)


class AlertConfigurationTests(unittest.TestCase):
    @staticmethod
    def _write_env(directory, recipients="+15551232832"):
        path = Path(directory) / "twilio.env"
        path.write_text(
            "export TWILIO_ACCOUNT_SID=AC-file-account\n"
            "export TWILIO_AUTH_TOKEN=file-secret-token\n"
            "export TWILIO_FROM_NUMBER=+15551230000\n"
            f"export ALERT_RECIPIENTS={recipients}\n",
            encoding="utf-8",
        )
        return path

    def test_dotenv_file_loads_and_validates_automatically(self):
        with tempfile.TemporaryDirectory() as directory:
            env_path = self._write_env(directory)
            with patch.dict(os.environ, {}, clear=True):
                config = load_alert_config(env_path)

        self.assertEqual(config.account_sid, "AC-file-account")
        self.assertEqual(config.auth_token, "file-secret-token")
        self.assertEqual(config.from_number, "+15551230000")
        self.assertEqual(config.recipients, ("+15551232832",))
        self.assertFalse(hasattr(config, "enabled"))

    def test_existing_environment_takes_precedence_over_file(self):
        with tempfile.TemporaryDirectory() as directory:
            env_path = self._write_env(directory)
            existing = {"TWILIO_AUTH_TOKEN": "deployment-secret-token"}
            with patch.dict(os.environ, existing, clear=True):
                config = load_alert_config(env_path)

        self.assertEqual(config.auth_token, "deployment-secret-token")

    def test_missing_dotenv_file_fails_clearly(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.env"
            with self.assertRaisesRegex(
                AlertConfigurationError, "Missing Twilio configuration file"
            ):
                load_alert_config(missing)

    def test_invalid_phone_number_fails_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            env_path = self._write_env(directory, recipients="not-a-number")
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(
                    AlertConfigurationError, "E.164"
                ):
                    load_alert_config(env_path)

    def test_provider_error_redacts_credentials(self):
        class FailingMessages:
            def create(self, **kwargs):
                raise RuntimeError(
                    "request exposed AC-file-account file-secret-token "
                    "+15551230000 +15551232832"
                )

        class FailingClient:
            messages = FailingMessages()

        config = AlertConfig(
            account_sid="AC-file-account",
            auth_token="file-secret-token",
            from_number="+15551230000",
            recipients=("+15551232832",),
        )
        result = SMSAlertSender(config, client=FailingClient()).send("test")[0]

        self.assertIn("<redacted>", result.error)
        for secret in (
            config.account_sid,
            config.auth_token,
            config.from_number,
            *config.recipients,
        ):
            self.assertNotIn(secret, result.error)


if __name__ == "__main__":
    unittest.main()
