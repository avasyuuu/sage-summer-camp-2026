"""State-machine tests for tracking and multi-channel wildlife alerts."""

import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from alerts import (  # noqa: E402
    AlertConfig,
    AlertConfigurationError,
    AlertManager,
    DeliveryResult,
    SlackAlertSender,
    SlackDeliveryResult,
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


TEST_BOT_TOKEN = "xoxb-test-bot-token"
TEST_CHANNEL_ID = "C0123456789"


class FakeSMSSender:
    def __init__(self, outcomes=None, recipients=("+15551232832",)):
        self.config = type("Config", (), {"recipients": recipients})()
        self.outcomes = list(outcomes or [True])
        self.calls = []

    def send(self, body, recipients=None):
        targets = tuple(recipients or ())
        self.calls.append((body, targets))
        succeeds = self.outcomes.pop(0) if self.outcomes else True
        if succeeds:
            return [
                DeliveryResult(recipient, "SM-test", "queued")
                for recipient in targets
            ]
        return [
            DeliveryResult(recipient, error="test failure")
            for recipient in targets
        ]


class FakeSlackSender:
    def __init__(self, outcomes=None):
        self.outcomes = list(outcomes or [True])
        self.calls = []

    def send(self, body, image_path):
        self.calls.append((body, Path(image_path)))
        succeeds = self.outcomes.pop(0) if self.outcomes else True
        if succeeds:
            return SlackDeliveryResult(status="sent")
        return SlackDeliveryResult(error="test failure")


class TrackRegistryTests(unittest.TestCase):
    def test_default_confirms_on_first_frame(self):
        registry = AnimalTrackRegistry()

        confirmed = registry.update([detection(1)], FRAME, "first.jpg")

        self.assertEqual([track.track_id for track in confirmed], [1])
        self.assertTrue(registry.tracks[1].confirmed)
        self.assertEqual(registry.tracks[1].consecutive_seen, 1)

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
        registry = AnimalTrackRegistry()
        track = registry.update([detection(1)], FRAME, "bear.jpg")[0]
        classify_as_dangerous(track)
        return track

    def test_successful_delivery_marks_track_alerted(self):
        track = self._confirmed_track()
        sms = FakeSMSSender()
        slack = FakeSlackSender()
        manager = AlertManager(sms_sender=sms, slack_sender=slack)

        first = manager.handle_tracks("bear.jpg", [track])
        second = manager.handle_tracks("bear2.jpg", [track])

        self.assertEqual(first[0].status, "sent")
        self.assertEqual(second, [])
        self.assertTrue(track.alerted)
        self.assertEqual(len(sms.calls), 1)
        self.assertEqual(len(slack.calls), 1)
        self.assertEqual(track.sms_delivered_recipients, {"+15551232832"})
        self.assertTrue(track.slack_alerted)

    def test_slack_message_has_timestamp_without_image_reference(self):
        track = self._confirmed_track()

        message = AlertManager._slack_message("secret-image-url.jpg", track)

        self.assertIn("*Time:*", message)
        self.assertNotIn("*Image:*", message)
        self.assertNotIn("*Track:*", message)
        self.assertNotIn("secret-image-url.jpg", message)

    def test_failed_delivery_remains_eligible_for_retry(self):
        track = self._confirmed_track()
        sms = FakeSMSSender(outcomes=[False, False])
        slack = FakeSlackSender(outcomes=[False, False])
        manager = AlertManager(sms_sender=sms, slack_sender=slack)

        first = manager.handle_tracks("bear.jpg", [track])
        second = manager.handle_tracks("bear2.jpg", [track])

        self.assertEqual(first[0].status, "failed")
        self.assertEqual(second[0].status, "failed")
        self.assertFalse(track.alerted)
        self.assertEqual(len(sms.calls), 2)
        self.assertEqual(len(slack.calls), 2)

    def test_successful_channel_is_not_duplicated_while_other_retries(self):
        track = self._confirmed_track()
        sms = FakeSMSSender(outcomes=[True])
        slack = FakeSlackSender(outcomes=[False, True])
        manager = AlertManager(sms_sender=sms, slack_sender=slack)

        first = manager.handle_tracks("bear.jpg", [track])
        second = manager.handle_tracks("bear2.jpg", [track])

        self.assertEqual(first[0].status, "partial")
        self.assertEqual(second[0].status, "sent")
        self.assertEqual(len(sms.calls), 1)
        self.assertEqual(len(slack.calls), 2)
        self.assertTrue(track.alerted)

    def test_only_failed_sms_recipient_is_retried(self):
        first_number = "+15551232832"
        second_number = "+15551232833"

        class PartialSMSSender(FakeSMSSender):
            def send(self, body, recipients=None):
                targets = tuple(recipients or ())
                self.calls.append((body, targets))
                if len(self.calls) == 1:
                    return [
                        DeliveryResult(first_number, "SM-test", "queued"),
                        DeliveryResult(second_number, error="test failure"),
                    ]
                return [DeliveryResult(second_number, "SM-retry", "queued")]

        track = self._confirmed_track()
        sms = PartialSMSSender(recipients=(first_number, second_number))
        slack = FakeSlackSender()
        manager = AlertManager(sms_sender=sms, slack_sender=slack)

        manager.handle_tracks("bear.jpg", [track])
        manager.handle_tracks("bear2.jpg", [track])

        self.assertEqual(sms.calls[0][1], (first_number, second_number))
        self.assertEqual(sms.calls[1][1], (second_number,))
        self.assertEqual(len(slack.calls), 1)
        self.assertTrue(track.alerted)

    def test_safe_track_does_not_alert(self):
        track = self._confirmed_track()
        track.hazard = "safe"
        track.danger_score = 3
        sms = FakeSMSSender()
        slack = FakeSlackSender()

        results = AlertManager(sms_sender=sms, slack_sender=slack).handle_tracks(
            "elk.jpg", [track]
        )

        self.assertEqual(results, [])
        self.assertEqual(sms.calls, [])
        self.assertEqual(slack.calls, [])

    def test_phone_number_is_masked_in_logs(self):
        track = self._confirmed_track()
        output = io.StringIO()

        with redirect_stdout(output):
            AlertManager(
                sms_sender=FakeSMSSender(),
                slack_sender=FakeSlackSender(),
            ).handle_tracks("bear.jpg", [track])

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
            f"export ALERT_RECIPIENTS={recipients}\n"
            f"export SLACK_BOT_TOKEN={TEST_BOT_TOKEN}\n"
            f"export SLACK_CHANNEL_ID={TEST_CHANNEL_ID}\n",
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
        self.assertEqual(config.slack_bot_token, TEST_BOT_TOKEN)
        self.assertEqual(config.slack_channel_id, TEST_CHANNEL_ID)
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
                AlertConfigurationError, "Missing alert configuration file"
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

    def test_invalid_slack_bot_token_fails_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            env_path = self._write_env(directory)
            env_path.write_text(
                env_path.read_text(encoding="utf-8").replace(
                    TEST_BOT_TOKEN, "not-a-bot-token"
                ),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(
                    AlertConfigurationError, "SLACK_BOT_TOKEN"
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
            slack_bot_token=TEST_BOT_TOKEN,
            slack_channel_id=TEST_CHANNEL_ID,
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

    def test_slack_provider_error_redacts_bot_token(self):
        class FailingClient:
            def files_upload_v2(self, **kwargs):
                raise RuntimeError(f"request exposed {TEST_BOT_TOKEN}")

        config = AlertConfig(
            account_sid="AC-file-account",
            auth_token="file-secret-token",
            from_number="+15551230000",
            recipients=("+15551232832",),
            slack_bot_token=TEST_BOT_TOKEN,
            slack_channel_id=TEST_CHANNEL_ID,
        )
        with tempfile.NamedTemporaryFile(suffix=".jpg") as image:
            result = SlackAlertSender(config, client=FailingClient()).send(
                "test", image.name
            )

        self.assertIn("<redacted>", result.error)
        self.assertNotIn(TEST_BOT_TOKEN, result.error)

    def test_slack_requires_ok_response(self):
        class Client:
            def files_upload_v2(self, **kwargs):
                return {"ok": False, "error": "channel_not_found"}

        config = AlertConfig(
            account_sid="AC-file-account",
            auth_token="file-secret-token",
            from_number="+15551230000",
            recipients=("+15551232832",),
            slack_bot_token=TEST_BOT_TOKEN,
            slack_channel_id=TEST_CHANNEL_ID,
        )
        with tempfile.NamedTemporaryFile(suffix=".jpg") as image:
            result = SlackAlertSender(config, client=Client()).send(
                "test", image.name
            )

        self.assertFalse(result.succeeded)

    def test_slack_uploads_image_with_alert_message(self):
        class Client:
            def __init__(self):
                self.calls = []

            def files_upload_v2(self, **kwargs):
                self.calls.append(kwargs)
                return {"ok": True}

        config = AlertConfig(
            account_sid="AC-file-account",
            auth_token="file-secret-token",
            from_number="+15551230000",
            recipients=("+15551232832",),
            slack_bot_token=TEST_BOT_TOKEN,
            slack_channel_id=TEST_CHANNEL_ID,
        )
        client = Client()
        with tempfile.NamedTemporaryFile(suffix=".jpg") as image:
            result = SlackAlertSender(config, client=client).send(
                "wildlife alert", image.name
            )

            self.assertTrue(result.succeeded)
            self.assertEqual(client.calls[0]["channel"], TEST_CHANNEL_ID)
            self.assertEqual(client.calls[0]["file"], image.name)
            self.assertEqual(client.calls[0]["initial_comment"], "wildlife alert")


if __name__ == "__main__":
    unittest.main()
