"""Load protected configuration and send wildlife alerts through SMS and Slack."""

import argparse
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv


TEST_MESSAGE = "TEST WILDLIFE ALERT"
TWILIO_TRIAL_TEMPLATE = "sms_internal_alerts"
E164_PATTERN = re.compile(r"^\+[1-9]\d{7,14}$")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TWILIO_ENV = PROJECT_ROOT / "twilio.env"


class AlertConfigurationError(ValueError):
    """Raised when alert configuration is missing or unsafe."""


@dataclass(frozen=True)
class AlertConfig:
    account_sid: str
    auth_token: str
    from_number: str
    recipients: tuple[str, ...]
    slack_bot_token: str
    slack_channel_id: str

    @classmethod
    def from_env(cls):
        """Load and validate Twilio and Slack settings from the environment."""
        account_sid = os.environ.get("TWILIO_ACCOUNT_SID", "").strip()
        auth_token = os.environ.get("TWILIO_AUTH_TOKEN", "").strip()
        from_number = os.environ.get("TWILIO_FROM_NUMBER", "").strip()
        recipients = tuple(
            number.strip()
            for number in os.environ.get("ALERT_RECIPIENTS", "").split(",")
            if number.strip()
        )
        slack_bot_token = os.environ.get("SLACK_BOT_TOKEN", "").strip()
        slack_channel_id = os.environ.get("SLACK_CHANNEL_ID", "").strip()
        missing = []
        if not account_sid:
            missing.append("TWILIO_ACCOUNT_SID")
        if not auth_token:
            missing.append("TWILIO_AUTH_TOKEN")
        if not from_number:
            missing.append("TWILIO_FROM_NUMBER")
        if not recipients:
            missing.append("ALERT_RECIPIENTS")
        if not slack_bot_token:
            missing.append("SLACK_BOT_TOKEN")
        if not slack_channel_id:
            missing.append("SLACK_CHANNEL_ID")
        if missing:
            raise AlertConfigurationError(
                "Missing required environment variables: " + ", ".join(missing)
            )

        if not account_sid.startswith("AC"):
            raise AlertConfigurationError(
                "TWILIO_ACCOUNT_SID must start with 'AC'"
            )
        invalid_numbers = [
            number
            for number in (from_number, *recipients)
            if not E164_PATTERN.fullmatch(number)
        ]
        if invalid_numbers:
            raise AlertConfigurationError(
                "Phone numbers must use E.164 format, such as +12125551234"
            )
        if not slack_bot_token.startswith("xoxb-"):
            raise AlertConfigurationError(
                "SLACK_BOT_TOKEN must be a Slack bot token beginning with 'xoxb-'"
            )

        return cls(
            account_sid=account_sid,
            auth_token=auth_token,
            from_number=from_number,
            recipients=recipients,
            slack_bot_token=slack_bot_token,
            slack_channel_id=slack_channel_id,
        )


def load_alert_config(env_path=DEFAULT_TWILIO_ENV):
    """Load the node's protected dotenv file and validate all alert settings."""
    env_path = Path(env_path)
    if not env_path.is_file():
        raise AlertConfigurationError(
            f"Missing alert configuration file: {env_path}"
        )

    # Explicit deployment environment variables take precedence over the file.
    load_dotenv(dotenv_path=env_path, override=False)
    return AlertConfig.from_env()


@dataclass(frozen=True)
class DeliveryResult:
    recipient: str
    message_sid: str = ""
    status: str = ""
    error: str = ""

    @property
    def succeeded(self):
        return not self.error


class SMSAlertSender:
    """Send SMS alerts without exposing Twilio credentials."""

    def __init__(self, config, client=None):
        self.config = config
        self._client = client

    @property
    def client(self):
        if self._client is None:
            from twilio.rest import Client

            self._client = Client(
                self.config.account_sid,
                self.config.auth_token,
            )
        return self._client

    def send(self, body, recipients=None):
        """Send one message to the requested recipients (all by default)."""
        if not body.strip():
            raise ValueError("Alert message cannot be empty")

        results = []
        targets = self.config.recipients if recipients is None else tuple(recipients)
        for recipient in targets:
            try:
                message = self.client.messages.create(
                    body=body,
                    from_=self.config.from_number,
                    to=recipient,
                )
                results.append(
                    DeliveryResult(
                        recipient=recipient,
                        message_sid=str(message.sid),
                        status=str(message.status or "queued"),
                    )
                )
            except Exception as exc:
                results.append(
                    DeliveryResult(
                        recipient=recipient,
                        error=self._safe_error(exc),
                    )
                )
        return results

    def _safe_error(self, error):
        """Redact configured secrets and phone numbers from provider errors."""
        detail = f"{type(error).__name__}: {error}"
        protected_values = (
            self.config.account_sid,
            self.config.auth_token,
            self.config.from_number,
            *self.config.recipients,
        )
        for value in protected_values:
            if value:
                detail = detail.replace(value, "<redacted>")
        return detail


@dataclass(frozen=True)
class SlackDeliveryResult:
    status: str = ""
    error: str = ""

    @property
    def succeeded(self):
        return not self.error


class SlackAlertSender:
    """Upload alert images to Slack without exposing the bot token."""

    def __init__(self, config, client=None):
        self.config = config
        self._client = client

    @property
    def client(self):
        if self._client is None:
            from slack_sdk import WebClient

            self._client = WebClient(token=self.config.slack_bot_token)
        return self._client

    def send(self, body, image_path):
        if not body.strip():
            raise ValueError("Alert message cannot be empty")
        image_path = Path(image_path)
        if not image_path.is_file():
            return SlackDeliveryResult(error="Slack alert image is unavailable")

        try:
            response = self.client.files_upload_v2(
                channel=self.config.slack_channel_id,
                file=str(image_path),
                filename=image_path.name,
                title="Wildlife detection",
                initial_comment=body,
            )
            if response.get("ok"):
                return SlackDeliveryResult(status="sent")
            return SlackDeliveryResult(
                error="Slack image upload was not accepted"
            )
        except Exception as exc:
            return SlackDeliveryResult(error=self._safe_error(exc))

    def _safe_error(self, error):
        detail = f"{type(error).__name__}: {error}"
        return detail.replace(self.config.slack_bot_token, "<redacted>")


@dataclass(frozen=True)
class AlertResult:
    status: str
    track_id: int | None = None
    intended_message: str = ""
    deliveries: tuple[DeliveryResult, ...] = ()
    slack_delivery: SlackDeliveryResult | None = None
    error: str = ""


class AlertManager:
    """Deliver each dangerous-track alert once through every configured channel."""

    def __init__(self, sms_sender, slack_sender, enabled=True):
        self._sms_sender = sms_sender
        self._slack_sender = slack_sender
        # When credentials are missing/invalid we still run detection and report
        # what *would* have been sent, instead of failing the whole pipeline.
        self._enabled = enabled

    @staticmethod
    def _intended_message(image_path, track):
        name = track.common_name or track.species or track.label or "animal"
        return (
            "WILDLIFE ALERT: Dangerous species detected. "
            f"Animal: {name}. "
            f"Danger score: {track.danger_score}/10. "
            f"Track: {track.track_id}. "
            f"Image: {Path(image_path).name}."
        )

    @staticmethod
    def _slack_message(image_path, track):
        name = track.common_name or track.species or track.label or "animal"
        timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        return (
            ":warning: *WILDLIFE ALERT*\n"
            "Dangerous species detected\n"
            f"*Animal:* {name}\n"
            f"*Danger score:* {track.danger_score}/10\n"
            f"*Time:* {timestamp}"
        )

    @staticmethod
    def _safe_attempt_error(error):
        return f"{type(error).__name__}: delivery attempt failed"

    def handle_tracks(self, image_path, tracks, slack_image_path=None):
        """Attempt one alert for each confirmed, dangerous, unalerted track."""
        results = []
        for track in tracks:
            if track.alerted or track.hazard != "dangerous":
                continue

            intended = self._intended_message(image_path, track)
            print(f"  [alert] intended message: {intended}")

            if not self._enabled:
                # No usable credentials: report the failure and keep going so the
                # annotated image and CSV row are still produced.
                print("  [alert] Twilio message failed: alerts disabled "
                      "(missing or invalid credentials)")
                print("  [alert] Slack message failed: alerts disabled "
                      "(missing or invalid credentials)")
                results.append(
                    AlertResult(
                        status="disabled",
                        track_id=track.track_id,
                        intended_message=intended,
                    )
                )
                continue

            required_recipients = self._sms_sender.config.recipients
            pending_recipients = tuple(
                recipient
                for recipient in required_recipients
                if recipient not in track.sms_delivered_recipients
            )
            sms_attempt_error = ""
            try:
                deliveries = tuple(
                    self._sms_sender.send(
                        TWILIO_TRIAL_TEMPLATE,
                        recipients=pending_recipients,
                    )
                ) if pending_recipients else ()
            except Exception as exc:
                deliveries = ()
                sms_attempt_error = self._safe_attempt_error(exc)
                print(f"  [alert] SMS failed: {sms_attempt_error}")

            failed = [delivery for delivery in deliveries if not delivery.succeeded]
            for delivery in deliveries:
                recipient = _masked_phone(delivery.recipient)
                if delivery.succeeded:
                    track.sms_delivered_recipients.add(delivery.recipient)
                    print(
                        f"  [alert] SMS sent to {recipient}: sid={delivery.message_sid} "
                        f"status={delivery.status}"
                    )
                else:
                    print(f"  [alert] SMS failed for {recipient}: {delivery.error}")

            slack_delivery = None
            slack_attempt_error = ""
            if not track.slack_alerted:
                try:
                    slack_delivery = self._slack_sender.send(
                        self._slack_message(image_path, track),
                        slack_image_path or image_path,
                    )
                except Exception as exc:
                    slack_attempt_error = self._safe_attempt_error(exc)
                    print(f"  [alert] Slack failed: {slack_attempt_error}")
                else:
                    if slack_delivery.succeeded:
                        track.slack_alerted = True
                        print("  [alert] Slack sent")
                    else:
                        print(f"  [alert] Slack failed: {slack_delivery.error}")

            sms_complete = all(
                recipient in track.sms_delivered_recipients
                for recipient in required_recipients
            )
            track.alerted = sms_complete and track.slack_alerted

            delivery_has_progress = bool(track.sms_delivered_recipients) or (
                track.slack_alerted
            )
            errors = []
            if failed:
                errors.append("one or more SMS messages failed")
            if sms_attempt_error:
                errors.append(sms_attempt_error)
            if slack_delivery and not slack_delivery.succeeded:
                errors.append("Slack message failed")
            if slack_attempt_error:
                errors.append(slack_attempt_error)

            if track.alerted:
                status = "sent"
            elif delivery_has_progress:
                status = "partial"
            else:
                status = "failed"

            results.append(
                AlertResult(
                    status=status,
                    track_id=track.track_id,
                    intended_message=intended,
                    deliveries=deliveries,
                    slack_delivery=slack_delivery,
                    error="; ".join(errors),
                )
            )

        return results


def _masked_phone(number):
    """Return a phone number safe for terminal status output."""
    return f"***{number[-4:]}"


def main():
    parser = argparse.ArgumentParser(
        description="Send SMS and Slack wildlife-alert tests using twilio.env."
    )
    parser.add_argument(
        "--send-test",
        action="store_true",
        help="send a real test through every configured alert channel",
    )
    args = parser.parse_args()
    if not args.send_test:
        parser.error("no action selected; use --send-test to send real alerts")

    try:
        config = load_alert_config()
        results = SMSAlertSender(config).send(TWILIO_TRIAL_TEMPLATE)
        test_images = sorted((PROJECT_ROOT / "test_images").glob("*"))
        test_image = next((path for path in test_images if path.is_file()), None)
        if test_image is None:
            parser.exit(2, "configuration error: no test image is available\n")
        slack_result = SlackAlertSender(config).send(TEST_MESSAGE, test_image)
    except AlertConfigurationError as exc:
        parser.exit(2, f"configuration error: {exc}\n")

    failed = False
    for result in results:
        recipient = _masked_phone(result.recipient)
        if result.succeeded:
            print(
                f"sent to {recipient}: sid={result.message_sid} "
                f"status={result.status}"
            )
        else:
            failed = True
            print(f"failed for {recipient}: {result.error}")
    if slack_result.succeeded:
        print("sent to Slack")
    else:
        failed = True
        print(f"Slack failed: {slack_result.error}")
    if failed:
        parser.exit(1, "one or more test messages failed\n")


if __name__ == "__main__":
    main()
