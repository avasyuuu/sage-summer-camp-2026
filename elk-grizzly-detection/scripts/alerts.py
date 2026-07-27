"""Send explicitly enabled wildlife test alerts through Twilio."""

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path


TEST_MESSAGE = "sms_internal_alerts"
TWILIO_TRIAL_TEMPLATE = "sms_internal_alerts"
E164_PATTERN = re.compile(r"^\+[1-9]\d{7,14}$")


class AlertConfigurationError(ValueError):
    """Raised when alert configuration is missing or unsafe."""


@dataclass(frozen=True)
class AlertConfig:
    account_sid: str
    auth_token: str
    from_number: str
    recipients: tuple[str, ...]
    enabled: bool

    @classmethod
    def from_env(cls):
        """Load and validate Twilio configuration from environment variables."""
        account_sid = os.environ.get("TWILIO_ACCOUNT_SID", "").strip()
        auth_token = os.environ.get("TWILIO_AUTH_TOKEN", "").strip()
        from_number = os.environ.get("TWILIO_FROM_NUMBER", "").strip()
        recipients = tuple(
            number.strip()
            for number in os.environ.get("ALERT_RECIPIENTS", "").split(",")
            if number.strip()
        )
        enabled = os.environ.get("ALERTS_ENABLED", "").strip().lower() == "true"

        missing = []
        if not account_sid:
            missing.append("TWILIO_ACCOUNT_SID")
        if not auth_token:
            missing.append("TWILIO_AUTH_TOKEN")
        if not from_number:
            missing.append("TWILIO_FROM_NUMBER")
        if not recipients:
            missing.append("ALERT_RECIPIENTS")
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

        return cls(
            account_sid=account_sid,
            auth_token=auth_token,
            from_number=from_number,
            recipients=recipients,
            enabled=enabled,
        )


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

    def send(self, body):
        """Send one message to every configured recipient."""
        if not self.config.enabled:
            raise AlertConfigurationError(
                "Real SMS is disabled. Set ALERTS_ENABLED=true to send."
            )
        if not body.strip():
            raise ValueError("Alert message cannot be empty")

        results = []
        for recipient in self.config.recipients:
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
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
        return results


@dataclass(frozen=True)
class AlertResult:
    status: str
    track_id: int | None = None
    intended_message: str = ""
    deliveries: tuple[DeliveryResult, ...] = ()
    error: str = ""


class AlertManager:
    """Send at most one alert during the lifetime of each dangerous track."""

    VALID_MODES = {"off", "dry-run", "send"}

    def __init__(self, mode="off", sender=None):
        if mode not in self.VALID_MODES:
            raise ValueError(f"Unknown alert mode: {mode!r}")
        self.mode = mode
        self._sender = sender

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

    def handle_tracks(self, image_path, tracks):
        """Attempt one alert for each confirmed, dangerous, unalerted track."""
        if self.mode == "off":
            return [AlertResult(status="disabled")]

        results = []
        for track in tracks:
            if track.alerted or track.hazard != "dangerous":
                continue

            intended = self._intended_message(image_path, track)
            print(f"  [alert] intended message: {intended}")

            if self.mode == "dry-run":
                track.alerted = True
                print("  [alert] dry run: SMS not sent")
                results.append(
                    AlertResult(
                        status="dry_run",
                        track_id=track.track_id,
                        intended_message=intended,
                    )
                )
                continue

            try:
                if self._sender is None:
                    self._sender = SMSAlertSender(AlertConfig.from_env())
                deliveries = tuple(self._sender.send(TWILIO_TRIAL_TEMPLATE))
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                print(f"  [alert] failed: {error}")
                results.append(
                    AlertResult(
                        status="failed",
                        track_id=track.track_id,
                        intended_message=intended,
                        error=error,
                    )
                )
                continue

            failed = [delivery for delivery in deliveries if not delivery.succeeded]
            for delivery in deliveries:
                recipient = _masked_phone(delivery.recipient)
                if delivery.succeeded:
                    print(
                        f"  [alert] sent to {recipient}: sid={delivery.message_sid} "
                        f"status={delivery.status}"
                    )
                else:
                    print(f"  [alert] failed for {recipient}: {delivery.error}")

            if any(delivery.succeeded for delivery in deliveries):
                track.alerted = True

            results.append(
                AlertResult(
                    status="failed" if failed else "sent",
                    track_id=track.track_id,
                    intended_message=intended,
                    deliveries=deliveries,
                    error="one or more messages failed" if failed else "",
                )
            )

        return results


def _masked_phone(number):
    """Return a phone number safe for terminal status output."""
    return f"***{number[-4:]}"


def main():
    parser = argparse.ArgumentParser(
        description="Send an explicitly enabled Twilio wildlife-alert test."
    )
    parser.add_argument(
        "--send-test",
        action="store_true",
        help="send the predefined TEST WILDLIFE ALERT to configured recipients",
    )
    args = parser.parse_args()
    if not args.send_test:
        parser.error("no action selected; use --send-test to send a real test SMS")

    try:
        config = AlertConfig.from_env()
        results = SMSAlertSender(config).send(TEST_MESSAGE)
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
    if failed:
        parser.exit(1, "one or more test messages failed\n")


if __name__ == "__main__":
    main()
