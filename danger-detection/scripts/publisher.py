"""Publish detections to the Sage beehive.

Sage nodes have restricted outbound network access, so the plugin does not
contact Twilio or Slack directly. Instead it publishes measurements (and the
annotated image) to the beehive, and a watcher running outside the node polls
the Sage data API and sends the alerts.

Measurements published per confirmed dangerous track:
  env.detection.dangerous  = danger score (1-10)
with meta carrying species, common name, and the track id the watcher uses to
avoid alerting twice for the same animal.
"""

# Sage uses dot-separated measurement names. We publish two:
#   env.detection.species    every identified animal (value = species confidence)
#   env.detection.dangerous  only dangerous ones      (value = danger score 1-10)
# Publishing both means a consumer can query all wildlife occurrences, or filter
# straight to the ones worth alerting on, without parsing meta.
MEASUREMENT_SPECIES = "env.detection.species"
MEASUREMENT_DANGEROUS = "env.detection.dangerous"


def _meta_for(track, image_name):
    """Build the metadata a watcher needs to alert and de-duplicate."""
    # Values must be strings: the beehive stores meta as string key/value pairs.
    return {
        "track_id": str(track.track_id),
        "species": str(track.species or ""),
        "common_name": str(track.common_name or ""),
        "hazard": str(track.hazard or ""),
        "danger_score": str(track.danger_score or ""),
        "yolo_label": str(getattr(track, "label", "") or ""),
        "image": str(image_name),
    }


class BeehivePublisher:
    """Publishes dangerous detections + annotated images to the beehive.

    Wrapped so a publish failure never takes down detection: on a node the
    plugin should keep running even if the messaging layer hiccups.
    """

    def __init__(self, plugin):
        self.plugin = plugin

    def _publish(self, name, value, meta, timestamp=None):
        kwargs = {"meta": meta}
        if timestamp is not None:
            kwargs["timestamp"] = timestamp
        self.plugin.publish(name, value, **kwargs)

    def publish_track(self, track, image_name, annotated_path=None,
                      timestamp=None):
        """Publish one classified track; upload the image if it's dangerous."""
        meta = _meta_for(track, image_name)
        dangerous = str(track.hazard or "").lower() == "dangerous"

        # Every identified animal is occurrence data worth keeping, not just
        # the dangerous ones.
        try:
            self._publish(
                MEASUREMENT_SPECIES,
                float(getattr(track, "species_confidence", 0.0) or 0.0),
                meta,
                timestamp,
            )
            print(f"  [beehive] published {MEASUREMENT_SPECIES} "
                  f"track={track.track_id} {track.species or 'unknown'}")
        except Exception as exc:
            print(f"  [beehive] publish failed: {type(exc).__name__}: {exc}")

        if not dangerous:
            return

        try:
            # Value is the danger score so a consumer can filter on severity.
            self._publish(
                MEASUREMENT_DANGEROUS,
                int(track.danger_score or 0),
                meta,
                timestamp,
            )
            print(f"  [beehive] published {MEASUREMENT_DANGEROUS} "
                  f"track={track.track_id} score={track.danger_score}")
        except Exception as exc:
            print(f"  [beehive] publish failed: {type(exc).__name__}: {exc}")

        if annotated_path is None:
            return

        try:
            # The watcher downloads this to attach to the Slack post.
            self.plugin.upload_file(str(annotated_path), meta=meta)
            print(f"  [beehive] uploaded image {annotated_path.name}")
        except Exception as exc:
            print(f"  [beehive] image upload failed: {type(exc).__name__}: {exc}")


class NullPublisher:
    """Used when --publish is off, so the pipeline needs no special-casing."""

    def publish_track(self, track, image_name, annotated_path=None,
                      timestamp=None):
        return


def open_publisher(enabled):
    """Return (publisher, plugin_context).

    `plugin_context` is the pywaggle Plugin to close when the run ends, or None.
    Falls back to a NullPublisher if pywaggle is unavailable, so local runs and
    machines without the Waggle runtime behave normally.
    """
    if not enabled:
        return NullPublisher(), None

    try:
        from waggle.plugin import Plugin
    except Exception as exc:
        print(f"[beehive] pywaggle unavailable, publishing disabled: {exc}")
        return NullPublisher(), None

    try:
        plugin = Plugin()
        plugin.__enter__()
    except Exception as exc:
        print(f"[beehive] could not start plugin, publishing disabled: {exc}")
        return NullPublisher(), None

    print("[beehive] publishing enabled")
    return BeehivePublisher(plugin), plugin
