"""Sparse-frame animal identity and application-level tracking state."""

import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from lap import lapjv


@dataclass
class TrackedAnimal:
    track_id: int
    label: str
    first_seen_image: str
    last_seen_image: str
    box: tuple[int, int, int, int]
    confidence: float
    consecutive_seen: int = 1
    missed_frames: int = 0
    confirmed: bool = False
    best_crop: object = None
    best_confidence: float = 0.0
    species: str = ""
    common_name: str = ""
    species_confidence: float = 0.0
    hazard: str = ""
    danger_score: int | None = None
    classified: bool = False
    sms_delivered_recipients: set[str] = field(default_factory=set)
    slack_alerted: bool = False
    alerted: bool = False
    center_history: list[tuple[float, float]] = field(default_factory=list)
    appearance: object = None
    scene_signature: object = None

    def apply_classification(self, species, verdict=None):
        """Store BioCLIP/Gemma output and release the saved confirmation crop."""
        self.species = species["species"]
        self.common_name = species["common_name"]
        self.species_confidence = species["score"]
        if verdict:
            self.hazard = verdict["hazard"]
            self.danger_score = verdict["danger_score"]
        self.classified = True
        self.best_crop = None


class AnimalTrackRegistry:
    """Associate sparse still-image detections and retain per-object state."""

    def __init__(self, confirmation_frames=1, max_missed_frames=30,
                 max_center_distance=0.35, max_match_cost=0.72,
                 min_scene_similarity=0.80):
        if confirmation_frames < 1:
            raise ValueError("confirmation_frames must be at least 1")
        if max_missed_frames < 1:
            raise ValueError("max_missed_frames must be at least 1")
        self.confirmation_frames = confirmation_frames
        self.max_missed_frames = max_missed_frames
        self.max_center_distance = max_center_distance
        self.max_match_cost = max_match_cost
        self.min_scene_similarity = min_scene_similarity
        self.tracks = {}
        self._next_track_id = 1

    @staticmethod
    def _crop(image, box, padding=0.05):
        """Copy a padded BGR crop so confirmation can use an earlier frame."""
        h, w = image.shape[:2]
        x1, y1, x2, y2 = box
        pad_x = int((x2 - x1) * padding)
        pad_y = int((y2 - y1) * padding)
        x1 = max(x1 - pad_x, 0)
        y1 = max(y1 - pad_y, 0)
        x2 = min(x2 + pad_x, w)
        y2 = min(y2 + pad_y, h)
        return image[y1:y2, x1:x2].copy()

    def _remember_best_crop(self, track, detection, image):
        confidence = detection["confidence"]
        if track.best_crop is None or confidence > track.best_confidence:
            crop = self._crop(image, detection["box"])
            if crop.size:
                track.best_crop = crop
                track.best_confidence = confidence

    @staticmethod
    def _center(box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @staticmethod
    def _area(box):
        x1, y1, x2, y2 = box
        return max(x2 - x1, 1) * max(y2 - y1, 1)

    @staticmethod
    def _iou(first, second):
        ax1, ay1, ax2, ay2 = first
        bx1, by1, bx2, by2 = second
        intersection = max(min(ax2, bx2) - max(ax1, bx1), 0) * max(
            min(ay2, by2) - max(ay1, by1), 0
        )
        union = AnimalTrackRegistry._area(first) + AnimalTrackRegistry._area(
            second
        ) - intersection
        return intersection / union if union else 0.0

    @staticmethod
    def _appearance_signature(image, box):
        """Return a normalized HSV histogram for lightweight re-identification."""
        try:
            crop = AnimalTrackRegistry._crop(image, box, padding=0)
            if not getattr(crop, "size", 0):
                return None
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            histogram = cv2.calcHist(
                [hsv], [0, 1], None, [16, 8], [0, 180, 0, 256]
            )
            cv2.normalize(histogram, histogram)
            return histogram
        except (cv2.error, TypeError, ValueError):
            # State-machine tests use lightweight fake images; geometry remains
            # enough for those tests and for any unreadable appearance crop.
            return None

    @staticmethod
    def _appearance_distance(first, second):
        if first is None or second is None:
            return 0.5
        distance = cv2.compareHist(first, second, cv2.HISTCMP_BHATTACHARYYA)
        return min(max(float(distance), 0.0), 1.0)

    @staticmethod
    def _scene_signature(image):
        """Return a small normalized grayscale view of the fixed-camera scene."""
        try:
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            thumbnail = cv2.resize(
                grayscale,
                (64, 36),
                interpolation=cv2.INTER_AREA,
            ).astype(np.float32)
            thumbnail -= float(thumbnail.mean())
            deviation = float(thumbnail.std())
            if deviation < 1e-6:
                return None
            return (thumbnail / deviation).reshape(-1)
        except (cv2.error, TypeError, ValueError, AttributeError):
            return None

    @staticmethod
    def _scene_similarity(first, second):
        if first is None or second is None:
            return 1.0
        return float(np.dot(first, second) / len(first))

    @staticmethod
    def _predicted_center(track):
        if not track.center_history:
            return AnimalTrackRegistry._center(track.box)
        latest = track.center_history[-1]
        if len(track.center_history) < 2:
            return latest
        previous = track.center_history[-2]
        steps = track.missed_frames + 1
        return (
            latest[0] + (latest[0] - previous[0]) * steps,
            latest[1] + (latest[1] - previous[1]) * steps,
        )

    def _match_cost(self, track, detection, signature, scene_signature,
                    image_diagonal):
        if (
            self._scene_similarity(track.scene_signature, scene_signature)
            < self.min_scene_similarity
        ):
            return math.inf
        predicted = self._predicted_center(track)
        current = self._center(detection["box"])
        distance = math.dist(predicted, current) / max(image_diagonal, 1.0)
        distance_limit = min(
            self.max_center_distance * (1 + 0.25 * track.missed_frames),
            0.65,
        )
        if distance > distance_limit:
            return math.inf

        first_area = self._area(track.box)
        second_area = self._area(detection["box"])
        area_ratio = min(first_area, second_area) / max(first_area, second_area)
        if area_ratio < 0.2:
            return math.inf

        appearance = self._appearance_distance(track.appearance, signature)
        if track.appearance is not None and signature is not None and appearance > 0.75:
            return math.inf

        normalized_distance = distance / max(distance_limit, 1e-6)
        size_cost = min(abs(math.log(second_area / first_area)) / math.log(5), 1.0)
        overlap_cost = 1.0 - self._iou(track.box, detection["box"])
        return (
            0.45 * normalized_distance
            + 0.20 * size_cost
            + 0.15 * overlap_cost
            + 0.20 * appearance
        )

    def _assign(self, tracks, detections, signatures, scene_signature,
                image_shape):
        if not tracks or not detections:
            return []
        height, width = image_shape[:2]
        diagonal = math.hypot(width, height)
        costs = np.full((len(tracks), len(detections)), 1e6, dtype=float)
        for row, track in enumerate(tracks):
            for column, detection in enumerate(detections):
                cost = self._match_cost(
                    track,
                    detection,
                    signatures[column],
                    scene_signature,
                    diagonal,
                )
                if math.isfinite(cost):
                    costs[row, column] = cost

        _, track_to_detection, _ = lapjv(
            costs,
            extend_cost=True,
            cost_limit=self.max_match_cost,
        )
        return [
            (row, int(column))
            for row, column in enumerate(track_to_detection)
            if column >= 0 and costs[row, column] <= self.max_match_cost
        ]

    def _new_track(self, detection, signature, scene_signature, image,
                   image_path):
        track_id = self._next_track_id
        self._next_track_id += 1
        detection["track_id"] = track_id
        track = TrackedAnimal(
            track_id=track_id,
            label=detection["label"],
            first_seen_image=str(image_path),
            last_seen_image=str(image_path),
            box=detection["box"],
            confidence=detection["confidence"],
            center_history=[self._center(detection["box"])],
            appearance=signature,
            scene_signature=scene_signature,
        )
        self.tracks[track_id] = track
        self._remember_best_crop(track, detection, image)
        return track

    def _update_track(self, track, detection, signature, scene_signature, image,
                      image_path):
        detection["track_id"] = track.track_id
        track.consecutive_seen = (
            1 if track.missed_frames else track.consecutive_seen + 1
        )
        track.missed_frames = 0
        track.label = detection["label"]
        track.last_seen_image = str(image_path)
        track.box = detection["box"]
        track.confidence = detection["confidence"]
        track.scene_signature = scene_signature
        track.center_history.append(self._center(detection["box"]))
        track.center_history = track.center_history[-3:]
        if signature is not None:
            if track.appearance is None:
                track.appearance = signature
            else:
                track.appearance = 0.7 * track.appearance + 0.3 * signature
                cv2.normalize(track.appearance, track.appearance)
        if not track.confirmed:
            self._remember_best_crop(track, detection, image)

    def update(self, detections, image, image_path):
        """Advance the registry by one frame and return newly confirmed tracks."""
        signatures = [
            self._appearance_signature(image, detection["box"])
            for detection in detections
        ]
        scene_signature = self._scene_signature(image)
        active_tracks = list(self.tracks.values())
        matches = self._assign(
            active_tracks,
            detections,
            signatures,
            scene_signature,
            image.shape,
        )
        matched_track_rows = {row for row, _ in matches}
        matched_detection_columns = {column for _, column in matches}
        newly_confirmed = []

        observed_tracks = []
        for row, column in matches:
            track = active_tracks[row]
            self._update_track(
                track,
                detections[column],
                signatures[column],
                scene_signature,
                image,
                image_path,
            )
            observed_tracks.append(track)

        for column, detection in enumerate(detections):
            if column in matched_detection_columns:
                continue
            observed_tracks.append(
                self._new_track(
                    detection,
                    signatures[column],
                    scene_signature,
                    image,
                    image_path,
                )
            )

        for row, track in enumerate(active_tracks):
            if row in matched_track_rows:
                continue
            track.missed_frames += 1
            track.consecutive_seen = 0
            if track.missed_frames >= self.max_missed_frames:
                del self.tracks[track.track_id]

        for track in observed_tracks:
            if (
                not track.confirmed
                and track.consecutive_seen >= self.confirmation_frames
            ):
                track.confirmed = True
                newly_confirmed.append(track)

        self.enrich_detections(detections)
        return newly_confirmed

    def enrich_detections(self, detections):
        """Copy known tracking/classification state onto current detections."""
        for detection in detections:
            track = self.tracks.get(detection.get("track_id"))
            if track is None:
                continue
            detection["confirmed"] = track.confirmed
            detection["consecutive_seen"] = track.consecutive_seen
            if track.classified:
                detection.update(
                    species=track.species,
                    common_name=track.common_name,
                    species_confidence=track.species_confidence,
                    hazard=track.hazard,
                    danger_score=track.danger_score,
                )

    def needing_classification(self):
        """Return visible confirmed tracks whose best crop has not been used."""
        return [
            track
            for track in self.tracks.values()
            if track.confirmed
            and not track.classified
            and track.missed_frames == 0
            and track.best_crop is not None
        ]

    def alert_candidates(self):
        """Return visible dangerous tracks that have not delivered an alert."""
        return [
            track
            for track in self.tracks.values()
            if track.confirmed
            and track.classified
            and track.hazard == "dangerous"
            and not track.alerted
            and track.missed_frames == 0
        ]
