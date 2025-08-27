"""Motion tracking for temporal consistency in detection."""

from __future__ import annotations

import logging
import math

from pydantic import BaseModel, Field, field_validator

from mivideoeditor.core.models import BoundingBox
from mivideoeditor.detection.base import DetectionConfig
from mivideoeditor.detection.constants import (
    DEFAULT_MAX_FRAMES_WITHOUT_UPDATE,
    DEFAULT_MAX_TRACKING_DISTANCE,
    FRAMES_UNTIL_LOST,
    FRAMES_UNTIL_OCCLUDED,
    MAX_FRAMES_FOR_PREDICTION,
    MAX_TRACK_HISTORY,
    MIN_DETECTIONS_FOR_ACCELERATION,
    MIN_DETECTIONS_FOR_VELOCITY,
    REDUCED_CONFIDENCE_PREDICTION,
    VELOCITY_SMOOTHING_FACTOR,
)

logger = logging.getLogger(__name__)


class TrackedDetection(BaseModel):
    """Detection with tracking information."""

    bbox: BoundingBox
    confidence: float = Field(..., ge=0.0, le=1.0)
    track_id: int = Field(..., ge=0)
    velocity: tuple[float, float] = Field(
        default=(0.0, 0.0), description="(vx, vy) in pixels per second"
    )
    frames_tracked: int = Field(..., ge=1)
    last_seen_timestamp: float = Field(..., ge=0.0)
    prediction_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    is_predicted: bool = Field(
        default=False, description="True if this is a predicted position"
    )

    class Config:
        """Configuration for tracked detection."""

        frozen = True
        arbitrary_types_allowed = True


class Track(BaseModel):
    """Individual object track with state history."""

    id: int = Field(..., ge=0)
    detections_history: list[BoundingBox] = Field(default_factory=list)
    timestamps_history: list[float] = Field(default_factory=list)
    confidence_history: list[float] = Field(default_factory=list)
    velocity: tuple[float, float] = Field(default=(0.0, 0.0))
    acceleration: tuple[float, float] = Field(default=(0.0, 0.0))
    predicted_position: BoundingBox | None = None
    frames_since_update: int = Field(default=0, ge=0)
    state: str = Field(default="active", pattern="^(active|lost|occluded)$")
    creation_timestamp: float = Field(..., ge=0.0)
    last_update_timestamp: float = Field(..., ge=0.0)

    class Config:
        """Configuration for track."""

        arbitrary_types_allowed = True

    @field_validator("confidence_history")
    def validate_confidences(cls, v: list[float]) -> list[float]:
        """Ensure all confidence values are in valid range."""
        if not all(0.0 <= conf <= 1.0 for conf in v):
            msg = "All confidence values must be between 0.0 and 1.0"
            raise ValueError(msg)
        return v

    @field_validator("last_update_timestamp")
    def validate_update_timestamp(cls, v: float) -> float:
        """Ensure update timestamp is reasonable."""
        if v < 0:
            msg = "Update timestamp cannot be negative"
            raise ValueError(msg)
        return v

    @property
    def current_position(self) -> BoundingBox | None:
        """Get most recent position."""
        return self.detections_history[-1] if self.detections_history else None

    @property
    def track_confidence(self) -> float:
        """Get overall track confidence."""
        if not self.confidence_history:
            return 0.0
        return sum(self.confidence_history) / len(self.confidence_history)

    @property
    def track_duration(self) -> float:
        """Get track duration in seconds."""
        return self.last_update_timestamp - self.creation_timestamp

    def predict_position(self, dt: float) -> BoundingBox:
        """Predict position after time dt using motion model."""
        if not self.current_position:
            msg = "Cannot predict position: no current position"
            raise ValueError(msg)

        if dt < 0:
            msg = f"Time delta must be non-negative, got {dt}"
            raise ValueError(msg)

        # Simple constant velocity + acceleration model
        vx, vy = self.velocity
        ax, ay = self.acceleration

        # New position: x = x0 + vx*dt + 0.5*ax*dt^2
        new_x = self.current_position.x + vx * dt + 0.5 * ax * dt * dt
        new_y = self.current_position.y + vy * dt + 0.5 * ay * dt * dt

        return BoundingBox(
            x=int(new_x),
            y=int(new_y),
            width=self.current_position.width,
            height=self.current_position.height,
        )

    def update_with_detection(
        self, bbox: BoundingBox, confidence: float, timestamp: float
    ) -> Track:
        """Update track with new detection. Returns new Track instance."""
        if not 0.0 <= confidence <= 1.0:
            msg = f"Confidence must be between 0.0 and 1.0, got {confidence}"
            raise ValueError(msg)

        if timestamp < self.last_update_timestamp:
            msg = (
                f"Timestamp cannot go backwards: {timestamp} "
                f"< {self.last_update_timestamp}"
            )
            raise ValueError(msg)

        # Update history
        new_detections = [*self.detections_history, bbox]
        new_timestamps = [*self.timestamps_history, timestamp]
        new_confidences = [*self.confidence_history, confidence]

        # Limit history size to prevent memory growth
        max_history = MAX_TRACK_HISTORY
        if len(new_detections) > max_history:
            new_detections = new_detections[-max_history:]
            new_timestamps = new_timestamps[-max_history:]
            new_confidences = new_confidences[-max_history:]

        # Calculate new velocity and acceleration
        new_velocity, new_acceleration = self._calculate_motion_model(
            new_detections, new_timestamps, timestamp
        )

        return self.copy(
            update={
                "detections_history": new_detections,
                "timestamps_history": new_timestamps,
                "confidence_history": new_confidences,
                "velocity": new_velocity,
                "acceleration": new_acceleration,
                "frames_since_update": 0,
                "last_update_timestamp": timestamp,
                "state": "active",
            }
        )

    def _calculate_motion_model(
        self,
        detections: list[BoundingBox],
        timestamps: list[float],
        current_timestamp: float,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Calculate velocity and acceleration from position history."""
        if len(detections) < MIN_DETECTIONS_FOR_VELOCITY:
            return (0.0, 0.0), (0.0, 0.0)

        # Calculate velocity from last two positions
        dt = current_timestamp - timestamps[-2]
        if dt <= 0:
            return self.velocity, self.acceleration

        curr_pos = detections[-1]
        prev_pos = detections[-2]

        new_vx = (curr_pos.x - prev_pos.x) / dt
        new_vy = (curr_pos.y - prev_pos.y) / dt

        # Smooth velocity with exponential moving average
        alpha = VELOCITY_SMOOTHING_FACTOR
        old_vx, old_vy = self.velocity
        smoothed_velocity = (
            alpha * new_vx + (1 - alpha) * old_vx,
            alpha * new_vy + (1 - alpha) * old_vy,
        )

        # Calculate acceleration if we have enough history
        new_acceleration = self.acceleration
        if len(detections) >= MIN_DETECTIONS_FOR_ACCELERATION:
            prev_dt = timestamps[-2] - timestamps[-3]
            if prev_dt > 0:
                prev_prev_pos = detections[-3]
                prev_vx = (prev_pos.x - prev_prev_pos.x) / prev_dt
                prev_vy = (prev_pos.y - prev_prev_pos.y) / prev_dt

                ax = (new_vx - prev_vx) / dt
                ay = (new_vy - prev_vy) / dt

                # Smooth acceleration
                old_ax, old_ay = self.acceleration
                new_acceleration = (
                    alpha * ax + (1 - alpha) * old_ax,
                    alpha * ay + (1 - alpha) * old_ay,
                )

        return smoothed_velocity, new_acceleration

    def mark_lost(self, timestamp: float) -> Track:
        """Mark track as lost. Returns new Track instance."""
        if timestamp < self.last_update_timestamp:
            msg = (
                f"Timestamp cannot go backwards: {timestamp} "
                f"< {self.last_update_timestamp}"
            )
            raise ValueError(msg)

        new_frames_since_update = self.frames_since_update + 1

        # Update state based on how long since last update
        if new_frames_since_update > FRAMES_UNTIL_LOST:
            new_state = "lost"
        elif new_frames_since_update > FRAMES_UNTIL_OCCLUDED:
            new_state = "occluded"
        else:
            new_state = self.state

        return self.copy(
            update={
                "frames_since_update": new_frames_since_update,
                "last_update_timestamp": timestamp,
                "state": new_state,
            }
        )


class TrackingStats(BaseModel):
    """Motion tracking performance statistics."""

    total_tracks_created: int = Field(default=0, ge=0)
    total_tracks_lost: int = Field(default=0, ge=0)
    average_track_duration: float = Field(default=0.0, ge=0.0)
    active_tracks: int = Field(default=0, ge=0)
    average_speed_px_per_sec: float = Field(default=0.0, ge=0.0)
    tracking_enabled: bool = True

    class Config:
        """Configuration for tracking stats."""

        frozen = True


class MotionTracker(BaseModel):
    """Track detected regions across frames for temporal consistency."""

    config: DetectionConfig
    active_tracks: dict[int, Track] = Field(default_factory=dict)
    next_track_id: int = Field(default=0, ge=0)
    max_tracking_distance: float = Field(default=DEFAULT_MAX_TRACKING_DISTANCE, gt=0.0)
    max_frames_without_update: int = Field(
        default=DEFAULT_MAX_FRAMES_WITHOUT_UPDATE, ge=1
    )
    tracking_stats: TrackingStats = Field(default_factory=TrackingStats)

    class Config:
        """Configuration for motion tracker."""

        arbitrary_types_allowed = True

    def update(
        self, detections: list[tuple[BoundingBox, float]], timestamp: float
    ) -> tuple[MotionTracker, list[TrackedDetection]]:
        """Update tracking with new frame detections."""
        if timestamp < 0:
            msg = f"Timestamp must be non-negative, got {timestamp}"
            raise ValueError(msg)

        # Validate detections
        for _bbox, confidence in detections:
            if not 0.0 <= confidence <= 1.0:
                msg = f"Confidence must be between 0.0 and 1.0, got {confidence}"
                raise ValueError(msg)

        if not self.config.enable_motion_tracking:
            # Return detections without tracking
            untracked_detections = [
                TrackedDetection(
                    bbox=bbox,
                    confidence=conf,
                    track_id=-1,  # No tracking
                    velocity=(0.0, 0.0),
                    frames_tracked=1,
                    last_seen_timestamp=timestamp,
                )
                for bbox, conf in detections
            ]
            return self, untracked_detections

        # Step 1: Predict current positions of existing tracks
        predictions = self._predict_track_positions(timestamp)

        # Step 2: Associate detections with existing tracks
        associations = self._associate_detections_to_tracks(detections, predictions)

        # Step 3: Update associated tracks
        updated_tracks = dict(self.active_tracks.items())

        for (bbox, confidence), track_id in associations:
            if track_id in updated_tracks:
                updated_tracks[track_id] = updated_tracks[
                    track_id
                ].update_with_detection(bbox, confidence, timestamp)

        # Step 4: Create new tracks for unassociated detections
        unassociated_detections = [
            det for det in detections if det not in [assoc[0] for assoc in associations]
        ]

        new_next_track_id = self.next_track_id
        new_stats = self.tracking_stats

        for bbox, confidence in unassociated_detections:
            new_track = Track(
                id=new_next_track_id,
                detections_history=[bbox],
                timestamps_history=[timestamp],
                confidence_history=[confidence],
                velocity=(0.0, 0.0),
                acceleration=(0.0, 0.0),
                creation_timestamp=timestamp,
                last_update_timestamp=timestamp,
            )

            updated_tracks[new_next_track_id] = new_track
            new_next_track_id += 1

            # Update stats
            new_stats = new_stats.copy(
                update={"total_tracks_created": new_stats.total_tracks_created + 1}
            )

            logger.debug("Created new track %d at %s", new_track.id, bbox)

        # Step 5: Update tracks that weren't associated (mark as lost)
        associated_track_ids = {track_id for _, track_id in associations}
        for track_id, track in updated_tracks.items():
            if track_id not in associated_track_ids:
                updated_tracks[track_id] = track.mark_lost(timestamp)

        # Step 6: Remove lost tracks and update stats
        final_tracks = {}
        tracks_lost = 0
        total_duration = 0.0

        for track_id, track in updated_tracks.items():
            if (
                track.state == "lost"
                or track.frames_since_update > self.max_frames_without_update
            ):
                # Track is lost
                tracks_lost += 1
                total_duration += track.track_duration
                logger.debug(
                    "Removed lost track %d (duration: %.2fs)",
                    track_id,
                    track.track_duration,
                )
            else:
                final_tracks[track_id] = track

        # Update tracking stats
        if tracks_lost > 0:
            old_total_lost = new_stats.total_tracks_lost
            new_total_lost = old_total_lost + tracks_lost
            old_avg_duration = new_stats.average_track_duration

            if new_total_lost > 0:
                new_avg_duration = (
                    old_avg_duration * old_total_lost + total_duration
                ) / new_total_lost
            else:
                new_avg_duration = 0.0

            new_stats = new_stats.copy(
                update={
                    "total_tracks_lost": new_total_lost,
                    "average_track_duration": new_avg_duration,
                }
            )

        # Calculate current stats
        velocities = [track.velocity for track in final_tracks.values()]
        avg_speed = 0.0
        if velocities:
            speeds = [math.sqrt(vx * vx + vy * vy) for vx, vy in velocities]
            avg_speed = sum(speeds) / len(speeds)

        final_stats = new_stats.copy(
            update={
                "active_tracks": len(final_tracks),
                "average_speed_px_per_sec": avg_speed,
                "tracking_enabled": self.config.enable_motion_tracking,
            }
        )

        # Create new instance with updated state
        updated_tracker = self.copy(
            update={
                "active_tracks": final_tracks,
                "next_track_id": new_next_track_id,
                "tracking_stats": final_stats,
            }
        )

        # Step 7: Return updated tracker and tracked detections
        tracked_detections = updated_tracker._get_tracked_detections(timestamp)
        return updated_tracker, tracked_detections

    def _predict_track_positions(self, timestamp: float) -> dict[int, BoundingBox]:
        """Predict current positions of all active tracks."""
        predictions = {}

        for track_id, track in self.active_tracks.items():
            if track.current_position:
                dt = timestamp - track.last_update_timestamp
                try:
                    predicted_pos = track.predict_position(dt)
                    predictions[track_id] = predicted_pos
                except ValueError:
                    # Prediction failed, use last known position
                    predictions[track_id] = track.current_position

        return predictions

    def _associate_detections_to_tracks(
        self,
        detections: list[tuple[BoundingBox, float]],
        predictions: dict[int, BoundingBox],
    ) -> list[tuple[tuple[BoundingBox, float], int]]:
        """Associate detections to tracks using distance-based matching."""
        if not predictions or not detections:
            return []

        # Calculate cost matrix (distance between detections and predictions)
        cost_matrix = []
        track_ids = list(predictions.keys())

        for bbox, conf in detections:
            row = []
            for track_id in track_ids:
                predicted_bbox = predictions[track_id]
                distance = self._calculate_distance(bbox, predicted_bbox)
                row.append(distance)
            cost_matrix.append(row)

        # Simple greedy assignment (in production, use Hungarian algorithm)
        associations = []
        used_tracks = set()
        used_detections = set()

        # Sort detections by confidence (higher confidence first)
        sorted_detections = sorted(
            enumerate(detections), key=lambda x: x[1][1], reverse=True
        )

        for det_idx, (bbox, conf) in sorted_detections:
            if det_idx in used_detections:
                continue

            best_track_id = None
            best_distance = float("inf")

            for track_idx, track_id in enumerate(track_ids):
                if track_id in used_tracks:
                    continue

                distance = cost_matrix[det_idx][track_idx]
                if distance < best_distance and distance < self.max_tracking_distance:
                    best_distance = distance
                    best_track_id = track_id

            if best_track_id is not None:
                associations.append(((bbox, conf), best_track_id))
                used_tracks.add(best_track_id)
                used_detections.add(det_idx)

        return associations

    def _calculate_distance(self, bbox1: BoundingBox, bbox2: BoundingBox) -> float:
        """Calculate distance between two bounding boxes."""
        # Use center-to-center distance
        center1 = bbox1.center
        center2 = bbox2.center

        dx = center1[0] - center2[0]
        dy = center1[1] - center2[1]

        return math.sqrt(dx * dx + dy * dy)

    def _get_tracked_detections(self, timestamp: float) -> list[TrackedDetection]:
        """Get current tracked detections with all tracking information."""
        tracked_detections = []

        for track_id, track in self.active_tracks.items():
            if track.current_position is None:
                continue

            # Use predicted position if track was lost recently
            if (
                track.state in ["occluded", "lost"]
                and track.frames_since_update <= MAX_FRAMES_FOR_PREDICTION
            ):
                dt = timestamp - track.last_update_timestamp
                try:
                    predicted_bbox = track.predict_position(dt)
                    tracked_detection = TrackedDetection(
                        bbox=predicted_bbox,
                        confidence=track.track_confidence
                        * REDUCED_CONFIDENCE_PREDICTION,
                        track_id=track_id,
                        velocity=track.velocity,
                        frames_tracked=len(track.detections_history),
                        last_seen_timestamp=track.last_update_timestamp,
                        prediction_confidence=max(
                            0.1, 1.0 - track.frames_since_update * 0.2
                        ),
                        is_predicted=True,
                    )
                    tracked_detections.append(tracked_detection)
                except ValueError:
                    # Prediction failed, skip this track
                    continue
            else:
                # Use actual detection
                tracked_detection = TrackedDetection(
                    bbox=track.current_position,
                    confidence=track.confidence_history[-1]
                    if track.confidence_history
                    else 0.5,
                    track_id=track_id,
                    velocity=track.velocity,
                    frames_tracked=len(track.detections_history),
                    last_seen_timestamp=track.last_update_timestamp,
                    prediction_confidence=1.0,
                    is_predicted=False,
                )
                tracked_detections.append(tracked_detection)

        return tracked_detections

    def reset_tracking(self) -> MotionTracker:
        """Reset all tracking state. Returns new MotionTracker instance."""
        return self.copy(
            update={
                "active_tracks": {},
                "next_track_id": 0,
                "tracking_stats": TrackingStats(),
            }
        )

    def get_track_by_id(self, track_id: int) -> Track | None:
        """Get track by ID."""
        return self.active_tracks.get(track_id)

    def get_track_history(
        self, track_id: int, max_frames: int = 10
    ) -> list[BoundingBox]:
        """Get recent position history for a track."""
        track = self.active_tracks.get(track_id)
        if not track:
            return []

        return track.detections_history[-max_frames:]
