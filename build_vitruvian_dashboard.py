#!/usr/bin/env python3
"""Build an interactive Vitruvian history dashboard from a backup export."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


PERTH = timezone(timedelta(hours=8), "Australia/Perth")


def as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def clean_name(value: Any) -> str:
    text = str(value or "").strip()
    return text or "Unknown"


def local_dt(ms: Any, tz: timezone = PERTH) -> datetime | None:
    if ms is None:
        return None
    try:
        return datetime.fromtimestamp(float(ms) / 1000, tz=timezone.utc).astimezone(tz)
    except (TypeError, ValueError, OSError):
        return None


def fmt_dt(ms: Any, tz: timezone = PERTH) -> dict[str, str]:
    dt = local_dt(ms, tz)
    if dt is None:
        return {"date": "", "time": "", "dateTime": "", "label": ""}
    return {
        "date": dt.strftime("%Y-%m-%d"),
        "time": dt.strftime("%H:%M"),
        "dateTime": dt.isoformat(timespec="seconds"),
        "label": dt.strftime("%Y-%m-%d %H:%M"),
    }


def rounded(value: Any, places: int = 2) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return round(number, places)


def mean(values: list[float]) -> float | None:
    usable = [v for v in values if v is not None and not math.isnan(v)]
    if not usable:
        return None
    return statistics.fmean(usable)


def median(values: list[float]) -> float | None:
    usable = [v for v in values if v is not None and not math.isnan(v)]
    if not usable:
        return None
    return statistics.median(usable)


def is_echo_mode(value: Any) -> bool:
    return "echo" in str(value or "").strip().lower()


def moving_average(values: list[float], span: int = 5) -> list[float]:
    if not values:
        return []
    half = max(1, span // 2)
    smoothed: list[float] = []
    for idx in range(len(values)):
        start = max(0, idx - half)
        end = min(len(values), idx + half + 1)
        smoothed.append(statistics.fmean(values[start:end]))
    return smoothed


def downsample_indices(length: int, max_points: int = 180) -> list[int]:
    if length <= max_points:
        return list(range(length))
    stride = math.ceil(length / max_points)
    indices = list(range(0, length, stride))
    if indices[-1] != length - 1:
        indices.append(length - 1)
    return indices


def collapse_close_centers(
    centers: list[int],
    timestamps: list[int],
    positions: list[float],
    min_gap_ms: int = 650,
) -> list[int]:
    if not centers:
        return []
    centers = sorted(centers, key=lambda i: timestamps[i])
    merged: list[int] = [centers[0]]
    for idx in centers[1:]:
        previous = merged[-1]
        if timestamps[idx] - timestamps[previous] < min_gap_ms:
            if positions[idx] < positions[previous]:
                merged[-1] = idx
        else:
            merged.append(idx)
    return merged


def detect_rep_centers(samples: list[dict[str, Any]], expected_reps: int) -> tuple[list[int], str]:
    if len(samples) < 8 or expected_reps <= 0:
        return [], "none"

    timestamps = [as_int(row.get("timestamp")) for row in samples]
    positions = [
        (as_float(row.get("position")) + as_float(row.get("positionB"))) / 2
        for row in samples
    ]
    velocities = [
        (as_float(row.get("velocity")) + as_float(row.get("velocityB"))) / 2
        for row in samples
    ]
    smooth_pos = moving_average(positions, 5)
    rom = max(smooth_pos) - min(smooth_pos)
    if rom < 3:
        return [], "low-rom"

    low_threshold = min(smooth_pos) + max(2.0, rom * 0.38)
    centers: list[int] = []
    start: int | None = None
    for idx, pos in enumerate(smooth_pos):
        if pos <= low_threshold:
            if start is None:
                start = idx
        elif start is not None:
            group = range(start, idx)
            centers.append(min(group, key=lambda i: smooth_pos[i]))
            start = None
    if start is not None:
        group = range(start, len(smooth_pos))
        centers.append(min(group, key=lambda i: smooth_pos[i]))

    centers = collapse_close_centers(centers, timestamps, smooth_pos)
    method = "position-valleys"

    if len(centers) < expected_reps:
        crossing_centers: list[int] = []
        for idx in range(1, len(velocities) - 1):
            if velocities[idx - 1] < -2.0 and velocities[idx + 1] > 2.0:
                window_start = max(0, idx - 5)
                window_end = min(len(smooth_pos), idx + 6)
                crossing_centers.append(
                    min(range(window_start, window_end), key=lambda i: smooth_pos[i])
                )
        centers = collapse_close_centers(centers + crossing_centers, timestamps, smooth_pos)
        method = "position-valleys-and-velocity"

    if len(centers) < expected_reps:
        centers = [
            min(len(samples) - 1, max(0, round((idx + 0.5) * len(samples) / expected_reps)))
            for idx in range(expected_reps)
        ]
        method = "estimated-even-split"

    if len(centers) > expected_reps:
        centers = centers[-expected_reps:]

    return sorted(centers, key=lambda i: timestamps[i]), method


def segment_working_reps(
    session: dict[str, Any],
    samples: list[dict[str, Any]],
    set_number: int | None,
) -> list[dict[str, Any]]:
    expected_reps = as_int(session.get("workingReps") or session.get("totalReps"))
    target_per_cable = as_float(session.get("weightPerCableKg"))
    cable_count = max(1, as_int(session.get("cableCount"), 1))
    if expected_reps <= 0 or target_per_cable <= 0:
        return []

    sorted_samples = sorted(samples, key=lambda row: as_int(row.get("timestamp")))
    load_threshold = max(1.0, target_per_cable * 0.85)
    working_samples = []
    for row in sorted_samples:
        load_a = as_float(row.get("load"))
        load_b = as_float(row.get("loadB"), load_a)
        avg_load = (load_a + load_b) / 2 if cable_count >= 2 else load_a
        if avg_load >= load_threshold:
            working_samples.append(row)

    if len(working_samples) < expected_reps * 6:
        return []

    centers, method = detect_rep_centers(working_samples, expected_reps)
    if not centers:
        return []

    timestamps = [as_int(row.get("timestamp")) for row in working_samples]
    avg_velocities = [
        (as_float(row.get("velocity")) + as_float(row.get("velocityB"))) / 2
        for row in working_samples
    ]
    traces: list[dict[str, Any]] = []
    parts = fmt_dt(session.get("timestamp"))
    exercise_name = clean_name(session.get("exerciseName"))

    for rep_idx, center_idx in enumerate(centers, start=1):
        start_idx = 0 if rep_idx == 1 else (centers[rep_idx - 2] + center_idx) // 2
        end_idx = len(working_samples) - 1
        if rep_idx < len(centers):
            end_idx = (center_idx + centers[rep_idx]) // 2

        active_indices = [
            idx for idx in range(start_idx, end_idx + 1) if abs(avg_velocities[idx]) > 1.5
        ]
        if active_indices:
            start_idx = max(start_idx, active_indices[0] - 4)
            end_idx = min(end_idx, active_indices[-1] + 4)

        segment = working_samples[start_idx : end_idx + 1]
        if len(segment) < 6:
            continue

        full_per_cable_loads: list[float] = []
        full_total_loads: list[float] = []
        for row in segment:
            load_a = as_float(row.get("load"))
            load_b = as_float(row.get("loadB"), load_a)
            full_per_cable_loads.append((load_a + load_b) / 2 if cable_count >= 2 else load_a)
            full_total_loads.append(load_a + load_b if cable_count >= 2 else load_a)

        selected = downsample_indices(len(segment))
        start_ms = as_int(segment[0].get("timestamp"))
        times: list[float] = []
        per_cable_loads: list[float] = []
        total_loads: list[float] = []
        positions: list[float] = []
        velocities: list[float] = []

        for sample_idx in selected:
            row = segment[sample_idx]
            load_a = as_float(row.get("load"))
            load_b = as_float(row.get("loadB"), load_a)
            per_cable_load = (load_a + load_b) / 2 if cable_count >= 2 else load_a
            total_load = load_a + load_b if cable_count >= 2 else load_a
            times.append(round((as_int(row.get("timestamp")) - start_ms) / 1000, 3))
            per_cable_loads.append(round(per_cable_load, 2))
            total_loads.append(round(total_load, 2))
            positions.append(round((as_float(row.get("position")) + as_float(row.get("positionB"))) / 2, 2))
            velocities.append(round((as_float(row.get("velocity")) + as_float(row.get("velocityB"))) / 2, 2))

        traces.append(
            {
                "sessionId": session["id"],
                "exerciseId": session["exerciseId"],
                "exerciseName": exercise_name,
                "date": parts["date"],
                "dateTime": parts["dateTime"],
                "label": parts["label"],
                "routineSessionId": session.get("routineSessionId") or "",
                "routineName": session.get("routineName") or "",
                "mode": session.get("mode") or "",
                "setNumber": None if set_number is None else set_number + 1,
                "repIndex": rep_idx,
                "weightPerCableKg": rounded(session.get("weightPerCableKg"), 2),
                "cableCount": cable_count,
                "durationSec": rounded(times[-1] if times else 0, 3),
                "avgPerCableLoadKg": rounded(mean(full_per_cable_loads), 2),
                "medianPerCableLoadKg": rounded(median(full_per_cable_loads), 2),
                "peakPerCableLoadKg": rounded(max(full_per_cable_loads), 2) if full_per_cable_loads else None,
                "avgTotalLoadKg": rounded(mean(full_total_loads), 2),
                "medianTotalLoadKg": rounded(median(full_total_loads), 2),
                "peakTotalLoadKg": rounded(max(full_total_loads), 2) if full_total_loads else None,
                "timeSec": times,
                "perCableLoadKg": per_cable_loads,
                "totalLoadKg": total_loads,
                "position": positions,
                "velocity": velocities,
                "segmentationMethod": method,
            }
        )

    return traces


def build_tables(raw_data: dict[str, Any]) -> dict[str, Any]:
    data = raw_data.get("data", {})
    raw_sessions = data.get("workoutSessions", [])
    raw_sets = data.get("completedSets", [])
    raw_samples = data.get("metricSamples", [])

    sessions: list[dict[str, Any]] = []
    session_by_id: dict[str, dict[str, Any]] = {}
    for raw in raw_sessions:
        if as_int(raw.get("totalReps")) <= 0:
            continue
        parts = fmt_dt(raw.get("timestamp"))
        cable_count = max(1, as_int(raw.get("cableCount"), 1))
        per_cable = as_float(raw.get("weightPerCableKg"))
        total_load = per_cable * cable_count
        reps = as_int(raw.get("workingReps") or raw.get("totalReps"))
        estimated_1rm = total_load * (1 + reps / 30) if total_load and reps else None
        row = {
            "id": raw.get("id") or "",
            "timestamp": as_int(raw.get("timestamp")),
            "localDateTime": parts["dateTime"],
            "localDate": parts["date"],
            "localTime": parts["time"],
            "label": parts["label"],
            "exerciseId": raw.get("exerciseId") or "",
            "exerciseName": clean_name(raw.get("exerciseName")),
            "routineSessionId": raw.get("routineSessionId") or "",
            "routineName": raw.get("routineName") or "",
            "routineId": raw.get("routineId") or "",
            "mode": raw.get("mode") or "",
            "targetReps": as_int(raw.get("targetReps")),
            "totalReps": as_int(raw.get("totalReps")),
            "workingReps": reps,
            "warmupReps": as_int(raw.get("warmupReps")),
            "weightPerCableKg": rounded(per_cable, 2),
            "cableCount": cable_count,
            "totalLoadKg": rounded(total_load, 2),
            "totalVolumeKg": rounded(raw.get("totalVolumeKg"), 2),
            "heaviestLiftKg": rounded(raw.get("heaviestLiftKg"), 2),
            "durationSec": rounded(as_float(raw.get("duration")) / 1000, 2),
            "avgMcvMmS": rounded(raw.get("avgMcvMmS"), 2),
            "avgAsymmetryPercent": rounded(raw.get("avgAsymmetryPercent"), 3),
            "totalVelocityLossPercent": rounded(raw.get("totalVelocityLossPercent"), 2),
            "dominantSide": raw.get("dominantSide") or "",
            "strengthProfile": raw.get("strengthProfile") or "",
            "estimatedOneRepMaxKg": rounded(estimated_1rm, 2),
        }
        sessions.append(row)
        session_by_id[row["id"]] = row

    completed_sets: list[dict[str, Any]] = []
    set_number_by_session: dict[str, int] = {}
    for raw in raw_sets:
        session = session_by_id.get(raw.get("sessionId"))
        if not session:
            continue
        completed = fmt_dt(raw.get("completedAt"))
        actual_weight = as_float(raw.get("actualWeightKg"))
        total_actual_load = actual_weight * max(1, as_int(session.get("cableCount"), 1))
        actual_reps = as_int(raw.get("actualReps"))
        one_rm = total_actual_load * (1 + actual_reps / 30) if total_actual_load and actual_reps else None
        row = {
            "id": raw.get("id") or "",
            "sessionId": raw.get("sessionId") or "",
            "setNumber": as_int(raw.get("setNumber")),
            "displaySetNumber": as_int(raw.get("setNumber")) + 1,
            "setType": raw.get("setType") or "",
            "actualReps": actual_reps,
            "actualWeightPerCableKg": rounded(actual_weight, 2),
            "actualTotalLoadKg": rounded(total_actual_load, 2),
            "actualVolumeKg": rounded(actual_reps * total_actual_load, 2),
            "loggedRpe": rounded(raw.get("loggedRpe"), 1),
            "isPr": bool(raw.get("isPr")),
            "completedLocalDateTime": completed["dateTime"],
            "completedLocalDate": completed["date"],
            "exerciseId": session["exerciseId"],
            "exerciseName": session["exerciseName"],
            "routineSessionId": session["routineSessionId"],
            "routineName": session["routineName"],
            "mode": session["mode"],
        }
        completed_sets.append(row)
        set_number_by_session[row["sessionId"]] = as_int(raw.get("setNumber"))

    workout_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    daily_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for session in sessions:
        workout_id = session["routineSessionId"] or session["id"]
        workout_groups[(workout_id, session["exerciseId"])].append(session)
        daily_groups[(session["localDate"], session["exerciseId"])].append(session)

    workout_summary: list[dict[str, Any]] = []
    for (workout_id, exercise_id), rows in workout_groups.items():
        rows = sorted(rows, key=lambda row: row["timestamp"])
        volume = sum(as_float(row.get("totalVolumeKg")) for row in rows)
        reps = sum(as_int(row.get("workingReps")) for row in rows)
        loads = [as_float(row.get("weightPerCableKg")) for row in rows]
        total_loads = [as_float(row.get("totalLoadKg")) for row in rows]
        e1rms = [as_float(row.get("estimatedOneRepMaxKg")) for row in rows]
        speeds = [as_float(row.get("avgMcvMmS"), math.nan) for row in rows if row.get("avgMcvMmS") is not None]
        velocity_losses = [
            as_float(row.get("totalVelocityLossPercent"), math.nan)
            for row in rows
            if row.get("totalVelocityLossPercent") is not None
        ]
        workout_summary.append(
            {
                "workoutId": workout_id,
                "timestamp": rows[0]["timestamp"],
                "localDateTime": rows[0]["localDateTime"],
                "localDate": rows[0]["localDate"],
                "label": rows[0]["label"],
                "exerciseId": exercise_id,
                "exerciseName": rows[0]["exerciseName"],
                "routineName": rows[0]["routineName"],
                "mode": rows[0]["mode"],
                "sets": len(rows),
                "workingReps": reps,
                "maxWeightPerCableKg": rounded(max(loads), 2),
                "maxTotalLoadKg": rounded(max(total_loads), 2),
                "totalVolumeKg": rounded(volume, 2),
                "estimatedOneRepMaxKg": rounded(max(e1rms), 2) if e1rms else None,
                "avgMcvMmS": rounded(mean(speeds), 2),
                "avgVelocityLossPercent": rounded(mean(velocity_losses), 2),
            }
        )

    daily_summary: list[dict[str, Any]] = []
    for (date, exercise_id), rows in daily_groups.items():
        daily_summary.append(
            {
                "localDate": date,
                "exerciseId": exercise_id,
                "exerciseName": rows[0]["exerciseName"],
                "sets": len(rows),
                "workingReps": sum(as_int(row.get("workingReps")) for row in rows),
                "maxWeightPerCableKg": rounded(max(as_float(row.get("weightPerCableKg")) for row in rows), 2),
                "maxTotalLoadKg": rounded(max(as_float(row.get("totalLoadKg")) for row in rows), 2),
                "totalVolumeKg": rounded(sum(as_float(row.get("totalVolumeKg")) for row in rows), 2),
            }
        )

    samples_by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in raw_samples:
        session_id = raw.get("sessionId")
        if session_id in session_by_id:
            samples_by_session[session_id].append(raw)

    raw_session_by_id = {raw.get("id"): raw for raw in raw_sessions}
    rep_traces: list[dict[str, Any]] = []
    for session in sessions:
        raw_session = raw_session_by_id.get(session["id"])
        if not raw_session:
            continue
        traces = segment_working_reps(
            raw_session,
            samples_by_session.get(session["id"], []),
            set_number_by_session.get(session["id"]),
        )
        rep_traces.extend(traces)

    rep_loads_by_workout: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for trace in rep_traces:
        session = session_by_id.get(trace["sessionId"])
        if not session:
            continue
        workout_id = session["routineSessionId"] or session["id"]
        rep_loads_by_workout[(workout_id, trace["exerciseId"])].append(trace)

    for row in workout_summary:
        traces = rep_loads_by_workout.get((row["workoutId"], row["exerciseId"]), [])
        echo_traces = [trace for trace in traces if is_echo_mode(trace.get("mode"))]
        non_echo_traces = [trace for trace in traces if not is_echo_mode(trace.get("mode"))]
        row["echoRepCount"] = len(echo_traces)
        row["nonEchoRepCount"] = len(non_echo_traces)
        row["largestRepMedianWeightPerCableKg"] = rounded(
            max((as_float(trace.get("medianPerCableLoadKg")) for trace in traces), default=math.nan),
            2,
        )
        row["largestRepAverageWeightPerCableKg"] = rounded(
            max((as_float(trace.get("avgPerCableLoadKg")) for trace in traces), default=math.nan),
            2,
        )
        row["largestRepMedianTotalLoadKg"] = rounded(
            max((as_float(trace.get("medianTotalLoadKg")) for trace in traces), default=math.nan),
            2,
        )
        row["largestRepAverageTotalLoadKg"] = rounded(
            max((as_float(trace.get("avgTotalLoadKg")) for trace in traces), default=math.nan),
            2,
        )
        row["largestEchoRepMedianWeightPerCableKg"] = rounded(
            max((as_float(trace.get("medianPerCableLoadKg")) for trace in echo_traces), default=math.nan),
            2,
        )
        row["largestEchoRepAverageWeightPerCableKg"] = rounded(
            max((as_float(trace.get("avgPerCableLoadKg")) for trace in echo_traces), default=math.nan),
            2,
        )
        row["largestEchoRepPeakWeightPerCableKg"] = rounded(
            max((as_float(trace.get("peakPerCableLoadKg")) for trace in echo_traces), default=math.nan),
            2,
        )
        row["largestEchoRepMedianTotalLoadKg"] = rounded(
            max((as_float(trace.get("medianTotalLoadKg")) for trace in echo_traces), default=math.nan),
            2,
        )
        row["largestEchoRepAverageTotalLoadKg"] = rounded(
            max((as_float(trace.get("avgTotalLoadKg")) for trace in echo_traces), default=math.nan),
            2,
        )
        row["largestEchoRepPeakTotalLoadKg"] = rounded(
            max((as_float(trace.get("peakTotalLoadKg")) for trace in echo_traces), default=math.nan),
            2,
        )
        row["largestNonEchoRepMedianWeightPerCableKg"] = rounded(
            max((as_float(trace.get("medianPerCableLoadKg")) for trace in non_echo_traces), default=math.nan),
            2,
        )
        row["largestNonEchoRepAverageWeightPerCableKg"] = rounded(
            max((as_float(trace.get("avgPerCableLoadKg")) for trace in non_echo_traces), default=math.nan),
            2,
        )
        row["largestNonEchoRepMedianTotalLoadKg"] = rounded(
            max((as_float(trace.get("medianTotalLoadKg")) for trace in non_echo_traces), default=math.nan),
            2,
        )
        row["largestNonEchoRepAverageTotalLoadKg"] = rounded(
            max((as_float(trace.get("avgTotalLoadKg")) for trace in non_echo_traces), default=math.nan),
            2,
        )

    exercise_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for session in sessions:
        exercise_groups[session["exerciseId"]].append(session)

    exercises: list[dict[str, Any]] = []
    for exercise_id, rows in exercise_groups.items():
        workout_ids = {row["routineSessionId"] or row["id"] for row in rows}
        exercises.append(
            {
                "id": exercise_id,
                "name": rows[0]["exerciseName"],
                "sets": len(rows),
                "workouts": len(workout_ids),
                "workingReps": sum(as_int(row.get("workingReps")) for row in rows),
                "totalVolumeKg": rounded(sum(as_float(row.get("totalVolumeKg")) for row in rows), 2),
                "maxWeightPerCableKg": rounded(max(as_float(row.get("weightPerCableKg")) for row in rows), 2),
                "maxTotalLoadKg": rounded(max(as_float(row.get("totalLoadKg")) for row in rows), 2),
                "firstDate": min(row["localDate"] for row in rows),
                "lastDate": max(row["localDate"] for row in rows),
            }
        )

    sessions.sort(key=lambda row: row["timestamp"])
    completed_sets.sort(key=lambda row: row["completedLocalDateTime"])
    workout_summary.sort(key=lambda row: row["timestamp"])
    daily_summary.sort(key=lambda row: (row["localDate"], row["exerciseName"]))
    exercises.sort(key=lambda row: (-row["sets"], row["name"]))

    return {
        "metadata": {
            "exportedAt": raw_data.get("exportedAt") or "",
            "version": raw_data.get("version"),
            "appVersion": raw_data.get("appVersion") or "",
            "validSessions": len(sessions),
            "completedSets": len(completed_sets),
            "repTraces": len(rep_traces),
            "timezone": "Australia/Perth",
        },
        "exercises": exercises,
        "sessions": sessions,
        "sets": completed_sets,
        "workoutExerciseSummary": workout_summary,
        "dailyExerciseSummary": daily_summary,
        "repTraces": rep_traces,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def dashboard_html(data_json: str, muscle_map_json: str) -> str:
    template = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Vitruvian Training Dashboard</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f8fa;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #667085;
      --line: #d9dee7;
      --blue: #2563eb;
      --teal: #0f766e;
      --red: #dc2626;
      --gold: #b45309;
      --green: #059669;
      --violet: #7c3aed;
      --shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      font-family: Inter, Segoe UI, Arial, sans-serif;
      background: var(--bg);
      color: var(--ink);
    }

    header {
      background: #111827;
      color: #fff;
      padding: 22px 24px 18px;
    }

    .header-inner {
      max-width: 1400px;
      margin: 0 auto;
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 20px;
      align-items: end;
    }

    h1 {
      margin: 0 0 8px;
      font-size: 26px;
      line-height: 1.15;
      letter-spacing: 0;
    }

    .subtle {
      margin: 0;
      color: #cbd5e1;
      font-size: 13px;
    }

    .controls {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
      justify-content: flex-end;
    }

    .control {
      display: grid;
      gap: 6px;
      min-width: 178px;
    }

    .stacked-control {
      min-width: 210px;
      align-self: stretch;
      align-content: end;
    }

    label {
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      color: #94a3b8;
      letter-spacing: 0.02em;
    }

    select {
      width: 100%;
      min-height: 38px;
      border: 1px solid #334155;
      border-radius: 6px;
      background: #0f172a;
      color: #fff;
      padding: 8px 34px 8px 10px;
      font-size: 14px;
    }

    .switch-row {
      min-height: 38px;
      display: flex;
      align-items: center;
      gap: 10px;
      color: #fff;
      font-size: 13px;
      white-space: nowrap;
    }

    .switch {
      position: relative;
      width: 54px;
      height: 28px;
      flex: 0 0 auto;
    }

    .switch input {
      opacity: 0;
      width: 0;
      height: 0;
    }

    .slider {
      position: absolute;
      cursor: pointer;
      inset: 0;
      background: #334155;
      border-radius: 28px;
      transition: 0.18s ease;
    }

    .slider:before {
      content: "";
      position: absolute;
      height: 22px;
      width: 22px;
      left: 3px;
      top: 3px;
      background: #fff;
      border-radius: 50%;
      transition: 0.18s ease;
    }

    input:checked + .slider { background: var(--teal); }
    input:checked + .slider:before { transform: translateX(26px); }

    .file-input {
      display: none;
    }

    .file-button {
      width: 100%;
      min-height: 38px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      border: 1px solid #334155;
      border-radius: 6px;
      background: #0f172a;
      color: #fff;
      padding: 8px 10px;
      font-size: 14px;
      font-weight: 600;
      text-transform: none;
      letter-spacing: 0;
      cursor: pointer;
    }

    .file-button:hover {
      border-color: #64748b;
    }

    main {
      max-width: 1400px;
      margin: 0 auto;
      padding: 18px 24px 32px;
    }

    .kpis {
      display: grid;
      grid-template-columns: repeat(5, minmax(120px, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }

    .kpi, .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }

    .kpi {
      padding: 14px 14px 12px;
      min-height: 86px;
    }

    .kpi-title {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.02em;
      margin-bottom: 8px;
    }

    .kpi-value {
      font-size: 24px;
      line-height: 1.1;
      font-weight: 800;
    }

    .kpi-note {
      color: var(--muted);
      font-size: 12px;
      margin-top: 6px;
    }

    .grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }

    .panel {
      min-width: 0;
      padding: 14px;
    }

    .wide { grid-column: 1 / -1; }

    .panel-head {
      display: flex;
      align-items: start;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 10px;
    }

    h2 {
      margin: 0;
      font-size: 16px;
      line-height: 1.25;
      letter-spacing: 0;
    }

    .panel-note {
      margin: 4px 0 0;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
    }

    .chart-options {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      justify-content: flex-end;
      align-items: center;
      color: var(--muted);
      font-size: 12px;
    }

    .check-control {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      min-height: 30px;
      padding: 5px 8px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: none;
      letter-spacing: 0;
      cursor: pointer;
    }

    .mini-control {
      display: flex;
      gap: 8px;
      align-items: center;
      white-space: nowrap;
    }

    .mini-control label {
      color: var(--muted);
    }

    .mini-control select {
      min-width: 110px;
      min-height: 32px;
      background: #fff;
      color: var(--ink);
      border-color: var(--line);
      padding: 6px 28px 6px 8px;
    }

    canvas {
      display: block;
      width: 100%;
      height: 320px;
      border-radius: 6px;
      background: #fff;
    }

    .chart-tooltip {
      position: fixed;
      z-index: 50;
      display: none;
      max-width: 240px;
      padding: 8px 10px;
      border: 1px solid #c7ced9;
      border-radius: 6px;
      background: rgba(17, 24, 39, 0.94);
      color: #fff;
      box-shadow: 0 10px 24px rgba(15, 23, 42, 0.18);
      font-size: 12px;
      line-height: 1.35;
      white-space: pre-line;
      pointer-events: none;
    }

    #repOverlay { height: 430px; }
    #muscleBalanceChart { height: 360px; }

    .muscle-balance-list {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
      margin-top: 10px;
    }

    .muscle-balance-item {
      display: grid;
      gap: 4px;
      padding: 8px 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      min-width: 0;
    }

    .muscle-balance-item strong {
      font-size: 12px;
      color: var(--ink);
    }

    .muscle-balance-item span {
      color: var(--muted);
      font-size: 12px;
    }

    .legend {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 10px;
      color: var(--muted);
      font-size: 12px;
      max-height: 74px;
      overflow: auto;
    }

    .legend-item {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 7px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--muted);
      cursor: pointer;
      font: inherit;
      appearance: none;
    }

    .legend-item:hover {
      border-color: #9aa4b2;
    }

    .legend-item:focus-visible {
      outline: 2px solid var(--blue);
      outline-offset: 2px;
    }

    .legend-item.is-muted {
      background: #f1f3f6;
      color: #8a94a3;
    }

    .legend-toggle-all {
      font-weight: 700;
      color: var(--ink);
    }

    .swatch {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      flex: 0 0 auto;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }

    th, td {
      padding: 9px 8px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      white-space: nowrap;
    }

    th {
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.02em;
    }

    th.group-separator, td.group-separator {
      border-left: 2px solid #b8c0cc;
    }

    .table-wrap {
      overflow: auto;
      max-height: 380px;
    }

    .empty {
      display: grid;
      place-items: center;
      height: 300px;
      color: var(--muted);
      text-align: center;
      border: 1px dashed var(--line);
      border-radius: 6px;
    }

    @media (max-width: 1000px) {
      .header-inner { grid-template-columns: 1fr; }
      .controls { justify-content: flex-start; }
      .kpis { grid-template-columns: repeat(2, minmax(120px, 1fr)); }
      .grid { grid-template-columns: 1fr; }
      .muscle-balance-list { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }

    @media (max-width: 560px) {
      header { padding: 18px 16px; }
      main { padding: 14px 12px 28px; }
      h1 { font-size: 22px; }
      .kpis { grid-template-columns: 1fr; }
      .panel-head { display: grid; }
      canvas, #repOverlay { height: 300px; }
      .control { width: 100%; }
      .switch-row { white-space: normal; }
      .muscle-balance-list { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <div class="header-inner">
      <div>
        <h1>Vitruvian Training Dashboard</h1>
        <p class="subtle" id="sourceMeta"></p>
      </div>
      <div class="controls">
        <div class="control stacked-control">
          <label for="backupFileInput">Backup File</label>
          <label class="file-button" for="backupFileInput">Load JSON/TXT</label>
          <input class="file-input" id="backupFileInput" type="file" accept=".json,.txt,application/json,text/plain">
          <label for="exerciseSelect">Exercise</label>
          <select id="exerciseSelect"></select>
        </div>
        <div class="control">
          <label for="historyWindow">History Window</label>
          <select id="historyWindow">
            <option value="5">Last 5 workouts</option>
            <option value="10" selected>Last 10 workouts</option>
            <option value="20">Last 20 workouts</option>
            <option value="all">All workouts</option>
          </select>
        </div>
        <div class="control stacked-control">
          <label for="loadUnitToggle">Load Units</label>
          <div class="switch-row">
            <span>kg</span>
            <label class="switch">
              <input id="loadUnitToggle" type="checkbox">
              <span class="slider"></span>
            </label>
            <span>lbs</span>
          </div>
          <label for="loadToggle">Load basis</label>
          <div class="switch-row">
            <span>Per cable</span>
            <label class="switch">
              <input id="loadToggle" type="checkbox" checked>
              <span class="slider"></span>
            </label>
            <span>Total load</span>
          </div>
        </div>
      </div>
    </div>
  </header>

  <main>
    <section class="kpis" id="kpis"></section>

    <section class="grid">
      <article class="panel">
        <div class="panel-head">
          <div>
            <h2>Total Reps Per Workout</h2>
            <p class="panel-note" id="repsChartNote">Grouped by routine workout and selected exercise.</p>
          </div>
        </div>
        <canvas id="repsChart"></canvas>
      </article>

      <article class="panel">
        <div class="panel-head">
          <div>
            <h2>Load Progression</h2>
            <p class="panel-note" id="loadProgressionNote"></p>
          </div>
          <div class="chart-options" aria-label="Load progression options">
            <label class="check-control"><input id="loadEchoMedianToggle" type="checkbox"> Echo median</label>
            <label class="check-control"><input id="loadEchoAverageToggle" type="checkbox"> Echo average</label>
          </div>
        </div>
        <canvas id="loadChart"></canvas>
      </article>

      <article class="panel">
        <div class="panel-head">
          <div>
            <h2>Volume And Estimated Strength</h2>
            <p class="panel-note" id="volumeChartNote">Volume bars with estimated 1RM trend from completed working sets.</p>
          </div>
        </div>
        <canvas id="volumeChart"></canvas>
      </article>

      <article class="panel">
        <div class="panel-head">
          <div>
            <h2>Velocity Trend</h2>
            <p class="panel-note">Average concentric velocity from session summaries.</p>
          </div>
        </div>
        <canvas id="velocityChart"></canvas>
      </article>

      <article class="panel wide">
        <div class="panel-head">
          <div>
            <h2>Muscle Balance</h2>
            <p class="panel-note" id="muscleBalanceNote">Relative training focus by body part.</p>
          </div>
        </div>
        <canvas id="muscleBalanceChart"></canvas>
        <div class="muscle-balance-list" id="muscleBalanceList"></div>
      </article>

      <article class="panel wide">
        <div class="panel-head">
          <div>
            <h2>Working Rep Load Overlay</h2>
            <p class="panel-note" id="repOverlayNote">Load over time for working reps only. Colours identify the workout date.</p>
          </div>
          <div class="chart-options" aria-label="Working rep overlay options">
            <div class="mini-control">
              <label for="repOverlayMode">Overlay</label>
              <select id="repOverlayMode">
                <option value="all">All</option>
                <option value="maxAverage">Max average</option>
                <option value="maxMedian">Max median</option>
              </select>
            </div>
          </div>
        </div>
        <canvas id="repOverlay"></canvas>
        <div class="legend" id="overlayLegend"></div>
      </article>

      <article class="panel wide">
        <div class="panel-head">
          <div>
            <h2>Workout History Table</h2>
            <p class="panel-note">Recent grouped workouts for the selected exercise.</p>
          </div>
        </div>
        <div class="table-wrap">
          <table>
            <thead>
              <tr id="historyHeader"></tr>
            </thead>
            <tbody id="historyRows"></tbody>
          </table>
        </div>
      </article>
    </section>
  </main>
  <div class="chart-tooltip" id="chartTooltip"></div>

  <script>
    let DATA = __DATA__;
    const PROJECT_PHOENIX_EXERCISE_MUSCLE_MAP = __MUSCLE_MAP__;

    const palette = [
      "#2563eb", "#dc2626", "#0f766e", "#b45309", "#7c3aed",
      "#059669", "#be123c", "#0891b2", "#9333ea", "#ca8a04",
      "#1d4ed8", "#c2410c", "#047857", "#6d28d9", "#0e7490"
    ];
    const MUSCLE_GROUP_ORDER = ["CHEST", "BACK", "SHOULDERS", "ARMS", "CORE", "LEGS"];
    const EXERCISE_MUSCLE_WEIGHT_OVERRIDES = {
      "high bar squat": [
        { group: "LEGS", weight: 0.75 },
        { group: "CORE", weight: 0.25 }
      ],
      "conventional deadlift": [
        { group: "LEGS", weight: 0.40 },
        { group: "BACK", weight: 0.30 },
        { group: "CORE", weight: 0.30 }
      ]
    };
    const KG_TO_LB = 2.2046226218;
    const CACHE_KEY = "vitruvianTrainingDashboardCache:v1";
    const ALL_EXERCISES_ID = "__all_exercises__";

    function loadCachedDashboard() {
      try {
        const raw = localStorage.getItem(CACHE_KEY);
        if (!raw) return null;
        const parsed = JSON.parse(raw);
        if (!parsed?.data?.workoutExerciseSummary || !Array.isArray(parsed.data.exercises)) return null;
        return parsed;
      } catch (error) {
        return null;
      }
    }

    const cachedDashboard = loadCachedDashboard();
    if (cachedDashboard?.data) DATA = cachedDashboard.data;
    const cachedSettings = cachedDashboard?.settings || {};

    const state = {
      exerciseId: cachedSettings.exerciseId || ALL_EXERCISES_ID,
      loadBasis: cachedSettings.loadBasis === "perCable" ? "perCable" : "total",
      loadUnit: cachedSettings.loadUnit === "lbs" ? "lbs" : "kg",
      historyWindow: ["5", "10", "20", "all"].includes(cachedSettings.historyWindow) ? cachedSettings.historyWindow : "10",
      showLoadEchoMedian: Boolean(cachedSettings.showLoadEchoMedian),
      showLoadEchoAverage: Boolean(cachedSettings.showLoadEchoAverage),
      repOverlayMode: ["all", "maxAverage", "maxMedian"].includes(cachedSettings.repOverlayMode) ? cachedSettings.repOverlayMode : "all",
      dimmedOverlayDates: new Set(Array.isArray(cachedSettings.dimmedOverlayDates) ? cachedSettings.dimmedOverlayDates : [])
    };

    function dashboardCacheSettings() {
      return {
        exerciseId: state.exerciseId,
        loadBasis: state.loadBasis,
        loadUnit: state.loadUnit,
        historyWindow: state.historyWindow,
        showLoadEchoMedian: state.showLoadEchoMedian,
        showLoadEchoAverage: state.showLoadEchoAverage,
        repOverlayMode: state.repOverlayMode,
        dimmedOverlayDates: [...state.dimmedOverlayDates]
      };
    }

    function saveDashboardCache() {
      try {
        localStorage.setItem(CACHE_KEY, JSON.stringify({
          data: DATA,
          settings: dashboardCacheSettings(),
          savedAt: new Date().toISOString()
        }));
      } catch (error) {
        console.warn("Dashboard cache could not be saved", error);
      }
    }

    const exerciseSelect = document.getElementById("exerciseSelect");
    const backupFileInput = document.getElementById("backupFileInput");
    const historyWindow = document.getElementById("historyWindow");
    const loadUnitToggle = document.getElementById("loadUnitToggle");
    const loadToggle = document.getElementById("loadToggle");
    const loadEchoMedianToggle = document.getElementById("loadEchoMedianToggle");
    const loadEchoAverageToggle = document.getElementById("loadEchoAverageToggle");
    const repOverlayMode = document.getElementById("repOverlayMode");
    const chartTooltip = document.getElementById("chartTooltip");
    const chartHitAreas = new Map();
    const exerciseMuscleLookup = buildExerciseMuscleLookup();

    const PERTH_OFFSET_MS = 8 * 60 * 60 * 1000;

    function asNumber(value, fallback = 0) {
      if (value === null || value === undefined || value === "") return fallback;
      const number = Number(value);
      return Number.isFinite(number) ? number : fallback;
    }

    function asInt(value, fallback = 0) {
      return Math.trunc(asNumber(value, fallback));
    }

    function cleanName(value) {
      const text = String(value || "").trim();
      return text || "Unknown";
    }

    function normalizeExerciseName(value) {
      return String(value || "")
        .toLowerCase()
        .replace(/&/g, " and ")
        .replace(/\([^)]*\)/g, " ")
        .replace(/[^a-z0-9]+/g, " ")
        .replace(/\s+/g, " ")
        .trim();
    }

    function singularExerciseKey(value) {
      return normalizeExerciseName(value)
        .split(" ")
        .map(word => word.length > 3 && word.endsWith("s") && !word.endsWith("ss") ? word.slice(0, -1) : word)
        .join(" ");
    }

    function titleCaseGroup(group) {
      const text = String(group || "").toLowerCase();
      return text ? text.charAt(0).toUpperCase() + text.slice(1) : "Unknown";
    }

    function buildExerciseMuscleLookup() {
      const lookup = new Map();
      PROJECT_PHOENIX_EXERCISE_MUSCLE_MAP.forEach(entry => {
        const groups = (entry.muscleGroups || []).map(group => String(group || "").toUpperCase()).filter(Boolean);
        if (!groups.length) return;
        [entry.name, ...(entry.aliases || [])].forEach(name => {
          const key = normalizeExerciseName(name);
          if (key && !lookup.has(key)) lookup.set(key, groups);
          const singularKey = singularExerciseKey(name);
          if (singularKey && !lookup.has(singularKey)) lookup.set(singularKey, groups);
        });
      });
      return lookup;
    }

    function pad2(value) {
      return String(value).padStart(2, "0");
    }

    function fmtLocalParts(ms) {
      if (ms === null || ms === undefined || Number.isNaN(Number(ms))) {
        return { date: "", time: "", dateTime: "", label: "" };
      }
      const date = new Date(Number(ms) + PERTH_OFFSET_MS);
      const yyyy = date.getUTCFullYear();
      const mm = pad2(date.getUTCMonth() + 1);
      const dd = pad2(date.getUTCDate());
      const hh = pad2(date.getUTCHours());
      const min = pad2(date.getUTCMinutes());
      const ss = pad2(date.getUTCSeconds());
      return {
        date: `${yyyy}-${mm}-${dd}`,
        time: `${hh}:${min}`,
        dateTime: `${yyyy}-${mm}-${dd}T${hh}:${min}:${ss}+08:00`,
        label: `${yyyy}-${mm}-${dd} ${hh}:${min}`
      };
    }

    function roundOrNull(value, digits = 2) {
      if (value === null || value === undefined) return null;
      const number = Number(value);
      if (!Number.isFinite(number)) return null;
      const factor = 10 ** digits;
      return Math.round(number * factor) / factor;
    }

    function meanValue(values) {
      const usable = values.filter(value => Number.isFinite(Number(value))).map(Number);
      if (!usable.length) return null;
      return usable.reduce((sum, value) => sum + value, 0) / usable.length;
    }

    function medianValue(values) {
      const usable = values.filter(value => Number.isFinite(Number(value))).map(Number).sort((a, b) => a - b);
      if (!usable.length) return null;
      const mid = Math.floor(usable.length / 2);
      return usable.length % 2 ? usable[mid] : (usable[mid - 1] + usable[mid]) / 2;
    }

    function maxOrNull(values) {
      const usable = values.filter(value => Number.isFinite(Number(value))).map(Number);
      return usable.length ? Math.max(...usable) : null;
    }

    function movingAverage(values, span = 5) {
      if (!values.length) return [];
      const half = Math.max(1, Math.floor(span / 2));
      return values.map((_, idx) => {
        const start = Math.max(0, idx - half);
        const end = Math.min(values.length, idx + half + 1);
        return meanValue(values.slice(start, end));
      });
    }

    function downsampleIndices(length, maxPoints = 180) {
      if (length <= maxPoints) return Array.from({ length }, (_, idx) => idx);
      const stride = Math.ceil(length / maxPoints);
      const indices = [];
      for (let idx = 0; idx < length; idx += stride) indices.push(idx);
      if (indices[indices.length - 1] !== length - 1) indices.push(length - 1);
      return indices;
    }

    function collapseCloseCenters(centers, timestamps, positions, minGapMs = 650) {
      if (!centers.length) return [];
      const sorted = [...centers].sort((a, b) => timestamps[a] - timestamps[b]);
      const merged = [sorted[0]];
      sorted.slice(1).forEach(idx => {
        const previous = merged[merged.length - 1];
        if (timestamps[idx] - timestamps[previous] < minGapMs) {
          if (positions[idx] < positions[previous]) merged[merged.length - 1] = idx;
        } else {
          merged.push(idx);
        }
      });
      return merged;
    }

    function detectRepCenters(samples, expectedReps) {
      if (samples.length < 8 || expectedReps <= 0) return { centers: [], method: "none" };
      const timestamps = samples.map(row => asInt(row.timestamp));
      const positions = samples.map(row => (asNumber(row.position) + asNumber(row.positionB)) / 2);
      const velocities = samples.map(row => (asNumber(row.velocity) + asNumber(row.velocityB)) / 2);
      const smoothPos = movingAverage(positions, 5);
      const rom = Math.max(...smoothPos) - Math.min(...smoothPos);
      if (rom < 3) return { centers: [], method: "low-rom" };

      const lowThreshold = Math.min(...smoothPos) + Math.max(2, rom * 0.38);
      let centers = [];
      let start = null;
      smoothPos.forEach((pos, idx) => {
        if (pos <= lowThreshold) {
          if (start === null) start = idx;
        } else if (start !== null) {
          let best = start;
          for (let scan = start + 1; scan < idx; scan += 1) {
            if (smoothPos[scan] < smoothPos[best]) best = scan;
          }
          centers.push(best);
          start = null;
        }
      });
      if (start !== null) {
        let best = start;
        for (let scan = start + 1; scan < smoothPos.length; scan += 1) {
          if (smoothPos[scan] < smoothPos[best]) best = scan;
        }
        centers.push(best);
      }
      centers = collapseCloseCenters(centers, timestamps, smoothPos);
      let method = "position-valleys";

      if (centers.length < expectedReps) {
        const crossingCenters = [];
        for (let idx = 1; idx < velocities.length - 1; idx += 1) {
          if (velocities[idx - 1] < -2 && velocities[idx + 1] > 2) {
            const windowStart = Math.max(0, idx - 5);
            const windowEnd = Math.min(smoothPos.length, idx + 6);
            let best = windowStart;
            for (let scan = windowStart + 1; scan < windowEnd; scan += 1) {
              if (smoothPos[scan] < smoothPos[best]) best = scan;
            }
            crossingCenters.push(best);
          }
        }
        centers = collapseCloseCenters([...centers, ...crossingCenters], timestamps, smoothPos);
        method = "position-valleys-and-velocity";
      }

      if (centers.length < expectedReps) {
        centers = Array.from({ length: expectedReps }, (_, idx) =>
          Math.min(samples.length - 1, Math.max(0, Math.round((idx + 0.5) * samples.length / expectedReps)))
        );
        method = "estimated-even-split";
      }

      if (centers.length > expectedReps) centers = centers.slice(-expectedReps);
      return { centers: centers.sort((a, b) => timestamps[a] - timestamps[b]), method };
    }

    function isEchoMode(value) {
      return String(value || "").trim().toLowerCase().includes("echo");
    }

    function segmentWorkingReps(session, samples, setNumber) {
      const expectedReps = asInt(session.workingReps || session.totalReps);
      const targetPerCable = asNumber(session.weightPerCableKg);
      const cableCount = Math.max(1, asInt(session.cableCount, 1));
      if (expectedReps <= 0 || targetPerCable <= 0) return [];

      const sortedSamples = [...samples].sort((a, b) => asInt(a.timestamp) - asInt(b.timestamp));
      const loadThreshold = Math.max(1, targetPerCable * 0.85);
      const workingSamples = sortedSamples.filter(row => {
        const loadA = asNumber(row.load);
        const loadB = asNumber(row.loadB, loadA);
        const avgLoad = cableCount >= 2 ? (loadA + loadB) / 2 : loadA;
        return avgLoad >= loadThreshold;
      });
      if (workingSamples.length < expectedReps * 6) return [];

      const detected = detectRepCenters(workingSamples, expectedReps);
      const centers = detected.centers;
      if (!centers.length) return [];

      const avgVelocities = workingSamples.map(row => (asNumber(row.velocity) + asNumber(row.velocityB)) / 2);
      const parts = fmtLocalParts(session.timestamp);
      const exerciseName = cleanName(session.exerciseName);

      return centers.map((centerIdx, idx) => {
        let startIdx = idx === 0 ? 0 : Math.floor((centers[idx - 1] + centerIdx) / 2);
        let endIdx = idx < centers.length - 1 ? Math.floor((centerIdx + centers[idx + 1]) / 2) : workingSamples.length - 1;
        const activeIndices = [];
        for (let scan = startIdx; scan <= endIdx; scan += 1) {
          if (Math.abs(avgVelocities[scan]) > 1.5) activeIndices.push(scan);
        }
        if (activeIndices.length) {
          startIdx = Math.max(startIdx, activeIndices[0] - 4);
          endIdx = Math.min(endIdx, activeIndices[activeIndices.length - 1] + 4);
        }
        const segment = workingSamples.slice(startIdx, endIdx + 1);
        if (segment.length < 6) return null;

        const fullPerCableLoads = [];
        const fullTotalLoads = [];
        segment.forEach(row => {
          const loadA = asNumber(row.load);
          const loadB = asNumber(row.loadB, loadA);
          fullPerCableLoads.push(cableCount >= 2 ? (loadA + loadB) / 2 : loadA);
          fullTotalLoads.push(cableCount >= 2 ? loadA + loadB : loadA);
        });

        const selected = downsampleIndices(segment.length);
        const startMs = asInt(segment[0].timestamp);
        const times = [];
        const perCableLoads = [];
        const totalLoads = [];
        const positions = [];
        const velocities = [];
        selected.forEach(sampleIdx => {
          const row = segment[sampleIdx];
          const loadA = asNumber(row.load);
          const loadB = asNumber(row.loadB, loadA);
          times.push(roundOrNull((asInt(row.timestamp) - startMs) / 1000, 3));
          perCableLoads.push(roundOrNull(cableCount >= 2 ? (loadA + loadB) / 2 : loadA, 2));
          totalLoads.push(roundOrNull(cableCount >= 2 ? loadA + loadB : loadA, 2));
          positions.push(roundOrNull((asNumber(row.position) + asNumber(row.positionB)) / 2, 2));
          velocities.push(roundOrNull((asNumber(row.velocity) + asNumber(row.velocityB)) / 2, 2));
        });

        return {
          sessionId: session.id,
          exerciseId: session.exerciseId,
          exerciseName,
          date: parts.date,
          dateTime: parts.dateTime,
          label: parts.label,
          routineSessionId: session.routineSessionId || "",
          routineName: session.routineName || "",
          mode: session.mode || "",
          setNumber: setNumber === null || setNumber === undefined ? null : setNumber + 1,
          repIndex: idx + 1,
          weightPerCableKg: roundOrNull(session.weightPerCableKg, 2),
          cableCount,
          durationSec: roundOrNull(times[times.length - 1] || 0, 3),
          avgPerCableLoadKg: roundOrNull(meanValue(fullPerCableLoads), 2),
          medianPerCableLoadKg: roundOrNull(medianValue(fullPerCableLoads), 2),
          peakPerCableLoadKg: roundOrNull(maxOrNull(fullPerCableLoads), 2),
          avgTotalLoadKg: roundOrNull(meanValue(fullTotalLoads), 2),
          medianTotalLoadKg: roundOrNull(medianValue(fullTotalLoads), 2),
          peakTotalLoadKg: roundOrNull(maxOrNull(fullTotalLoads), 2),
          timeSec: times,
          perCableLoadKg: perCableLoads,
          totalLoadKg: totalLoads,
          position: positions,
          velocity: velocities,
          segmentationMethod: detected.method
        };
      }).filter(Boolean);
    }

    function buildDashboardData(rawData) {
      const source = rawData && rawData.data ? rawData.data : null;
      if (!source || !Array.isArray(source.workoutSessions)) {
        throw new Error("This does not look like a Vitruvian backup export.");
      }

      const rawSessions = source.workoutSessions || [];
      const rawSets = source.completedSets || [];
      const rawSamples = source.metricSamples || [];
      const sessions = [];
      const sessionById = new Map();

      rawSessions.forEach(raw => {
        if (asInt(raw.totalReps) <= 0) return;
        const parts = fmtLocalParts(raw.timestamp);
        const cableCount = Math.max(1, asInt(raw.cableCount, 1));
        const perCable = asNumber(raw.weightPerCableKg);
        const totalLoad = perCable * cableCount;
        const reps = asInt(raw.workingReps || raw.totalReps);
        const estimatedOneRepMax = totalLoad && reps ? totalLoad * (1 + reps / 30) : null;
        const row = {
          id: raw.id || "",
          timestamp: asInt(raw.timestamp),
          localDateTime: parts.dateTime,
          localDate: parts.date,
          localTime: parts.time,
          label: parts.label,
          exerciseId: raw.exerciseId || "",
          exerciseName: cleanName(raw.exerciseName),
          routineSessionId: raw.routineSessionId || "",
          routineName: raw.routineName || "",
          routineId: raw.routineId || "",
          mode: raw.mode || "",
          targetReps: asInt(raw.targetReps),
          totalReps: asInt(raw.totalReps),
          workingReps: reps,
          warmupReps: asInt(raw.warmupReps),
          weightPerCableKg: roundOrNull(perCable, 2),
          cableCount,
          totalLoadKg: roundOrNull(totalLoad, 2),
          totalVolumeKg: roundOrNull(raw.totalVolumeKg, 2),
          heaviestLiftKg: roundOrNull(raw.heaviestLiftKg, 2),
          durationSec: roundOrNull(asNumber(raw.duration) / 1000, 2),
          avgMcvMmS: roundOrNull(raw.avgMcvMmS, 2),
          avgAsymmetryPercent: roundOrNull(raw.avgAsymmetryPercent, 3),
          totalVelocityLossPercent: roundOrNull(raw.totalVelocityLossPercent, 2),
          dominantSide: raw.dominantSide || "",
          strengthProfile: raw.strengthProfile || "",
          estimatedOneRepMaxKg: roundOrNull(estimatedOneRepMax, 2)
        };
        sessions.push(row);
        sessionById.set(row.id, row);
      });

      const completedSets = [];
      const setNumberBySession = new Map();
      rawSets.forEach(raw => {
        const session = sessionById.get(raw.sessionId);
        if (!session) return;
        const completed = fmtLocalParts(raw.completedAt);
        const actualWeight = asNumber(raw.actualWeightKg);
        const totalActualLoad = actualWeight * Math.max(1, asInt(session.cableCount, 1));
        const actualReps = asInt(raw.actualReps);
        const oneRm = totalActualLoad && actualReps ? totalActualLoad * (1 + actualReps / 30) : null;
        completedSets.push({
          id: raw.id || "",
          sessionId: raw.sessionId || "",
          setNumber: asInt(raw.setNumber),
          displaySetNumber: asInt(raw.setNumber) + 1,
          setType: raw.setType || "",
          actualReps,
          actualWeightPerCableKg: roundOrNull(actualWeight, 2),
          actualTotalLoadKg: roundOrNull(totalActualLoad, 2),
          actualVolumeKg: roundOrNull(actualReps * totalActualLoad, 2),
          loggedRpe: roundOrNull(raw.loggedRpe, 1),
          isPr: Boolean(raw.isPr),
          completedLocalDateTime: completed.dateTime,
          completedLocalDate: completed.date,
          exerciseId: session.exerciseId,
          exerciseName: session.exerciseName,
          routineSessionId: session.routineSessionId,
          routineName: session.routineName,
          mode: session.mode,
          estimatedOneRepMaxKg: roundOrNull(oneRm, 2)
        });
        setNumberBySession.set(raw.sessionId, asInt(raw.setNumber));
      });

      const workoutGroups = new Map();
      const dailyGroups = new Map();
      sessions.forEach(session => {
        const workoutId = session.routineSessionId || session.id;
        const workoutKey = `${workoutId}::${session.exerciseId}`;
        const dailyKey = `${session.localDate}::${session.exerciseId}`;
        if (!workoutGroups.has(workoutKey)) workoutGroups.set(workoutKey, []);
        if (!dailyGroups.has(dailyKey)) dailyGroups.set(dailyKey, []);
        workoutGroups.get(workoutKey).push(session);
        dailyGroups.get(dailyKey).push(session);
      });

      const workoutSummary = [];
      workoutGroups.forEach(rows => {
        rows.sort((a, b) => a.timestamp - b.timestamp);
        const loads = rows.map(row => asNumber(row.weightPerCableKg));
        const totalLoads = rows.map(row => asNumber(row.totalLoadKg));
        const e1rms = rows.map(row => asNumber(row.estimatedOneRepMaxKg));
        const speeds = rows.map(row => row.avgMcvMmS).filter(value => value !== null);
        const velocityLosses = rows.map(row => row.totalVelocityLossPercent).filter(value => value !== null);
        workoutSummary.push({
          workoutId: rows[0].routineSessionId || rows[0].id,
          timestamp: rows[0].timestamp,
          localDateTime: rows[0].localDateTime,
          localDate: rows[0].localDate,
          label: rows[0].label,
          exerciseId: rows[0].exerciseId,
          exerciseName: rows[0].exerciseName,
          routineName: rows[0].routineName,
          mode: rows[0].mode,
          sets: rows.length,
          workingReps: rows.reduce((sum, row) => sum + asInt(row.workingReps), 0),
          maxWeightPerCableKg: roundOrNull(maxOrNull(loads), 2),
          maxTotalLoadKg: roundOrNull(maxOrNull(totalLoads), 2),
          totalVolumeKg: roundOrNull(rows.reduce((sum, row) => sum + asNumber(row.totalVolumeKg), 0), 2),
          estimatedOneRepMaxKg: roundOrNull(maxOrNull(e1rms), 2),
          avgMcvMmS: roundOrNull(meanValue(speeds), 2),
          avgVelocityLossPercent: roundOrNull(meanValue(velocityLosses), 2)
        });
      });

      const dailySummary = [];
      dailyGroups.forEach(rows => {
        dailySummary.push({
          localDate: rows[0].localDate,
          exerciseId: rows[0].exerciseId,
          exerciseName: rows[0].exerciseName,
          sets: rows.length,
          workingReps: rows.reduce((sum, row) => sum + asInt(row.workingReps), 0),
          maxWeightPerCableKg: roundOrNull(maxOrNull(rows.map(row => asNumber(row.weightPerCableKg))), 2),
          maxTotalLoadKg: roundOrNull(maxOrNull(rows.map(row => asNumber(row.totalLoadKg))), 2),
          totalVolumeKg: roundOrNull(rows.reduce((sum, row) => sum + asNumber(row.totalVolumeKg), 0), 2)
        });
      });

      const samplesBySession = new Map();
      rawSamples.forEach(raw => {
        if (!sessionById.has(raw.sessionId)) return;
        if (!samplesBySession.has(raw.sessionId)) samplesBySession.set(raw.sessionId, []);
        samplesBySession.get(raw.sessionId).push(raw);
      });

      const rawSessionById = new Map(rawSessions.map(raw => [raw.id, raw]));
      const repTraces = [];
      sessions.forEach(session => {
        const rawSession = rawSessionById.get(session.id);
        if (!rawSession) return;
        repTraces.push(...segmentWorkingReps(
          rawSession,
          samplesBySession.get(session.id) || [],
          setNumberBySession.get(session.id)
        ));
      });

      const repLoadsByWorkout = new Map();
      repTraces.forEach(trace => {
        const session = sessionById.get(trace.sessionId);
        if (!session) return;
        const workoutId = session.routineSessionId || session.id;
        const key = `${workoutId}::${trace.exerciseId}`;
        if (!repLoadsByWorkout.has(key)) repLoadsByWorkout.set(key, []);
        repLoadsByWorkout.get(key).push(trace);
      });

      workoutSummary.forEach(row => {
        const traces = repLoadsByWorkout.get(`${row.workoutId}::${row.exerciseId}`) || [];
        const echoTraces = traces.filter(trace => isEchoMode(trace.mode));
        const nonEchoTraces = traces.filter(trace => !isEchoMode(trace.mode));
        row.echoRepCount = echoTraces.length;
        row.nonEchoRepCount = nonEchoTraces.length;
        row.largestRepMedianWeightPerCableKg = roundOrNull(maxOrNull(traces.map(trace => trace.medianPerCableLoadKg)), 2);
        row.largestRepAverageWeightPerCableKg = roundOrNull(maxOrNull(traces.map(trace => trace.avgPerCableLoadKg)), 2);
        row.largestRepMedianTotalLoadKg = roundOrNull(maxOrNull(traces.map(trace => trace.medianTotalLoadKg)), 2);
        row.largestRepAverageTotalLoadKg = roundOrNull(maxOrNull(traces.map(trace => trace.avgTotalLoadKg)), 2);
        row.largestEchoRepMedianWeightPerCableKg = roundOrNull(maxOrNull(echoTraces.map(trace => trace.medianPerCableLoadKg)), 2);
        row.largestEchoRepAverageWeightPerCableKg = roundOrNull(maxOrNull(echoTraces.map(trace => trace.avgPerCableLoadKg)), 2);
        row.largestEchoRepPeakWeightPerCableKg = roundOrNull(maxOrNull(echoTraces.map(trace => trace.peakPerCableLoadKg)), 2);
        row.largestEchoRepMedianTotalLoadKg = roundOrNull(maxOrNull(echoTraces.map(trace => trace.medianTotalLoadKg)), 2);
        row.largestEchoRepAverageTotalLoadKg = roundOrNull(maxOrNull(echoTraces.map(trace => trace.avgTotalLoadKg)), 2);
        row.largestEchoRepPeakTotalLoadKg = roundOrNull(maxOrNull(echoTraces.map(trace => trace.peakTotalLoadKg)), 2);
        row.largestNonEchoRepMedianWeightPerCableKg = roundOrNull(maxOrNull(nonEchoTraces.map(trace => trace.medianPerCableLoadKg)), 2);
        row.largestNonEchoRepAverageWeightPerCableKg = roundOrNull(maxOrNull(nonEchoTraces.map(trace => trace.avgPerCableLoadKg)), 2);
        row.largestNonEchoRepMedianTotalLoadKg = roundOrNull(maxOrNull(nonEchoTraces.map(trace => trace.medianTotalLoadKg)), 2);
        row.largestNonEchoRepAverageTotalLoadKg = roundOrNull(maxOrNull(nonEchoTraces.map(trace => trace.avgTotalLoadKg)), 2);
      });

      const exerciseGroups = new Map();
      sessions.forEach(session => {
        if (!exerciseGroups.has(session.exerciseId)) exerciseGroups.set(session.exerciseId, []);
        exerciseGroups.get(session.exerciseId).push(session);
      });

      const exercises = [];
      exerciseGroups.forEach((rows, exerciseId) => {
        const workoutIds = new Set(rows.map(row => row.routineSessionId || row.id));
        exercises.push({
          id: exerciseId,
          name: rows[0].exerciseName,
          sets: rows.length,
          workouts: workoutIds.size,
          workingReps: rows.reduce((sum, row) => sum + asInt(row.workingReps), 0),
          totalVolumeKg: roundOrNull(rows.reduce((sum, row) => sum + asNumber(row.totalVolumeKg), 0), 2),
          maxWeightPerCableKg: roundOrNull(maxOrNull(rows.map(row => asNumber(row.weightPerCableKg))), 2),
          maxTotalLoadKg: roundOrNull(maxOrNull(rows.map(row => asNumber(row.totalLoadKg))), 2),
          firstDate: rows.map(row => row.localDate).sort()[0],
          lastDate: rows.map(row => row.localDate).sort().slice(-1)[0]
        });
      });

      sessions.sort((a, b) => a.timestamp - b.timestamp);
      completedSets.sort((a, b) => String(a.completedLocalDateTime).localeCompare(String(b.completedLocalDateTime)));
      workoutSummary.sort((a, b) => a.timestamp - b.timestamp);
      dailySummary.sort((a, b) => String(a.localDate + a.exerciseName).localeCompare(String(b.localDate + b.exerciseName)));
      exercises.sort((a, b) => b.sets - a.sets || a.name.localeCompare(b.name));

      return {
        metadata: {
          exportedAt: rawData.exportedAt || "",
          version: rawData.version,
          appVersion: rawData.appVersion || "",
          validSessions: sessions.length,
          completedSets: completedSets.length,
          repTraces: repTraces.length,
          timezone: "Australia/Perth"
        },
        exercises,
        sessions,
        sets: completedSets,
        workoutExerciseSummary: workoutSummary,
        dailyExerciseSummary: dailySummary,
        repTraces
      };
    }

    function fmtNumber(value, digits = 0) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
      return Number(value).toLocaleString(undefined, {
        maximumFractionDigits: digits,
        minimumFractionDigits: 0
      });
    }

    function displayLoadValue(value) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return null;
      return state.loadUnit === "lbs" ? Number(value) * KG_TO_LB : Number(value);
    }

    function loadUnitText() {
      return state.loadUnit === "lbs" ? "lbs" : "kg";
    }

    function fmtLoad(value, digits = 1) {
      const text = fmtNumber(displayLoadValue(value), digits);
      return text === "-" ? "-" : `${text} ${loadUnitText()}`;
    }

    function tooltipLoad(value, digits = 1) {
      return fmtLoad(value, digits);
    }

    function hideChartTooltip() {
      chartTooltip.style.display = "none";
    }

    function showChartTooltip(event, item) {
      chartTooltip.textContent = item.lines.join("\n");
      chartTooltip.style.display = "block";
      const margin = 14;
      const rect = chartTooltip.getBoundingClientRect();
      let left = event.clientX + margin;
      let top = event.clientY + margin;
      if (left + rect.width > window.innerWidth - 8) left = event.clientX - rect.width - margin;
      if (top + rect.height > window.innerHeight - 8) top = event.clientY - rect.height - margin;
      chartTooltip.style.left = `${Math.max(8, left)}px`;
      chartTooltip.style.top = `${Math.max(8, top)}px`;
    }

    function setChartHitAreas(canvas, items) {
      chartHitAreas.set(canvas.id, items);
    }

    function nearestChartItem(canvas, mouseX, mouseY) {
      const items = chartHitAreas.get(canvas.id) || [];
      let best = null;
      items.forEach(item => {
        let score;
        if (item.bounds) {
          const inside = mouseX >= item.bounds.left && mouseX <= item.bounds.right && mouseY >= item.bounds.top && mouseY <= item.bounds.bottom;
          const dx = mouseX - item.x;
          const dy = mouseY - item.y;
          score = inside ? 0 : Math.hypot(dx, dy);
        } else {
          const dx = mouseX - item.x;
          const dy = mouseY - item.y;
          score = Math.hypot(dx, dy);
        }
        if (!best || score < best.score) best = { item, score };
      });
      return best && best.score <= 36 ? best.item : null;
    }

    function setupChartTooltips() {
      ["repsChart", "loadChart", "volumeChart", "velocityChart", "muscleBalanceChart"].forEach(id => {
        const canvas = document.getElementById(id);
        canvas.addEventListener("mousemove", event => {
          const rect = canvas.getBoundingClientRect();
          const item = nearestChartItem(canvas, event.clientX - rect.left, event.clientY - rect.top);
          if (item) showChartTooltip(event, item);
          else hideChartTooltip();
        });
        canvas.addEventListener("mouseleave", hideChartTooltip);
      });
    }

    function loadField() {
      return state.loadBasis === "total" ? "maxTotalLoadKg" : "maxWeightPerCableKg";
    }

    function bestEchoRepMedianLoadField() {
      return state.loadBasis === "total" ? "largestEchoRepMedianTotalLoadKg" : "largestEchoRepMedianWeightPerCableKg";
    }

    function bestEchoRepAverageLoadField() {
      return state.loadBasis === "total" ? "largestEchoRepAverageTotalLoadKg" : "largestEchoRepAverageWeightPerCableKg";
    }

    function bestEchoRepPeakLoadField() {
      return state.loadBasis === "total" ? "largestEchoRepPeakTotalLoadKg" : "largestEchoRepPeakWeightPerCableKg";
    }

    function bestNonEchoRepMedianLoadField() {
      return state.loadBasis === "total" ? "largestNonEchoRepMedianTotalLoadKg" : "largestNonEchoRepMedianWeightPerCableKg";
    }

    function bestNonEchoRepAverageLoadField() {
      return state.loadBasis === "total" ? "largestNonEchoRepAverageTotalLoadKg" : "largestNonEchoRepAverageWeightPerCableKg";
    }

    function traceLoadField() {
      return state.loadBasis === "total" ? "totalLoadKg" : "perCableLoadKg";
    }

    function loadUnitLabel() {
      return state.loadBasis === "total" ? `total ${loadUnitText()}` : `${loadUnitText()} per cable`;
    }

    function isAllExercises() {
      return state.exerciseId === ALL_EXERCISES_ID;
    }

    function isValidExerciseId(exerciseId) {
      return exerciseId === ALL_EXERCISES_ID || DATA.exercises.some(exercise => exercise.id === exerciseId);
    }

    function chartRowLabel(row) {
      const base = row.label || row.localDate;
      if (row.isAggregateTotal || row.isDailyTotal) return base;
      return isAllExercises() ? `${base} - ${row.exerciseName || "Unknown"}` : base;
    }

    function populateExerciseSelect() {
      const previous = state.exerciseId;
      exerciseSelect.innerHTML = [
        `<option value="${ALL_EXERCISES_ID}">All Exercises</option>`,
        ...DATA.exercises.map(exercise =>
        `<option value="${exercise.id}">${exercise.name}</option>`
        )
      ].join("");
      state.exerciseId = isValidExerciseId(previous)
        ? previous
        : ALL_EXERCISES_ID;
      exerciseSelect.value = state.exerciseId;
    }

    async function loadBackupFile(file) {
      if (!file) return;
      const sourceMeta = document.getElementById("sourceMeta");
      try {
        sourceMeta.textContent = `Loading ${file.name}...`;
        const text = await file.text();
        const raw = JSON.parse(text);
        const nextData = buildDashboardData(raw);
        DATA = nextData;
        state.exerciseId = ALL_EXERCISES_ID;
        state.dimmedOverlayDates = new Set();
        populateExerciseSelect();
        document.querySelector("label[for='backupFileInput'].file-button").textContent = file.name;
        saveDashboardCache();
        render();
      } catch (error) {
        sourceMeta.textContent = `Could not load ${file.name}: ${error.message}`;
      } finally {
        backupFileInput.value = "";
      }
    }

    function setupControls() {
      populateExerciseSelect();
      historyWindow.value = state.historyWindow;
      loadUnitToggle.checked = state.loadUnit === "lbs";
      loadToggle.checked = state.loadBasis === "total";
      loadEchoMedianToggle.checked = state.showLoadEchoMedian;
      loadEchoAverageToggle.checked = state.showLoadEchoAverage;
      repOverlayMode.value = state.repOverlayMode;
      if (cachedDashboard?.data) {
        document.querySelector("label[for='backupFileInput'].file-button").textContent = "Cached data";
      }
      exerciseSelect.addEventListener("change", () => {
        state.exerciseId = exerciseSelect.value;
        saveDashboardCache();
        render();
      });
      backupFileInput.addEventListener("change", event => {
        loadBackupFile(event.target.files?.[0]);
      });
      historyWindow.addEventListener("change", () => {
        state.historyWindow = historyWindow.value;
        saveDashboardCache();
        render();
      });
      loadUnitToggle.addEventListener("change", () => {
        state.loadUnit = loadUnitToggle.checked ? "lbs" : "kg";
        saveDashboardCache();
        render();
      });
      loadToggle.addEventListener("change", () => {
        state.loadBasis = loadToggle.checked ? "total" : "perCable";
        saveDashboardCache();
        render();
      });
      loadEchoMedianToggle.addEventListener("change", () => {
        state.showLoadEchoMedian = loadEchoMedianToggle.checked;
        saveDashboardCache();
        render();
      });
      loadEchoAverageToggle.addEventListener("change", () => {
        state.showLoadEchoAverage = loadEchoAverageToggle.checked;
        saveDashboardCache();
        render();
      });
      repOverlayMode.addEventListener("change", () => {
        state.repOverlayMode = repOverlayMode.value;
        saveDashboardCache();
        render();
      });
    }

    function allRowsForExercise() {
      return DATA.workoutExerciseSummary
        .filter(row => isAllExercises() || row.exerciseId === state.exerciseId)
        .sort((a, b) => a.timestamp - b.timestamp || String(a.exerciseName).localeCompare(String(b.exerciseName)));
    }

    function rowsForExercise() {
      const rows = allRowsForExercise();
      if (state.historyWindow === "all") return rows;
      return rows.slice(-Number(state.historyWindow));
    }

    function rowsForRepsChart() {
      if (!isAllExercises()) return rowsForExercise();
      const workoutRows = new Map();
      allRowsForExercise().forEach(row => {
        const key = row.workoutId || `${row.localDate || ""}::${row.routineName || ""}::${row.timestamp || ""}`;
        if (!workoutRows.has(key)) {
          workoutRows.set(key, {
            workoutId: row.workoutId,
            timestamp: row.timestamp,
            localDate: row.localDate,
            label: row.label || row.localDate,
            routineName: row.routineName,
            exerciseName: "All Exercises",
            workingReps: 0,
            isAggregateTotal: true
          });
        }
        const workout = workoutRows.get(key);
        workout.timestamp = Math.min(workout.timestamp, row.timestamp);
        workout.workingReps += Number(row.workingReps || 0);
      });
      const rows = [...workoutRows.values()].sort((a, b) => a.timestamp - b.timestamp);
      if (state.historyWindow === "all") return rows;
      return rows.slice(-Number(state.historyWindow));
    }

    function rowsForVolumeChart() {
      if (!isAllExercises()) return rowsForExercise();
      const dailyRows = new Map();
      allRowsForExercise().forEach(row => {
        const key = row.localDate || row.label || "";
        if (!dailyRows.has(key)) {
          dailyRows.set(key, {
            timestamp: row.timestamp,
            localDate: row.localDate,
            label: row.localDate,
            exerciseName: "All Exercises",
            totalVolumeKg: 0,
            estimatedOneRepMaxKg: null,
            isDailyTotal: true
          });
        }
        const daily = dailyRows.get(key);
        const estimatedOneRepMax = Number(row.estimatedOneRepMaxKg || 0);
        daily.timestamp = Math.min(daily.timestamp, row.timestamp);
        daily.totalVolumeKg += Number(row.totalVolumeKg || 0);
        daily.estimatedOneRepMaxKg = Math.max(Number(daily.estimatedOneRepMaxKg || 0), estimatedOneRepMax) || null;
      });
      const rows = [...dailyRows.values()]
        .map(row => ({
          ...row,
          totalVolumeKg: Math.round(row.totalVolumeKg * 100) / 100,
          estimatedOneRepMaxKg: row.estimatedOneRepMaxKg === null ? null : Math.round(row.estimatedOneRepMaxKg * 100) / 100
        }))
        .sort((a, b) => a.timestamp - b.timestamp);
      if (state.historyWindow === "all") return rows;
      return rows.slice(-Number(state.historyWindow));
    }

    function exerciseInfo() {
      if (isAllExercises()) return { id: ALL_EXERCISES_ID, name: "All Exercises" };
      return DATA.exercises.find(exercise => exercise.id === state.exerciseId) || DATA.exercises[0];
    }

    function renderKpis(rows) {
      const setCount = rows.reduce((sum, row) => sum + Number(row.sets || 0), 0);
      const totalReps = rows.reduce((sum, row) => sum + Number(row.workingReps || 0), 0);
      const totalVolume = rows.reduce((sum, row) => sum + Number(row.totalVolumeKg || 0), 0);
      const maxLoad = Math.max(...rows.map(row => Number(row[loadField()] || 0)), 0);
      const bestE1rm = Math.max(...rows.map(row => Number(row.estimatedOneRepMaxKg || 0)), 0);
      const latest = rows.length ? rows[rows.length - 1].localDate : "-";
      const kpis = [
        ["Workouts", rows.length, `${setCount} completed sets`],
        ["Working Reps", totalReps, "warmup and zero-rep sessions excluded"],
        ["Best Load", fmtLoad(maxLoad, 1), loadUnitLabel()],
        ["Total Volume", fmtLoad(totalVolume, 0), "from Vitruvian volume"],
        ["Best Est. 1RM", fmtLoad(bestE1rm, 1), `latest ${latest}`]
      ];
      document.getElementById("kpis").innerHTML = kpis.map(([title, value, note]) => `
        <div class="kpi">
          <div class="kpi-title">${title}</div>
          <div class="kpi-value">${value}</div>
          <div class="kpi-note">${note}</div>
        </div>
      `).join("");
    }

    function canvasSetup(canvas) {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, Math.floor(rect.width * dpr));
      canvas.height = Math.max(1, Math.floor(rect.height * dpr));
      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, rect.width, rect.height);
      return { ctx, width: rect.width, height: rect.height };
    }

    function drawEmpty(canvas, text) {
      const { ctx, width, height } = canvasSetup(canvas);
      setChartHitAreas(canvas, []);
      ctx.fillStyle = "#667085";
      ctx.font = "14px Segoe UI, Arial";
      ctx.textAlign = "center";
      ctx.fillText(text, width / 2, height / 2);
    }

    function scaleLinear(domainMin, domainMax, rangeMin, rangeMax) {
      const span = domainMax - domainMin || 1;
      return value => rangeMin + ((value - domainMin) / span) * (rangeMax - rangeMin);
    }

    function drawAxes(ctx, box, yTicks, yLabel) {
      ctx.strokeStyle = "#d9dee7";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(box.left, box.top);
      ctx.lineTo(box.left, box.bottom);
      ctx.lineTo(box.right, box.bottom);
      ctx.stroke();

      ctx.fillStyle = "#667085";
      ctx.font = "12px Segoe UI, Arial";
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      yTicks.forEach(tick => {
        ctx.strokeStyle = "#eef1f5";
        ctx.beginPath();
        ctx.moveTo(box.left, tick.y);
        ctx.lineTo(box.right, tick.y);
        ctx.stroke();
        ctx.fillText(tick.label, box.left - 8, tick.y);
      });

      ctx.save();
      ctx.translate(15, (box.top + box.bottom) / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = "center";
      ctx.fillText(yLabel, 0, 0);
      ctx.restore();
    }

    function drawLineSeries(canvas, rows, config) {
      if (!rows.length) {
        drawEmpty(canvas, "No data for this exercise.");
        return;
      }
      const { ctx, width, height } = canvasSetup(canvas);
      const box = { left: 58, right: width - 18, top: 20, bottom: height - 44 };
      const numericSeriesValue = (series, row) => {
        const value = series.value(row);
        if (value === null || value === undefined || value === "") return null;
        const number = Number(value);
        return Number.isFinite(number) ? number : null;
      };
      const yValues = rows
        .flatMap(row => config.series.map(series => numericSeriesValue(series, row)))
        .filter(value => Number.isFinite(value));
      if (!yValues.length) {
        drawEmpty(canvas, "No plottable values for this exercise.");
        return;
      }
      const yMax = Math.max(...yValues, 1);
      const yMin = config.zeroBase ? 0 : Math.min(...yValues, 0);
      const yPad = (yMax - yMin || 1) * 0.12;
      const yScale = scaleLinear(yMin, yMax + yPad, box.bottom, box.top);
      const xScale = scaleLinear(0, Math.max(rows.length - 1, 1), box.left, box.right);
      const hitItems = [];

      const yTicks = [];
      for (let idx = 0; idx <= 4; idx++) {
        const value = yMin + ((yMax + yPad - yMin) * idx / 4);
        yTicks.push({ y: yScale(value), label: fmtNumber(value, config.digits || 0) });
      }
      drawAxes(ctx, box, yTicks, config.yLabel);

      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.fillStyle = "#667085";
      ctx.font = "11px Segoe UI, Arial";
      const labelStep = Math.max(1, Math.ceil(rows.length / 6));
      rows.forEach((row, idx) => {
        if (idx % labelStep === 0 || idx === rows.length - 1) {
          ctx.fillText(row.localDate.slice(5), xScale(idx), box.bottom + 12);
        }
      });

      config.series.forEach((series, seriesIndex) => {
        ctx.strokeStyle = series.color || palette[seriesIndex % palette.length];
        ctx.fillStyle = series.color || palette[seriesIndex % palette.length];
        ctx.lineWidth = series.width || 2;
        ctx.beginPath();
        let hasStarted = false;
        rows.forEach((row, idx) => {
          const rawValue = numericSeriesValue(series, row);
          if (rawValue === null) {
            hasStarted = false;
            return;
          }
          const x = xScale(idx);
          const y = yScale(rawValue);
          if (!hasStarted) {
            ctx.moveTo(x, y);
            hasStarted = true;
          }
          else ctx.lineTo(x, y);
        });
        ctx.stroke();

        rows.forEach((row, idx) => {
          const rawValue = numericSeriesValue(series, row);
          if (rawValue === null) return;
          const x = xScale(idx);
          const y = yScale(rawValue);
          ctx.beginPath();
          ctx.arc(x, y, 3, 0, Math.PI * 2);
          ctx.fill();
          const renderedValue = series.formatValue ? series.formatValue(rawValue, row) : fmtNumber(rawValue, config.digits || 0);
          hitItems.push({
            x,
            y,
            lines: [
              chartRowLabel(row),
              series.name,
              String(renderedValue)
            ]
          });
        });
      });

      if (config.legend) {
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.font = "12px Segoe UI, Arial";
        let x = box.left;
        config.series.forEach((series, idx) => {
          const color = series.color || palette[idx % palette.length];
          ctx.fillStyle = color;
          ctx.fillRect(x, height - 20, 12, 3);
          ctx.fillStyle = "#667085";
          ctx.fillText(series.name, x + 18, height - 18);
          x += ctx.measureText(series.name).width + 64;
        });
      }
      setChartHitAreas(canvas, hitItems);
    }

    function drawBarLine(canvas, rows) {
      if (!rows.length) {
        drawEmpty(canvas, "No data for this exercise.");
        return;
      }
      const { ctx, width, height } = canvasSetup(canvas);
      const box = { left: 62, right: width - 22, top: 18, bottom: height - 44 };
      const maxVolume = Math.max(...rows.map(row => Number(displayLoadValue(row.totalVolumeKg) || 0)), 1);
      const maxE1rm = Math.max(...rows.map(row => Number(displayLoadValue(row.estimatedOneRepMaxKg) || 0)), 1);
      const yScale = scaleLinear(0, maxVolume * 1.16, box.bottom, box.top);
      const xScale = scaleLinear(0, Math.max(rows.length - 1, 1), box.left, box.right);
      const yTicks = [0, 0.25, 0.5, 0.75, 1].map(part => {
        const value = maxVolume * 1.16 * part;
        return { y: yScale(value), label: fmtNumber(value, 0) };
      });
      drawAxes(ctx, box, yTicks, `volume ${loadUnitText()}`);

      const barWidth = Math.max(4, Math.min(28, (box.right - box.left) / rows.length * 0.48));
      const hitItems = [];
      ctx.fillStyle = "rgba(37, 99, 235, 0.28)";
      rows.forEach((row, idx) => {
        const x = xScale(idx) - barWidth / 2;
        const y = yScale(Number(displayLoadValue(row.totalVolumeKg) || 0));
        ctx.fillRect(x, y, barWidth, box.bottom - y);
        hitItems.push({
          x: x + barWidth / 2,
          y: y + (box.bottom - y) / 2,
          bounds: { left: x, right: x + barWidth, top: y, bottom: box.bottom },
          lines: [
            chartRowLabel(row),
            "Volume",
            fmtLoad(row.totalVolumeKg, 0)
          ]
        });
      });

      const eScale = scaleLinear(0, maxE1rm * 1.12, box.bottom, box.top);
      ctx.strokeStyle = "#dc2626";
      ctx.fillStyle = "#dc2626";
      ctx.lineWidth = 2;
      ctx.beginPath();
      rows.forEach((row, idx) => {
        const x = xScale(idx);
        const y = eScale(Number(displayLoadValue(row.estimatedOneRepMaxKg) || 0));
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
      rows.forEach((row, idx) => {
        const x = xScale(idx);
        const y = eScale(Number(displayLoadValue(row.estimatedOneRepMaxKg) || 0));
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, Math.PI * 2);
        ctx.fill();
        hitItems.push({
          x,
          y,
          lines: [
            chartRowLabel(row),
            "Est. 1RM",
            fmtLoad(row.estimatedOneRepMaxKg, 1)
          ]
        });
      });

      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.fillStyle = "#667085";
      ctx.font = "11px Segoe UI, Arial";
      const labelStep = Math.max(1, Math.ceil(rows.length / 6));
      rows.forEach((row, idx) => {
        if (idx % labelStep === 0 || idx === rows.length - 1) ctx.fillText(row.localDate.slice(5), xScale(idx), box.bottom + 12);
      });

      ctx.textAlign = "left";
      ctx.fillStyle = "#2563eb";
      ctx.fillRect(box.left, height - 20, 12, 8);
      ctx.fillStyle = "#667085";
      ctx.fillText("Volume", box.left + 18, height - 22);
      ctx.fillStyle = "#dc2626";
      ctx.fillRect(box.left + 92, height - 17, 12, 3);
      ctx.fillStyle = "#667085";
      ctx.fillText("Est. 1RM", box.left + 110, height - 22);
      setChartHitAreas(canvas, hitItems);
    }

    function muscleGroupsForExercise(exerciseName) {
      const key = normalizeExerciseName(exerciseName);
      if (exerciseMuscleLookup.has(key)) return exerciseMuscleLookup.get(key);
      const singularKey = singularExerciseKey(exerciseName);
      return exerciseMuscleLookup.get(singularKey) || [];
    }

    function muscleGroupWeightsForExercise(exerciseName) {
      const key = normalizeExerciseName(exerciseName);
      const singularKey = singularExerciseKey(exerciseName);
      const override = EXERCISE_MUSCLE_WEIGHT_OVERRIDES[key] || EXERCISE_MUSCLE_WEIGHT_OVERRIDES[singularKey];
      if (override) return override;
      const groups = muscleGroupsForExercise(exerciseName);
      return groups.map(group => ({ group, weight: 1 / groups.length }));
    }

    function muscleBalanceData(rows) {
      const groupTotals = new Map(MUSCLE_GROUP_ORDER.map(group => [group, 0]));
      const groupExercises = new Map(MUSCLE_GROUP_ORDER.map(group => [group, new Set()]));
      const unmatched = new Set();

      rows.forEach(row => {
        const groupWeights = muscleGroupWeightsForExercise(row.exerciseName);
        if (!groupWeights.length) {
          unmatched.add(row.exerciseName || "Unknown");
          return;
        }
        const rawValue = Number(row.totalVolumeKg || 0);
        if (!rawValue) return;
        groupWeights.forEach(({ group, weight }) => {
          if (!groupTotals.has(group)) {
            groupTotals.set(group, 0);
            groupExercises.set(group, new Set());
          }
          groupTotals.set(group, groupTotals.get(group) + rawValue * weight);
          groupExercises.get(group).add(row.exerciseName || "Unknown");
        });
      });

      const orderedGroups = [
        ...MUSCLE_GROUP_ORDER,
        ...[...groupTotals.keys()].filter(group => !MUSCLE_GROUP_ORDER.includes(group)).sort()
      ];
      const total = [...groupTotals.values()].reduce((sum, value) => sum + value, 0);
      const maxValue = Math.max(...groupTotals.values(), 0);
      const items = orderedGroups.map((group, idx) => {
        const value = groupTotals.get(group) || 0;
        return {
          group,
          label: titleCaseGroup(group),
          value,
          relative: maxValue ? value / maxValue : 0,
          share: total ? value / total : 0,
          exercises: [...(groupExercises.get(group) || [])].sort(),
          color: palette[idx % palette.length]
        };
      });
      return { items, total, maxValue, unmatched: [...unmatched].sort() };
    }

    function drawMuscleRadar(canvas, items) {
      if (!items.some(item => item.value > 0)) {
        drawEmpty(canvas, "No muscle-group matches found for this selection.");
        return;
      }
      const { ctx, width, height } = canvasSetup(canvas);
      const radius = Math.max(70, Math.min(width - 190, height - 72) / 2);
      const centerX = width / 2;
      const centerY = height / 2 + 8;
      const labelRadius = radius + 38;
      const angleStep = (Math.PI * 2) / items.length;
      const hitItems = [];

      ctx.strokeStyle = "#d9dee7";
      ctx.lineWidth = 1;
      for (let ring = 1; ring <= 5; ring++) {
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius * ring / 5, 0, Math.PI * 2);
        ctx.stroke();
      }

      items.forEach((item, idx) => {
        const angle = idx * angleStep - Math.PI / 2;
        const outerX = centerX + radius * Math.cos(angle);
        const outerY = centerY + radius * Math.sin(angle);
        ctx.beginPath();
        ctx.moveTo(centerX, centerY);
        ctx.lineTo(outerX, outerY);
        ctx.stroke();
      });

      ctx.beginPath();
      items.forEach((item, idx) => {
        const angle = idx * angleStep - Math.PI / 2;
        const distance = radius * item.relative;
        const x = centerX + distance * Math.cos(angle);
        const y = centerY + distance * Math.sin(angle);
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.closePath();
      ctx.fillStyle = "rgba(37, 99, 235, 0.18)";
      ctx.fill();
      ctx.strokeStyle = "#2563eb";
      ctx.lineWidth = 2.5;
      ctx.stroke();

      ctx.font = "12px Segoe UI, Arial";
      items.forEach((item, idx) => {
        const angle = idx * angleStep - Math.PI / 2;
        const pointX = centerX + radius * item.relative * Math.cos(angle);
        const pointY = centerY + radius * item.relative * Math.sin(angle);
        ctx.fillStyle = item.value > 0 ? "#2563eb" : "#c9d1dc";
        ctx.beginPath();
        ctx.arc(pointX, pointY, 4, 0, Math.PI * 2);
        ctx.fill();

        const labelX = centerX + labelRadius * Math.cos(angle);
        const labelY = centerY + labelRadius * Math.sin(angle);
        ctx.fillStyle = "#475467";
        ctx.textAlign = Math.cos(angle) > 0.25 ? "left" : (Math.cos(angle) < -0.25 ? "right" : "center");
        ctx.textBaseline = Math.sin(angle) > 0.4 ? "top" : (Math.sin(angle) < -0.4 ? "bottom" : "middle");
        ctx.fillText(item.label, labelX, labelY);

        hitItems.push({
          x: pointX,
          y: pointY,
          lines: [
            item.label,
            `Relative focus: ${fmtNumber(item.relative * 100, 0)}%`,
            `Share: ${fmtNumber(item.share * 100, 0)}%`,
            `Volume: ${fmtLoad(item.value, 0)}`,
            `Exercises: ${item.exercises.length ? item.exercises.join(", ") : "-"}`
          ]
        });
      });
      setChartHitAreas(canvas, hitItems);
    }

    function renderMuscleBalance(rows) {
      const balance = muscleBalanceData(rows);
      drawMuscleRadar(document.getElementById("muscleBalanceChart"), balance.items);
      const matchedCount = new Set(balance.items.flatMap(item => item.exercises)).size;
      const unmatchedText = balance.unmatched.length
        ? ` ${balance.unmatched.length} unmatched: ${balance.unmatched.slice(0, 4).join(", ")}${balance.unmatched.length > 4 ? ", ..." : ""}.`
        : "";
      document.getElementById("muscleBalanceNote").textContent =
        `Relative focus from Project Phoenix muscle groups; ${matchedCount} matched exercise${matchedCount === 1 ? "" : "s"}.${unmatchedText}`;
      document.getElementById("muscleBalanceList").innerHTML = balance.items.map(item => `
        <div class="muscle-balance-item">
          <strong>${item.label}</strong>
          <span>${fmtNumber(item.share * 100, 0)}% share · ${fmtLoad(item.value, 0)}</span>
        </div>
      `).join("");
    }

    function traceWorkoutId(trace) {
      return trace.routineSessionId || trace.sessionId;
    }

    function overlayDateKey(date) {
      return `${state.exerciseId}::${date}`;
    }

    function isOverlayDateDimmed(date) {
      return state.dimmedOverlayDates.has(overlayDateKey(date));
    }

    function overlayRankField() {
      if (state.repOverlayMode === "maxAverage") {
        return state.loadBasis === "total" ? "avgTotalLoadKg" : "avgPerCableLoadKg";
      }
      if (state.repOverlayMode === "maxMedian") {
        return state.loadBasis === "total" ? "medianTotalLoadKg" : "medianPerCableLoadKg";
      }
      return null;
    }

    function overlayModeLabel() {
      if (state.repOverlayMode === "maxAverage") return "Max average";
      if (state.repOverlayMode === "maxMedian") return "Max median";
      return "All";
    }

    function traceModeKey(trace) {
      return isEchoMode(trace.mode) ? "Echo" : "Non-echo";
    }

    function traceMetricValue(trace, field) {
      const value = Number(trace[field]);
      return Number.isFinite(value) ? value : -Infinity;
    }

    function selectRepOverlayTraces(traces) {
      const field = overlayRankField();
      if (!field) return traces.map(trace => ({ ...trace, overlayModeLabel: "All" }));
      const bestByDateAndMode = new Map();
      traces.forEach(trace => {
        const key = `${trace.date}::${traceModeKey(trace)}`;
        const current = bestByDateAndMode.get(key);
        if (!current || traceMetricValue(trace, field) > traceMetricValue(current, field)) {
          bestByDateAndMode.set(key, trace);
        }
      });
      return [...bestByDateAndMode.values()]
        .map(trace => ({ ...trace, overlayModeLabel: traceModeKey(trace) }))
        .sort((a, b) => new Date(a.dateTime) - new Date(b.dateTime) || traceModeKey(a).localeCompare(traceModeKey(b)));
    }

    function renderRepOverlay(activeRows) {
      const canvas = document.getElementById("repOverlay");
      const activeWorkoutExerciseKeys = new Set(activeRows.map(row => `${row.workoutId}::${row.exerciseId}`));
      const sourceTraces = DATA.repTraces
        .filter(trace =>
          (isAllExercises() || trace.exerciseId === state.exerciseId) &&
          activeWorkoutExerciseKeys.has(`${traceWorkoutId(trace)}::${trace.exerciseId}`)
        )
        .sort((a, b) => new Date(a.dateTime) - new Date(b.dateTime));
      const traces = selectRepOverlayTraces(sourceTraces);

      document.getElementById("repOverlayNote").textContent =
        state.repOverlayMode === "all"
          ? "Load over time for all working reps. Colours identify the workout date."
          : `${overlayModeLabel()} shows the top echo rep and top non-echo rep for each selected workout date. Echo traces are dashed.`;
      if (!traces.length) {
        drawEmpty(canvas, "No working rep traces were detected for this selection.");
        document.getElementById("overlayLegend").innerHTML = "";
        return;
      }

      const dates = [...new Set(traces.map(trace => trace.date))];
      const { ctx, width, height } = canvasSetup(canvas);
      const box = { left: 62, right: width - 22, top: 20, bottom: height - 46 };
      const loadKey = traceLoadField();
      const maxTime = Math.max(...traces.flatMap(trace => trace.timeSec), 1);
      const yValues = traces.flatMap(trace => trace[loadKey].map(value => displayLoadValue(value)));
      const yMin = Math.max(0, Math.min(...yValues) - 2);
      const yMax = Math.max(...yValues, 1) + 2;
      const xScale = scaleLinear(0, maxTime, box.left, box.right);
      const yScale = scaleLinear(yMin, yMax, box.bottom, box.top);
      const yTicks = [0, 0.25, 0.5, 0.75, 1].map(part => {
        const value = yMin + (yMax - yMin) * part;
        return { y: yScale(value), label: fmtNumber(value, 1) };
      });
      drawAxes(ctx, box, yTicks, loadUnitLabel());

      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.fillStyle = "#667085";
      ctx.font = "12px Segoe UI, Arial";
      ctx.fillText("seconds from rep start", (box.left + box.right) / 2, height - 22);

      const dateColors = new Map(dates.map((date, idx) => [date, palette[idx % palette.length]]));
      traces.forEach(trace => {
        const color = isOverlayDateDimmed(trace.date) ? "#9ca3af" : (dateColors.get(trace.date) || "#2563eb");
        ctx.strokeStyle = `${color}${isOverlayDateDimmed(trace.date) ? "88" : "aa"}`;
        ctx.lineWidth = state.repOverlayMode === "all"
          ? (isOverlayDateDimmed(trace.date) ? 1.1 : 1.4)
          : (isOverlayDateDimmed(trace.date) ? 1.4 : 2.3);
        ctx.setLineDash(state.repOverlayMode !== "all" && isEchoMode(trace.mode) ? [6, 4] : []);
        ctx.beginPath();
        trace.timeSec.forEach((time, idx) => {
          const x = xScale(time);
          const y = yScale(displayLoadValue(trace[loadKey][idx]));
          if (idx === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.stroke();
      });
      ctx.setLineDash([]);

      const allDatesDimmed = dates.every(date => isOverlayDateDimmed(date));
      const toggleAllLabel = allDatesDimmed ? "All on" : "All off";
      document.getElementById("overlayLegend").innerHTML = `
        <button type="button" class="legend-item legend-toggle-all" data-action="toggle-all" aria-pressed="${!allDatesDimmed}" title="Toggle all dates">
          <span class="swatch" style="background:${allDatesDimmed ? "#9ca3af" : "#17202a"}"></span>${toggleAllLabel}
        </button>
      ` + dates.map(date => `
        <button type="button" class="legend-item ${isOverlayDateDimmed(date) ? "is-muted" : ""}" data-date="${date}" aria-pressed="${!isOverlayDateDimmed(date)}" title="Toggle ${date}">
          <span class="swatch" style="background:${isOverlayDateDimmed(date) ? "#9ca3af" : dateColors.get(date)}"></span>${date}
        </button>
      `).join("");
      document.querySelectorAll("#overlayLegend .legend-item").forEach(button => {
        button.addEventListener("click", () => {
          if (button.dataset.action === "toggle-all") {
            if (allDatesDimmed) {
              dates.forEach(date => state.dimmedOverlayDates.delete(overlayDateKey(date)));
            } else {
              dates.forEach(date => state.dimmedOverlayDates.add(overlayDateKey(date)));
            }
            saveDashboardCache();
            renderRepOverlay(activeRows);
            return;
          }
          const date = button.dataset.date;
          const key = overlayDateKey(date);
          if (state.dimmedOverlayDates.has(key)) state.dimmedOverlayDates.delete(key);
          else state.dimmedOverlayDates.add(key);
          saveDashboardCache();
          renderRepOverlay(activeRows);
        });
      });
    }

    function renderHistoryTable(rows) {
      document.getElementById("historyHeader").innerHTML = `
        <th>Date</th>
        ${isAllExercises() ? "<th>Exercise</th>" : ""}
        <th>Routine</th>
        <th>Sets</th>
        <th>Reps</th>
        <th>Max Load</th>
        <th class="group-separator">Non-Echo Reps</th>
        <th>Volume</th>
        <th>Est. 1RM</th>
        <th>Avg MCV</th>
        <th class="group-separator">Echo Reps</th>
        <th>Echo Median</th>
        <th>Echo Avg</th>
        <th>Echo Peak</th>
      `;
      const newest = [...rows].sort((a, b) => b.timestamp - a.timestamp);
      document.getElementById("historyRows").innerHTML = newest.map(row => `
        <tr>
          <td>${row.label}</td>
          ${isAllExercises() ? `<td>${row.exerciseName || "-"}</td>` : ""}
          <td>${row.routineName || "-"}</td>
          <td>${row.sets}</td>
          <td>${row.workingReps}</td>
          <td>${fmtLoad(row[loadField()], 1)}</td>
          <td class="group-separator">${row.nonEchoRepCount || 0}</td>
          <td>${fmtLoad(row.totalVolumeKg, 0)}</td>
          <td>${fmtLoad(row.estimatedOneRepMaxKg, 1)}</td>
          <td>${fmtNumber(row.avgMcvMmS, 1)} mm/s</td>
          <td class="group-separator">${row.echoRepCount || 0}</td>
          <td>${fmtLoad(row[bestEchoRepMedianLoadField()], 1)}</td>
          <td>${fmtLoad(row[bestEchoRepAverageLoadField()], 1)}</td>
          <td>${fmtLoad(row[bestEchoRepPeakLoadField()], 1)}</td>
        </tr>
      `).join("");
    }

    function render() {
      const rows = rowsForExercise();
      const repsRows = rowsForRepsChart();
      const volumeRows = rowsForVolumeChart();
      const source = DATA.metadata;
      document.getElementById("sourceMeta").textContent =
        `${source.validSessions} valid sessions, ${source.completedSets} completed sets, ${source.repTraces} working rep traces. Exported ${source.exportedAt}.`;
      const windowLabel = state.historyWindow === "all" ? "all workouts" : `last ${state.historyWindow} workouts`;
      document.getElementById("repsChartNote").textContent = isAllExercises()
        ? "Sum of all working reps across all exercises in each workout."
        : "Grouped by routine workout and selected exercise.";
      document.getElementById("volumeChartNote").textContent = isAllExercises()
        ? "Daily total volume across all exercises; red line shows the best estimated 1RM that day."
        : "Volume bars with estimated 1RM trend from completed working sets.";
      document.getElementById("loadProgressionNote").textContent =
        `Showing ${loadUnitLabel()} for ${windowLabel}, with incomplete and zero-rep sessions excluded.`;
      renderKpis(rows);
      drawLineSeries(document.getElementById("repsChart"), repsRows, {
        yLabel: "working reps",
        zeroBase: true,
        digits: 0,
        series: [{
          name: isAllExercises() ? "Total working reps" : "Working reps",
          color: "#2563eb",
          value: row => row.workingReps,
          formatValue: value => `${fmtNumber(value, 0)} reps`
        }]
      });
      const loadSeries = [{
        name: "Max load",
        color: "#0f766e",
        value: row => displayLoadValue(row[loadField()]),
        formatValue: value => `${fmtNumber(value, 1)} ${loadUnitText()}`
      }];
      if (state.showLoadEchoMedian) {
        loadSeries.push({
          name: "Echo median",
          color: "#7c3aed",
          value: row => row.echoRepCount ? displayLoadValue(row[bestEchoRepMedianLoadField()]) : null,
          formatValue: value => `${fmtNumber(value, 1)} ${loadUnitText()}`
        });
      }
      if (state.showLoadEchoAverage) {
        loadSeries.push({
          name: "Echo average",
          color: "#b45309",
          value: row => row.echoRepCount ? displayLoadValue(row[bestEchoRepAverageLoadField()]) : null,
          formatValue: value => `${fmtNumber(value, 1)} ${loadUnitText()}`
        });
      }
      drawLineSeries(document.getElementById("loadChart"), rows, {
        yLabel: loadUnitLabel(),
        zeroBase: true,
        digits: 1,
        legend: loadSeries.length > 1,
        series: loadSeries
      });
      drawBarLine(document.getElementById("volumeChart"), volumeRows);
      drawLineSeries(document.getElementById("velocityChart"), rows.filter(row => row.avgMcvMmS !== null), {
        yLabel: "mm/s",
        zeroBase: false,
        digits: 1,
        series: [{
          name: "Avg MCV",
          color: "#b45309",
          value: row => row.avgMcvMmS,
          formatValue: value => `${fmtNumber(value, 1)} mm/s`
        }]
      });
      renderMuscleBalance(rows);
      renderRepOverlay(rows);
      renderHistoryTable(rows);
    }

    window.addEventListener("resize", () => render());
    setupControls();
    setupChartTooltips();
    render();
    saveDashboardCache();
  </script>
</body>
</html>
"""
    return template.replace("__DATA__", data_json).replace("__MUSCLE_MAP__", muscle_map_json)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Vitruvian interactive training dashboard.")
    parser.add_argument(
        "backup",
        nargs="?",
        default=r"C:\Users\f677716\Downloads\vitruvian_backup_20260521_062930.txt",
        help="Path to a Vitruvian JSON backup .txt file.",
    )
    parser.add_argument(
        "--out",
        default="vitruvian_dashboard.html",
        help="Output HTML dashboard path.",
    )
    parser.add_argument(
        "--tables-dir",
        default="vitruvian_tables",
        help="Directory for cleaned CSV table exports.",
    )
    parser.add_argument(
        "--exercise-map",
        default="project_phoenix_exercise_muscle_map.json",
        help="Project Phoenix exercise-to-muscle-group mapping JSON.",
    )
    args = parser.parse_args()

    backup_path = Path(args.backup)
    output_path = Path(args.out)
    tables_dir = Path(args.tables_dir)
    exercise_map_path = Path(args.exercise_map)

    with backup_path.open("r", encoding="utf-8") as handle:
        raw_data = json.load(handle)
    if exercise_map_path.exists():
        with exercise_map_path.open("r", encoding="utf-8-sig") as handle:
            exercise_muscle_map = json.load(handle)
    else:
        exercise_muscle_map = []

    dashboard = build_tables(raw_data)

    write_csv(tables_dir / "sessions.csv", dashboard["sessions"], list(dashboard["sessions"][0].keys()))
    write_csv(tables_dir / "sets.csv", dashboard["sets"], list(dashboard["sets"][0].keys()))
    write_csv(
        tables_dir / "workout_exercise_summary.csv",
        dashboard["workoutExerciseSummary"],
        list(dashboard["workoutExerciseSummary"][0].keys()),
    )
    write_csv(
        tables_dir / "daily_exercise_summary.csv",
        dashboard["dailyExerciseSummary"],
        list(dashboard["dailyExerciseSummary"][0].keys()),
    )
    write_csv(
        tables_dir / "rep_traces_summary.csv",
        [
            {
                key: value
                for key, value in trace.items()
                if key not in {"timeSec", "perCableLoadKg", "totalLoadKg", "position", "velocity"}
            }
            for trace in dashboard["repTraces"]
        ],
        [
            key
            for key in dashboard["repTraces"][0].keys()
            if key not in {"timeSec", "perCableLoadKg", "totalLoadKg", "position", "velocity"}
        ],
    )

    data_json = json.dumps(dashboard, separators=(",", ":"), ensure_ascii=False)
    muscle_map_json = json.dumps(exercise_muscle_map, separators=(",", ":"), ensure_ascii=False)
    output_path.write_text(dashboard_html(data_json, muscle_map_json), encoding="utf-8")

    print(f"Wrote {output_path.resolve()}")
    print(f"Wrote CSV tables to {tables_dir.resolve()}")
    print(
        f"Included {dashboard['metadata']['validSessions']} sessions, "
        f"{dashboard['metadata']['completedSets']} completed sets, "
        f"{dashboard['metadata']['repTraces']} rep traces."
    )


if __name__ == "__main__":
    main()
