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
GRAVITY_M_S2 = 9.80665


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


def add_cable_segment_energy(
    totals: dict[str, float],
    previous: dict[str, Any],
    current: dict[str, Any],
    load_key: str,
    position_key: str,
    fallback_position_key: str | None = None,
) -> None:
    previous_position = as_float(
        previous.get(position_key),
        as_float(previous.get(fallback_position_key)) if fallback_position_key else 0.0,
    )
    current_position = as_float(
        current.get(position_key),
        as_float(current.get(fallback_position_key)) if fallback_position_key else previous_position,
    )
    delta_cm = current_position - previous_position
    if abs(delta_cm) <= 0.05:
        return
    previous_load = as_float(previous.get(load_key))
    current_load = as_float(current.get(load_key), previous_load)
    average_load_kg = max(0.0, (previous_load + current_load) / 2)
    energy_j = average_load_kg * GRAVITY_M_S2 * abs(delta_cm) / 100
    if delta_cm > 0:
        totals["concentric"] += energy_j
    else:
        totals["eccentric"] += energy_j


def segment_energy_joules(samples: list[dict[str, Any]], cable_count: int) -> dict[str, float]:
    totals = {"concentric": 0.0, "eccentric": 0.0}
    for idx in range(1, len(samples)):
        previous = samples[idx - 1]
        current = samples[idx]
        add_cable_segment_energy(totals, previous, current, "load", "position")
        if cable_count >= 2:
            add_cable_segment_energy(totals, previous, current, "loadB", "positionB", "position")
    totals["total"] = totals["concentric"] + totals["eccentric"]
    return totals


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

        energy = segment_energy_joules(segment, cable_count)
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
        positions_a: list[float] = []
        positions_b: list[float] = []
        positions: list[float] = []
        velocities: list[float] = []

        for sample_idx in selected:
            row = segment[sample_idx]
            load_a = as_float(row.get("load"))
            load_b = as_float(row.get("loadB"), load_a)
            position_a = as_float(row.get("position"))
            position_b = as_float(row.get("positionB"), position_a)
            per_cable_load = (load_a + load_b) / 2 if cable_count >= 2 else load_a
            total_load = load_a + load_b if cable_count >= 2 else load_a
            times.append(round((as_int(row.get("timestamp")) - start_ms) / 1000, 3))
            per_cable_loads.append(round(per_cable_load, 2))
            total_loads.append(round(total_load, 2))
            positions_a.append(round(position_a, 2))
            positions_b.append(round(position_b, 2))
            positions.append(round((position_a + position_b) / 2, 2))
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
                "concentricEnergyJ": rounded(energy["concentric"], 1),
                "eccentricEnergyJ": rounded(energy["eccentric"], 1),
                "totalEnergyJ": rounded(energy["total"], 1),
                "timeSec": times,
                "perCableLoadKg": per_cable_loads,
                "totalLoadKg": total_loads,
                "positionA": positions_a,
                "positionB": positions_b,
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
        row["largestEchoRepEnergyJ"] = rounded(
            max((as_float(trace.get("totalEnergyJ")) for trace in echo_traces), default=math.nan),
            1,
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
        row["largestNonEchoRepEnergyJ"] = rounded(
            max((as_float(trace.get("totalEnergyJ")) for trace in non_echo_traces), default=math.nan),
            1,
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


def dashboard_html(data_json: str, muscle_map_json: str, refined_muscle_map_json: str) -> str:
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
      --grid-line: #eef1f5;
      --canvas: #ffffff;
      --control-bg: #ffffff;
      --control-ink: #17202a;
      --table-group-bg: #f8fafc;
      --hover-bg: #f8fafc;
      --legend-muted-bg: #f1f3f6;
      --legend-muted-ink: #8a94a3;
      --separator-line: #b8c0cc;
      --inactive-point: #c9d1dc;
      --body-region-idle: #15191f;
      --body-outline: #6b7280;
      --blue: #2563eb;
      --teal: #0f766e;
      --red: #dc2626;
      --gold: #b45309;
      --green: #059669;
      --violet: #7c3aed;
      --shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
      --rep-filter-sticky-top: 72px;
    }

    body.theme-dark {
      color-scheme: dark;
      --bg: #0b1120;
      --panel: #111827;
      --ink: #e5e7eb;
      --muted: #a6b0bf;
      --line: #2b3648;
      --grid-line: #1f2937;
      --canvas: #0f172a;
      --control-bg: #0f172a;
      --control-ink: #e5e7eb;
      --table-group-bg: #111827;
      --hover-bg: #1f2937;
      --legend-muted-bg: #172033;
      --legend-muted-ink: #818da0;
      --separator-line: #4b5563;
      --inactive-point: #4b5563;
      --body-region-idle: #15191f;
      --body-outline: #7b8494;
      --shadow: 0 8px 24px rgba(0, 0, 0, 0.32);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      font-family: Inter, Segoe UI, Arial, sans-serif;
      background: var(--bg);
      color: var(--ink);
      transition: background 160ms ease, color 160ms ease;
    }

    header {
      background: #111827;
      color: #fff;
      padding: 22px 24px 18px;
      position: sticky;
      top: 0;
      z-index: 50;
      box-shadow: 0 8px 22px rgba(15, 23, 42, 0.18);
    }

    .top-ribbon {
      transition: padding 160ms ease;
      max-height: 82vh;
      overflow-y: auto;
    }

    .top-ribbon.is-collapsed {
      padding-top: 12px;
      padding-bottom: 12px;
    }

    .header-inner {
      max-width: 1400px;
      margin: 0 auto;
      display: grid;
      grid-template-columns: minmax(260px, 1fr) minmax(560px, auto);
      gap: 20px;
      align-items: center;
    }

    .header-copy {
      min-width: 0;
    }

    .header-title-row {
      display: flex;
      gap: 12px;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 8px;
    }

    .top-ribbon.is-collapsed .header-title-row {
      margin-bottom: 0;
    }

    h1 {
      margin: 0;
      font-size: 26px;
      line-height: 1.15;
      letter-spacing: 0;
    }

    .ribbon-toggle {
      width: 32px;
      height: 32px;
      display: grid;
      place-items: center;
      flex: 0 0 auto;
      border: 1px solid rgba(203, 213, 225, 0.45);
      border-radius: 6px;
      background: rgba(255, 255, 255, 0.08);
      color: #fff;
      font-size: 20px;
      line-height: 1;
      cursor: pointer;
    }

    .ribbon-toggle:hover {
      background: rgba(255, 255, 255, 0.16);
    }

    .top-ribbon.is-collapsed .subtle,
    .top-ribbon.is-collapsed .controls {
      display: none;
    }

    .subtle {
      margin: 0;
      color: #cbd5e1;
      font-size: 13px;
    }

    .controls {
      display: grid;
      grid-template-columns: minmax(170px, 210px) minmax(430px, 1fr);
      gap: 10px;
      align-items: stretch;
      justify-content: end;
    }

    .control {
      display: grid;
      gap: 8px;
      min-width: 0;
    }

    .banner-control {
      padding: 10px;
      border: 1px solid rgba(148, 163, 184, 0.32);
      border-radius: 8px;
      background: rgba(15, 23, 42, 0.74);
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
    }

    .file-control {
      align-content: stretch;
    }

    .switch-control {
      min-width: 0;
    }

    .switch-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(140px, 1fr));
      gap: 8px;
    }

    .switch-card {
      display: grid;
      gap: 6px;
      min-width: 0;
      padding: 8px 10px;
      border: 1px solid #334155;
      border-radius: 7px;
      background: #0f172a;
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
      min-height: 30px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      color: #fff;
      font-size: 13px;
      white-space: nowrap;
    }

    .switch-card .switch-row span {
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
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

    .switch:focus-within .slider {
      outline: 2px solid rgba(45, 212, 191, 0.65);
      outline-offset: 2px;
    }

    .file-input {
      display: none;
    }

    .file-button {
      width: 100%;
      min-height: 62px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      border: 1px solid #3b82f6;
      border-radius: 6px;
      background: #1d4ed8;
      color: #fff;
      padding: 10px 12px;
      font-size: 14px;
      font-weight: 800;
      text-transform: none;
      letter-spacing: 0;
      cursor: pointer;
      text-align: center;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .file-button:hover {
      background: #2563eb;
      border-color: #60a5fa;
    }

    .file-button:focus-visible {
      outline: 2px solid rgba(96, 165, 250, 0.85);
      outline-offset: 2px;
    }

    main {
      max-width: 1400px;
      margin: 0 auto;
      padding: 18px 24px 96px;
    }

    .dashboard-tab {
      display: block;
    }

    .dashboard-tab.is-hidden {
      display: none;
    }

    .kpis {
      display: grid;
      grid-template-columns: repeat(6, minmax(120px, 1fr));
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

    .sticky-filter-panel {
      position: sticky;
      top: var(--rep-filter-sticky-top);
      z-index: 35;
      align-self: start;
      width: 100%;
    }

    .summary-filter-panel {
      margin-bottom: 16px;
    }

    .filter-panel:not(.is-collapsed) {
      padding: 12px 14px 14px;
    }

    .filter-panel:not(.is-collapsed) .panel-head {
      align-items: center;
      margin-bottom: 12px;
      padding-bottom: 10px;
      border-bottom: 1px solid var(--grid-line);
    }

    .filter-panel.is-collapsed .filter-body {
      display: none;
    }

    .filter-panel.is-collapsed {
      width: fit-content;
      min-width: 0;
      padding: 0;
      margin-left: auto;
      justify-self: end;
    }

    .filter-panel.is-collapsed .panel-head {
      display: block;
      margin-bottom: 0;
    }

    .filter-panel.is-collapsed .panel-head > div {
      display: none;
    }

    .filter-toggle {
      width: 32px;
      height: 32px;
      display: grid;
      place-items: center;
      flex: 0 0 auto;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--control-bg);
      color: var(--ink);
      font-size: 20px;
      line-height: 1;
      cursor: pointer;
    }

    .filter-panel.is-collapsed .filter-toggle {
      width: auto;
      min-width: 104px;
      height: 36px;
      padding: 0 12px;
      border: 0;
      border-radius: 6px;
      font-size: 13px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.02em;
    }

    .filter-toggle:hover {
      border-color: var(--separator-line);
      background: var(--hover-bg);
    }

    .filter-body {
      display: grid;
      gap: 12px;
    }

    .filter-controls {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 12px;
      align-items: end;
    }

    .summary-filter-panel .filter-controls {
      grid-template-columns: minmax(240px, 1.4fr) minmax(180px, 0.8fr);
      max-width: 760px;
    }

    .filter-body .legend {
      margin-top: 2px;
      padding-top: 10px;
      border-top: 1px solid var(--grid-line);
      max-height: 104px;
    }

    .filter-body .legend:empty {
      display: none;
    }

    .check-control {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      min-height: 30px;
      padding: 5px 8px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--control-bg);
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

    .filter-controls .mini-control {
      display: grid;
      gap: 5px;
      min-width: 0;
      white-space: normal;
    }

    .mini-control label {
      color: var(--muted);
    }

    .filter-controls .mini-control label {
      line-height: 1;
    }

    .mini-control select {
      min-width: 110px;
      min-height: 32px;
      background: var(--control-bg);
      color: var(--control-ink);
      border-color: var(--line);
      padding: 6px 28px 6px 8px;
    }

    .filter-controls .mini-control select {
      width: 100%;
      min-width: 0;
      min-height: 36px;
    }

    canvas {
      display: block;
      width: 100%;
      height: 320px;
      border-radius: 6px;
      background: var(--canvas);
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

    #repOverlay, #positionOverlay { height: 430px; }
    #muscleBalanceChart { height: 360px; }

    .phase-chart-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }

    .phase-chart-title {
      margin: 0 0 8px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.02em;
    }

    .phase-chart-grid canvas {
      height: 330px;
    }

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
      background: var(--control-bg);
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

    .muscle-breakdown-layout {
      display: grid;
      grid-template-columns: minmax(320px, 1.3fr) minmax(240px, 0.7fr);
      gap: 16px;
      align-items: start;
    }

    .body-map {
      min-width: 0;
      padding: 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #050505;
    }

    .body-map svg {
      display: block;
      width: 100%;
      height: auto;
      min-height: 420px;
    }

    .body-chart-pair {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      align-items: stretch;
    }

    .body-chart-panel {
      display: grid;
      grid-template-rows: auto 1fr;
      gap: 8px;
      min-width: 0;
    }

    .body-view-label {
      color: var(--muted);
      font-size: 13px;
      font-weight: 800;
      letter-spacing: 0.02em;
      text-align: center;
      text-transform: uppercase;
    }

    .body-chart-host {
      min-height: 410px;
      height: min(64vh, 460px);
    }

    .body-map .body-chart-container {
      min-height: 100%;
      padding: 0 !important;
    }

    .body-map .body-chart-svg {
      max-width: 100% !important;
      max-height: 100% !important;
      height: 100% !important;
      filter: none !important;
    }

    .body-label {
      fill: var(--muted);
      font-size: 13px;
      font-weight: 800;
      letter-spacing: 0.02em;
      text-transform: uppercase;
    }

    .body-outline {
      fill: none;
      stroke: var(--body-outline);
      stroke-width: 2.2;
      stroke-linecap: round;
      stroke-linejoin: round;
      opacity: 0.8;
    }

    .body-detail {
      fill: none;
      stroke: var(--body-outline);
      stroke-width: 1;
      stroke-linecap: round;
      stroke-linejoin: round;
      opacity: 0.34;
      pointer-events: none;
    }

    .body-region {
      fill: var(--body-region-idle);
      stroke: #050505;
      stroke-width: 1.2;
      cursor: default;
      transition: fill 160ms ease, opacity 160ms ease, stroke 160ms ease;
    }

    .body-region:hover {
      stroke: var(--ink);
      stroke-width: 1.8;
    }

    .muscle-breakdown-side {
      display: grid;
      gap: 12px;
    }

    .muscle-scale {
      display: grid;
      gap: 6px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }

    .muscle-scale-bar {
      height: 12px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: linear-gradient(90deg, var(--body-region-idle), #fde68a, #f97316, #ef4444);
    }

    .muscle-scale-labels {
      display: flex;
      justify-content: space-between;
    }

    .muscle-breakdown-list {
      display: grid;
      gap: 8px;
      max-height: min(72vh, 760px);
      overflow: auto;
      padding-right: 2px;
    }

    .muscle-breakdown-row {
      display: grid;
      grid-template-columns: 12px minmax(0, 1fr) auto 18px;
      gap: 8px;
      align-items: center;
      padding: 8px 9px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--control-bg);
      color: var(--muted);
      font-size: 12px;
      transition: background 160ms ease, border-color 160ms ease, box-shadow 160ms ease;
      scroll-margin: 16px;
      cursor: pointer;
    }

    .muscle-breakdown-row:hover {
      background: var(--hover-bg);
    }

    .muscle-breakdown-row:focus-visible {
      outline: 2px solid rgba(37, 99, 235, 0.55);
      outline-offset: 2px;
    }

    .muscle-breakdown-row.is-selected {
      border-color: #2563eb;
      background: rgba(37, 99, 235, 0.12);
      box-shadow: inset 0 0 0 1px #2563eb;
      color: var(--ink);
    }

    .muscle-breakdown-row strong {
      color: var(--ink);
      font-size: 12px;
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .muscle-breakdown-row.is-selected strong {
      color: #1d4ed8;
    }

    .theme-dark .muscle-breakdown-row.is-selected strong {
      color: #93c5fd;
    }

    .muscle-breakdown-row .swatch {
      width: 12px;
      height: 12px;
    }

    .muscle-breakdown-toggle {
      opacity: 0;
      color: #2563eb;
      font-size: 14px;
      font-weight: 900;
      text-align: center;
    }

    .muscle-breakdown-row.is-selected .muscle-breakdown-toggle {
      opacity: 1;
    }

    .muscle-contribution-insert {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel);
      margin: -4px 0 8px;
      padding: 10px;
      min-height: 150px;
      max-height: 280px;
      overflow: auto;
      overscroll-behavior: contain;
    }

    .muscle-contribution-inner {
      min-width: 520px;
      overflow-x: auto;
    }

    .muscle-contribution-table {
      width: 100%;
      border-collapse: collapse;
      color: var(--muted);
      font-size: 11px;
    }

    .muscle-contribution-table th,
    .muscle-contribution-table td {
      padding: 5px 6px;
      border-bottom: 1px solid var(--grid-line);
      text-align: left;
      white-space: nowrap;
    }

    .muscle-contribution-table th:nth-child(2),
    .muscle-contribution-table td:nth-child(2) {
      white-space: normal;
      min-width: 120px;
    }

    .muscle-contribution-table th {
      color: var(--muted);
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 0.02em;
    }

    .muscle-contribution-table td:nth-child(3),
    .muscle-contribution-table td:nth-child(4),
    .muscle-contribution-table td:nth-child(5),
    .muscle-contribution-table td:nth-child(6),
    .muscle-contribution-table td:nth-child(7) {
      text-align: right;
    }

    .muscle-contribution-table tr:last-child td {
      border-bottom: 0;
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
      background: var(--control-bg);
      color: var(--muted);
      cursor: pointer;
      font: inherit;
      appearance: none;
    }

    .legend-item:hover {
      border-color: var(--separator-line);
    }

    .legend-item:focus-visible {
      outline: 2px solid var(--blue);
      outline-offset: 2px;
    }

    .legend-item.is-muted {
      background: var(--legend-muted-bg);
      color: var(--legend-muted-ink);
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
      border-left: 2px solid var(--separator-line);
    }

    .history-group-row th {
      padding-top: 7px;
      padding-bottom: 6px;
      text-align: center;
      background: var(--table-group-bg);
      color: var(--ink);
      font-weight: 800;
    }

    .history-group-row .group-spacer {
      background: var(--panel);
      border-bottom-color: transparent;
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

    .bottom-ribbon {
      position: fixed;
      left: 0;
      right: 0;
      bottom: 0;
      z-index: 60;
      background: #111827;
      border-top: 1px solid rgba(203, 213, 225, 0.22);
      box-shadow: 0 -8px 22px rgba(15, 23, 42, 0.18);
      padding: 10px 16px;
    }

    .bottom-ribbon-inner {
      max-width: 1400px;
      margin: 0 auto;
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
    }

    .tab-button {
      min-width: 0;
      min-height: 42px;
      border: 1px solid rgba(203, 213, 225, 0.35);
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.08);
      color: #cbd5e1;
      font-size: 14px;
      font-weight: 800;
      cursor: pointer;
      padding: 0 10px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .tab-button:hover {
      background: rgba(255, 255, 255, 0.14);
    }

    .tab-button.is-active {
      background: #ffffff;
      color: #111827;
      border-color: #ffffff;
    }

    @media (max-width: 1000px) {
      .header-inner { grid-template-columns: 1fr; }
      .controls {
        grid-template-columns: minmax(170px, 220px) minmax(0, 1fr);
        justify-content: stretch;
      }
      .kpis { grid-template-columns: repeat(2, minmax(120px, 1fr)); }
      .grid { grid-template-columns: 1fr; }
      .phase-chart-grid { grid-template-columns: 1fr; }
      .muscle-breakdown-layout { grid-template-columns: 1fr; }
      .muscle-balance-list { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }

    @media (max-width: 760px) {
      .controls { grid-template-columns: 1fr; }
      .file-button { min-height: 44px; }
      .body-chart-pair { grid-template-columns: 1fr; }
      .body-chart-host { height: 390px; }
    }

    @media (max-width: 560px) {
      header { padding: 18px 16px; }
      main { padding: 14px 12px 92px; }
      .bottom-ribbon { padding: 8px 10px; }
      .bottom-ribbon-inner { gap: 6px; }
      .tab-button {
        min-height: 38px;
        padding: 0 4px;
        font-size: 11px;
      }
      h1 { font-size: 22px; }
      .kpis { grid-template-columns: 1fr; }
      .panel-head { display: grid; }
      .filter-panel:not(.is-collapsed) .panel-head { display: flex; }
      .filter-controls,
      .summary-filter-panel .filter-controls { grid-template-columns: 1fr; }
      canvas, #repOverlay, #positionOverlay { height: 300px; }
      .controls,
      .switch-grid { grid-template-columns: 1fr; }
      .control { width: 100%; }
      .switch-row { white-space: normal; }
      .muscle-balance-list { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header id="topRibbon" class="top-ribbon">
    <div class="header-inner">
      <div class="header-copy">
        <div class="header-title-row">
          <h1>Vitruvian Training Dashboard</h1>
          <button class="ribbon-toggle" id="ribbonToggle" type="button" aria-expanded="true" aria-controls="dashboardRibbonControls" title="Collapse top ribbon">-</button>
        </div>
        <p class="subtle" id="sourceMeta"></p>
      </div>
      <div class="controls" id="dashboardRibbonControls">
        <div class="control banner-control file-control">
          <label for="backupFileInput">Upload File</label>
          <label class="file-button" for="backupFileInput" title="Load JSON/TXT">Load JSON/TXT</label>
          <input class="file-input" id="backupFileInput" type="file" accept=".json,.txt,application/json,text/plain">
        </div>
        <div class="control banner-control switch-control">
          <div class="switch-grid">
            <div class="switch-card">
              <label for="loadUnitToggle">Load Units</label>
              <div class="switch-row">
                <span>kg</span>
                <label class="switch">
                  <input id="loadUnitToggle" type="checkbox">
                  <span class="slider"></span>
                </label>
                <span>lbs</span>
              </div>
            </div>
            <div class="switch-card">
              <label for="loadToggle">Load basis</label>
              <div class="switch-row">
                <span>Per cable</span>
                <label class="switch">
                  <input id="loadToggle" type="checkbox" checked>
                  <span class="slider"></span>
                </label>
                <span>Total</span>
              </div>
            </div>
            <div class="switch-card">
              <label for="themeToggle">Theme</label>
              <div class="switch-row">
                <span>Day</span>
                <label class="switch">
                  <input id="themeToggle" type="checkbox">
                  <span class="slider"></span>
                </label>
                <span>Night</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </header>

  <main>
    <section class="dashboard-tab" id="summaryTabPanel">
      <article class="panel summary-filter-panel filter-panel sticky-filter-panel" id="summaryFilterPanel">
        <div class="panel-head">
          <div>
            <h2>Filters</h2>
          </div>
          <button class="filter-toggle" id="summaryFilterToggle" type="button" aria-expanded="true" aria-controls="summaryFilterBody" title="Collapse filters">-</button>
        </div>
        <div class="filter-body" id="summaryFilterBody">
          <div class="filter-controls">
            <div class="mini-control">
              <label for="summaryExerciseSelect">Exercise</label>
              <select id="summaryExerciseSelect"></select>
            </div>
            <div class="mini-control">
              <label for="summaryHistoryWindow">History</label>
              <select id="summaryHistoryWindow">
                <option value="5">Last 5 workouts</option>
                <option value="10" selected>Last 10 workouts</option>
                <option value="20">Last 20 workouts</option>
                <option value="all">All workouts</option>
              </select>
            </div>
          </div>
        </div>
      </article>
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
            <h2>Workout History Table</h2>
            <p class="panel-note">Recent grouped workouts for the selected exercise.</p>
          </div>
        </div>
        <div class="table-wrap">
          <table>
            <thead>
              <tr class="history-group-row" id="historyGroupHeader"></tr>
              <tr id="historyHeader"></tr>
            </thead>
            <tbody id="historyRows"></tbody>
          </table>
        </div>
      </article>
      </section>
    </section>

    <section class="dashboard-tab is-hidden" id="repAnalyticsTabPanel">
      <section class="grid">
      <article class="panel wide filter-panel sticky-filter-panel" id="repAnalyticsFilterPanel">
        <div class="panel-head">
          <div>
            <h2>Filters</h2>
          </div>
          <button class="filter-toggle" id="repAnalyticsFilterToggle" type="button" aria-expanded="true" aria-controls="repAnalyticsFilterBody" title="Collapse filters">-</button>
        </div>
        <div class="filter-body" id="repAnalyticsFilterBody">
          <div class="filter-controls">
            <div class="mini-control">
              <label for="repExerciseSelect">Exercise</label>
              <select id="repExerciseSelect"></select>
            </div>
            <div class="mini-control">
              <label for="repHistoryWindow">History</label>
              <select id="repHistoryWindow">
                <option value="5">Last 5 workouts</option>
                <option value="10" selected>Last 10 workouts</option>
                <option value="20">Last 20 workouts</option>
                <option value="all">All workouts</option>
              </select>
            </div>
            <div class="mini-control">
              <label for="repOverlayMode">Overlay</label>
              <select id="repOverlayMode">
                <option value="all">All</option>
                <option value="maxAverage">Max average</option>
                <option value="maxMedian">Max median</option>
                <option value="maxEnergy">Max energy</option>
              </select>
            </div>
            <div class="mini-control">
              <label for="repTypeFilter">Rep type</label>
              <select id="repTypeFilter">
                <option value="all">All</option>
                <option value="nonEcho">Non echo</option>
                <option value="echo">Echo</option>
              </select>
            </div>
          </div>
          <div class="legend" id="overlayLegend"></div>
        </div>
      </article>

      <article class="panel wide">
        <div class="panel-head">
          <div>
            <h2>Working Rep Load Overlay</h2>
            <p class="panel-note" id="repOverlayNote">Load over time for working reps only. Colours identify the workout date.</p>
          </div>
        </div>
        <canvas id="repOverlay"></canvas>
      </article>

      <article class="panel wide">
        <div class="panel-head">
          <div>
            <h2>Working Rep Position Overlay</h2>
            <p class="panel-note" id="positionOverlayNote">Left and right cable position over time for the selected working reps.</p>
          </div>
          <div class="chart-options" aria-label="Working rep position overlay options">
            <div class="mini-control">
              <label for="positionOverlayBasis">Position</label>
              <select id="positionOverlayBasis">
                <option value="average">Average</option>
                <option value="left">Left cable</option>
                <option value="right">Right cable</option>
              </select>
            </div>
          </div>
        </div>
        <canvas id="positionOverlay"></canvas>
      </article>

      <article class="panel wide">
        <div class="panel-head">
          <div>
            <h2>Load vs Position</h2>
            <p class="panel-note" id="loadPositionNote">Load against cable position for the same selected working reps.</p>
          </div>
        </div>
        <div class="phase-chart-grid">
          <div>
            <h3 class="phase-chart-title">Concentric</h3>
            <canvas id="concentricLoadPositionChart"></canvas>
          </div>
          <div>
            <h3 class="phase-chart-title">Eccentric</h3>
            <canvas id="eccentricLoadPositionChart"></canvas>
          </div>
        </div>
      </article>

      <article class="panel wide">
        <div class="panel-head">
          <div>
            <h2>Energy Per Rep</h2>
            <p class="panel-note" id="repEnergyNote">Total energy for each selected working rep over workout time.</p>
          </div>
        </div>
        <canvas id="repEnergyChart"></canvas>
      </article>

      </section>
    </section>

    <section class="dashboard-tab is-hidden" id="muscleBreakdownTabPanel">
      <section class="grid">
      <article class="panel wide filter-panel sticky-filter-panel" id="muscleBreakdownFilterPanel">
        <div class="panel-head">
          <div>
            <h2>Filters</h2>
          </div>
          <button class="filter-toggle" id="muscleBreakdownFilterToggle" type="button" aria-expanded="true" aria-controls="muscleBreakdownFilterBody" title="Collapse filters">-</button>
        </div>
        <div class="filter-body" id="muscleBreakdownFilterBody">
          <div class="filter-controls">
            <div class="mini-control">
              <label for="muscleExerciseSelect">Exercise</label>
              <select id="muscleExerciseSelect"></select>
            </div>
            <div class="mini-control">
              <label for="muscleHistoryWindow">History</label>
              <select id="muscleHistoryWindow">
                <option value="5">Last 5 workouts</option>
                <option value="10" selected>Last 10 workouts</option>
                <option value="20">Last 20 workouts</option>
                <option value="all">All workouts</option>
              </select>
            </div>
          </div>
        </div>
      </article>

      <article class="panel wide">
        <div class="panel-head">
          <div>
            <h2>Muscle Breakdown</h2>
            <p class="panel-note" id="muscleBreakdownNote">Front and back body map coloured by filtered workout focus.</p>
          </div>
        </div>
        <div class="muscle-breakdown-layout">
          <div class="body-map" aria-label="Front and back muscle focus body map">
            <div class="body-chart-pair">
              <div class="body-chart-panel">
                <div class="body-view-label">Front</div>
                <div class="body-chart-host" id="bodyMusclesFront"></div>
              </div>
              <div class="body-chart-panel">
                <div class="body-view-label">Back</div>
                <div class="body-chart-host" id="bodyMusclesBack"></div>
              </div>
            </div>
          </div>
          <aside class="muscle-breakdown-side">
            <div class="muscle-scale">
              <div class="muscle-scale-bar" aria-hidden="true"></div>
              <div class="muscle-scale-labels">
                <span>No focus</span>
                <span>High focus</span>
              </div>
            </div>
            <div class="muscle-breakdown-list" id="muscleBreakdownList"></div>
          </aside>
        </div>
      </article>
      </section>
    </section>
  </main>
  <nav class="bottom-ribbon" aria-label="Dashboard tabs">
    <div class="bottom-ribbon-inner">
      <button class="tab-button is-active" id="summaryTabButton" type="button" data-tab="summary" aria-pressed="true">Summary</button>
      <button class="tab-button" id="repAnalyticsTabButton" type="button" data-tab="repAnalytics" aria-pressed="false">Rep Analytics</button>
      <button class="tab-button" id="muscleBreakdownTabButton" type="button" data-tab="muscleBreakdown" aria-pressed="false">Muscle Breakdown</button>
    </div>
  </nav>
  <div class="chart-tooltip" id="chartTooltip"></div>

  <script src="body-muscles.umd.min.js"></script>
  <script>
    let DATA = __DATA__;
    const PROJECT_PHOENIX_EXERCISE_MUSCLE_MAP = __MUSCLE_MAP__;
    const REFINED_EXERCISE_BODY_MUSCLE_MAP = __REFINED_MUSCLE_MAP__;

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
    const GRAVITY_M_S2 = 9.80665;
    const CACHE_KEY = "vitruvianTrainingDashboardCache:v1";
    const ALL_EXERCISES_ID = "__all_exercises__";

    function loadCachedDashboard() {
      try {
        const raw = localStorage.getItem(CACHE_KEY);
        if (!raw) return null;
        const parsed = JSON.parse(raw);
        if (!parsed?.data?.workoutExerciseSummary || !Array.isArray(parsed.data.exercises)) return null;
        const hasPositionChannels = !parsed.data.repTraces?.length ||
          parsed.data.repTraces.some(trace => Array.isArray(trace.positionA) && Array.isArray(trace.positionB));
        const hasEnergyFields = !parsed.data.repTraces?.length ||
          parsed.data.repTraces.some(trace => Number.isFinite(Number(trace.totalEnergyJ)));
        const hasSummaryEnergyFields = !parsed.data.workoutExerciseSummary.length ||
          parsed.data.workoutExerciseSummary.every(row =>
            Object.prototype.hasOwnProperty.call(row, "largestEchoRepEnergyJ") &&
            Object.prototype.hasOwnProperty.call(row, "largestNonEchoRepEnergyJ")
          );
        if (!hasPositionChannels || !hasEnergyFields || !hasSummaryEnergyFields) return { settings: parsed.settings || {} };
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
      theme: cachedSettings.theme === "dark" ? "dark" : "light",
      historyWindow: ["5", "10", "20", "all"].includes(cachedSettings.historyWindow) ? cachedSettings.historyWindow : "10",
      showLoadEchoMedian: Boolean(cachedSettings.showLoadEchoMedian),
      showLoadEchoAverage: Boolean(cachedSettings.showLoadEchoAverage),
      repOverlayMode: ["all", "maxAverage", "maxMedian", "maxEnergy"].includes(cachedSettings.repOverlayMode) ? cachedSettings.repOverlayMode : "all",
      repTypeFilter: ["all", "echo", "nonEcho"].includes(cachedSettings.repTypeFilter) ? cachedSettings.repTypeFilter : "all",
      positionOverlayBasis: ["left", "right", "average"].includes(cachedSettings.positionOverlayBasis) ? cachedSettings.positionOverlayBasis : "average",
      ribbonCollapsed: Boolean(cachedSettings.ribbonCollapsed),
      activeTab: ["summary", "repAnalytics", "muscleBreakdown"].includes(cachedSettings.activeTab) ? cachedSettings.activeTab : "summary",
      filtersCollapsed: Boolean(cachedSettings.filtersCollapsed ?? cachedSettings.summaryFiltersCollapsed ?? cachedSettings.repAnalyticsFiltersCollapsed),
      selectedBodyMuscleId: "",
      expandedBodyMuscleId: "",
      dimmedOverlayDates: new Set(Array.isArray(cachedSettings.dimmedOverlayDates) ? cachedSettings.dimmedOverlayDates : [])
    };

    function dashboardCacheSettings() {
      return {
        exerciseId: state.exerciseId,
        loadBasis: state.loadBasis,
        loadUnit: state.loadUnit,
        theme: state.theme,
        historyWindow: state.historyWindow,
        showLoadEchoMedian: state.showLoadEchoMedian,
        showLoadEchoAverage: state.showLoadEchoAverage,
        repOverlayMode: state.repOverlayMode,
        repTypeFilter: state.repTypeFilter,
        positionOverlayBasis: state.positionOverlayBasis,
        ribbonCollapsed: state.ribbonCollapsed,
        activeTab: state.activeTab,
        filtersCollapsed: state.filtersCollapsed,
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

    const topRibbon = document.getElementById("topRibbon");
    const ribbonToggle = document.getElementById("ribbonToggle");
    const summaryTabPanel = document.getElementById("summaryTabPanel");
    const repAnalyticsTabPanel = document.getElementById("repAnalyticsTabPanel");
    const muscleBreakdownTabPanel = document.getElementById("muscleBreakdownTabPanel");
    const summaryFilterPanel = document.getElementById("summaryFilterPanel");
    const summaryFilterToggle = document.getElementById("summaryFilterToggle");
    const repAnalyticsFilterPanel = document.getElementById("repAnalyticsFilterPanel");
    const repAnalyticsFilterToggle = document.getElementById("repAnalyticsFilterToggle");
    const muscleBreakdownFilterPanel = document.getElementById("muscleBreakdownFilterPanel");
    const muscleBreakdownFilterToggle = document.getElementById("muscleBreakdownFilterToggle");
    const summaryTabButton = document.getElementById("summaryTabButton");
    const repAnalyticsTabButton = document.getElementById("repAnalyticsTabButton");
    const muscleBreakdownTabButton = document.getElementById("muscleBreakdownTabButton");
    const summaryExerciseSelect = document.getElementById("summaryExerciseSelect");
    const repExerciseSelect = document.getElementById("repExerciseSelect");
    const muscleExerciseSelect = document.getElementById("muscleExerciseSelect");
    const backupFileInput = document.getElementById("backupFileInput");
    const summaryHistoryWindow = document.getElementById("summaryHistoryWindow");
    const repHistoryWindow = document.getElementById("repHistoryWindow");
    const muscleHistoryWindow = document.getElementById("muscleHistoryWindow");
    const loadUnitToggle = document.getElementById("loadUnitToggle");
    const loadToggle = document.getElementById("loadToggle");
    const themeToggle = document.getElementById("themeToggle");
    const loadEchoMedianToggle = document.getElementById("loadEchoMedianToggle");
    const loadEchoAverageToggle = document.getElementById("loadEchoAverageToggle");
    const repOverlayMode = document.getElementById("repOverlayMode");
    const repTypeFilter = document.getElementById("repTypeFilter");
    const positionOverlayBasis = document.getElementById("positionOverlayBasis");
    const muscleBreakdownList = document.getElementById("muscleBreakdownList");
    const bodyMusclesFront = document.getElementById("bodyMusclesFront");
    const bodyMusclesBack = document.getElementById("bodyMusclesBack");
    const chartTooltip = document.getElementById("chartTooltip");
    const chartHitAreas = new Map();
    const exerciseMuscleLookup = buildExerciseMuscleLookup();
    const refinedExerciseBodyMuscleLookup = buildRefinedExerciseBodyMuscleLookup();
    const mainContent = document.querySelector("main");
    const exerciseSelects = [summaryExerciseSelect, repExerciseSelect, muscleExerciseSelect];
    const historyWindowSelects = [summaryHistoryWindow, repHistoryWindow, muscleHistoryWindow];
    let ribbonAutoCollapsePausedUntil = 0;
    let frontBodyChart = null;
    let backBodyChart = null;
    let currentMuscleFocusItems = [];

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

    function buildRefinedExerciseBodyMuscleLookup() {
      const byId = new Map();
      const byName = new Map();
      REFINED_EXERCISE_BODY_MUSCLE_MAP.forEach(entry => {
        const bodyMuscles = (entry.bodyMuscles || [])
          .map(item => ({ id: String(item.id || ""), weight: Number(item.weight) }))
          .filter(item => item.id && Number.isFinite(item.weight) && item.weight > 0);
        if (!bodyMuscles.length) return;
        const normalizedEntry = { ...entry, bodyMuscles };
        if (entry.id) byId.set(String(entry.id), normalizedEntry);
        [entry.name, ...(entry.aliases || [])].forEach(name => {
          const key = normalizeExerciseName(name);
          if (key && !byName.has(key)) byName.set(key, normalizedEntry);
          const singularKey = singularExerciseKey(name);
          if (singularKey && !byName.has(singularKey)) byName.set(singularKey, normalizedEntry);
        });
      });
      return { byId, byName };
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

    function addCableSegmentEnergy(totals, previous, current, loadKey, positionKey, fallbackPositionKey = null) {
      const fallbackPrevious = fallbackPositionKey ? asNumber(previous[fallbackPositionKey]) : 0;
      const fallbackCurrent = fallbackPositionKey ? asNumber(current[fallbackPositionKey], fallbackPrevious) : fallbackPrevious;
      const previousPosition = asNumber(previous[positionKey], fallbackPrevious);
      const currentPosition = asNumber(current[positionKey], fallbackCurrent);
      const deltaCm = currentPosition - previousPosition;
      if (Math.abs(deltaCm) <= 0.05) return;
      const previousLoad = asNumber(previous[loadKey]);
      const currentLoad = asNumber(current[loadKey], previousLoad);
      const averageLoadKg = Math.max(0, (previousLoad + currentLoad) / 2);
      const energyJ = averageLoadKg * GRAVITY_M_S2 * Math.abs(deltaCm) / 100;
      if (deltaCm > 0) totals.concentric += energyJ;
      else totals.eccentric += energyJ;
    }

    function segmentEnergyJoules(samples, cableCount) {
      const totals = { concentric: 0, eccentric: 0 };
      for (let idx = 1; idx < samples.length; idx += 1) {
        const previous = samples[idx - 1];
        const current = samples[idx];
        addCableSegmentEnergy(totals, previous, current, "load", "position");
        if (cableCount >= 2) addCableSegmentEnergy(totals, previous, current, "loadB", "positionB", "position");
      }
      totals.total = totals.concentric + totals.eccentric;
      return totals;
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

        const energy = segmentEnergyJoules(segment, cableCount);
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
        const positionsA = [];
        const positionsB = [];
        const positions = [];
        const velocities = [];
        selected.forEach(sampleIdx => {
          const row = segment[sampleIdx];
          const loadA = asNumber(row.load);
          const loadB = asNumber(row.loadB, loadA);
          const positionA = asNumber(row.position);
          const positionB = asNumber(row.positionB, positionA);
          times.push(roundOrNull((asInt(row.timestamp) - startMs) / 1000, 3));
          perCableLoads.push(roundOrNull(cableCount >= 2 ? (loadA + loadB) / 2 : loadA, 2));
          totalLoads.push(roundOrNull(cableCount >= 2 ? loadA + loadB : loadA, 2));
          positionsA.push(roundOrNull(positionA, 2));
          positionsB.push(roundOrNull(positionB, 2));
          positions.push(roundOrNull((positionA + positionB) / 2, 2));
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
          concentricEnergyJ: roundOrNull(energy.concentric, 1),
          eccentricEnergyJ: roundOrNull(energy.eccentric, 1),
          totalEnergyJ: roundOrNull(energy.total, 1),
          timeSec: times,
          perCableLoadKg: perCableLoads,
          totalLoadKg: totalLoads,
          positionA: positionsA,
          positionB: positionsB,
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
        row.largestEchoRepEnergyJ = roundOrNull(maxOrNull(echoTraces.map(trace => trace.totalEnergyJ)), 1);
        row.largestNonEchoRepMedianWeightPerCableKg = roundOrNull(maxOrNull(nonEchoTraces.map(trace => trace.medianPerCableLoadKg)), 2);
        row.largestNonEchoRepAverageWeightPerCableKg = roundOrNull(maxOrNull(nonEchoTraces.map(trace => trace.avgPerCableLoadKg)), 2);
        row.largestNonEchoRepMedianTotalLoadKg = roundOrNull(maxOrNull(nonEchoTraces.map(trace => trace.medianTotalLoadKg)), 2);
        row.largestNonEchoRepAverageTotalLoadKg = roundOrNull(maxOrNull(nonEchoTraces.map(trace => trace.avgTotalLoadKg)), 2);
        row.largestNonEchoRepEnergyJ = roundOrNull(maxOrNull(nonEchoTraces.map(trace => trace.totalEnergyJ)), 1);
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

    function fmtEnergy(value, digits = 1) {
      const number = Number(value);
      const text = fmtNumber(Number.isFinite(number) ? number / 1000 : null, digits);
      return text === "-" ? "-" : `${text} kJ`;
    }

    function fmtEnergyJ(value, digits = 0) {
      const text = fmtNumber(value, digits);
      return text === "-" ? "-" : `${text} J`;
    }

    function escapeHtml(value) {
      return String(value ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
    }

    function themeVar(name, fallback) {
      return getComputedStyle(document.body).getPropertyValue(name).trim() || fallback;
    }

    function mutedColor() {
      return themeVar("--muted", "#667085");
    }

    function lineColor() {
      return themeVar("--line", "#d9dee7");
    }

    function gridColor() {
      return themeVar("--grid-line", "#eef1f5");
    }

    function canvasColor() {
      return themeVar("--canvas", "#ffffff");
    }

    function inactivePointColor() {
      return themeVar("--inactive-point", "#c9d1dc");
    }

    function bodyRegionIdleColor() {
      return themeVar("--body-region-idle", "#d4dae3");
    }

    function hexToRgb(hex) {
      const clean = String(hex || "").replace("#", "").trim();
      if (clean.length !== 6) return { r: 212, g: 218, b: 227 };
      return {
        r: parseInt(clean.slice(0, 2), 16),
        g: parseInt(clean.slice(2, 4), 16),
        b: parseInt(clean.slice(4, 6), 16)
      };
    }

    function mixColor(startHex, endHex, amount) {
      const start = hexToRgb(startHex);
      const end = hexToRgb(endHex);
      const mix = Math.max(0, Math.min(1, amount));
      const channel = key => Math.round(start[key] + (end[key] - start[key]) * mix);
      return `rgb(${channel("r")}, ${channel("g")}, ${channel("b")})`;
    }

    function muscleFocusColor(relative) {
      const focus = Math.max(0, Math.min(1, Number(relative) || 0));
      if (!focus) return bodyRegionIdleColor();
      if (focus < 0.5) return mixColor("#fde68a", "#f97316", focus * 2);
      return mixColor("#f97316", "#ef4444", (focus - 0.5) * 2);
    }

    function bodyMusclesLibrary() {
      return window.BodyMuscles || null;
    }

    function bodyMuscleIdsForGroup(group) {
      const library = bodyMusclesLibrary();
      const groups = library?.MUSCLE_GROUPS || {};
      if (group === "CHEST") return groups.Chest || [];
      if (group === "SHOULDERS") return groups.Shoulders || [];
      if (group === "ARMS") return groups.Arms || [];
      if (group === "LEGS") return groups.Legs || [];
      if (group === "BACK") return groups.Back || [];
      if (group === "CORE") {
        return [
          ...(groups.Abdominals || []),
          "spine",
          "lower-back-erectors-left",
          "lower-back-erectors-right",
          "lower-back-ql-left",
          "lower-back-ql-right"
        ];
      }
      return [];
    }

    function fallbackBodyMusclesForExercise(exerciseName) {
      return muscleGroupWeightsForExercise(exerciseName)
        .flatMap(({ group, weight }) => {
          const ids = bodyMuscleIdsForGroup(group);
          if (!ids.length) return [];
          return ids.map(id => ({ id, weight: weight / ids.length }));
        });
    }

    function refinedBodyMusclesForExercise(exerciseId, exerciseName) {
      const byId = refinedExerciseBodyMuscleLookup.byId.get(String(exerciseId || ""));
      if (byId) return byId.bodyMuscles;
      const key = normalizeExerciseName(exerciseName);
      const byName = refinedExerciseBodyMuscleLookup.byName.get(key) ||
        refinedExerciseBodyMuscleLookup.byName.get(singularExerciseKey(exerciseName));
      if (byName) return byName.bodyMuscles;
      return fallbackBodyMusclesForExercise(exerciseName);
    }

    function bodyMuscleNameLookup() {
      const library = bodyMusclesLibrary();
      const lookup = new Map();
      (library?.MUSCLE_MAP || []).forEach(item => lookup.set(item.id, item.name));
      return lookup;
    }

    function bodyMuscleFocusData(rows) {
      const totals = new Map();
      const exerciseSets = new Map();
      const contributionRows = new Map();
      const unmatched = new Set();

      rows.forEach(row => {
        const volume = Number(row.totalVolumeKg || 0);
        if (!volume) return;
        const bodyMuscles = refinedBodyMusclesForExercise(row.exerciseId, row.exerciseName);
        if (!bodyMuscles.length) {
          unmatched.add(row.exerciseName || "Unknown");
          return;
        }
        bodyMuscles.forEach(({ id, weight }) => {
          const amount = volume * Number(weight || 0);
          if (!id || !Number.isFinite(amount) || amount <= 0) return;
          totals.set(id, (totals.get(id) || 0) + amount);
          if (!exerciseSets.has(id)) exerciseSets.set(id, new Set());
          exerciseSets.get(id).add(row.exerciseName || "Unknown");
          if (!contributionRows.has(id)) contributionRows.set(id, []);
          contributionRows.get(id).push({
            date: row.localDate || "",
            label: row.label || row.localDate || "",
            exerciseName: row.exerciseName || "Unknown",
            sets: asInt(row.sets),
            reps: asInt(row.workingReps),
            workoutVolumeKg: volume,
            allocatedVolumeKg: amount
          });
        });
      });

      const nameLookup = bodyMuscleNameLookup();
      const total = [...totals.values()].reduce((sum, value) => sum + value, 0);
      const maxValue = Math.max(...totals.values(), 0);
      const items = [...totals.entries()]
        .map(([id, value]) => ({
          id,
          label: nameLookup.get(id) || id,
          value,
          relative: maxValue ? value / maxValue : 0,
          share: total ? value / total : 0,
          exercises: [...(exerciseSets.get(id) || [])].sort(),
          contributions: (contributionRows.get(id) || [])
            .map(entry => ({
              ...entry,
              shareOfMuscle: value ? entry.allocatedVolumeKg / value : 0
            }))
            .sort((a, b) =>
              b.allocatedVolumeKg - a.allocatedVolumeKg ||
              String(b.date).localeCompare(String(a.date)) ||
              a.exerciseName.localeCompare(b.exerciseName)
            )
        }))
        .sort((a, b) => b.value - a.value || a.label.localeCompare(b.label));
      return { items, total, maxValue, unmatched: [...unmatched].sort() };
    }

    function bodyMuscleStateFromFocus(items) {
      const bodyState = {};
      items.forEach(item => {
        if (!item.value) return;
        const intensity = Math.max(1, Math.min(10, Math.round(item.relative * 10)));
        bodyState[item.id] = { intensity, selected: item.id === state.selectedBodyMuscleId };
      });
      return bodyState;
    }

    function updateBodyMuscleChartSelection() {
      if (!frontBodyChart || !backBodyChart) return;
      const bodyState = bodyMuscleStateFromFocus(currentMuscleFocusItems);
      frontBodyChart.update({ bodyState });
      backBodyChart.update({ bodyState });
    }

    function applyMuscleListSelection({ scroll = false } = {}) {
      document.querySelectorAll(".muscle-contribution-insert").forEach(panel => panel.remove());
      document.querySelectorAll(".muscle-breakdown-row.is-selected")
        .forEach(row => row.classList.remove("is-selected"));
      document.querySelectorAll(".muscle-breakdown-toggle")
        .forEach(toggle => { toggle.textContent = ""; });
      if (!state.selectedBodyMuscleId) return;
      const row = [...document.querySelectorAll(".muscle-breakdown-row")]
        .find(item => item.dataset.muscleId === state.selectedBodyMuscleId);
      if (!row) return;
      row.classList.add("is-selected");
      const toggle = row.querySelector(".muscle-breakdown-toggle");
      if (toggle) toggle.textContent = state.expandedBodyMuscleId === state.selectedBodyMuscleId ? "-" : "+";
      if (state.expandedBodyMuscleId === state.selectedBodyMuscleId) {
        const item = currentMuscleFocusItems.find(entry => entry.id === state.selectedBodyMuscleId);
        if (item) row.insertAdjacentHTML("afterend", renderMuscleContributionTable(item));
      }
      if (scroll) {
        row.scrollIntoView({ behavior: "smooth", block: "center", inline: "nearest" });
      }
    }

    function selectBodyMuscle(muscleId, { scroll = true, collapseExpanded = true } = {}) {
      state.selectedBodyMuscleId = String(muscleId || "");
      if (collapseExpanded) state.expandedBodyMuscleId = "";
      updateBodyMuscleChartSelection();
      applyMuscleListSelection({ scroll });
    }

    function activateMuscleListRow(row) {
      const muscleId = row?.dataset?.muscleId || "";
      if (!muscleId) return;
      if (state.selectedBodyMuscleId === muscleId) {
        state.expandedBodyMuscleId = state.expandedBodyMuscleId === muscleId ? "" : muscleId;
        updateBodyMuscleChartSelection();
        applyMuscleListSelection();
      } else {
        selectBodyMuscle(muscleId, { scroll: false, collapseExpanded: true });
      }
    }

    function clearBodyMuscleSelection() {
      if (!state.selectedBodyMuscleId) return;
      state.selectedBodyMuscleId = "";
      state.expandedBodyMuscleId = "";
      updateBodyMuscleChartSelection();
      applyMuscleListSelection();
    }

    function isInsideMuscleBreakdownTarget(target) {
      if (!(target instanceof Element)) return false;
      return Boolean(
        target.closest("#muscleBreakdownList") ||
        target.closest("#bodyMusclesFront") ||
        target.closest("#bodyMusclesBack")
      );
    }

    function ensureBodyMuscleCharts() {
      const library = bodyMusclesLibrary();
      if (!library || !bodyMusclesFront || !bodyMusclesBack) return false;
      if (!frontBodyChart) {
        frontBodyChart = new library.BodyChart(bodyMusclesFront, {
          view: library.ViewSide.FRONT,
          bodyState: {},
          ariaLabel: "Front muscle focus body map",
          enableTransitions: true,
          onMuscleClick: selectBodyMuscle
        });
      }
      if (!backBodyChart) {
        backBodyChart = new library.BodyChart(bodyMusclesBack, {
          view: library.ViewSide.BACK,
          bodyState: {},
          ariaLabel: "Back muscle focus body map",
          enableTransitions: true,
          onMuscleClick: selectBodyMuscle
        });
      }
      return true;
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
      ["repsChart", "loadChart", "volumeChart", "velocityChart", "muscleBalanceChart", "repEnergyChart"].forEach(id => {
        const canvas = document.getElementById(id);
        if (!canvas) return;
        canvas.addEventListener("mousemove", event => {
          const rect = canvas.getBoundingClientRect();
          const item = nearestChartItem(canvas, event.clientX - rect.left, event.clientY - rect.top);
          if (item) showChartTooltip(event, item);
          else hideChartTooltip();
        });
        canvas.addEventListener("mouseleave", hideChartTooltip);
      });
    }

    function setupMuscleBreakdownTooltips() {
      document.querySelectorAll(".body-region[data-muscle]").forEach(region => {
        region.addEventListener("mousemove", event => {
          const lines = (region.dataset.tooltip || titleCaseGroup(region.dataset.muscle)).split("|");
          showChartTooltip(event, { lines });
        });
        region.addEventListener("mouseleave", hideChartTooltip);
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
      const options = [
        `<option value="${ALL_EXERCISES_ID}">All Exercises</option>`,
        ...DATA.exercises.map(exercise =>
        `<option value="${exercise.id}">${exercise.name}</option>`
        )
      ].join("");
      state.exerciseId = isValidExerciseId(previous)
        ? previous
        : ALL_EXERCISES_ID;
      exerciseSelects.forEach(select => {
        select.innerHTML = options;
        select.value = state.exerciseId;
      });
    }

    function syncExerciseSelects() {
      exerciseSelects.forEach(select => {
        select.value = state.exerciseId;
      });
    }

    function syncHistoryWindowSelects() {
      historyWindowSelects.forEach(select => {
        select.value = state.historyWindow;
      });
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
        const fileButton = document.querySelector("label[for='backupFileInput'].file-button");
        fileButton.textContent = file.name;
        fileButton.title = file.name;
        saveDashboardCache();
        render();
      } catch (error) {
        sourceMeta.textContent = `Could not load ${file.name}: ${error.message}`;
      } finally {
        backupFileInput.value = "";
      }
    }

    function updateStickyOffsets() {
      const topOffset = Math.ceil(topRibbon.getBoundingClientRect().height + 12);
      document.documentElement.style.setProperty("--rep-filter-sticky-top", `${topOffset}px`);
    }

    function applyRibbonState() {
      topRibbon.classList.toggle("is-collapsed", state.ribbonCollapsed);
      ribbonToggle.setAttribute("aria-expanded", String(!state.ribbonCollapsed));
      ribbonToggle.textContent = state.ribbonCollapsed ? "+" : "-";
      ribbonToggle.title = state.ribbonCollapsed ? "Expand top ribbon" : "Collapse top ribbon";
      updateStickyOffsets();
      window.requestAnimationFrame(updateStickyOffsets);
      window.setTimeout(updateStickyOffsets, 200);
    }

    function autoCollapseRibbon() {
      if (state.ribbonCollapsed || Date.now() < ribbonAutoCollapsePausedUntil) return;
      state.ribbonCollapsed = true;
      applyRibbonState();
      saveDashboardCache();
    }

    function applyDashboardTabState() {
      [
        ["summary", summaryTabPanel, summaryTabButton],
        ["repAnalytics", repAnalyticsTabPanel, repAnalyticsTabButton],
        ["muscleBreakdown", muscleBreakdownTabPanel, muscleBreakdownTabButton]
      ].forEach(([tabKey, panel, button]) => {
        const isActive = state.activeTab === tabKey;
        panel.classList.toggle("is-hidden", !isActive);
        button.classList.toggle("is-active", isActive);
        button.setAttribute("aria-pressed", String(isActive));
      });
    }

    function applyFilterCollapseState() {
      const expanded = !state.filtersCollapsed;
      [
        [summaryFilterPanel, summaryFilterToggle],
        [repAnalyticsFilterPanel, repAnalyticsFilterToggle],
        [muscleBreakdownFilterPanel, muscleBreakdownFilterToggle]
      ].forEach(([panel, toggle]) => {
        panel.classList.toggle("is-collapsed", state.filtersCollapsed);
        toggle.setAttribute("aria-expanded", String(expanded));
        toggle.textContent = expanded ? "-" : "Filters +";
        toggle.title = expanded ? "Collapse filters" : "Expand filters";
      });
    }

    function applyTheme() {
      document.body.classList.toggle("theme-dark", state.theme === "dark");
    }

    function toggleFiltersCollapsed() {
      state.filtersCollapsed = !state.filtersCollapsed;
      applyFilterCollapseState();
      saveDashboardCache();
    }

    function setupControls() {
      populateExerciseSelect();
      syncHistoryWindowSelects();
      loadUnitToggle.checked = state.loadUnit === "lbs";
      loadToggle.checked = state.loadBasis === "total";
      themeToggle.checked = state.theme === "dark";
      loadEchoMedianToggle.checked = state.showLoadEchoMedian;
      loadEchoAverageToggle.checked = state.showLoadEchoAverage;
      repOverlayMode.value = state.repOverlayMode;
      repTypeFilter.value = state.repTypeFilter;
      positionOverlayBasis.value = state.positionOverlayBasis;
      applyRibbonState();
      applyDashboardTabState();
      applyFilterCollapseState();
      applyTheme();
      if (cachedDashboard?.data) {
        const fileButton = document.querySelector("label[for='backupFileInput'].file-button");
        fileButton.textContent = "Cached data";
        fileButton.title = "Cached data";
      }
      ribbonToggle.addEventListener("click", () => {
        ribbonAutoCollapsePausedUntil = Date.now() + 450;
        state.ribbonCollapsed = !state.ribbonCollapsed;
        applyRibbonState();
        saveDashboardCache();
      });
      summaryFilterToggle.addEventListener("click", () => {
        toggleFiltersCollapsed();
      });
      repAnalyticsFilterToggle.addEventListener("click", () => {
        toggleFiltersCollapsed();
      });
      muscleBreakdownFilterToggle.addEventListener("click", () => {
        toggleFiltersCollapsed();
      });
      muscleBreakdownList.addEventListener("click", event => {
        const target = event.target instanceof Element ? event.target : event.target?.parentElement;
        const row = target?.closest(".muscle-breakdown-row");
        if (!row || !muscleBreakdownList.contains(row)) return;
        activateMuscleListRow(row);
      });
      muscleBreakdownList.addEventListener("keydown", event => {
        if (event.key !== "Enter" && event.key !== " ") return;
        const target = event.target instanceof Element ? event.target : null;
        const row = target?.closest(".muscle-breakdown-row");
        if (!row || !muscleBreakdownList.contains(row)) return;
        event.preventDefault();
        activateMuscleListRow(row);
      });
      document.addEventListener("click", event => {
        if (isInsideMuscleBreakdownTarget(event.target)) return;
        clearBodyMuscleSelection();
      });
      mainContent.addEventListener("pointerdown", autoCollapseRibbon);
      window.addEventListener("scroll", autoCollapseRibbon, { passive: true });
      [summaryTabButton, repAnalyticsTabButton, muscleBreakdownTabButton].forEach(button => {
        button.addEventListener("click", () => {
          const nextTab = button.dataset.tab;
          if (state.activeTab === nextTab) return;
          state.activeTab = nextTab;
          applyDashboardTabState();
          chartTooltip.style.display = "none";
          saveDashboardCache();
          window.scrollTo({ top: 0, behavior: "smooth" });
          render();
        });
      });
      exerciseSelects.forEach(select => {
        select.addEventListener("change", () => {
          state.exerciseId = select.value;
          syncExerciseSelects();
          saveDashboardCache();
          render();
        });
      });
      backupFileInput.addEventListener("change", event => {
        loadBackupFile(event.target.files?.[0]);
      });
      historyWindowSelects.forEach(select => {
        select.addEventListener("change", () => {
          state.historyWindow = select.value;
          syncHistoryWindowSelects();
          saveDashboardCache();
          render();
        });
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
      themeToggle.addEventListener("change", () => {
        state.theme = themeToggle.checked ? "dark" : "light";
        applyTheme();
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
      repTypeFilter.addEventListener("change", () => {
        state.repTypeFilter = repTypeFilter.value;
        saveDashboardCache();
        render();
      });
      positionOverlayBasis.addEventListener("change", () => {
        state.positionOverlayBasis = positionOverlayBasis.value;
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

    function totalEnergyForRows(rows) {
      const activeWorkoutExerciseKeys = new Set(rows.map(row => `${row.workoutId}::${row.exerciseId}`));
      return DATA.repTraces
        .filter(trace => activeWorkoutExerciseKeys.has(`${traceWorkoutId(trace)}::${trace.exerciseId}`))
        .reduce((sum, trace) => {
          const energy = Number(trace.totalEnergyJ);
          return Number.isFinite(energy) ? sum + energy : sum;
        }, 0);
    }

    function renderKpis(rows) {
      const setCount = rows.reduce((sum, row) => sum + Number(row.sets || 0), 0);
      const totalReps = rows.reduce((sum, row) => sum + Number(row.workingReps || 0), 0);
      const totalVolume = rows.reduce((sum, row) => sum + Number(row.totalVolumeKg || 0), 0);
      const maxLoad = Math.max(...rows.map(row => Number(row[loadField()] || 0)), 0);
      const bestE1rm = Math.max(...rows.map(row => Number(row.estimatedOneRepMaxKg || 0)), 0);
      const totalEnergy = totalEnergyForRows(rows);
      const latest = rows.length ? rows[rows.length - 1].localDate : "-";
      const kpis = [
        ["Workouts", rows.length, `${setCount} completed sets`],
        ["Working Reps", totalReps, "warmup and zero-rep sessions excluded"],
        ["Best Load", fmtLoad(maxLoad, 1), loadUnitLabel()],
        ["Total Volume", fmtLoad(totalVolume, 0), "from Vitruvian volume"],
        ["Best Est. 1RM", fmtLoad(bestE1rm, 1), `latest ${latest}`],
        ["Total Energy", fmtEnergy(totalEnergy, 1), "selected exercise and window"]
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
      ctx.fillStyle = canvasColor();
      ctx.fillRect(0, 0, rect.width, rect.height);
      return { ctx, width: rect.width, height: rect.height };
    }

    function drawEmpty(canvas, text) {
      const { ctx, width, height } = canvasSetup(canvas);
      setChartHitAreas(canvas, []);
      ctx.fillStyle = mutedColor();
      ctx.font = "14px Segoe UI, Arial";
      ctx.textAlign = "center";
      ctx.fillText(text, width / 2, height / 2);
    }

    function scaleLinear(domainMin, domainMax, rangeMin, rangeMax) {
      const span = domainMax - domainMin || 1;
      return value => rangeMin + ((value - domainMin) / span) * (rangeMax - rangeMin);
    }

    function drawAxes(ctx, box, yTicks, yLabel) {
      ctx.strokeStyle = lineColor();
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(box.left, box.top);
      ctx.lineTo(box.left, box.bottom);
      ctx.lineTo(box.right, box.bottom);
      ctx.stroke();

      ctx.fillStyle = mutedColor();
      ctx.font = "12px Segoe UI, Arial";
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      yTicks.forEach(tick => {
        ctx.strokeStyle = gridColor();
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
      ctx.fillStyle = mutedColor();
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
          ctx.fillStyle = mutedColor();
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
      ctx.fillStyle = mutedColor();
      ctx.font = "11px Segoe UI, Arial";
      const labelStep = Math.max(1, Math.ceil(rows.length / 6));
      rows.forEach((row, idx) => {
        if (idx % labelStep === 0 || idx === rows.length - 1) ctx.fillText(row.localDate.slice(5), xScale(idx), box.bottom + 12);
      });

      ctx.textAlign = "left";
      ctx.fillStyle = "#2563eb";
      ctx.fillRect(box.left, height - 20, 12, 8);
      ctx.fillStyle = mutedColor();
      ctx.fillText("Volume", box.left + 18, height - 22);
      ctx.fillStyle = "#dc2626";
      ctx.fillRect(box.left + 92, height - 17, 12, 3);
      ctx.fillStyle = mutedColor();
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

    function balanceGroupForBodyMuscle(id) {
      if (/^chest-/.test(id)) return "CHEST";
      if (/^(lats|traps|lower-back|nape|head-back)/.test(id)) return "BACK";
      if (/^(shoulder|deltoid-rear)/.test(id)) return "SHOULDERS";
      if (/^(biceps|triceps|forearm|elbow|hand)/.test(id)) return "ARMS";
      if (/^(abs|obliques|serratus|spine)/.test(id)) return "CORE";
      if (/^(quads|hamstrings|gluteus|adductors|calves|tibialis|knee|hip-flexor|foot)/.test(id)) return "LEGS";
      return null;
    }

    function balanceGroupWeightsForExercise(exerciseId, exerciseName) {
      const bodyMuscles = refinedBodyMusclesForExercise(exerciseId, exerciseName);
      const groupWeights = new Map(MUSCLE_GROUP_ORDER.map(group => [group, 0]));
      bodyMuscles.forEach(({ id, weight }) => {
        const group = balanceGroupForBodyMuscle(id);
        const value = Number(weight || 0);
        if (!group || !Number.isFinite(value) || value <= 0) return;
        groupWeights.set(group, groupWeights.get(group) + value);
      });
      const total = [...groupWeights.values()].reduce((sum, value) => sum + value, 0);
      if (!total) return muscleGroupWeightsForExercise(exerciseName);
      return MUSCLE_GROUP_ORDER
        .map(group => ({ group, weight: groupWeights.get(group) / total }))
        .filter(item => item.weight > 0);
    }

    function muscleBalanceData(rows) {
      const groupTotals = new Map(MUSCLE_GROUP_ORDER.map(group => [group, 0]));
      const groupExercises = new Map(MUSCLE_GROUP_ORDER.map(group => [group, new Set()]));
      const unmatched = new Set();

      rows.forEach(row => {
        const groupWeights = balanceGroupWeightsForExercise(row.exerciseId, row.exerciseName);
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

      ctx.strokeStyle = lineColor();
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
        ctx.fillStyle = item.value > 0 ? "#2563eb" : inactivePointColor();
        ctx.beginPath();
        ctx.arc(pointX, pointY, 4, 0, Math.PI * 2);
        ctx.fill();

        const labelX = centerX + labelRadius * Math.cos(angle);
        const labelY = centerY + labelRadius * Math.sin(angle);
        ctx.fillStyle = mutedColor();
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
        `Relative focus from refined body-region mappings rolled up to six muscle groups; ${matchedCount} matched exercise${matchedCount === 1 ? "" : "s"}.${unmatchedText}`;
      document.getElementById("muscleBalanceList").innerHTML = balance.items.map(item => `
        <div class="muscle-balance-item">
          <strong>${item.label}</strong>
          <span>${fmtNumber(item.share * 100, 0)}% share · ${fmtLoad(item.value, 0)}</span>
        </div>
      `).join("");
    }

    function renderMuscleContributionTable(item) {
      const rows = item.contributions || [];
      if (!rows.length) {
        return `<div class="muscle-contribution-insert">No contributing workout rows.</div>`;
      }
      return `
        <div class="muscle-contribution-insert">
          <div class="muscle-contribution-inner">
            <table class="muscle-contribution-table">
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Exercise</th>
                  <th>Sets</th>
                  <th>Reps</th>
                  <th>Volume</th>
                  <th>Muscle Load</th>
                  <th>Muscle %</th>
                </tr>
              </thead>
              <tbody>
                ${rows.map(row => `
                  <tr>
                    <td>${escapeHtml(row.label || row.date)}</td>
                    <td>${escapeHtml(row.exerciseName)}</td>
                    <td>${fmtNumber(row.sets, 0)}</td>
                    <td>${fmtNumber(row.reps, 0)}</td>
                    <td>${fmtLoad(row.workoutVolumeKg, 0)}</td>
                    <td>${fmtLoad(row.allocatedVolumeKg, 0)}</td>
                    <td>${fmtNumber(row.shareOfMuscle * 100, 1)}%</td>
                  </tr>
                `).join("")}
              </tbody>
            </table>
          </div>
        </div>
      `;
    }

    function renderMuscleBreakdown(rows) {
      const focus = bodyMuscleFocusData(rows);
      currentMuscleFocusItems = focus.items;
      if (state.selectedBodyMuscleId && !currentMuscleFocusItems.some(item => item.id === state.selectedBodyMuscleId)) {
        state.selectedBodyMuscleId = "";
        state.expandedBodyMuscleId = "";
      }
      const matchedCount = new Set(focus.items.flatMap(item => item.exercises)).size;
      const unmatchedText = focus.unmatched.length
        ? ` ${focus.unmatched.length} unmatched exercise${focus.unmatched.length === 1 ? "" : "s"} not shown.`
        : "";

      if (ensureBodyMuscleCharts()) {
        const bodyState = bodyMuscleStateFromFocus(focus.items);
        frontBodyChart.update({ bodyState });
        backBodyChart.update({ bodyState });
        document.getElementById("muscleBreakdownNote").textContent =
          `Refined from ${REFINED_EXERCISE_BODY_MUSCLE_MAP.length} Project Phoenix exercises, Project Phoenix detailed muscles, free-exercise-db matches, and body-muscles anatomical regions; ${matchedCount} matched exercise${matchedCount === 1 ? "" : "s"}.${unmatchedText}`;
      } else {
        document.getElementById("muscleBreakdownNote").textContent =
          "Body map library could not be loaded. Keep body-muscles.umd.min.js beside this HTML file.";
      }

      const listItems = focus.items.filter(item => item.value > 0);
      document.getElementById("muscleBreakdownList").innerHTML = listItems.map(item => `
        <div class="muscle-breakdown-row" role="button" tabindex="0" data-muscle-id="${escapeHtml(item.id)}">
          <span class="swatch" style="background:${muscleFocusColor(item.relative)}"></span>
          <strong>${escapeHtml(item.label)}</strong>
          <span>${fmtNumber(item.share * 100, 0)}% &middot; ${fmtLoad(item.value, 0)}</span>
          <span class="muscle-breakdown-toggle" aria-hidden="true"></span>
        </div>
      `).join("");
      applyMuscleListSelection();
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
      if (state.repOverlayMode === "maxEnergy") {
        return "totalEnergyJ";
      }
      return null;
    }

    function overlayModeLabel() {
      if (state.repOverlayMode === "maxAverage") return "Max average";
      if (state.repOverlayMode === "maxMedian") return "Max median";
      if (state.repOverlayMode === "maxEnergy") return "Max energy";
      return "All";
    }

    function traceModeKey(trace) {
      return isEchoMode(trace.mode) ? "Echo" : "Non-echo";
    }

    function repTypeFilterLabel() {
      if (state.repTypeFilter === "echo") return "Echo reps";
      if (state.repTypeFilter === "nonEcho") return "Non-echo reps";
      return "All rep types";
    }

    function maxRepTypeDescription() {
      const qualifier = state.repOverlayMode === "maxEnergy"
        ? "highest-energy"
        : state.repOverlayMode === "maxAverage"
          ? "highest-average-load"
          : "highest-median-load";
      if (state.repTypeFilter === "echo") return `${qualifier} echo rep`;
      if (state.repTypeFilter === "nonEcho") return `${qualifier} non-echo rep`;
      return `${qualifier} echo rep and ${qualifier} non-echo rep`;
    }

    function traceMatchesRepTypeFilter(trace) {
      if (state.repTypeFilter === "echo") return isEchoMode(trace.mode);
      if (state.repTypeFilter === "nonEcho") return !isEchoMode(trace.mode);
      return true;
    }

    function traceMetricValue(trace, field) {
      const value = Number(trace[field]);
      return Number.isFinite(value) ? value : -Infinity;
    }

    function selectRepOverlayTraces(traces) {
      const field = overlayRankField();
      if (!field) {
        return traces.map(trace => ({ ...trace, overlayModeLabel: "All" }));
      }
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

    function selectedOverlayTraces(activeRows) {
      const activeWorkoutExerciseKeys = new Set(activeRows.map(row => `${row.workoutId}::${row.exerciseId}`));
      const sourceTraces = DATA.repTraces
        .filter(trace =>
          (isAllExercises() || trace.exerciseId === state.exerciseId) &&
          traceMatchesRepTypeFilter(trace) &&
          activeWorkoutExerciseKeys.has(`${traceWorkoutId(trace)}::${trace.exerciseId}`)
        )
        .sort((a, b) => new Date(a.dateTime) - new Date(b.dateTime));
      return selectRepOverlayTraces(sourceTraces);
    }

    function traceTimeLabel(trace) {
      const text = String(trace.dateTime || trace.label || trace.date || "");
      const match = text.match(/\b(\d{1,2}:\d{2})(?::\d{2})?\b/);
      if (match) return match[1];
      return trace.label || trace.date || "-";
    }

    function renderRepEnergyChart(activeRows, selectedTraces = null) {
      const canvas = document.getElementById("repEnergyChart");
      const traces = (selectedTraces || selectedOverlayTraces(activeRows))
        .map(trace => ({ ...trace, energyJ: Number(trace.totalEnergyJ) }))
        .filter(trace => Number.isFinite(trace.energyJ))
        .sort((a, b) => new Date(a.dateTime) - new Date(b.dateTime) || Number(a.repIndex || 0) - Number(b.repIndex || 0));
      document.getElementById("repEnergyNote").textContent =
        `${overlayModeLabel()} energy per selected ${repTypeFilterLabel().toLowerCase()} over workout time. Bars use workout date colours.`;
      if (!traces.length) {
        drawEmpty(canvas, "No rep energy values were detected for this selection.");
        return;
      }

      const dates = [...new Set(traces.map(trace => trace.date))];
      const dateColors = new Map(dates.map((date, idx) => [date, palette[idx % palette.length]]));
      const { ctx, width, height } = canvasSetup(canvas);
      const box = { left: 62, right: width - 22, top: 18, bottom: height - 52 };
      const energyValues = traces.map(trace => trace.energyJ);
      const yMax = Math.max(...energyValues, 1);
      const yScale = scaleLinear(0, yMax * 1.16, box.bottom, box.top);
      const xScale = traces.length === 1
        ? () => (box.left + box.right) / 2
        : scaleLinear(0, traces.length - 1, box.left, box.right);
      const slotWidth = (box.right - box.left) / Math.max(traces.length, 1);
      const barWidth = Math.max(2, Math.min(16, slotWidth * 0.72));
      const yTicks = [0, 0.25, 0.5, 0.75, 1].map(part => {
        const value = yMax * 1.16 * part;
        return { y: yScale(value), label: fmtNumber(value, 0) };
      });
      drawAxes(ctx, box, yTicks, "energy (J)");

      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.fillStyle = mutedColor();
      ctx.font = "11px Segoe UI, Arial";
      const labelStep = Math.max(1, Math.ceil(traces.length / 7));
      traces.forEach((trace, idx) => {
        if (idx % labelStep === 0 || idx === traces.length - 1) {
          const label = `${String(trace.date || "").slice(5)} ${traceTimeLabel(trace)}`.trim();
          ctx.fillText(label, xScale(idx), box.bottom + 12);
        }
      });
      ctx.fillText("workout time", (box.left + box.right) / 2, height - 18);

      const hitItems = [];
      traces.forEach((trace, idx) => {
        const x = xScale(idx);
        const energyJ = trace.energyJ;
        const y = yScale(energyJ);
        const color = isOverlayDateDimmed(trace.date) ? "#9ca3af" : (dateColors.get(trace.date) || "#2563eb");
        ctx.fillStyle = `${color}${isOverlayDateDimmed(trace.date) ? "90" : "d8"}`;
        ctx.fillRect(x - barWidth / 2, y, barWidth, box.bottom - y);
        hitItems.push({
          x,
          y,
          bounds: {
            left: x - barWidth / 2,
            right: x + barWidth / 2,
            top: y,
            bottom: box.bottom
          },
          lines: [
            `${trace.label || trace.date || "-"} ${traceTimeLabel(trace)}`,
            trace.exerciseName || "Working rep",
            `${traceModeKey(trace)} rep ${trace.repIndex || "-"}`,
            `Energy: ${fmtEnergyJ(trace.energyJ, 0)}`
          ]
        });
      });
      setChartHitAreas(canvas, hitItems);
    }

    function drawLoadTracePath(ctx, trace, loadKey, xScale, yScale, strokeStyle, lineWidth) {
      ctx.strokeStyle = strokeStyle;
      ctx.lineWidth = lineWidth;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.setLineDash(isEchoMode(trace.mode) ? [6, 4] : []);
      ctx.beginPath();
      let hasStarted = false;
      trace.timeSec.forEach((time, idx) => {
        const yValue = displayLoadValue(trace[loadKey][idx]);
        if (!Number.isFinite(yValue)) {
          hasStarted = false;
          return;
        }
        const x = xScale(time);
        const y = yScale(yValue);
        if (!hasStarted) {
          ctx.moveTo(x, y);
          hasStarted = true;
        } else {
          ctx.lineTo(x, y);
        }
      });
      ctx.stroke();
    }

    function renderRepOverlay(activeRows, selectedTraces = null) {
      const canvas = document.getElementById("repOverlay");
      const traces = selectedTraces || selectedOverlayTraces(activeRows);
      document.getElementById("repOverlayNote").textContent =
        state.repOverlayMode === "all"
          ? `Load over time for ${repTypeFilterLabel().toLowerCase()}. Colours identify the workout date. Echo traces are dashed.`
          : `${overlayModeLabel()} shows the ${maxRepTypeDescription()} for each selected workout date. Echo traces are dashed.`;
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
      ctx.fillStyle = mutedColor();
      ctx.font = "12px Segoe UI, Arial";
      ctx.fillText("seconds from rep start", (box.left + box.right) / 2, height - 22);

      const dateColors = new Map(dates.map((date, idx) => [date, palette[idx % palette.length]]));
      traces.forEach(trace => {
        const color = isOverlayDateDimmed(trace.date) ? "#9ca3af" : (dateColors.get(trace.date) || "#2563eb");
        const strokeAlpha = isOverlayDateDimmed(trace.date)
          ? "82"
          : "aa";
        const lineWidth = state.repOverlayMode === "all"
          ? (isOverlayDateDimmed(trace.date) ? 1.1 : 1.4)
          : (isOverlayDateDimmed(trace.date) ? 1.4 : 2.3);
        drawLoadTracePath(ctx, trace, loadKey, xScale, yScale, `${color}${strokeAlpha}`, lineWidth);
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
            renderRepEnergyChart(activeRows);
            renderRepOverlay(activeRows);
            renderPositionOverlay(activeRows);
            renderLoadPositionPhaseCharts(activeRows);
            return;
          }
          const date = button.dataset.date;
          const key = overlayDateKey(date);
          if (state.dimmedOverlayDates.has(key)) state.dimmedOverlayDates.delete(key);
          else state.dimmedOverlayDates.add(key);
          saveDashboardCache();
          renderRepEnergyChart(activeRows);
          renderRepOverlay(activeRows);
          renderPositionOverlay(activeRows);
          renderLoadPositionPhaseCharts(activeRows);
        });
      });
    }

    function tracePositionValues(trace, field) {
      if (Array.isArray(trace[field]) && trace[field].length) return trace[field];
      if (Array.isArray(trace.position) && trace.position.length) return trace.position;
      return [];
    }

    function positionOverlayLabel() {
      if (state.positionOverlayBasis === "left") return "Left cable";
      if (state.positionOverlayBasis === "right") return "Right cable";
      return "Average";
    }

    function positionOverlayField() {
      if (state.positionOverlayBasis === "left") return "positionA";
      if (state.positionOverlayBasis === "right") return "positionB";
      return "position";
    }

    function drawTracePath(ctx, trace, values, xScale, yScale, strokeStyle, lineWidth) {
      ctx.strokeStyle = strokeStyle;
      ctx.lineWidth = lineWidth;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.setLineDash(isEchoMode(trace.mode) ? [6, 4] : []);
      ctx.beginPath();
      let hasStarted = false;
      values.forEach((value, idx) => {
        const number = Number(value);
        if (!Number.isFinite(number)) {
          hasStarted = false;
          return;
        }
        const x = xScale(trace.timeSec[idx] || 0);
        const y = yScale(number);
        if (!hasStarted) {
          ctx.moveTo(x, y);
          hasStarted = true;
        }
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    }

    function renderPositionOverlay(activeRows, selectedTraces = null) {
      const canvas = document.getElementById("positionOverlay");
      const traces = selectedTraces || selectedOverlayTraces(activeRows);
      const positionField = positionOverlayField();
      const positionLabel = positionOverlayLabel();
      document.getElementById("positionOverlayNote").textContent =
        state.repOverlayMode === "all"
          ? `${positionLabel} position over time for ${repTypeFilterLabel().toLowerCase()}. Echo traces are dashed.`
          : `${overlayModeLabel()} uses the same ${maxRepTypeDescription()} as the load overlay; ${positionLabel.toLowerCase()} position is shown. Echo traces are dashed.`;
      if (!traces.length) {
        drawEmpty(canvas, "No working rep positions were detected for this selection.");
        return;
      }

      const dates = [...new Set(traces.map(trace => trace.date))];
      const { ctx, width, height } = canvasSetup(canvas);
      const box = { left: 62, right: width - 22, top: 34, bottom: height - 46 };
      const maxTime = Math.max(...traces.flatMap(trace => trace.timeSec), 1);
      const yValues = traces
        .flatMap(trace => tracePositionValues(trace, positionField))
        .map(value => Number(value))
        .filter(value => Number.isFinite(value));
      if (!yValues.length) {
        drawEmpty(canvas, "No working rep positions were detected for this selection.");
        return;
      }
      const yMinRaw = Math.min(...yValues);
      const yMaxRaw = Math.max(...yValues);
      const yPad = Math.max((yMaxRaw - yMinRaw || 1) * 0.12, 1);
      const yMin = yMinRaw - yPad;
      const yMax = yMaxRaw + yPad;
      const xScale = scaleLinear(0, maxTime, box.left, box.right);
      const yScale = scaleLinear(yMin, yMax, box.bottom, box.top);
      const yTicks = [0, 0.25, 0.5, 0.75, 1].map(part => {
        const value = yMin + (yMax - yMin) * part;
        return { y: yScale(value), label: fmtNumber(value, 0) };
      });
      drawAxes(ctx, box, yTicks, "cable position");

      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.fillStyle = mutedColor();
      ctx.font = "12px Segoe UI, Arial";
      ctx.fillText("seconds from rep start", (box.left + box.right) / 2, height - 22);

      const dateColors = new Map(dates.map((date, idx) => [date, palette[idx % palette.length]]));
      traces.forEach(trace => {
        const color = isOverlayDateDimmed(trace.date) ? "#9ca3af" : (dateColors.get(trace.date) || "#2563eb");
        const strokeAlpha = isOverlayDateDimmed(trace.date)
          ? "82"
          : "dd";
        const strokeStyle = `${color}${strokeAlpha}`;
        const lineWidth = state.repOverlayMode === "all"
          ? (isOverlayDateDimmed(trace.date) ? 1.1 : 1.4)
          : (isOverlayDateDimmed(trace.date) ? 1.4 : 2.3);
        drawTracePath(ctx, trace, tracePositionValues(trace, positionField), xScale, yScale, strokeStyle, lineWidth);
      });
      ctx.setLineDash([]);

      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ctx.font = "12px Segoe UI, Arial";
      ctx.strokeStyle = "#2563ebdd";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(box.left, 16);
      ctx.lineTo(box.left + 22, 16);
      ctx.stroke();
      ctx.fillStyle = mutedColor();
      ctx.fillText(positionLabel, box.left + 28, 16);
    }

    function phaseSegmentsForTrace(trace, phase) {
      const positions = tracePositionValues(trace, positionOverlayField());
      const loads = Array.isArray(trace[traceLoadField()]) ? trace[traceLoadField()] : [];
      const count = Math.min(positions.length, loads.length);
      const segments = [];
      for (let idx = 1; idx < count; idx += 1) {
        const p0 = Number(positions[idx - 1]);
        const p1 = Number(positions[idx]);
        const l0 = displayLoadValue(loads[idx - 1]);
        const l1 = displayLoadValue(loads[idx]);
        if (![p0, p1, l0, l1].every(value => Number.isFinite(value))) continue;
        const delta = p1 - p0;
        const isConcentric = delta > 0.05;
        const isEccentric = delta < -0.05;
        if ((phase === "concentric" && isConcentric) || (phase === "eccentric" && isEccentric)) {
          segments.push({ trace, p0, p1, l0, l1 });
        }
      }
      return segments;
    }

    function drawPhaseSegment(ctx, segment, xScale, yScale, strokeStyle, lineWidth) {
      const trace = segment.trace;
      ctx.strokeStyle = strokeStyle;
      ctx.lineWidth = lineWidth;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.setLineDash(isEchoMode(trace.mode) ? [6, 4] : []);
      ctx.beginPath();
      ctx.moveTo(xScale(segment.p0), yScale(segment.l0));
      ctx.lineTo(xScale(segment.p1), yScale(segment.l1));
      ctx.stroke();
    }

    function drawLoadPositionPhaseChart(canvas, traces, phase) {
      const segments = traces.flatMap(trace => phaseSegmentsForTrace(trace, phase));
      if (!segments.length) {
        drawEmpty(
          canvas,
          phase === "eccentric"
            ? "No eccentric segments found. Stop-at-top reps may omit lowering data."
            : "No concentric segments found for this selection."
        );
        return;
      }

      const { ctx, width, height } = canvasSetup(canvas);
      const box = { left: 62, right: width - 20, top: 18, bottom: height - 46 };
      const xValues = segments.flatMap(segment => [segment.p0, segment.p1]);
      const yValues = segments.flatMap(segment => [segment.l0, segment.l1]);
      const xMinRaw = Math.min(...xValues);
      const xMaxRaw = Math.max(...xValues);
      const xPad = Math.max((xMaxRaw - xMinRaw || 1) * 0.08, 1);
      const xMin = xMinRaw - xPad;
      const xMax = xMaxRaw + xPad;
      const yMin = Math.max(0, Math.min(...yValues) - 2);
      const yMax = Math.max(...yValues, 1) + 2;
      const xScale = phase === "eccentric"
        ? scaleLinear(xMin, xMax, box.right, box.left)
        : scaleLinear(xMin, xMax, box.left, box.right);
      const yScale = scaleLinear(yMin, yMax, box.bottom, box.top);
      const yTicks = [0, 0.25, 0.5, 0.75, 1].map(part => {
        const value = yMin + (yMax - yMin) * part;
        return { y: yScale(value), label: fmtNumber(value, 1) };
      });
      drawAxes(ctx, box, yTicks, loadUnitLabel());

      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.fillStyle = mutedColor();
      ctx.font = "11px Segoe UI, Arial";
      for (let idx = 0; idx <= 4; idx += 1) {
        const value = xMin + ((xMax - xMin) * idx / 4);
        const x = xScale(value);
        ctx.fillText(fmtNumber(value, 0), x, box.bottom + 12);
      }
      ctx.fillText(`${positionOverlayLabel()} position`, (box.left + box.right) / 2, height - 18);

      const dates = [...new Set(traces.map(trace => trace.date))];
      const dateColors = new Map(dates.map((date, idx) => [date, palette[idx % palette.length]]));
      segments.forEach(segment => {
        const trace = segment.trace;
        const color = isOverlayDateDimmed(trace.date) ? "#9ca3af" : (dateColors.get(trace.date) || "#2563eb");
        const strokeAlpha = isOverlayDateDimmed(trace.date)
          ? "82"
          : "cc";
        const lineWidth = state.repOverlayMode === "all"
          ? (isOverlayDateDimmed(trace.date) ? 1.1 : 1.4)
          : (isOverlayDateDimmed(trace.date) ? 1.4 : 2.2);
        drawPhaseSegment(ctx, segment, xScale, yScale, `${color}${strokeAlpha}`, lineWidth);
      });
      ctx.setLineDash([]);
    }

    function renderLoadPositionPhaseCharts(activeRows, selectedTraces = null) {
      const traces = selectedTraces || selectedOverlayTraces(activeRows);
      document.getElementById("loadPositionNote").textContent =
        `${positionOverlayLabel()} position versus ${loadUnitLabel()} for the same ${repTypeFilterLabel().toLowerCase()} shown in the load overlay. Eccentric may be empty when stop-at-top removes lowering data.`;
      drawLoadPositionPhaseChart(document.getElementById("concentricLoadPositionChart"), traces, "concentric");
      drawLoadPositionPhaseChart(document.getElementById("eccentricLoadPositionChart"), traces, "eccentric");
    }

    function historyRepEnergy(row, echoMode) {
      const field = echoMode ? "largestEchoRepEnergyJ" : "largestNonEchoRepEnergyJ";
      const direct = row[field];
      if (direct !== null && direct !== undefined && direct !== "" && Number.isFinite(Number(direct))) {
        return Number(direct);
      }
      const rowKey = `${row.workoutId}::${row.exerciseId}`;
      const energies = DATA.repTraces
        .filter(trace => `${traceWorkoutId(trace)}::${trace.exerciseId}` === rowKey && isEchoMode(trace.mode) === echoMode)
        .map(trace => Number(trace.totalEnergyJ))
        .filter(value => Number.isFinite(value));
      return energies.length ? Math.max(...energies) : null;
    }

    function renderHistoryTable(rows) {
      const baseColumnCount = isAllExercises() ? 7 : 6;
      document.getElementById("historyGroupHeader").innerHTML = `
        <th class="group-spacer" colspan="${baseColumnCount}"></th>
        <th class="group-separator" colspan="2">Non-Echo</th>
        <th class="group-spacer" colspan="3"></th>
        <th class="group-separator" colspan="5">Echo</th>
      `;
      document.getElementById("historyHeader").innerHTML = `
        <th>Date</th>
        ${isAllExercises() ? "<th>Exercise</th>" : ""}
        <th>Routine</th>
        <th>Sets</th>
        <th>Reps</th>
        <th>Max Load</th>
        <th class="group-separator">Reps</th>
        <th>Energy</th>
        <th>Volume</th>
        <th>Est. 1RM</th>
        <th>Avg MCV</th>
        <th class="group-separator">Reps</th>
        <th>Median</th>
        <th>Avg</th>
        <th>Peak</th>
        <th>Energy</th>
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
          <td>${fmtEnergyJ(historyRepEnergy(row, false), 0)}</td>
          <td>${fmtLoad(row.totalVolumeKg, 0)}</td>
          <td>${fmtLoad(row.estimatedOneRepMaxKg, 1)}</td>
          <td>${fmtNumber(row.avgMcvMmS, 1)} mm/s</td>
          <td class="group-separator">${row.echoRepCount || 0}</td>
          <td>${fmtLoad(row[bestEchoRepMedianLoadField()], 1)}</td>
          <td>${fmtLoad(row[bestEchoRepAverageLoadField()], 1)}</td>
          <td>${fmtLoad(row[bestEchoRepPeakLoadField()], 1)}</td>
          <td>${fmtEnergyJ(historyRepEnergy(row, true), 0)}</td>
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
      renderMuscleBreakdown(rows);
      const overlayTraces = selectedOverlayTraces(rows);
      renderRepEnergyChart(rows, overlayTraces);
      renderRepOverlay(rows, overlayTraces);
      renderPositionOverlay(rows, overlayTraces);
      renderLoadPositionPhaseCharts(rows, overlayTraces);
      renderHistoryTable(rows);
    }

    window.addEventListener("resize", () => {
      updateStickyOffsets();
      render();
    });
    setupControls();
    setupChartTooltips();
    setupMuscleBreakdownTooltips();
    render();
    saveDashboardCache();
  </script>
</body>
</html>
"""
    return (
        template
        .replace("__DATA__", data_json)
        .replace("__MUSCLE_MAP__", muscle_map_json)
        .replace("__REFINED_MUSCLE_MAP__", refined_muscle_map_json)
    )


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
    parser.add_argument(
        "--refined-muscle-map",
        default="refined_exercise_body_muscle_map.json",
        help="Exercise-to-body-muscles refined mapping JSON.",
    )
    args = parser.parse_args()

    backup_path = Path(args.backup)
    output_path = Path(args.out)
    tables_dir = Path(args.tables_dir)
    exercise_map_path = Path(args.exercise_map)
    refined_map_path = Path(args.refined_muscle_map)

    with backup_path.open("r", encoding="utf-8") as handle:
        raw_data = json.load(handle)
    if exercise_map_path.exists():
        with exercise_map_path.open("r", encoding="utf-8-sig") as handle:
            exercise_muscle_map = json.load(handle)
    else:
        exercise_muscle_map = []
    if refined_map_path.exists():
        with refined_map_path.open("r", encoding="utf-8-sig") as handle:
            refined_muscle_map = json.load(handle)
    else:
        refined_muscle_map = []

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
                if key not in {"timeSec", "perCableLoadKg", "totalLoadKg", "positionA", "positionB", "position", "velocity"}
            }
            for trace in dashboard["repTraces"]
        ],
        [
            key
            for key in dashboard["repTraces"][0].keys()
            if key not in {"timeSec", "perCableLoadKg", "totalLoadKg", "positionA", "positionB", "position", "velocity"}
        ],
    )

    data_json = json.dumps(dashboard, separators=(",", ":"), ensure_ascii=False)
    muscle_map_json = json.dumps(exercise_muscle_map, separators=(",", ":"), ensure_ascii=False)
    refined_muscle_map_json = json.dumps(refined_muscle_map, separators=(",", ":"), ensure_ascii=False)
    output_path.write_text(dashboard_html(data_json, muscle_map_json, refined_muscle_map_json), encoding="utf-8")

    print(f"Wrote {output_path.resolve()}")
    print(f"Wrote CSV tables to {tables_dir.resolve()}")
    print(
        f"Included {dashboard['metadata']['validSessions']} sessions, "
        f"{dashboard['metadata']['completedSets']} completed sets, "
        f"{dashboard['metadata']['repTraces']} rep traces."
    )


if __name__ == "__main__":
    main()
