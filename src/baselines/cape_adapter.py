"""Minimal CAPE-to-Speakeasy adapters for external baselines."""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional, Union

LOGGER = logging.getLogger(__name__)


def _extract_arg_values(raw_arguments) -> List[str]:
    """Extract argument values as simple strings, matching Speakeasy format."""
    args: List[str] = []
    if isinstance(raw_arguments, list):
        for arg in raw_arguments:
            if isinstance(arg, dict):
                # CAPE format: {"name": "...", "value": "..."}
                args.append(str(arg.get("value", "")))
            elif isinstance(arg, (str, int, float)):
                args.append(str(arg))
            else:
                args.append(str(arg))
    return args


def parse_api_calls_from_cape(report: Union[Dict, str]) -> List[Dict[str, object]]:
    """Extract API calls from a CAPE report in Speakeasy-compatible format.

    Preserves original API name casing and extracts ret_val to match
    the Speakeasy schema that Nebula's BPE tokenizer was trained on.
    """
    if isinstance(report, str):
        try:
            with open(report, "r", encoding="utf-8") as handle:
                report = json.load(handle)
        except Exception as exc:
            LOGGER.warning("Failed to load JSON from %s: %s", report, exc)
            return []

    if not isinstance(report, dict):
        return []

    api_calls: List[Dict[str, object]] = []
    behavior = report.get("behavior")
    if not isinstance(behavior, dict):
        return api_calls

    processes = behavior.get("processes", [])
    if not isinstance(processes, list):
        return api_calls

    for process in processes:
        if not isinstance(process, dict):
            continue
        calls = process.get("calls", [])
        if not isinstance(calls, list):
            continue
        for call in calls:
            if not isinstance(call, dict):
                continue
            api_name = call.get("api")
            if not api_name:
                continue
            args = _extract_arg_values(call.get("arguments", []))
            ret_val = call.get("return", "")

            api_calls.append(
                {
                    "api_name": str(api_name),  # preserve original casing
                    "args": args,
                    "ret_val": str(ret_val) if ret_val else "",
                }
            )

    return api_calls


def _extract_file_access(report: Dict) -> List[Dict[str, str]]:
    """Extract file access events from CAPE behavior."""
    events: List[Dict[str, str]] = []
    behavior = report.get("behavior", {})
    processes = behavior.get("processes", [])
    for proc in processes:
        if not isinstance(proc, dict):
            continue
        fa = proc.get("file_activities", {})
        if not isinstance(fa, dict):
            continue
        for event_type in ("read_files", "write_files", "delete_files"):
            files = fa.get(event_type, [])
            if not isinstance(files, list):
                continue
            event_label = event_type.replace("_files", "")
            for path in files:
                if isinstance(path, str):
                    events.append({"event": event_label, "path": path})
    return events


def _extract_registry_access(report: Dict) -> List[Dict[str, str]]:
    """Extract registry access events from CAPE behavior summary."""
    events: List[Dict[str, str]] = []
    behavior = report.get("behavior", {})
    summary = behavior.get("summary", {})
    if not isinstance(summary, dict):
        return events
    for event_type in ("regkey_read", "regkey_written", "regkey_deleted", "regkey_opened"):
        keys = summary.get(event_type, [])
        if not isinstance(keys, list):
            continue
        event_label = event_type.replace("regkey_", "")
        for path in keys:
            if isinstance(path, str):
                events.append({"event": event_label, "path": path})
    return events


def _extract_network_events(report: Dict) -> Dict[str, List[Dict]]:
    """Extract network events from CAPE report."""
    result: Dict[str, List[Dict]] = {}
    network = report.get("network", {})
    if not isinstance(network, dict):
        return result

    # DNS
    dns = network.get("dns", [])
    if isinstance(dns, list) and dns:
        result["dns"] = [
            {"query": str(d.get("request", ""))}
            for d in dns if isinstance(d, dict) and d.get("request")
        ]

    # TCP traffic
    tcp = network.get("tcp", [])
    if isinstance(tcp, list) and tcp:
        traffic = []
        for conn in tcp:
            if isinstance(conn, dict):
                entry = {}
                if conn.get("dst"):
                    entry["server"] = str(conn["dst"])
                if conn.get("dport"):
                    entry["port"] = str(conn["dport"])
                if entry:
                    traffic.append(entry)
        if traffic:
            result["traffic"] = traffic

    return result


def cape_to_speakeasy_format(report: Union[Dict, str]) -> Optional[List[Dict]]:
    """Convert a CAPE JSON report to Speakeasy entry_points list.

    Returns a LIST of entry points (not wrapped in a dict), so that
    Nebula.preprocess() correctly triggers filter_and_normalize_report().
    """
    if isinstance(report, str):
        try:
            with open(report, "r", encoding="utf-8") as handle:
                report = json.load(handle)
        except Exception as exc:
            LOGGER.warning("Failed to load JSON from %s: %s", report, exc)
            return None

    if not isinstance(report, dict):
        return None

    api_calls = parse_api_calls_from_cape(report)
    if not api_calls:
        return None

    # Build a single entry point with all data
    entry_point: Dict[str, object] = {"apis": api_calls}

    # Add file_access, registry_access, network_events if available
    file_access = _extract_file_access(report)
    if file_access:
        entry_point["file_access"] = file_access

    registry_access = _extract_registry_access(report)
    if registry_access:
        entry_point["registry_access"] = registry_access

    network_events = _extract_network_events(report)
    if network_events:
        entry_point["network_events"] = network_events

    # Return as LIST so Nebula.preprocess() calls filter_and_normalize_report()
    return [entry_point]
