from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

LOGGER = logging.getLogger(__name__)


DEFAULT_CATEGORY_ORDER = [
    "network",
    "file",
    "registry",
    "process",
    "memory",
    "crypto",
    "system",
    "api",
]


_CATEGORY_PATTERNS: Sequence[tuple[str, Sequence[str]]] = [
    (
        "network",
        (
            "http",
            "https",
            "internet",
            "wininet",
            "winhttp",
            "url",
            "socket",
            "connect",
            "send",
            "recv",
            "bind",
            "dns",
            "ftp",
            "winsock",
            "wsastartup",
            "inet",
        ),
    ),
    (
        "file",
        (
            "createfile",
            "writefile",
            "readfile",
            "deletefile",
            "copyfile",
            "movefile",
            "setfile",
            "openfile",
            "findfirstfile",
            "findnextfile",
            "getfile",
            "setendoffile",
            "flushfilebuffers",
        ),
    ),
    (
        "registry",
        (
            "reg",
            "ntsetvaluekey",
            "ntqueryvaluekey",
            "ntcreatekey",
            "ntopenkey",
            "ntdeletekey",
            "advapi32",
        ),
    ),
    (
        "process",
        (
            "createprocess",
            "createuserprocess",
            "createprocessasuser",
            "createremotethread",
            "shellexecute",
            "winexec",
            "terminateprocess",
            "process32",
            "getcommandline",
        ),
    ),
    (
        "memory",
        (
            "virtualalloc",
            "virtualprotect",
            "virtualfree",
            "heapalloc",
            "heapprepare",
            "heapfree",
            "mapviewoffile",
            "unmapviewoffile",
            "writeprocessmemory",
            "readprocessmemory",
        ),
    ),
    (
        "crypto",
        (
            "crypt",
            "bcrypt",
            "advcrypt",
            "hash",
            "md5",
            "sha",
        ),
    ),
    (
        "system",
        (
            "ntquerysystem",
            "ntsetinformation",
            "setwindowshook",
            "getsystemtime",
            "systemtime",
            "getversion",
            "gettickcount",
        ),
    ),
]


def _categorise_api(api_name: str) -> str:
    lowered = api_name.lower()
    for category, patterns in _CATEGORY_PATTERNS:
        if any(pattern in lowered for pattern in patterns):
            return category
    return "api"


def _normalise_entry_points(report: object) -> List[dict]:
    if isinstance(report, list):
        return [item for item in report if isinstance(item, dict)]
    if isinstance(report, dict):
        if "entry_points" in report and isinstance(report["entry_points"], list):
            return [item for item in report["entry_points"] if isinstance(item, dict)]
        if "behavior" in report and isinstance(report["behavior"], dict):
            processes = report["behavior"].get("processes", [])
            if isinstance(processes, list):
                normalised = []
                for proc in processes:
                    if not isinstance(proc, dict):
                        continue
                    apis = proc.get("calls") or proc.get("apis") or []
                    if isinstance(apis, list):
                        normalised.append({"ep_type": proc.get("process_name"), "apis": apis})
                if normalised:
                    return normalised
    return []


class SpeakeasyTextualizer:
    """Convert Speakeasy JSON into structured, normalized text for LEC."""

    def __init__(
        self,
        max_events: int = 800,
        max_args: int = 4,
        category_order: Sequence[str] | None = None,
        max_field_events: int = 256,
    ) -> None:
        self.max_events = max_events
        self.max_args = max_args
        self.category_order: List[str] = list(category_order or DEFAULT_CATEGORY_ORDER)
        self.max_field_events = max_field_events

    def textualize_file(self, path: Path, sample_id: str | None = None) -> str:
        try:
            with path.open("r", encoding="utf-8") as fh:
                report = json.load(fh)
        except Exception as exc:  # pragma: no cover - filesystem errors
            LOGGER.warning("读取 Speakeasy JSON 失败 %s: %s", path, exc)
            return ""
        sample = sample_id or self._derive_sample_id(path)
        return self.textualize(report, sample)

    def textualize(self, report: object, sample_id: str | None = None) -> str:
        entry_points = _normalise_entry_points(report)
        if not entry_points:
            return ""

        api_section, counts = self._format_api_section(entry_points)
        if not api_section:
            return ""

        file_section = self._format_file_access(entry_points)
        reg_section = self._format_registry_access(entry_points)
        net_section = self._format_network_events(entry_points)

        histogram_lines = [
            f"- {category}: {counts.get(category, 0)} events"
            for category in self.category_order
            if counts.get(category, 0)
        ]
        if not histogram_lines:
            histogram_lines = [f"- {category}: {count} events" for category, count in counts.most_common(5)]

        parts = [
            f"Sample {sample_id or ''}".strip(),
            "## Category Histogram",
            *histogram_lines,
            "## API Event Stream",
            *api_section,
        ]
        if file_section:
            parts.append("## File Activity")
            parts.extend(file_section)
        if reg_section:
            parts.append("## Registry Activity")
            parts.extend(reg_section)
        if net_section:
            parts.append("## Network Activity")
            parts.extend(net_section)
        return "\n".join(parts)

    def _derive_sample_id(self, path: Path) -> str:
        name = path.name
        for suffix in (".json", ".dat"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
        return name

    def _format_args(self, args: Iterable[object]) -> str:
        formatted = []
        for idx, arg in enumerate(args or []):
            if idx >= self.max_args:
                break
            formatted.append(_normalize_value(arg))
        return " | ".join(filter(None, formatted))

    def _format_api_section(self, entry_points: List[dict]) -> Tuple[List[str], Counter[str]]:
        records: List[str] = []
        counts: Counter[str] = Counter()
        for ep_index, entry in enumerate(entry_points):
            ep_label = entry.get("ep_type") or entry.get("process_name") or f"entry_{ep_index}"
            apis = entry.get("apis") or entry.get("calls") or []
            if not isinstance(apis, list):
                continue
            for api in apis:
                if not isinstance(api, dict):
                    continue
                name = str(api.get("api_name") or api.get("api") or "").strip()
                if not name:
                    continue
                category = _categorise_api(name)
                counts[category] += 1
                if self.max_events and len(records) >= self.max_events:
                    continue
                args = api.get("args") or api.get("arguments") or []
                formatted_args = self._format_args(args)
                record = f"[{category}] {name.lower()} ({ep_label})"
                if formatted_args:
                    record += f" args: {formatted_args}"
                ret_val = api.get("ret_val")
                if ret_val not in (None, "", "0x0"):
                    record += f" ret={_normalize_value(ret_val)}"
                records.append(record)
        lines = [f"{idx}. {record}" for idx, record in enumerate(records, start=1)]
        return lines, counts

    def _format_file_access(self, entry_points: List[dict]) -> List[str]:
        events: List[dict] = []
        for entry in entry_points:
            accesses = entry.get("file_access") or []
            if isinstance(accesses, list):
                events.extend(accesses)
        if not events:
            return []
        lines = []
        for idx, item in enumerate(events[: self.max_field_events], start=1):
            if not isinstance(item, dict):
                continue
            path = _normalize_value(item.get("path"))
            event = str(item.get("event", "")).lower() or "unknown"
            flags = item.get("access_flags") or item.get("open_flags") or []
            if isinstance(flags, list):
                flag_str = "|".join(flags)
            elif flags:
                flag_str = str(flags)
            else:
                flag_str = "-"
            lines.append(f"{idx}. [FILE] {event} {path} flags={flag_str}")
        return lines

    def _format_registry_access(self, entry_points: List[dict]) -> List[str]:
        events: List[dict] = []
        for entry in entry_points:
            reg = entry.get("registry_access") or []
            if isinstance(reg, list):
                events.extend(reg)
        if not events:
            return []
        lines = []
        for idx, item in enumerate(events[: self.max_field_events], start=1):
            if not isinstance(item, dict):
                continue
            path = _normalize_value(item.get("path"))
            event = str(item.get("event", "")).lower() or "unknown"
            data = item.get("data")
            data_str = f" data={_normalize_value(data)}" if data else ""
            lines.append(f"{idx}. [REG] {event} {path}{data_str}")
        return lines

    def _format_network_events(self, entry_points: List[dict]) -> List[str]:
        traffic_records: List[dict] = []
        for entry in entry_points:
            network = entry.get("network_events")
            traffic: List[dict] = []
            if isinstance(network, dict):
                traffic = network.get("traffic") or []
            elif isinstance(network, list):
                for item in network:
                    if isinstance(item, dict):
                        traffic.extend(item.get("traffic") or [])
            if isinstance(traffic, list):
                traffic_records.extend(traffic)
        if not traffic_records:
            return []
        lines = []
        for idx, item in enumerate(traffic_records[: self.max_field_events], start=1):
            if not isinstance(item, dict):
                continue
            server = _normalize_host(item.get("server") or item.get("domain") or "")
            port = item.get("port")
            proto = (item.get("proto") or item.get("protocol") or "").lower()
            method = item.get("method") or ""
            uri = _normalize_value(item.get("uri") or item.get("resource") or "")
            summary = f"{idx}. [NET] {server}"
            if port:
                summary += f":{port}"
            if proto:
                summary += f" proto={proto}"
            if method:
                summary += f" method={method}"
            if uri:
                summary += f" uri={uri}"
            lines.append(summary)
        return lines


_HEX64 = re.compile(r"\b[0-9a-fA-F]{64}\b")
_HEX40 = re.compile(r"\b[0-9a-fA-F]{40}\b")
_HEX32 = re.compile(r"\b[0-9a-fA-F]{32}\b")
_DOMAIN_RE = re.compile(r"\b([a-z0-9][a-z0-9-]{0,61}[a-z0-9]\.)+(com|net|org|gov|edu|uk|cn|ru|de|jp|io|co)\b")
_IP_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
_PRIVATE_IP = re.compile(r"^10\.|^172\.(1[6-9]|2[0-9]|3[01])\.|^192\.168\.")
_LOOPBACK_IP = re.compile(r"^127\.")
_ENV_MAP = {
    "%systemdrive%": "<drive>",
    "%systemroot%": "<drive>\\windows",
    "%windir%": "<drive>\\windows",
    "%programfiles%": "<drive>\\program files",
    "%programfiles(x86)%": "<drive>\\program files (x86)",
    "%appdata%": "<drive>\\users\\<user>\\appdata\\roaming",
    "%localappdata%": "<drive>\\users\\<user>\\appdata\\local",
    "%temp%": "<drive>\\users\\<user>\\appdata\\local\\temp",
    "%tmp%": "<drive>\\users\\<user>\\appdata\\local\\temp",
    "%userprofile%": "<drive>\\users\\<user>",
}
_DEFAULT_USERS = {"administrator", "public", "default"}


def _normalize_path(value: str) -> str:
    if not value:
        return value
    text = value.replace("/", "\\").lower()
    for env_name, replacement in _ENV_MAP.items():
        text = text.replace(env_name, replacement)
    text = re.sub(r"^[a-z]:", "<drive>", text)
    text = re.sub(r"\\\\[a-z0-9._-]+\\", r"<net>\\", text)
    text = re.sub(r"\\[\.\?]\\volume\{[a-f0-9-]{36}\}", "<drive>", text)
    text = re.sub(
        r"users\\([^\\]+)",
        lambda m: f"users\\<user>" if m.group(1) not in _DEFAULT_USERS else m.group(0),
        text,
    )
    return text


def _normalize_ip(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        ip = match.group(0)
        if _LOOPBACK_IP.match(ip):
            return "<loopIP>"
        if _PRIVATE_IP.match(ip):
            return "<prvIP>"
        return "<pubIP>"

    return _IP_RE.sub(repl, text)


def _normalize_domain(text: str) -> str:
    return _DOMAIN_RE.sub("<domain>", text)


def _normalize_hash(text: str) -> str:
    text = _HEX64.sub("<sha256>", text)
    text = _HEX40.sub("<sha1>", text)
    text = _HEX32.sub("<md5>", text)
    return text


def _normalize_value(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    lowered = text.lower()
    if "\\" in lowered or lowered.startswith("%"):
        text = _normalize_path(lowered)
    else:
        text = lowered
    text = _normalize_ip(text)
    text = _normalize_domain(text)
    text = _normalize_hash(text)
    return text


def _normalize_host(value: object) -> str:
    text = _normalize_value(value)
    if not text:
        return "<unknown>"
    return text


def batch_textualize(paths: Iterable[Path], *, max_events: int = 800, max_args: int = 4) -> List[str]:
    textualizer = SpeakeasyTextualizer(max_events=max_events, max_args=max_args)
    return [textualizer.textualize_file(path) for path in paths]
