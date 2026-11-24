"""Request templates with ``{placeholder}`` substitution.

Placeholders: ``{text}``, ``{audio}`` (raw bytes in ``dataset.audio_in`` format), ``{audio_wav}`` (WAV bytes),
``{messages}`` (list of {role, content}), ``{system_prompt}``, ``{uuid}`` (random hex per call),
``{session}`` (session/worker index), ``{model}`` (the service's model).

A string that is exactly one placeholder is replaced by the object itself (bytes, list, int, ...).
Otherwise each known placeholder inside the string is replaced by ``str(value)``; unknown ``{names}``
(and any other braces) are left untouched, so JSON-like text in templates is safe.
"""
from __future__ import annotations

import base64
import re

_WHOLE = re.compile(r"^\{(\w+)\}$")
_ANY = re.compile(r"\{(\w+)\}")


def render(template, variables: dict):
    """Recursively render dicts / lists / strings; other values are returned unchanged."""
    if isinstance(template, dict):
        return {k: render(v, variables) for k, v in template.items()}
    if isinstance(template, (list, tuple)):
        return [render(v, variables) for v in template]
    if not isinstance(template, str):
        return template
    m = _WHOLE.match(template)
    if m and m.group(1) in variables:
        return variables[m.group(1)]

    def sub(match):
        name = match.group(1)
        if name not in variables:
            return match.group(0)
        value = variables[name]
        return "" if value is None else str(value)

    return _ANY.sub(sub, template)


def placeholders(template) -> set[str]:
    """Names of all placeholders used anywhere in a template."""
    if isinstance(template, dict):
        return set().union(*(placeholders(v) for v in template.values())) if template else set()
    if isinstance(template, (list, tuple)):
        return set().union(*(placeholders(v) for v in template)) if template else set()
    if isinstance(template, str):
        return set(_ANY.findall(template))
    return set()


def drop_none(obj):
    """Remove keys whose value is None (e.g. a placeholder with no data for this call)."""
    if isinstance(obj, dict):
        return {k: drop_none(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [drop_none(v) for v in obj if v is not None]
    return obj


def jsonable(obj):
    """Make a rendered template JSON-serializable: bytes -> base64 text (also what protobuf JSON expects)."""
    if isinstance(obj, (bytes, bytearray)):
        return base64.b64encode(bytes(obj)).decode("ascii")
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    return obj
