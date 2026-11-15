"""Build JSON-schema function declarations from Python signatures + Google-style docstrings."""
from __future__ import annotations

import inspect
import re
import typing

_PY2JSON = {float: "number", int: "integer", str: "string", bool: "boolean", dict: "object", list: "array"}


def _json_type(tp) -> dict:
    origin = typing.get_origin(tp) or tp
    if origin is list:
        args = typing.get_args(tp)
        item = _json_type(args[0]) if args else {}
        return {"type": "array", "items": item or {"type": "object"}}
    if origin is dict:
        return {"type": "object"}
    return {"type": _PY2JSON.get(origin, "string")}


def parse_docstring(doc: str) -> tuple[str, dict[str, str]]:
    doc = inspect.cleandoc(doc or "")
    summary, _, rest = doc.partition("\nArgs:")
    params: dict[str, str] = {}
    for m in re.finditer(r"^\s{2,}(\w+):\s*(.+?)(?=^\s{2,}\w+:|\Z)", rest, re.M | re.S):
        params[m.group(1)] = " ".join(m.group(2).split())
    return " ".join(summary.split()), params


def function_schema(fn) -> dict:
    sig = inspect.signature(fn)
    hints = typing.get_type_hints(fn)
    summary, pdocs = parse_docstring(fn.__doc__)
    props, required = {}, []
    for name, p in sig.parameters.items():
        prop = _json_type(hints.get(name, str))
        if name in pdocs:
            prop["description"] = pdocs[name]
        if p.default is inspect.Parameter.empty:
            required.append(name)
        else:
            prop["default"] = p.default
        props[name] = prop
    return {
        "name": fn.__name__,
        "description": summary,
        "parameters": {"type": "object", "properties": props, "required": required},
    }
