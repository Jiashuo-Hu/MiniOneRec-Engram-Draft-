import argparse
import inspect

from sid_engram.sid_utils import bool_arg


def _convert(value, default):
    if isinstance(default, bool):
        return bool_arg(value)
    if isinstance(default, int):
        return int(value)
    if isinstance(default, float):
        return float(value)
    return value


def run_cli(fn):
    sig = inspect.signature(fn)
    parser = argparse.ArgumentParser()
    for name, param in sig.parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        arg = f"--{name}"
        if param.default is inspect._empty:
            parser.add_argument(arg, required=True)
        else:
            parser.add_argument(arg, default=param.default)
    ns = parser.parse_args()
    kwargs = {}
    for name, param in sig.parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        value = getattr(ns, name)
        if param.default is not inspect._empty:
            value = _convert(value, param.default)
        kwargs[name] = value
    return fn(**kwargs)
