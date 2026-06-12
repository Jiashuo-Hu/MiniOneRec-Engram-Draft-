import ast
import hashlib
import json
import os
import re
from collections import OrderedDict
from typing import Iterable, List, Sequence


SID_PATTERN = re.compile(r"<([abc])_(\d+)>")
PAD_SID_ID = -1
DEFAULT_HASH_BUCKETS = 262144


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def parse_sid_components(sid: str) -> List[str]:
    """Return SID components such as ['<a_1>', '<b_2>', '<c_3>']."""
    if sid is None:
        return []
    sid = str(sid).strip()
    comps = [f"<{name}_{idx}>" for name, idx in SID_PATTERN.findall(sid)]
    return comps


def normalize_sid(sid: str) -> str:
    comps = parse_sid_components(sid)
    return "".join(comps) if comps else str(sid).strip()


def sid_prefixes(sid: str) -> List[str]:
    comps = parse_sid_components(sid)
    return ["".join(comps[:i]) for i in range(1, len(comps) + 1)]


def safe_parse_sid_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [normalize_sid(x) for x in value]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [normalize_sid(x) for x in parsed]
    except (ValueError, SyntaxError):
        pass
    return [normalize_sid(x) for x in text.split(",") if x.strip()]


def sid_key(sid: str) -> str:
    return f"sid:{normalize_sid(sid)}"


def bigram_key(left: str, right: str) -> str:
    return f"bigram:{normalize_sid(left)}|{normalize_sid(right)}"


def prefix_key(prefix: str) -> str:
    return f"prefix:{normalize_sid(prefix)}"


def deterministic_hash(text: str, buckets: int = DEFAULT_HASH_BUCKETS) -> int:
    digest = hashlib.blake2b(str(text).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % int(buckets)


def ordered_unique(values: Iterable[str]) -> List[str]:
    seen = OrderedDict()
    for value in values:
        if value not in seen:
            seen[value] = None
    return list(seen.keys())


def parse_int_list(value) -> List[int]:
    if value is None or value == "":
        return []
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
    return [int(x.strip()) for x in str(value).split(",") if x.strip()]


def bool_arg(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y", "on"}


def build_trigger_keys(
    sid_history: Sequence[str],
    ngram_orders: Sequence[int],
    use_hierarchical_sid: bool = False,
) -> List[str]:
    keys = []
    hist = [normalize_sid(x) for x in sid_history if normalize_sid(x)]
    if 1 in ngram_orders:
        keys.extend(sid_key(sid) for sid in hist)
    if 2 in ngram_orders and len(hist) >= 2:
        keys.extend(bigram_key(hist[i - 1], hist[i]) for i in range(1, len(hist)))
    if use_hierarchical_sid:
        for sid in hist:
            keys.extend(prefix_key(prefix) for prefix in sid_prefixes(sid))
    return keys


def sid_history_to_ids(sid_history: Sequence[str], sid2idx: dict) -> List[int]:
    return [int(sid2idx.get(normalize_sid(sid), PAD_SID_ID)) for sid in sid_history]
