from __future__ import annotations

# Set native-thread preferences BEFORE importing the engine/numerical libraries.
# These are startup preferences, not a hard CPU or RAM limit.
import os

for _variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_variable] = "1"

os.environ["ARROW_DEFAULT_MEMORY_POOL"] = "system"

import gc
import hashlib
import importlib
import json
import math
import sys
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st


# ============================================================
# AEGIS ULTRA — APP V3.1
#
# Engine contract:
#   run_engine(input_dict) -> result_dict
#
# Expected existing result collections:
#   candidate_markets: list[dict]
#   recommendations: list[dict]
#   correct_scores: dict
#
# Unknown/new diagnostic fields remain available as raw JSON.
#
# Execution model:
#   - One active engine call per application process.
#   - Busy submissions are rejected, NOT silently queued.
#   - Results remain private to each Streamlit session.
#   - No automatic engine hot-reload.
#   - No automatic publication.
# ============================================================

APP_VERSION = "3.1.0"
ENGINE_MODULE = os.getenv(
    "AEGIS_ENGINE_MODULE", "aegisultra_enginev31"
).strip()

DEFAULT_API_URL = (
    "https://script.google.com/macros/s/"
    "AKfycbwhceZ9-Z-n4R7U-ctJsLrmZuSiy98MtCPgUIw26ZOM9tv2Y5WPt7af56mJJ8M4pbqfww/"
    "exec"
)

PERIODS = ("FT", "HT", "2H")
MARKETS = ("1X2", "AH", "OU", "HHAD", "TEAM_OU")
TIERS = ("OFFICIAL", "ALTERNATIVE", "CORRECT_SCORE")
MAX_INPUT_BYTES = 10 * 1024 * 1024

st.set_page_config(
    page_title="AEGIS ULTRA V3.1",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

if not hasattr(st, "fragment"):
    st.error("此版本需要 Streamlit 1.37 或以上。")
    st.stop()

st.markdown(
    """
    <style>
    .block-container {max-width:1500px; padding-top:1.3rem;}
    .stButton > button, .stFormSubmitButton > button {
        min-height:2.8rem; border-radius:10px;
    }
    textarea {
        font-family:Consolas,monospace !important;
        font-size:0.88rem !important;
    }
    [data-testid="stMetric"] {
        border:1px solid rgba(128,128,128,.22);
        border-radius:12px; padding:.75rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 1. Small, strict helpers
# ============================================================

def text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def as_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def records(value: Any) -> list[dict]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError, OverflowError):
        return None


def boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return text(value).lower() in {
        "true", "1", "yes", "y", "on", "heavy", "重心"
    }


def get_path(value: Any, *keys: str) -> Any:
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def first_value(*values: Any) -> Any:
    for value in values:
        if value is not None and not (
            isinstance(value, str) and not value.strip()
        ):
            return value
    return None


def item_id(item: dict) -> str:
    for key in ("id", "candidate_id", "market_id", "rec_id"):
        if text(item.get(key)):
            return text(item[key])
    return ""


def normalize_period(value: Any) -> str:
    value = text(value).upper().replace("-", "_").replace(" ", "_")
    aliases = {
        "FULL": "FT", "FULL_TIME": "FT", "FULLTIME": "FT",
        "MATCH": "FT", "90": "FT",
        "1H": "HT", "H1": "HT", "HALF_TIME": "HT",
        "HALFTIME": "HT", "FIRST_HALF": "HT",
        "H2": "2H", "SECOND_HALF": "2H",
    }
    return aliases.get(value, value)


def normalize_market(value: Any) -> str:
    value = (
        text(value).upper().replace("-", "_")
        .replace(" ", "_").replace("/", "_")
    )
    aliases = {
        "HAD": "1X2", "H2H": "1X2", "MATCH_ODDS": "1X2",
        "MONEYLINE": "1X2", "THREE_WAY": "1X2",
        "ASIAN_HANDICAP": "AH", "HANDICAP": "AH", "HDP": "AH",
        "OVER_UNDER": "OU", "TOTAL": "OU", "TOTALS": "OU",
        "GOALS": "OU", "HANDICAP_1X2": "HHAD",
        "HOME_HANDICAP_DRAW_AWAY": "HHAD",
        "TEAM_TOTAL": "TEAM_OU", "TEAM_TOTALS": "TEAM_OU",
        "TEAM_OVER_UNDER": "TEAM_OU",
        "SCORE": "CORRECT_SCORE", "CS": "CORRECT_SCORE",
    }
    return aliases.get(value, value)


def pct(value: Any) -> str:
    """Probability contract: fraction in [0, 1]. Never guess units."""
    value = number(value)
    if value is None:
        return "—"
    if not 0 <= value <= 1:
        return f"單位待確認：{value:g}"
    return f"{value:.2%}"


def ev_text(value: Any) -> str:
    """Expected return contract: fractional net return."""
    value = number(value)
    return "—" if value is None else f"{value:+.2%}"


def odds_text(value: Any) -> str:
    value = number(value)
    return "—" if value is None else f"{value:.3f}"


def scalar_summary(item: dict, field: str, statistic: str) -> Any:
    value = item.get(field)
    return value.get(statistic) if isinstance(value, dict) else value


def json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, set):
        return sorted(value, key=str)
    if isinstance(value, (datetime, Path)):
        return str(value)
    raise TypeError(f"不能轉換成 JSON：{type(value).__name__}")


def dumps(value: Any, *, pretty: bool = True) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
        default=json_default,
        allow_nan=False,
    )


def reject_constant(value: str) -> None:
    raise ValueError(f"JSON 不接受非有限數值：{value}")


def unique_object(pairs: list) -> dict:
    output = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"JSON 出現重複欄位：{key}")
        output[key] = value
    return output


def loads(value: str) -> Any:
    if len(value.encode("utf-8")) > MAX_INPUT_BYTES:
        raise ValueError("輸入 JSON 超過 10 MB。")
    return json.loads(
        value,
        parse_constant=reject_constant,
        object_pairs_hook=unique_object,
    )


def digest(value: Any) -> str:
    return hashlib.sha256(
        dumps(value, pretty=False).encode("utf-8")
    ).hexdigest()


def download_filename(prefix: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    return f"{prefix}_{stamp}.json"


def merge_settings(base: dict, override: dict) -> dict:
    output = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(output.get(key), dict):
            output[key] = merge_settings(output[key], value)
        else:
            output[key] = value
    return output


# ============================================================
# 2. Engine startup and process-wide execution lock
# ============================================================

def engine_file_manifest(paths: tuple[str, ...]) -> dict:
    output = {}
    for filename in paths:
        path = Path(filename)
        output[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return output


@st.cache_resource(show_spinner=False)
def engine_runtime(module_name: str) -> dict:
    # Cached engine module, NOT cached analysis results.
    module = importlib.import_module(module_name)

    if not callable(getattr(module, "run_engine", None)):
        raise RuntimeError(
            f"{module_name} 沒有可呼叫的 run_engine(input_dict)。"
        )

    paths = {str(Path(module.__file__).resolve())}

    # Include imported AEGIS engine modules and sibling engine files.
    for name, imported in list(sys.modules.items()):
        if name.startswith("aegisultra_engine"):
            filename = getattr(imported, "__file__", None)
            if filename and str(filename).endswith(".py"):
                paths.add(str(Path(filename).resolve()))

    for path in Path(module.__file__).resolve().parent.glob(
        "aegisultra_engine*.py"
    ):
        paths.add(str(path.resolve()))

    paths = tuple(sorted(paths))
    manifest = engine_file_manifest(paths)

    return {
        "module": module,
        "paths": paths,
        "manifest": manifest,
        "fingerprint": digest(manifest),
        "name": text(getattr(module, "ENGINE_NAME", module_name)),
        "version": text(getattr(module, "ENGINE_VERSION", "Unknown")),
    }


@st.cache_resource(show_spinner=False)
def calculation_gate():
    # Independent of engine filename/version: one gate for this app process.
    return threading.BoundedSemaphore(1)


try:
    RUNTIME = engine_runtime(ENGINE_MODULE)
except Exception as error:
    st.error(f"無法載入引擎：{error}")
    st.info(
        "請將 aegisultra_enginev31.py 放在 app 同一目錄，"
        "並保留它需要的所有引擎依賴。此 app 不會自動改用 V3。"
    )
    st.code(traceback.format_exc(), language="text")
    st.stop()


def require_current_engine() -> None:
    current = engine_file_manifest(RUNTIME["paths"])
    if current != RUNTIME["manifest"]:
        raise RuntimeError(
            "引擎檔案已更改，但伺服器仍持有舊模組。"
            "請先在 Community Cloud reboot app，再重新分析。"
        )


# ============================================================
# 3. Session state
# ============================================================

DEFAULT_SETTINGS = {
    "minimum_odds": 1.50,
    "maximum_odds": None,
    "max_recommendations": 3,
    "minimum_shortlist_hit_probability": 0.50,
    "correct_score_count": 2,
    "devig_methods": ["MULTIPLICATIVE", "POWER"],
    "forecast_priors": [
        "INDEPENDENT_POISSON",
        "DIXON_COLES",
        "COM_POISSON_SHARED",
    ],
    "audit_priors": ["FLAT_GRID_MAXENT"],
    "primary_source": "pinnacle",
    "ev_rejection_floor": None,
    "require_market_supported_for_shortlist": False,
    "support_thresholds": {
        period: {
            "max_prior_spread": None,
            "max_feasible_width": None,
            "max_maxent_gap": None,
            "max_hidden_line_absolute_error": None,
        }
        for period in ("FT", "HT")
    },
    "features": {
        "quality_gate": True,
        "stress_audit": True,
        "family_out_audit": True,
        "adaptive_grids": True,
        "ht_ft_coherence": True,
        "feasible_bounds": True,
        "maxent_audit": True,
    },
}


def example_input() -> dict:
    return {
        "match": {
            "name": "主隊 vs 客隊",
            "home": "主隊",
            "away": "客隊",
            "competition": "示例賽事",
            "kickoff": "",
            "snapshot_time": "",
        },
        "sharp_books": [{
            "key": "pinnacle",
            "title": "Pinnacle",
            "markets": {
                "FT": {
                    "1X2": {"home": 2.12, "draw": 3.35, "away": 3.55},
                    "AH": [{"line": -0.25, "home": 1.95, "away": 1.95}],
                    "OU": [{"line": 2.25, "over": 1.92, "under": 1.98}],
                    "HHAD": [],
                    "TEAM_OU": [],
                },
            },
        }],
        "hkjc_markets": [{
            "id": "M001",
            "period": "FT",
            "market": "AH",
            "selection": "HOME",
            "line": -0.25,
            "odds": 1.95,
            "label": "主隊 全場 AH -0.25",
        }],
        "settings": deepcopy(DEFAULT_SETTINGS),
    }


SESSION_DEFAULTS = {
    "a31_result": None,
    "a31_input": None,
    "a31_hash": None,
    "a31_engine_fingerprint": None,
    "a31_meta": None,
    "a31_error": None,
    "a31_prepared": None,
    "a31_editor_saved": None,
    "a31_publish_meta": None,
    "a31_portal_response": None,
    "a31_publish_uncertain": False,
    "a31_json_saved": dumps(example_input()),
}

for key, value in SESSION_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = deepcopy(value)

S = st.session_state


def clear_results() -> None:
    for key in (
        "a31_result", "a31_input", "a31_hash",
        "a31_engine_fingerprint", "a31_meta", "a31_error",
        "a31_prepared", "a31_editor_saved", "a31_publish_meta",
        "a31_portal_response",
    ):
        S[key] = None

    S.a31_publish_uncertain = False

    for key in list(S.keys()):
        if str(key).startswith(("a31pub_", "a31export_")):
            del S[key]

    gc.collect()


# ============================================================
# 4. Bulk parsers — same home-line convention as original app
# ============================================================

SKIPS = {"", "-", "x", "na", "n/a", "none", "null"}


def tokens(value: str) -> list[str]:
    return value.replace(",", " ").replace("\t", " ").split()


def price(value: Any, label: str, *, allow_skip: bool) -> float | None:
    if text(value).lower() in SKIPS:
        if allow_skip:
            return None
        raise ValueError(f"{label}：尖銳市場不可省略單邊賠率。")
    result = number(value)
    if result is None or result <= 1:
        raise ValueError(f"{label}：十進制賠率必須大於 1。")
    return result


def market_line(value: Any, market: str, label: str) -> float:
    result = number(value)
    if result is None:
        raise ValueError(f"{label}：無效盤口。")

    multiplier = 1 if market == "HHAD" else 4
    if not math.isclose(
        result * multiplier,
        round(result * multiplier),
        rel_tol=0,
        abs_tol=1e-8,
    ):
        unit = "整數" if market == "HHAD" else "0.25"
        raise ValueError(f"{label}：盤口必須以 {unit} 為單位。")

    if market in {"OU", "TEAM_OU"} and result < 0:
        raise ValueError(f"{label}：入球線不可為負數。")

    return round(result * multiplier) / multiplier


def parse_market(
    raw: str, market: str, *, sharp: bool, label: str
) -> Any:
    if not text(raw):
        return None if market == "1X2" else []

    if market == "1X2":
        values = tokens(raw)
        if len(values) != 3:
            raise ValueError(f"{label}：請輸入 主／和／客 三個賠率。")
        output = {
            side: price(value, label, allow_skip=not sharp)
            for side, value in zip(("home", "draw", "away"), values)
        }
        return output if any(v is not None for v in output.values()) else None

    output, seen = [], set()

    for row_number, row in enumerate(raw.splitlines(), start=1):
        if not row.strip():
            continue

        values = tokens(row)
        expected = 4 if market in {"HHAD", "TEAM_OU"} else 3
        row_label = f"{label} 第 {row_number} 行"

        if len(values) != expected:
            raise ValueError(f"{row_label}：需要 {expected} 個欄位。")

        team = ""
        if market == "TEAM_OU":
            team = values.pop(0).upper()
            if team not in {"HOME", "AWAY"}:
                raise ValueError(f"{row_label}：球隊必須為 HOME/AWAY。")

        line = market_line(values.pop(0), market, row_label)
        identity = (team, line)

        if identity in seen:
            raise ValueError(f"{row_label}：重複盤口。")
        seen.add(identity)

        sides = (
            ("home", "draw", "away") if market == "HHAD"
            else ("home", "away") if market == "AH"
            else ("over", "under")
        )

        parsed = {
            side: price(value, row_label, allow_skip=not sharp)
            for side, value in zip(sides, values)
        }

        if all(value is None for value in parsed.values()):
            raise ValueError(f"{row_label}：最少需要一個賠率。")

        parsed["line"] = line
        if team:
            parsed["team"] = team
        output.append(parsed)

    return output


def bulk_market_fields(prefix: str) -> dict:
    raw = {}
    for period in ("FT", "HT"):
        with st.expander(f"{period} 市場輸入", expanded=period == "FT"):
            period_raw = {}
            for market in MARKETS:
                if period == "HT" and market == "HHAD":
                    period_raw[market] = ""
                    continue

                hints = {
                    "1X2": "2.12 3.35 3.55",
                    "AH": "-0.25 1.95 1.95",
                    "OU": "2.25 1.92 1.98",
                    "HHAD": "-1 3.60 3.75 1.72",
                    "TEAM_OU": "HOME 1.50 2.05 1.78",
                }

                period_raw[market] = st.text_area(
                    f"{period} {market}",
                    placeholder=hints[market],
                    height=90,
                    key=f"{prefix}_{period}_{market}",
                )
            raw[period] = period_raw
    return raw


def build_bulk_input(
    match: dict,
    sources: list[dict],
    target_raw: dict,
    settings: dict,
    primary_index: int,
) -> dict:
    if not text(match["home"]) or not text(match["away"]):
        raise ValueError("主隊及客隊不可留空。")
    if text(match["home"]).casefold() == text(match["away"]).casefold():
        raise ValueError("主隊及客隊不可相同。")

    books, seen = [], set()

    for source in sources:
        key = text(source["key"]).lower()
        if not key or key in seen:
            raise ValueError("尖銳來源識別碼不可留空或重複。")
        seen.add(key)

        book = {
            "key": key,
            "title": text(source["title"]) or key,
            "timestamp": match["snapshot_time"],
            "markets": {},
        }

        for period, raw in source["raw"].items():
            parsed = {
                market: parse_market(
                    value, market, sharp=True, label=f"{key} {period} {market}"
                )
                for market, value in raw.items()
            }
            if any(bool(value) for value in parsed.values()):
                book["markets"][period] = parsed

        if "FT" not in book["markets"]:
            raise ValueError(f"{key} 最少需要一個 FT 市場。")
        books.append(book)

    candidates = []

    for period, raw in target_raw.items():
        for market, value in raw.items():
            parsed = parse_market(
                value, market, sharp=False, label=f"HKJC {period} {market}"
            )
            ladder = [parsed] if market == "1X2" and parsed else (
                parsed if isinstance(parsed, list) else []
            )

            for row in ladder:
                sides = (
                    ("home", "draw", "away")
                    if market in {"1X2", "HHAD"}
                    else ("home", "away") if market == "AH"
                    else ("over", "under")
                )

                for side in sides:
                    if row.get(side) is None:
                        continue

                    item = {
                        "id": f"M{len(candidates) + 1:03d}",
                        "period": period,
                        "market": market,
                        "selection": side.upper(),
                        "odds": row[side],
                    }

                    if "line" in row:
                        # Stored AH line is always the HOME line.
                        item["line"] = row["line"]

                    if row.get("team"):
                        item["team"] = row["team"]
                        item["market_scope"] = row["team"]

                    display_line = row.get("line")
                    if market == "AH" and side == "away":
                        display_line = -display_line

                    team_label = (
                        match["home"] if row.get("team") == "HOME"
                        else match["away"] if row.get("team") == "AWAY"
                        else ""
                    )
                    side_label = {
                        "home": match["home"],
                        "away": match["away"],
                        "draw": "和",
                        "over": "大",
                        "under": "細",
                    }[side]

                    item["label"] = " ".join(
                        part for part in (
                            period, market, team_label, side_label,
                            "" if display_line is None else f"{display_line:+g}",
                        ) if part
                    )
                    candidates.append(item)

    settings = deepcopy(settings)
    settings["primary_source"] = books[primary_index]["key"]

    return {
        "match": match,
        "sharp_books": books,
        "hkjc_markets": candidates,
        "settings": settings,
    }


def validate_input(data: Any) -> dict:
    if not isinstance(data, dict):
        raise ValueError("JSON 最外層必須是物件。")

    if not isinstance(data.get("match"), dict):
        raise ValueError("缺少 match 物件。")

    books = data.get("sharp_books")
    if not isinstance(books, list) or not books:
        raise ValueError("sharp_books 必須是非空清單。")
    if any(not isinstance(book, dict) for book in books):
        raise ValueError("sharp_books 每項必須是物件。")

    candidates = data.get("hkjc_markets")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("最少需要一個 HKJC 候選盤。")

    seen = set()
    for item in candidates:
        if not isinstance(item, dict):
            raise ValueError("hkjc_markets 每項必須是物件。")

        identity = item_id(item)
        if not identity or identity in seen:
            raise ValueError("每個 HKJC 候選盤必須有唯一 ID。")
        seen.add(identity)

        if normalize_period(item.get("period")) not in PERIODS:
            raise ValueError(f"{identity} 必須明確指定 FT、HT 或 2H。")

        market = normalize_market(item.get("market"))
        if market not in MARKETS:
            raise ValueError(f"{identity}：不支援的候選市場 {market}。")

        selection = text(item.get("selection")).upper()
        allowed = (
            {"HOME", "DRAW", "AWAY"} if market in {"1X2", "HHAD"}
            else {"HOME", "AWAY"} if market == "AH"
            else {"OVER", "UNDER"}
        )
        if selection not in allowed:
            raise ValueError(f"{identity}：selection 與市場不相符。")

        price(
            first_value(item.get("odds"), item.get("hkjc_odds")),
            identity,
            allow_skip=False,
        )

        if market != "1X2":
            market_line(item.get("line"), market, identity)

        if market == "TEAM_OU":
            scope = text(first_value(
                item.get("team"), item.get("market_scope")
            )).upper()
            if scope not in {"HOME", "AWAY"}:
                raise ValueError(f"{identity}：TEAM_OU 缺少 HOME/AWAY。")

    settings = data.get("settings", {})
    if not isinstance(settings, dict):
        raise ValueError("settings 必須是物件。")

    minimum = number(settings.get("minimum_odds"))
    maximum = number(settings.get("maximum_odds"))
    if minimum is not None and maximum is not None and maximum < minimum:
        raise ValueError("最高賠率不可低於最低賠率。")

    # Confirm strict JSON serializability; do not rewrite engine settings.
    dumps(data, pretty=False)
    return data


# ============================================================
# 5. Controlled engine execution
# ============================================================

def execute(data: dict) -> str:
    require_current_engine()
    canonical = dumps(data, pretty=False)
    analysis_hash = digest({
        "input": data,
        "engine": RUNTIME["fingerprint"],
        "app": APP_VERSION,
    })

    if S.a31_hash == analysis_hash and S.a31_result is not None:
        return "已使用本工作階段的相同分析結果，沒有重跑引擎。"

    gate = calculation_gate()

    if not gate.acquire(blocking=False):
        raise RuntimeError(
            "伺服器正在分析另一場賽事。今次沒有開始計算，也沒有排隊。"
            "請稍後再次按開始分析；目前已完成的結果不受影響。"
        )

    try:
        # Do not keep two full engine results in this session.
        clear_results()

        submitted = loads(canonical)
        del canonical

        started = time.perf_counter()
        result = RUNTIME["module"].run_engine(submitted)
        elapsed = time.perf_counter() - started

        if not isinstance(result, dict):
            raise TypeError("引擎輸出必須是 dict。")

        # Detect source changes during a long calculation.
        require_current_engine()

        snapshot = result.get("input_snapshot")
        if not isinstance(snapshot, dict):
            snapshot = loads(dumps(data, pretty=False))

        S.a31_result = result
        S.a31_input = snapshot
        S.a31_hash = analysis_hash
        S.a31_engine_fingerprint = RUNTIME["fingerprint"]
        S.a31_meta = {
            "engine_module": ENGINE_MODULE,
            "engine_version": RUNTIME["version"],
            "engine_fingerprint": RUNTIME["fingerprint"],
            "analysis_hash": analysis_hash,
            "elapsed_seconds": elapsed,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "submitted_candidates": len(data["hkjc_markets"]),
        }
        return f"分析已完成，用時 {elapsed:.1f} 秒。"

    finally:
        gate.release()
        gc.collect()


# ============================================================
# 6. Portal API
# ============================================================

def portal_setting(name: str, environment: str, default: str = "") -> str:
    try:
        value = text(st.secrets["portal_api"][name])
        if value:
            return value
    except Exception:
        pass
    return text(os.getenv(environment)) or default


def portal_request(payload: dict) -> dict:
    url = portal_setting("url", "AEGIS_API_URL", DEFAULT_API_URL)
    token = portal_setting("token", "AEGIS_API_TOKEN")

    parsed_url = urllib.parse.urlparse(url)
    if parsed_url.scheme != "https" or not parsed_url.netloc:
        raise ValueError("Portal API URL 必須是有效 HTTPS URL。")
    if not token:
        raise ValueError("缺少 portal_api.token / AEGIS_API_TOKEN。")

    body = dict(payload)
    body["token"] = token

    request = urllib.request.Request(
        url,
        data=dumps(body, pretty=False).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json",
            "User-Agent": f"AEGIS-ULTRA/{APP_VERSION}",
        },
    )

    def redact(value: str) -> str:
        return value.replace(token, "[REDACTED]")[:1200]

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            response_text = response.read(2 * 1024 * 1024).decode(
                "utf-8-sig", errors="replace"
            )
    except urllib.error.HTTPError as error:
        details = error.read(4096).decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Portal HTTP {error.code}：{redact(details)}"
        ) from None
    except Exception as error:
        raise RuntimeError(f"Portal 連線失敗：{redact(str(error))}") from None

    try:
        result = json.loads(response_text)
    except json.JSONDecodeError:
        raise RuntimeError(
            "Portal 未回傳有效 JSON：" + redact(response_text)
        ) from None

    if not isinstance(result, dict) or result.get("ok") is not True:
        raise RuntimeError("Portal 拒絕要求：" + redact(response_text))

    # Do not retain an echoed credential.
    result.pop("token", None)
    return result


# ============================================================
# 7. Candidate/result adapters
# ============================================================

def candidates_for_display(result: dict, snapshot: dict) -> list[dict]:
    original = {
        item_id(item): item
        for item in records(snapshot.get("hkjc_markets"))
        if item_id(item)
    }
    shortlisted = {
        item_id(item): item
        for item in records(result.get("recommendations"))
        if item_id(item)
    }

    output, seen = [], set()

    source = records(result.get("candidate_markets"))
    source += records(result.get("recommendations"))

    for index, item in enumerate(source):
        identity = item_id(item)
        marker = identity or f"anonymous_{index}"

        if marker in seen:
            continue
        seen.add(marker)

        merged = dict(original.get(identity, {}))

        # Keep input identity fields when result supplies null/blank values.
        merged.update({
            key: value for key, value in item.items()
            if value is not None and value != ""
        })

        if identity in shortlisted:
            merged.update({
                key: value for key, value in shortlisted[identity].items()
                if value is not None and value != ""
            })

        merged["_shortlisted"] = (
            identity in shortlisted
            or boolean(item.get("shortlisted"))
            or boolean(item.get("official"))
        )
        merged["_editor_key"] = marker
        output.append(merged)

    return output


def candidate_row(item: dict) -> dict:
    reasons = []
    for field in (
        "shortlist_exclusion_reasons",
        "official_exclusion_reasons",
        "exclusion_reasons",
    ):
        value = item.get(field)
        if isinstance(value, list):
            reasons.extend(text(reason) for reason in value)

    return {
        "ID": item_id(item),
        "候選": boolean(item.get("_shortlisted")),
        "時段": normalize_period(item.get("period")) or "MISSING",
        "市場": normalize_market(item.get("market")),
        "項目": text(first_value(item.get("label"), item_id(item))),
        "賠率": odds_text(first_value(
            item.get("hkjc_odds"), item.get("odds")
        )),
        "情境最低命中": pct(get_path(item, "probability", "hit", "minimum")),
        "情境中位命中": pct(get_path(item, "probability", "hit", "median")),
        "最低不輸率": pct(get_path(item, "probability", "nonloss", "minimum")),
        "最高全輸率": pct(get_path(item, "probability", "full_loss", "maximum")),
        "最低EV": ev_text(scalar_summary(item, "expected_return", "minimum")),
        "市場支持": text(get_path(item, "market_support", "status")),
        "價格": text(item.get("price_status")),
        "排除原因": "；".join(dict.fromkeys(reasons)),
    }


def commentary(item: dict) -> str:
    for field in ("commentary", "analysis", "summary", "explanation", "reason"):
        value = item.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def conflict_ids(item: dict) -> list[str]:
    output = []
    explicit = item.get("conflict_ids")
    if isinstance(explicit, str):
        output.extend(part.strip() for part in explicit.split(",") if part.strip())
    elif isinstance(explicit, list):
        output.extend(text(part) for part in explicit if text(part))

    for conflict in item.get("conflicts_with") or []:
        if isinstance(conflict, dict):
            identity = text(conflict.get("selected_id")) or item_id(conflict)
        else:
            identity = text(conflict)
        if identity:
            output.append(identity)
    return list(dict.fromkeys(output))


def publication_sources(candidates: list[dict], result: dict) -> list[dict]:
    output = list(candidates)
    scores = records(as_dict(result.get("correct_scores")).get("recommendations"))

    for index, score in enumerate(scores):
        selection = text(score.get("score"))
        if not selection:
            continue

        output.append({
            **score,
            "id": f"CS_{selection}",
            "_editor_key": f"CS_{selection}_{index}",
            "_shortlisted": False,
            "_score": True,
            "label": f"波膽 {selection}",
            "period": "FT",
            "market": "CORRECT_SCORE",
            "selection": selection,
            "market_scope": "SCORE",
        })

    return output


def initial_editor_rows(sources: list[dict]) -> list[dict]:
    output = []
    for index, item in enumerate(sources, start=1):
        is_score = boolean(item.get("_score"))
        hit = (
            get_path(item, "probability", "minimum") if is_score
            else get_path(item, "probability", "hit", "minimum")
        )
        stars = number(item.get("stars"))
        output.append({
            "_key": item["_editor_key"],
            "發佈": False,
            "類別": (
                "CORRECT_SCORE" if is_score
                else "OFFICIAL" if item.get("_shortlisted")
                else "ALTERNATIVE"
            ),
            "排序": index,
            "時段": normalize_period(item.get("period")) or "MISSING",
            "標題": text(first_value(item.get("label"), item_id(item))),
            "市場": normalize_market(item.get("market")),
            "選擇": text(item.get("selection")),
            "盤口": text(item.get("line")),
            "賠率": "" if is_score else odds_text(first_value(
                item.get("hkjc_odds"), item.get("odds")
            )),
            "情境最低命中": pct(hit),
            "市場支持": text(get_path(item, "market_support", "status")),
            "星級": max(1, min(5, int(stars if stars is not None else 3))),
            "重心": boolean(item.get("is_heavy")),
            "短評": (
                commentary(item) if not is_score
                else "高風險波膽參考；不同波膽結果互相排斥。"
            ),
        })
    return output


def match_record(result: dict, snapshot: dict, override: str) -> dict:
    match = dict(as_dict(snapshot.get("match")))
    match.update({
        key: value
        for key, value in as_dict(result.get("match")).items()
        if value is not None and value != ""
    })

    explicit = text(override)
    if not explicit:
        explicit = text(first_value(
            match.get("match_id"), match.get("id"), match.get("fixture_id")
        ))

    home, away = text(match.get("home")), text(match.get("away"))
    kickoff = text(match.get("kickoff"))

    if not home or not away:
        raise ValueError("發佈需要明確主隊及客隊。")

    if not explicit:
        if not kickoff:
            raise ValueError(
                "沒有 kickoff。請在發佈表格填寫唯一 match_id，"
                "避免同一對球隊的不同賽事互相覆蓋。"
            )
        try:
            kickoff_dt = datetime.fromisoformat(kickoff.replace("Z", "+00:00"))
            if kickoff_dt.tzinfo is None:
                raise ValueError("missing timezone")
            normalized_kickoff = kickoff_dt.astimezone(timezone.utc).isoformat()
        except ValueError:
            raise ValueError(
                "自動 match_id 需要包含時區的 ISO kickoff；"
                "否則請人手指定唯一 match_id。"
            ) from None

        explicit = "match_" + digest([
            home.casefold(), away.casefold(),
            text(match.get("competition")).casefold(),
            normalized_kickoff,
        ])[:20]

    return {
        "match_id": explicit,
        "match_name": text(match.get("name")) or f"{home} vs {away}",
        "home_team": home,
        "away_team": away,
        "competition": text(match.get("competition")),
        "kickoff": kickoff,
        "final_score": "",
    }


def portal_record(item: dict, edit: dict, match_id: str, status: str) -> dict:
    is_score = boolean(item.get("_score"))
    period = normalize_period(item.get("period"))
    market = normalize_market(item.get("market"))
    selection = text(item.get("selection")).upper()
    scope = text(first_value(
        item.get("market_scope"), item.get("team"), item.get("scope")
    )).upper()

    if period not in PERIODS:
        raise ValueError(f"{edit['標題']} 缺少有效 period，拒絕假設為 FT。")
    if market not in {*MARKETS, "CORRECT_SCORE"} or not selection:
        raise ValueError(f"{edit['標題']} 缺少有效 market/selection。")
    if market == "TEAM_OU" and scope not in {"HOME", "AWAY"}:
        raise ValueError(f"{edit['標題']} 缺少 TEAM_OU scope。")

    line = None
    if market in {"AH", "OU", "HHAD", "TEAM_OU"}:
        line = market_line(item.get("line"), market, edit["標題"])

    odds = None if is_score else price(
        first_value(item.get("hkjc_odds"), item.get("odds")),
        edit["標題"],
        allow_skip=False,
    )

    # Scope is part of identity; source row number is deliberately not.
    rec_id = "rec_" + digest([
        match_id, period, market, selection, scope,
        "" if line is None else format(0.0 if line == 0 else line, ".12g"),
    ])[:22]

    if is_score:
        minimum = get_path(item, "probability", "minimum")
        median = get_path(item, "probability", "median")
        fair = first_value(
            item.get("central_fair_odds"),
            scalar_summary(item, "fair_odds", "maximum"),
        )
    else:
        minimum = get_path(item, "probability", "hit", "minimum")
        median = get_path(item, "probability", "hit", "median")
        fair = scalar_summary(item, "fair_odds", "maximum")

    def numeric_or_blank(value: Any) -> float | str:
        value = number(value)
        return value if value is not None else ""

    for probability in (minimum, median):
        value = number(probability)
        if value is not None and not 0 <= value <= 1:
            raise ValueError(
                f"{edit['標題']} 的概率不在 [0,1]；拒絕猜測百分比單位。"
            )

    tier = edit["類別"]
    if tier not in TIERS:
        raise ValueError("無效發佈類別。")
    if is_score != (tier == "CORRECT_SCORE"):
        raise ValueError(
            "波膽必須使用 CORRECT_SCORE；其他市場不可改為波膽類別。"
        )

    return {
        "rec_id": rec_id,
        "match_id": match_id,
        "tier": tier,
        "rank": 0,
        "rec_title": text(edit["標題"]),
        "market": market,
        "selection": selection,
        "line": "" if line is None else line,
        "odds": "" if odds is None else odds,
        # Preserve legacy Portal fields, without claiming calibration.
        "conservative_hit": numeric_or_blank(minimum),
        "median_hit": numeric_or_blank(median),
        "nonloss_probability": numeric_or_blank(
            get_path(item, "probability", "nonloss", "minimum")
        ),
        "full_loss_probability": numeric_or_blank(
            get_path(item, "probability", "full_loss", "maximum")
        ),
        "fair_odds": numeric_or_blank(fair),
        "price_status": (
            "REFERENCE_ONLY" if is_score else text(item.get("price_status"))
        ),
        "commentary": text(edit["短評"]),
        "stars": max(1, min(5, int(edit["星級"]))),
        "is_heavy": boolean(edit["重心"]),
        "compatibility_group": (
            f"{match_id}:FT:CORRECT_SCORE" if is_score
            else text(item.get("compatibility_group"))
        ),
        "conflict_ids": "",
        "result": "pending",
        "status": status,
        "period": period,
        "market_scope": scope,
        "edge": numeric_or_blank(scalar_summary(item, "edge", "minimum")),
        "expected_value": numeric_or_blank(first_value(
            scalar_summary(item, "expected_return", "minimum"),
            scalar_summary(item, "expected_value", "minimum"),
        )),
        "market_support": text(get_path(item, "market_support", "status")),
        "feasible_hit_lower": numeric_or_blank(get_path(
            item, "feasible_probability_bounds", "overall_envelope", "minimum"
        )),
        "feasible_hit_upper": numeric_or_blank(get_path(
            item, "feasible_probability_bounds", "overall_envelope", "maximum"
        )),
        "prior_spread": numeric_or_blank(get_path(
            item, "prior_diagnostics", "prior_spread", "maximum"
        )),
        # Additive metadata; an older Portal API may ignore these fields.
        "probability_semantics": "engine_scenario_summary_not_calibrated_claim",
        "analysis_hash": S.a31_hash,
        "engine_version": RUNTIME["version"],
    }


def build_bundle(
    result: dict, snapshot: dict, sources: list[dict],
    edits: list[dict], meta: dict,
) -> dict:
    source_lookup = {item["_editor_key"]: item for item in sources}
    chosen = [row for row in edits if boolean(row["發佈"])]

    if not chosen:
        raise ValueError("請最少勾選一項發佈項目。")

    for row in chosen:
        rank = number(row["排序"])
        if rank is None or rank < 1 or not rank.is_integer():
            raise ValueError("排序必須是正整數。")
        if not text(row["標題"]):
            raise ValueError("發佈標題不可留空。")

    chosen.sort(key=lambda row: (
        TIERS.index(row["類別"]),
        int(row["排序"]),
        row["_key"],
    ))

    match = match_record(result, snapshot, meta["match_id"])
    match.update({
        "status": meta["status"],
        "model_direction": meta["direction"],
        "model_summary": (
            f"App {APP_VERSION}｜Engine {RUNTIME['version']}｜"
            f"模型品質 {text(get_path(result, 'model_quality', 'status')) or 'UNKNOWN'}"
            f"｜分析 {S.a31_hash[:12]}"
        ),
        # Only publish scores that were explicitly selected.
        "top_scores": "；".join(
            text(source_lookup[row["_key"]].get("selection"))
            for row in chosen if row["類別"] == "CORRECT_SCORE"
        ),
    })

    output, mapping, item_pairs = [], {}, []
    ranks = {tier: 0 for tier in TIERS}
    seen_rec_ids = set()

    for edit in chosen:
        item = source_lookup[edit["_key"]]
        record = portal_record(item, edit, match["match_id"], meta["status"])

        if record["rec_id"] in seen_rec_ids:
            raise ValueError(
                f"重複發佈同一市場／盤口／選擇：{record['rec_title']}。"
                "請只保留其中一項。"
            )
        seen_rec_ids.add(record["rec_id"])

        ranks[record["tier"]] += 1
        record["rank"] = ranks[record["tier"]]
        output.append(record)
        item_pairs.append((item, record))

        if item_id(item):
            mapping[item_id(item)] = record["rec_id"]
        mapping[record["rec_id"]] = record["rec_id"]

    conflicts = {record["rec_id"]: set() for record in output}

    for item, record in item_pairs:
        for source_id in conflict_ids(item):
            resolved = mapping.get(source_id)
            if resolved and resolved != record["rec_id"]:
                conflicts[record["rec_id"]].add(resolved)
                conflicts[resolved].add(record["rec_id"])

    score_ids = {
        record["rec_id"] for record in output
        if record["market"] == "CORRECT_SCORE"
    }
    for rec_id in score_ids:
        conflicts[rec_id].update(score_ids - {rec_id})

    for record in output:
        record["conflict_ids"] = ",".join(sorted(conflicts[record["rec_id"]]))

    return {
        "action": "publish_bundle",
        "replace_recommendations": True,
        "match": match,
        "recommendations": output,
    }


# ============================================================
# 8. Header and sidebar
# ============================================================

st.title("🛡️ AEGIS ULTRA V3.1")
st.caption(
    f"App {APP_VERSION} · {ENGINE_MODULE} · Engine {RUNTIME['version']}"
)

with st.expander("V3.1 使用方法及改動", expanded=False):
    st.markdown(
        """
        1. **輸入**：批量貼上、貼上 JSON 或上載 JSON。
        2. **分析**：只按一次提交；伺服器忙碌時稍後重試。
        3. **審核**：所有候選、個別審核、組合、模型及原始 JSON 均可查看。
        4. **編輯**：VIP 表格逐項勾選，修改類別、排序、標題、星級、重心及短評。
        5. **建立快照**：按「建立／更新發佈快照」。
        6. **發佈**：核對下方快照，再於獨立確認表格提交。

        **注意**
        - 引擎候選不會自動勾選或自動發佈。
        - 情境最低值不是已校準概率，也不是統計信賴下限。
        - 表格內未提交的修改不屬於已建立的發佈快照。
        - 發佈會要求 Portal 取代相同 match_id 的舊推薦。
        - 本版本不提供背景排隊；忙碌提交不會自動稍後執行。
        - 新分析會釋放本工作階段的舊結果；需要保留時請先下載。
        """
    )

with st.sidebar:
    st.header("工作流程")
    mode = st.radio(
        "輸入模式",
        ["📋 貼上 JSON", "📁 上載 JSON", "🎛️ 批量貼上"],
    )

    st.info(
        "同一應用程式進程最多一個分析。\n\n"
        "沒有背景佇列；忙碌時請稍後重試。"
    )

    st.caption(f"引擎指紋：{RUNTIME['fingerprint'][:16]}")

    if ENGINE_MODULE != "aegisultra_enginev31":
        st.warning(f"目前明確設定使用：{ENGINE_MODULE}，並非預設 V3.1 模組。")

    if st.button("清除目前結果及發佈草稿", use_container_width=True):
        clear_results()
        st.rerun()

    st.divider()
    st.subheader("Portal")

    if portal_setting("token", "AEGIS_API_TOKEN"):
        st.success("已載入 API token")
    else:
        st.warning("未設定 API token；仍可分析及下載 payload。")

    if st.button("測試 Portal 連線", use_container_width=True):
        try:
            st.json(portal_request({"action": "ping"}))
        except Exception as error:
            st.error(str(error))


# ============================================================
# 9. Input interface
# ============================================================

submission = None

if mode == "📋 貼上 JSON":
    if "a31_json_widget" not in S:
        S.a31_json_widget = S.a31_json_saved

    with st.form("a31_json_form"):
        raw_json = st.text_area(
            "完整引擎輸入 JSON",
            key="a31_json_widget",
            height=520,
        )
        run = st.form_submit_button(
            "🚀 提交 JSON 開始分析",
            type="primary",
            use_container_width=True,
        )

    if run:
        S.a31_json_saved = raw_json
        try:
            submission = validate_input(loads(raw_json))
            S.a31_error = None
        except Exception:
            S.a31_error = traceback.format_exc()

elif mode == "📁 上載 JSON":
    with st.form("a31_upload_form"):
        uploaded = st.file_uploader("上載 JSON，最大 10 MB", type=["json"])
        run = st.form_submit_button(
            "🚀 提交檔案開始分析",
            type="primary",
            use_container_width=True,
        )

    if run:
        try:
            if uploaded is None:
                raise ValueError("請先選擇 JSON 檔案。")
            if uploaded.size > MAX_INPUT_BYTES:
                raise ValueError("JSON 檔案超過 10 MB。")
            raw_json = uploaded.getvalue().decode("utf-8-sig")
            submission = validate_input(loads(raw_json))
            S.a31_json_saved = raw_json
            S.a31_error = None
        except Exception:
            S.a31_error = traceback.format_exc()

else:
    # Layout-changing control is outside the submission form.
    source_count = st.number_input(
        "尖銳來源數目", min_value=1, max_value=5, value=1, step=1
    )
    st.caption("請先設定來源數目，再填寫表格；更改數目會重新載入頁面。")

    with st.form("a31_bulk_form"):
        match_tab, sharp_tab, target_tab, settings_tab = st.tabs(
            ["① 賽事", "② 尖銳市場", "③ HKJC", "④ 設定"]
        )

        with match_tab:
            a, b = st.columns(2)
            home = a.text_input("主隊", key="a31_home")
            away = b.text_input("客隊", key="a31_away")
            competition = a.text_input("賽事", key="a31_competition")
            kickoff = b.text_input(
                "Kickoff：含時區 ISO 格式",
                placeholder="YYYY-MM-DDTHH:MM:SS+08:00",
                key="a31_kickoff",
            )
            match_name = a.text_input("自訂名稱，可留空", key="a31_match_name")
            snapshot_time = b.text_input(
                "市場快照時間",
                value=datetime.now(timezone.utc).isoformat(timespec="minutes"),
                key="a31_snapshot_time",
            )

        sources = []
        with sharp_tab:
            st.caption(
                "1X2：主 和 客｜AH：主隊盤口 主賠 客賠｜"
                "OU：盤口 大賠 細賠｜HHAD：整數主隊盤口 主 和 客｜"
                "TEAM_OU：HOME/AWAY 盤口 大賠 細賠。"
                "HT 全部留空即不提供 HT。"
            )

            for index in range(int(source_count)):
                st.subheader(f"來源 {index + 1}")
                a, b = st.columns(2)
                key = a.text_input(
                    "識別碼",
                    value="pinnacle" if index == 0 else f"source_{index + 1}",
                    key=f"a31_source_key_{index}",
                )
                title = b.text_input(
                    "來源名稱",
                    value="Pinnacle" if index == 0 else f"Source {index + 1}",
                    key=f"a31_source_title_{index}",
                )
                raw = bulk_market_fields(f"a31_sharp_{index}")
                sources.append({"key": key, "title": title, "raw": raw})

            primary_index = st.selectbox(
                "主要來源",
                options=list(range(int(source_count))),
                format_func=lambda index: f"來源 {index + 1}",
            )

        with target_tab:
            st.caption(
                "可用 X 或 - 省略單邊賠率。"
                "一行有兩邊或三邊賠率時，會建立多個候選盤。"
            )
            target_raw = bulk_market_fields("a31_target")

        with settings_tab:
            st.info(
                "所有設定均開放。此處使用完整 settings JSON；"
                "V3.1 專有設定請依你的引擎規格加入。"
                "App 不會偷偷停用審核以換取速度。"
            )
            settings_raw = st.text_area(
                "完整 settings JSON",
                value=dumps(DEFAULT_SETTINGS),
                height=500,
                key="a31_bulk_settings",
            )
            st.caption("primary_source 會由上方「主要來源」覆寫。")

        run = st.form_submit_button(
            "🚀 提交批量資料開始分析",
            type="primary",
            use_container_width=True,
        )

    if run:
        try:
            settings = loads(settings_raw)
            if not isinstance(settings, dict):
                raise ValueError("settings 必須是 JSON 物件。")
            match = {
                "name": text(match_name) or f"{text(home)} vs {text(away)}",
                "home": text(home),
                "away": text(away),
                "competition": text(competition),
                "kickoff": text(kickoff),
                "snapshot_time": text(snapshot_time),
            }
            submission = validate_input(build_bulk_input(
                match, sources, target_raw, settings, primary_index
            ))
            S.a31_json_saved = dumps(submission)
            S.a31_error = None
        except Exception:
            S.a31_error = traceback.format_exc()

if submission is not None:
    try:
        st.caption(
            f"提交 {len(submission['hkjc_markets'])} 個候選盤；"
            "候選數目不等於原始貼上行數。"
        )
        with st.spinner("引擎分析中。請勿重複提交或重新整理瀏覽器……"):
            message = execute(submission)
        st.success(message)
        S.a31_error = None
    except Exception:
        S.a31_error = traceback.format_exc()
    finally:
        submission = None
        gc.collect()

if S.a31_error:
    st.error("今次提交未成功。下方如仍有結果，屬上一次已完成的分析。")
    with st.expander("技術錯誤詳情", expanded=True):
        st.code(S.a31_error, language="text")


# ============================================================
# 10. Manual publisher
# ============================================================

def render_publisher(result: dict, snapshot: dict, candidates: list[dict]):
    sources = publication_sources(candidates, result)
    if not sources:
        st.info("沒有可供發佈的候選盤。")
        return

    st.warning(
        "此處所有項目均可人手選擇，不會替你隱藏不合資格或未入選項目。"
        "請自行核對品質閘門、排除原因及衝突。"
    )
    st.caption(
        "雙擊儲存格可修改標題或短評。勾選發佈後，"
        "先建立快照，再到下方核對及確認。"
    )

    if S.a31_editor_saved is None:
        S.a31_editor_saved = initial_editor_rows(sources)

    meta_defaults = S.a31_publish_meta or {
        "status": "published",
        "match_id": "",
        "direction": "",
    }

    suffix = S.a31_hash[:16]

    with st.form(f"a31pub_editor_form_{suffix}"):
        a, b = st.columns(2)
        status = a.selectbox(
            "發佈狀態",
            ["published", "draft"],
            index=0 if meta_defaults["status"] == "published" else 1,
        )
        override = b.text_input(
            "自訂唯一 match_id（沒有有效 kickoff 時必填）",
            value=meta_defaults["match_id"],
        )
        direction = st.text_area(
            "人手投注方向",
            value=meta_defaults["direction"],
            height=90,
        )

        editable = {
            "發佈", "類別", "排序", "標題", "星級", "重心", "短評"
        }
        edits = st.data_editor(
            S.a31_editor_saved,
            key=f"a31pub_grid_{suffix}",
            hide_index=True,
            use_container_width=True,
            num_rows="fixed",
            height=560,
            disabled=[
                key for key in S.a31_editor_saved[0] if key not in editable
            ],
            column_config={
                "_key": None,
                "發佈": st.column_config.CheckboxColumn("發佈"),
                "類別": st.column_config.SelectboxColumn(
                    "類別",
                    options=list(TIERS),
                    required=True,
                ),
                "排序": st.column_config.NumberColumn(
                    "排序", min_value=1, step=1, required=True
                ),
                "星級": st.column_config.NumberColumn(
                    "星級", min_value=1, max_value=5, step=1, required=True
                ),
                "重心": st.column_config.CheckboxColumn("重心"),
                "短評": st.column_config.TextColumn("短評", width="large"),
            },
        )

        prepare = st.form_submit_button(
            "✅ 建立／更新發佈快照",
            type="primary",
            use_container_width=True,
        )

    if prepare:
        # Invalidate old payload immediately, even if new validation fails.
        S.a31_prepared = None
        S.a31_portal_response = None
        S.a31_publish_uncertain = False
        S.a31_editor_saved = deepcopy(edits)
        S.a31_publish_meta = {
            "status": status,
            "match_id": text(override),
            "direction": text(direction),
        }

        try:
            require_current_engine()
            bundle = build_bundle(
                result, snapshot, sources, edits, S.a31_publish_meta
            )
            S.a31_prepared = {
                "analysis_hash": S.a31_hash,
                "payload_hash": digest(bundle),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "bundle": bundle,
            }
            # Remount editor on next rerun using the saved complete rows.
            S.pop(f"a31pub_grid_{suffix}", None)
            st.rerun(scope="fragment")
        except Exception as error:
            st.error(f"無法建立快照：{error}")
            return

    prepared = S.a31_prepared
    if not isinstance(prepared, dict):
        st.info("尚未建立發佈快照。")
        return

    if prepared["analysis_hash"] != S.a31_hash:
        st.error("快照不屬於目前分析，請重新建立。")
        return

    bundle = prepared["bundle"]
    selected = bundle["recommendations"]
    payload_hash = prepared["payload_hash"]

    st.divider()
    st.subheader("實際將要發佈的快照")
    st.caption(
        f"建立：{prepared['created_at']}｜"
        f"快照：{payload_hash[:16]}｜"
        f"分析：{S.a31_hash[:16]}"
    )
    st.warning(
        "下面的快照才是實際發佈內容。"
        "上方表格若有新修改，必須再按「建立／更新發佈快照」。"
    )

    a, b, c = st.columns(3)
    a.metric("正式推薦", sum(row["tier"] == "OFFICIAL" for row in selected))
    b.metric("進取選擇", sum(row["tier"] == "ALTERNATIVE" for row in selected))
    c.metric("波膽參考", sum(row["tier"] == "CORRECT_SCORE" for row in selected))

    st.write("**Match ID：**", bundle["match"]["match_id"])
    st.write("**發佈狀態：**", bundle["match"]["status"])
    st.write("**人手方向：**", bundle["match"]["model_direction"] or "（空白）")

    st.dataframe(
        [{
            "類別": row["tier"],
            "排名": row["rank"],
            "時段": row["period"],
            "標題": row["rec_title"],
            "盤口": text(row["line"]),
            "賠率": text(row["odds"]),
            "星級": row["stars"],
            "重心": row["is_heavy"],
            "短評": row["commentary"],
            "衝突IDs": row["conflict_ids"],
        } for row in selected],
        use_container_width=True,
        hide_index=True,
    )

    has_conflicts = any(row["conflict_ids"] for row in selected)
    if has_conflicts:
        st.warning(
            "清單含引擎已標記的衝突或互斥波膽。"
            "App 保留你的選擇，但需要額外確認。"
        )

    if st.checkbox(
        "查看完整 Portal payload",
        key=f"a31pub_show_payload_{payload_hash[:16]}",
    ):
        st.json(bundle)

    st.download_button(
        "⬇️ 下載這份 Portal payload",
        data=dumps(bundle),
        file_name=download_filename("aegis_v31_portal"),
        mime="application/json",
        key=f"a31pub_download_{payload_hash[:16]}",
    )

    previous = S.a31_portal_response
    if isinstance(previous, dict) and previous.get("payload_hash") == payload_hash:
        st.success("這份快照已於本工作階段成功發佈；不會再次提交。")
        st.json(previous["response"])
        return

    if S.a31_publish_uncertain:
        st.error(
            "上次發佈回應不確定。請先到 VIP App 核對是否已成功，"
            "不要直接連續重試。確認後重新建立快照才可再次提交。"
        )
        return

    with st.form(f"a31pub_confirm_{payload_hash[:16]}"):
        acknowledged = st.checkbox(
            "我已核對以上快照，並同意取代相同 match_id 的舊推薦。"
        )
        conflict_acknowledged = (
            st.checkbox("我已檢查並接受上述互斥／衝突項目。")
            if has_conflicts else True
        )
        publish = st.form_submit_button(
            f"📤 確認提交以上 {len(selected)} 項",
            type="primary",
            use_container_width=True,
        )

    if publish:
        if not acknowledged or not conflict_acknowledged:
            st.error("請先完成確認。")
            return

        request_started = False
        try:
            require_current_engine()
            if S.a31_engine_fingerprint != RUNTIME["fingerprint"]:
                raise ValueError("引擎版本與分析不符，請重新分析。")
            if digest(bundle) != payload_hash:
                raise ValueError("快照內容已改變，請重新建立。")
            if not portal_setting("token", "AEGIS_API_TOKEN"):
                raise ValueError("缺少 Portal API token。")

            request_started = True
            with st.spinner("正在提交已確認快照……"):
                response = portal_request(bundle)

            S.a31_portal_response = {
                "payload_hash": payload_hash,
                "response": response,
            }
            st.rerun(scope="fragment")

        except Exception as error:
            S.a31_publish_uncertain = request_started
            st.error(f"發佈未確認成功：{error}")
            if request_started:
                st.warning(
                    "HTTP 超時或無效回應不代表伺服器沒有寫入。"
                    "請先核對 VIP App。"
                )


# ============================================================
# 11. Results fragment — one section rendered at a time
# ============================================================

@st.fragment
def render_results():
    result = S.a31_result
    if not isinstance(result, dict):
        st.info("提交資料後，分析結果會顯示在這裡。")
        return

    snapshot = as_dict(S.a31_input)
    meta = as_dict(S.a31_meta)
    candidates = candidates_for_display(result, snapshot)

    st.divider()
    match = as_dict(result.get("match")) or as_dict(snapshot.get("match"))
    st.header("📡 " + (text(match.get("name")) or "賽事分析"))

    st.caption(
        f"已完成分析：{text(meta.get('completed_at'))}｜"
        f"分析識別：{text(S.a31_hash)[:16]}"
    )

    a, b, c, d = st.columns(4)
    a.metric(
        "模型品質",
        text(get_path(result, "model_quality", "status")) or "UNKNOWN",
    )
    b.metric("候選盤", len(candidates))
    c.metric("引擎入選", sum(bool(item["_shortlisted"]) for item in candidates))
    d.metric("執行秒數", f"{meta.get('elapsed_seconds', 0):.1f}")

    section = st.radio(
        "結果頁面",
        [
            "引擎候選", "所有候選盤", "候選盤審核",
            "推薦組合", "波膽參考", "模型診斷",
            "發佈到 VIP App", "JSON／下載",
        ],
        horizontal=True,
        key="a31_result_section",
    )

    if section == "引擎候選":
        st.info(
            "情境最低／中位概率直接取自引擎原欄位。"
            "沒有自動視為校準概率，也沒有以其他欄位補造數值。"
        )
        shortlisted = [item for item in candidates if item["_shortlisted"]]
        if not shortlisted:
            st.warning("沒有引擎入選項目。")

        for item in shortlisted:
            with st.container(border=True):
                st.subheader(text(first_value(item.get("label"), item_id(item))))
                st.caption(
                    f"{normalize_period(item.get('period')) or 'MISSING'}｜"
                    f"{normalize_market(item.get('market'))}"
                )
                a, b, c, d = st.columns(4)
                a.metric("賠率", odds_text(first_value(
                    item.get("hkjc_odds"), item.get("odds")
                )))
                b.metric("情境最低命中", pct(get_path(
                    item, "probability", "hit", "minimum"
                )))
                c.metric("情境中位命中", pct(get_path(
                    item, "probability", "hit", "median"
                )))
                d.metric("最低 EV", ev_text(
                    scalar_summary(item, "expected_return", "minimum")
                ))
                st.write(
                    "市場支持：",
                    text(get_path(item, "market_support", "status")) or "—",
                )

                # Explicit fields shown exactly as supplied; no inferred semantics.
                extra = {
                    key: item[key]
                    for key in (
                        "ranking_probability",
                        "reference_probability",
                        "calibrated_probability",
                        "calibration",
                    )
                    if key in item
                }
                if extra:
                    st.write("引擎額外概率／校準欄位（原值）")
                    st.json(extra)

    elif section == "所有候選盤":
        st.dataframe(
            [candidate_row(item) for item in candidates],
            use_container_width=True,
            hide_index=True,
        )

    elif section == "候選盤審核":
        if not candidates:
            st.info("沒有候選盤。")
            return

        index = st.selectbox(
            "候選盤",
            range(len(candidates)),
            format_func=lambda i: (
                f"{item_id(candidates[i]) or i}｜"
                f"{normalize_period(candidates[i].get('period'))}｜"
                f"{text(candidates[i].get('label'))}"
            ),
        )
        item = candidates[index]

        fields = [
            key for key in item
            if not key.startswith("_")
        ]
        field = st.selectbox("查看原始欄位／審核", ["完整候選盤"] + fields)

        if field == "完整候選盤":
            st.json({key: value for key, value in item.items()
                     if not key.startswith("_")})
        else:
            value = item[field]
            if isinstance(value, (dict, list)):
                st.json(value)
            else:
                st.write(value)

        st.caption(
            "保留引擎原始結構，避免 V3／V3.1 審核欄位不同時顯示錯誤摘要。"
        )

    elif section == "推薦組合":
        st.json(result.get("recommendation_set", {}))
        st.caption("此為引擎候選組合，不會因 VIP 人手改選而自動重算。")

    elif section == "波膽參考":
        st.warning("波膽為高風險參考；不同比分互相排斥。")
        st.json(result.get("correct_scores", {}))

    elif section == "模型診斷":
        keys = [
            key for key in result
            if key not in {
                "candidate_markets", "recommendations", "input_snapshot"
            }
        ]
        if keys:
            field = st.selectbox("頂層診斷欄位", keys)
            value = result[field]

            if isinstance(value, dict) and value:
                child = st.selectbox(
                    "子欄位", ["全部"] + list(value.keys())
                )
                if child != "全部":
                    value = value[child]

            if isinstance(value, (dict, list)):
                st.json(value)
            else:
                st.write(value)

        st.caption("所有原始診斷仍保留；只建立目前選擇的畫面。")

    elif section == "發佈到 VIP App":
        render_publisher(result, snapshot, candidates)

    else:
        st.json(meta)
        kind = st.selectbox(
            "下載／查看內容",
            ["完整結果", "實際分析輸入", "分析執行資料"],
        )
        value = {
            "完整結果": result,
            "實際分析輸入": snapshot,
            "分析執行資料": meta,
        }[kind]

        export_key = "a31export_data"

        if st.button("準備所選 JSON 下載"):
            try:
                S[export_key] = {
                    "hash": S.a31_hash,
                    "kind": kind,
                    "data": dumps(value),
                }
            except Exception as error:
                st.error(f"JSON 序列化失敗：{error}")

        exported = S.get(export_key)
        if (
            isinstance(exported, dict)
            and exported["hash"] == S.a31_hash
            and exported["kind"] == kind
        ):
            st.download_button(
                "⬇️ 下載已準備 JSON",
                data=exported["data"],
                file_name=download_filename("aegis_v31"),
                mime="application/json",
            )
            if st.button("釋放已準備下載的 JSON 副本"):
                S.pop(export_key, None)
                gc.collect()
                st.rerun(scope="fragment")

        if st.checkbox("在頁面顯示所選完整 JSON"):
            st.json(value)

        st.caption(
            "完整結果仍保留在目前工作階段。"
            "這不是永久儲存；重要結果請下載保存。"
        )


render_results()

st.divider()
st.caption(
    f"AEGIS ULTRA App {APP_VERSION} · Engine {RUNTIME['version']} · "
    "市場重建及風險分析工具。引擎候選不等於自動正式推薦，"
    "情境最低概率不等於已校準預測，亦不保證投注結果。"
)
