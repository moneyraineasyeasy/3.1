from __future__ import annotations

import copy
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares, linprog, minimize
from scipy.sparse import coo_matrix
from scipy.special import gammaln, logsumexp
from scipy.stats import poisson


# ============================================================
# AEGIS ULTRA V3.0
#
# Multi-prior sharp-market reconstruction engine.
#
# Forecasting priors:
# - Independent Poisson
# - Dixon-Coles
# - Shared-dispersion COM-Poisson
#
# Audit prior:
# - Flat-grid maximum entropy
#
# Major controls:
# - Independent FT and HT reconstruction
# - Target-line-out reconstruction
# - Whole-family-out audit
# - Candidate-specific feasible probability bounds
# - Prior sensitivity diagnostics
# - MaxEnt audit
# - Light / Medium / Heavy structural stress
# - Adaptive score grids
# - HT-FT transport-coherence audit
#
# Important:
# - The engine produces an engine shortlist.
# - Final publication remains a manual decision.
# - MARKET_SUPPORTED requires user-calibrated thresholds.
# ============================================================


ENGINE_NAME = "Aegis Ultra Multi-Prior Reconstruction Engine"
ENGINE_VERSION = "3.0.0"

TOL = 1e-10

PARAMETRIC_PRIORS = (
    "INDEPENDENT_POISSON",
    "DIXON_COLES",
    "COM_POISSON_SHARED",
)

AUDIT_PRIORS = (
    "FLAT_GRID_MAXENT",
)

ALL_PRIORS = PARAMETRIC_PRIORS + AUDIT_PRIORS


CONFIG = {
    # Adaptive score grids.
    "FT_INITIAL_MAX_GOALS": 12,
    "HT_INITIAL_MAX_GOALS": 6,
    "FT_MAX_GOALS_SAFETY": 24,
    "HT_MAX_GOALS_SAFETY": 14,
    "GRID_EXPANSION_STEP": 2,
    "TAIL_TOLERANCE": 1e-8,
    "BOUNDARY_TOLERANCE": 1e-8,

    # Solvers.
    "FIT_MAX_EVALUATIONS": 800,
    "PROJECTION_MAX_ITERATIONS": 1800,
    "PROJECTION_BUFFER": 1e-8,
    "LP_NUMERICAL_BUFFER": 1e-8,

    # Prior parameter bounds.
    "MIN_GOAL_RATE": 0.01,
    "MAX_GOAL_RATE": 8.0,
    "MIN_COM_DISPERSION": 0.35,
    "MAX_COM_DISPERSION": 3.00,

    # De-vig.
    "DEFAULT_DEVIG_METHODS": (
        "MULTIPLICATIVE",
        "POWER",
    ),

    # Structural requirements.
    "MINIMUM_CONSTRAINTS": 3,
    "MINIMUM_MARKET_FAMILIES": 2,

    # Shortlist defaults.
    "DEFAULT_MAX_RECOMMENDATIONS": 3,
    "DEFAULT_CORRECT_SCORE_COUNT": 2,
    "DEFAULT_MINIMUM_SHORTLIST_HIT": 0.50,

    # Price warnings only.
    "POOR_PRICE_EV_WARNING": -0.05,
    "SEVERE_PRICE_EV_WARNING": -0.10,
}


# ============================================================
# 1. Generic utilities
# ============================================================

def to_builtin(value):
    if isinstance(value, dict):
        return {
            str(key): to_builtin(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            to_builtin(item)
            for item in value
        ]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.bool_):
        return bool(value)

    return value


def clamp(value, minimum, maximum):
    return max(
        minimum,
        min(maximum, value),
    )


def sigmoid(value):
    value = float(value)

    if value >= 0:
        exponential = math.exp(-value)
        return 1.0 / (1.0 + exponential)

    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def safe_float(value, default=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default

    if not math.isfinite(value):
        return default

    return value


def summary_statistics(values):
    values = np.asarray(
        list(values),
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return {
            "minimum": None,
            "median": None,
            "maximum": None,
            "count": 0,
        }

    return {
        "minimum": float(np.min(values)),
        "median": float(np.median(values)),
        "maximum": float(np.max(values)),
        "count": int(len(values)),
    }


def validate_probability_threshold(
    value,
    label,
    allow_none=True,
):
    if value in {None, ""}:
        if allow_none:
            return None

        raise ValueError(
            f"{label} is required."
        )

    value = float(value)

    if (
        not math.isfinite(value)
        or value < 0.0
        or value > 1.0
    ):
        raise ValueError(
            f"{label} must be between 0 and 1."
        )

    return value


def validate_odds(value, label="Odds"):
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"{label} must be numeric."
        )

    if (
        not math.isfinite(value)
        or value <= 1.0
    ):
        raise ValueError(
            f"{label} must exceed 1.00."
        )

    return value


def validate_quarter_line(
    value,
    label="Line",
):
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"{label} must be numeric."
        )

    if not math.isfinite(value):
        raise ValueError(
            f"{label} must be finite."
        )

    units = value * 4.0

    if not math.isclose(
        units,
        round(units),
        abs_tol=1e-8,
    ):
        raise ValueError(
            f"{label} must be a multiple of 0.25."
        )

    return round(units) / 4.0


def validate_integer_line(
    value,
    label="Handicap",
):
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"{label} must be numeric."
        )

    if (
        not math.isfinite(value)
        or not math.isclose(
            value,
            round(value),
            abs_tol=1e-8,
        )
    ):
        raise ValueError(
            f"{label} must be a whole-goal handicap."
        )

    return float(round(value))


def split_quarter_line(line):
    line = validate_quarter_line(line)
    units = int(round(line * 4.0))

    if abs(units) % 2 == 1:
        return [
            line - 0.25,
            line + 0.25,
        ]

    return [line]


def normalize_period(value):
    period = str(
        value or "FT"
    ).strip().upper()

    aliases = {
        "FULL": "FT",
        "FULL_TIME": "FT",
        "90": "FT",
        "FIRST_HALF": "HT",
        "HALF_TIME": "HT",
        "1H": "HT",
        "45": "HT",
    }

    period = aliases.get(
        period,
        period,
    )

    if period not in {"FT", "HT"}:
        raise ValueError(
            f"Unsupported period: {period}"
        )

    return period


def normalize_market(value):
    market = str(
        value
    ).strip().upper()

    aliases = {
        "H2H": "1X2",
        "3WAY": "1X2",
        "HAD": "1X2",

        "SPREAD": "AH",
        "SPREADS": "AH",
        "HANDICAP": "AH",
        "ASIAN_HANDICAP": "AH",

        "TOTAL": "OU",
        "TOTALS": "OU",
        "OVER_UNDER": "OU",

        "HANDICAP_HAD": "HHAD",
        "HANDICAP_1X2": "HHAD",
        "3WAY_HANDICAP": "HHAD",
        "HHC": "HHAD",

        "TEAM_TOTAL": "TEAM_OU",
        "TEAM_TOTALS": "TEAM_OU",
        "TEAM_HILO": "TEAM_OU",
        "TEAM_O/U": "TEAM_OU",
        "TEAMOU": "TEAM_OU",
    }

    market = aliases.get(
        market,
        market,
    )

    if market not in {
        "1X2",
        "AH",
        "OU",
        "HHAD",
        "TEAM_OU",
    }:
        raise ValueError(
            f"Unsupported market: {market}"
        )

    return market


def normalize_team(value):
    team = str(
        value
    ).strip().upper()

    aliases = {
        "H": "HOME",
        "HOME_TEAM": "HOME",
        "A": "AWAY",
        "AWAY_TEAM": "AWAY",
    }

    team = aliases.get(
        team,
        team,
    )

    if team not in {"HOME", "AWAY"}:
        raise ValueError(
            "Team must be HOME or AWAY."
        )

    return team


def normalize_selection(
    market,
    selection,
):
    market = normalize_market(market)

    selection = str(
        selection
    ).strip().upper()

    aliases = {
        "H": "HOME",
        "HOME": "HOME",
        "HOME_WIN": "HOME",

        "D": "DRAW",
        "DRAW": "DRAW",
        "X": "DRAW",

        "A": "AWAY",
        "AWAY": "AWAY",
        "AWAY_WIN": "AWAY",

        "O": "OVER",
        "OVER": "OVER",

        "U": "UNDER",
        "UNDER": "UNDER",
    }

    selection = aliases.get(
        selection,
        selection,
    )

    if market in {"1X2", "HHAD"}:
        allowed = {
            "HOME",
            "DRAW",
            "AWAY",
        }

    elif market == "AH":
        allowed = {
            "HOME",
            "AWAY",
        }

    else:
        allowed = {
            "OVER",
            "UNDER",
        }

    if selection not in allowed:
        raise ValueError(
            f"Invalid {market} selection: "
            f"{selection}"
        )

    return selection


def normalize_prior(value):
    prior = str(value).strip().upper()

    aliases = {
        "POISSON": "INDEPENDENT_POISSON",
        "INDEPENDENT": "INDEPENDENT_POISSON",
        "INDEPENDENT-POISSON": "INDEPENDENT_POISSON",

        "DC": "DIXON_COLES",
        "DIXON-COLES": "DIXON_COLES",

        "COM_POISSON": "COM_POISSON_SHARED",
        "COM-POISSON": "COM_POISSON_SHARED",
        "SHARED_COM_POISSON": "COM_POISSON_SHARED",

        "MAXENT": "FLAT_GRID_MAXENT",
        "MAX_ENTROPY": "FLAT_GRID_MAXENT",
        "FLAT_MAXENT": "FLAT_GRID_MAXENT",
    }

    prior = aliases.get(
        prior,
        prior,
    )

    if prior not in ALL_PRIORS:
        raise ValueError(
            f"Unsupported prior: {prior}"
        )

    return prior


def candidate_key(
    period,
    market,
    selection,
    line=None,
    team=None,
):
    period = normalize_period(period)
    market = normalize_market(market)
    selection = normalize_selection(
        market,
        selection,
    )

    line_key = (
        None
        if line is None
        else round(float(line), 4)
    )

    team_key = (
        None
        if team is None
        else normalize_team(team)
    )

    return (
        period,
        market,
        selection,
        line_key,
        team_key,
    )


def market_group_key(
    period,
    market,
    line=None,
    team=None,
):
    period = normalize_period(period)
    market = normalize_market(market)

    if market == "1X2":
        return f"{period}:1X2"

    if line is None:
        raise ValueError(
            f"{market} requires a line."
        )

    if market == "TEAM_OU":
        team = normalize_team(team)

        return (
            f"{period}:TEAM_OU:"
            f"{team}:{float(line):+.4f}"
        )

    return (
        f"{period}:{market}:"
        f"{float(line):+.4f}"
    )


def market_family_key(
    period,
    market,
    team=None,
):
    period = normalize_period(period)
    market = normalize_market(market)

    if market == "TEAM_OU":
        return (
            f"{period}:TEAM_OU:"
            f"{normalize_team(team)}"
        )

    return f"{period}:{market}"


def quote_identity(
    period,
    market,
    selection,
    line=None,
    team=None,
):
    return candidate_key(
        period,
        market,
        selection,
        line,
        team,
    )


def score_arrays(max_goals):
    home, away = np.indices(
        (
            max_goals + 1,
            max_goals + 1,
        )
    )

    return (
        home.ravel().astype(float),
        away.ravel().astype(float),
    )


def score_boundary_mass(
    probabilities,
    max_goals,
):
    home_scores, away_scores = (
        score_arrays(max_goals)
    )

    boundary = (
        (home_scores >= max_goals)
        | (away_scores >= max_goals)
    )

    return float(
        np.asarray(probabilities)
        @ boundary.astype(float)
    )


# ============================================================
# 2. Exact market settlement
# ============================================================

def settlement_coefficients(
    home_scores,
    away_scores,
    market,
    selection,
    line=None,
    team=None,
):
    market = normalize_market(market)
    selection = normalize_selection(
        market,
        selection,
    )

    home_scores = np.asarray(
        home_scores,
        dtype=float,
    )

    away_scores = np.asarray(
        away_scores,
        dtype=float,
    )

    if home_scores.shape != away_scores.shape:
        raise ValueError(
            "Score arrays must have identical shapes."
        )

    a_coeff = np.zeros(
        home_scores.shape,
        dtype=float,
    )

    b_coeff = np.zeros(
        home_scores.shape,
        dtype=float,
    )

    if market == "1X2":
        if selection == "HOME":
            wins = home_scores > away_scores
        elif selection == "DRAW":
            wins = home_scores == away_scores
        else:
            wins = home_scores < away_scores

        a_coeff[wins] = 1.0
        return a_coeff, b_coeff

    if market == "HHAD":
        if line is None:
            raise ValueError(
                "HHAD requires a handicap."
            )

        handicap = validate_integer_line(
            line
        )

        adjusted_margin = (
            home_scores
            - away_scores
            + handicap
        )

        if selection == "HOME":
            wins = adjusted_margin > TOL
        elif selection == "DRAW":
            wins = (
                np.abs(adjusted_margin)
                <= TOL
            )
        else:
            wins = adjusted_margin < -TOL

        a_coeff[wins] = 1.0
        return a_coeff, b_coeff

    if line is None:
        raise ValueError(
            f"{market} requires a line."
        )

    component_lines = split_quarter_line(
        line
    )

    component_weight = (
        1.0 / len(component_lines)
    )

    if market == "TEAM_OU":
        team = normalize_team(team)

        goal_values = (
            home_scores
            if team == "HOME"
            else away_scores
        )
    else:
        goal_values = None

    for component_line in component_lines:
        if market == "AH":
            adjusted_margin = (
                home_scores
                - away_scores
                + component_line
            )

            if selection == "HOME":
                wins = adjusted_margin > TOL
            else:
                wins = adjusted_margin < -TOL

            pushes = (
                np.abs(adjusted_margin)
                <= TOL
            )

        elif market == "OU":
            difference = (
                home_scores
                + away_scores
                - component_line
            )

            if selection == "OVER":
                wins = difference > TOL
            else:
                wins = difference < -TOL

            pushes = (
                np.abs(difference)
                <= TOL
            )

        else:
            difference = (
                goal_values
                - component_line
            )

            if selection == "OVER":
                wins = difference > TOL
            else:
                wins = difference < -TOL

            pushes = (
                np.abs(difference)
                <= TOL
            )

        a_coeff[wins] += component_weight
        b_coeff[pushes] += component_weight

    return a_coeff, b_coeff


def classify_settlement(
    a_coeff,
    b_coeff,
):
    a_coeff = np.asarray(
        a_coeff,
        dtype=float,
    )

    b_coeff = np.asarray(
        b_coeff,
        dtype=float,
    )

    return {
        "full_win": (
            np.isclose(a_coeff, 1.0)
            & np.isclose(b_coeff, 0.0)
        ),
        "half_win": (
            np.isclose(a_coeff, 0.5)
            & np.isclose(b_coeff, 0.5)
        ),
        "push": (
            np.isclose(a_coeff, 0.0)
            & np.isclose(b_coeff, 1.0)
        ),
        "half_loss": (
            np.isclose(a_coeff, 0.0)
            & np.isclose(b_coeff, 0.5)
        ),
        "full_loss": (
            np.isclose(a_coeff, 0.0)
            & np.isclose(b_coeff, 0.0)
        ),
    }


def effective_fair_probability(
    probabilities,
    a_coeff,
    b_coeff,
):
    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    expected_a = float(
        probabilities @ a_coeff
    )

    expected_b = float(
        probabilities @ b_coeff
    )

    denominator = 1.0 - expected_b

    if denominator <= TOL:
        return 0.0

    return expected_a / denominator


def candidate_metrics(
    probabilities,
    a_coeff,
    b_coeff,
    odds,
):
    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    odds = validate_odds(odds)

    categories = classify_settlement(
        a_coeff,
        b_coeff,
    )

    category_probabilities = {
        name: float(
            probabilities
            @ mask.astype(float)
        )
        for name, mask
        in categories.items()
    }

    payout = (
        a_coeff * odds
        + b_coeff
    )

    profit = payout - 1.0

    hit_mask = profit > TOL
    nonloss_mask = profit >= -TOL
    loss_mask = profit < -TOL

    expected_return = float(
        probabilities @ profit
    )

    expected_a = float(
        probabilities @ a_coeff
    )

    expected_b = float(
        probabilities @ b_coeff
    )

    fair_odds = (
        (1.0 - expected_b) / expected_a
        if expected_a > TOL
        else None
    )

    return {
        "hit_probability": float(
            probabilities
            @ hit_mask.astype(float)
        ),
        "nonloss_probability": float(
            probabilities
            @ nonloss_mask.astype(float)
        ),
        "loss_probability": float(
            probabilities
            @ loss_mask.astype(float)
        ),
        "expected_return": expected_return,
        "effective_fair_probability": (
            effective_fair_probability(
                probabilities,
                a_coeff,
                b_coeff,
            )
        ),
        "fair_odds": fair_odds,
        **category_probabilities,
    }


# ============================================================
# 3. De-vigging
# ============================================================

def multiplicative_devig(odds):
    odds = np.asarray(
        [
            validate_odds(
                value,
                f"Odds #{index + 1}",
            )
            for index, value
            in enumerate(odds)
        ],
        dtype=float,
    )

    inverse = 1.0 / odds

    return inverse / inverse.sum()


def power_devig(odds):
    odds = np.asarray(
        [
            validate_odds(
                value,
                f"Odds #{index + 1}",
            )
            for index, value
            in enumerate(odds)
        ],
        dtype=float,
    )

    inverse = 1.0 / odds

    def equation(power):
        return float(
            np.sum(
                inverse ** power
            )
            - 1.0
        )

    lower = 1e-8
    upper = 100.0

    while (
        equation(upper) >= 0
        and upper < 100000
    ):
        upper *= 2.0

    if equation(upper) >= 0:
        raise RuntimeError(
            "Power de-vig root was not found."
        )

    for _ in range(220):
        middle = (
            lower + upper
        ) / 2.0

        if equation(middle) > 0:
            lower = middle
        else:
            upper = middle

    power = (
        lower + upper
    ) / 2.0

    probabilities = inverse ** power
    probabilities /= probabilities.sum()

    return probabilities


def devig_probabilities(
    odds,
    method,
):
    method = str(
        method
    ).strip().upper()

    if method == "MULTIPLICATIVE":
        return multiplicative_devig(
            odds
        )

    if method == "POWER":
        return power_devig(
            odds
        )

    raise ValueError(
        f"Unsupported de-vig method: {method}"
    )


# ============================================================
# 4. Input normalization
# ============================================================

def normalize_one_x_two(
    raw,
    label,
    required=False,
):
    if raw is None or raw == "":
        if required:
            raise ValueError(
                f"{label} is required."
            )

        return None

    if not isinstance(raw, dict):
        raise ValueError(
            f"{label} must be an object."
        )

    home = (
        raw.get("home")
        if raw.get("home") is not None
        else raw.get("H")
    )

    draw = (
        raw.get("draw")
        if raw.get("draw") is not None
        else raw.get("D")
    )

    away = (
        raw.get("away")
        if raw.get("away") is not None
        else raw.get("A")
    )

    return {
        "home": validate_odds(
            home,
            f"{label} home",
        ),
        "draw": validate_odds(
            draw,
            f"{label} draw",
        ),
        "away": validate_odds(
            away,
            f"{label} away",
        ),
    }


def normalize_ah_rows(
    rows,
    label,
):
    if rows is None:
        return []

    if not isinstance(rows, list):
        raise ValueError(
            f"{label} must be a list."
        )

    output = []
    seen_lines = set()

    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(
                f"{label} row {index + 1} "
                "must be an object."
            )

        line = validate_quarter_line(
            row.get("line"),
            f"{label} row {index + 1} line",
        )

        if line in seen_lines:
            raise ValueError(
                f"{label} contains duplicate "
                f"line {line:g}."
            )

        seen_lines.add(line)

        output.append({
            "line": line,
            "home": validate_odds(
                row.get("home"),
                f"{label} {line:g} home",
            ),
            "away": validate_odds(
                row.get("away"),
                f"{label} {line:g} away",
            ),
        })

    return sorted(
        output,
        key=lambda item: item["line"],
    )


def normalize_ou_rows(
    rows,
    label,
):
    if rows is None:
        return []

    if not isinstance(rows, list):
        raise ValueError(
            f"{label} must be a list."
        )

    output = []
    seen_lines = set()

    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(
                f"{label} row {index + 1} "
                "must be an object."
            )

        line = validate_quarter_line(
            row.get("line"),
            f"{label} row {index + 1} line",
        )

        if line in seen_lines:
            raise ValueError(
                f"{label} contains duplicate "
                f"line {line:g}."
            )

        seen_lines.add(line)

        output.append({
            "line": line,
            "over": validate_odds(
                row.get("over"),
                f"{label} {line:g} over",
            ),
            "under": validate_odds(
                row.get("under"),
                f"{label} {line:g} under",
            ),
        })

    return sorted(
        output,
        key=lambda item: item["line"],
    )


def normalize_hhad_rows(
    rows,
    label,
):
    if rows is None:
        return []

    if not isinstance(rows, list):
        raise ValueError(
            f"{label} must be a list."
        )

    output = []
    seen_lines = set()

    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(
                f"{label} row {index + 1} "
                "must be an object."
            )

        line = validate_integer_line(
            row.get("line"),
            f"{label} row {index + 1} handicap",
        )

        if line in seen_lines:
            raise ValueError(
                f"{label} contains duplicate "
                f"handicap {line:g}."
            )

        seen_lines.add(line)

        output.append({
            "line": line,
            "home": validate_odds(
                row.get("home"),
                f"{label} {line:g} home",
            ),
            "draw": validate_odds(
                row.get("draw"),
                f"{label} {line:g} draw",
            ),
            "away": validate_odds(
                row.get("away"),
                f"{label} {line:g} away",
            ),
        })

    return sorted(
        output,
        key=lambda item: item["line"],
    )


def normalize_team_ou_rows(
    rows,
    label,
):
    if rows is None:
        return []

    if not isinstance(rows, list):
        raise ValueError(
            f"{label} must be a list."
        )

    output = []
    seen = set()

    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(
                f"{label} row {index + 1} "
                "must be an object."
            )

        team = normalize_team(
            row.get("team")
        )

        line = validate_quarter_line(
            row.get("line"),
            f"{label} row {index + 1} line",
        )

        key = (team, line)

        if key in seen:
            raise ValueError(
                f"{label} contains duplicate "
                f"{team} line {line:g}."
            )

        seen.add(key)

        output.append({
            "team": team,
            "line": line,
            "over": validate_odds(
                row.get("over"),
                f"{label} {team} {line:g} over",
            ),
            "under": validate_odds(
                row.get("under"),
                f"{label} {team} {line:g} under",
            ),
        })

    return sorted(
        output,
        key=lambda item: (
            item["team"],
            item["line"],
        ),
    )


def empty_period_markets():
    return {
        "1X2": None,
        "AH": [],
        "OU": [],
        "HHAD": [],
        "TEAM_OU": [],
    }


def normalize_period_market_block(
    raw,
    label,
):
    if raw is None:
        return empty_period_markets()

    if not isinstance(raw, dict):
        raise ValueError(
            f"{label} markets must be an object."
        )

    team_rows = raw.get(
        "TEAM_OU",
        raw.get(
            "TEAM_TOTAL",
            raw.get("TEAM_HILO", []),
        ),
    )

    hhad_rows = raw.get(
        "HHAD",
        raw.get(
            "HANDICAP_HAD",
            [],
        ),
    )

    return {
        "1X2": normalize_one_x_two(
            raw.get(
                "1X2",
                raw.get("HAD"),
            ),
            f"{label} 1X2",
        ),
        "AH": normalize_ah_rows(
            raw.get("AH", []),
            f"{label} AH",
        ),
        "OU": normalize_ou_rows(
            raw.get("OU", []),
            f"{label} O/U",
        ),
        "HHAD": normalize_hhad_rows(
            hhad_rows,
            f"{label} Handicap HAD",
        ),
        "TEAM_OU": normalize_team_ou_rows(
            team_rows,
            f"{label} Team O/U",
        ),
    }


def period_has_market_data(markets):
    return bool(
        markets.get("1X2")
        or markets.get("AH")
        or markets.get("OU")
        or markets.get("HHAD")
        or markets.get("TEAM_OU")
    )


def legacy_pinnacle_to_sharp_books(
    pinnacle,
):
    if not isinstance(pinnacle, dict):
        raise ValueError(
            "Pinnacle input must be an object."
        )

    if "FT" in pinnacle or "HT" in pinnacle:
        markets = copy.deepcopy(
            pinnacle
        )
    else:
        markets = {
            "FT": copy.deepcopy(
                pinnacle
            )
        }

    return [{
        "key": "pinnacle",
        "title": "Pinnacle",
        "markets": markets,
    }]


def normalize_sharp_books(data):
    sharp_books = data.get(
        "sharp_books"
    )

    if sharp_books is None:
        pinnacle = data.get(
            "pinnacle"
        )

        if pinnacle is None:
            raise ValueError(
                "Provide sharp_books or legacy "
                "pinnacle input."
            )

        sharp_books = (
            legacy_pinnacle_to_sharp_books(
                pinnacle
            )
        )

    if (
        not isinstance(sharp_books, list)
        or not sharp_books
    ):
        raise ValueError(
            "sharp_books must be a "
            "non-empty list."
        )

    output = []
    seen_keys = set()

    for index, raw_book in enumerate(
        sharp_books
    ):
        if not isinstance(raw_book, dict):
            raise ValueError(
                f"Sharp book #{index + 1} "
                "must be an object."
            )

        key = str(
            raw_book.get(
                "key",
                f"source_{index + 1}",
            )
        ).strip().lower()

        if not key:
            raise ValueError(
                f"Sharp book #{index + 1} "
                "requires a key."
            )

        if key in seen_keys:
            raise ValueError(
                f"Duplicate sharp source: {key}"
            )

        seen_keys.add(key)

        title = str(
            raw_book.get(
                "title",
                key,
            )
        ).strip()

        raw_markets = raw_book.get(
            "markets",
            raw_book,
        )

        if not isinstance(raw_markets, dict):
            raise ValueError(
                f"{title} markets must "
                "be an object."
            )

        if (
            "FT" in raw_markets
            or "HT" in raw_markets
        ):
            ft_raw = raw_markets.get("FT")
            ht_raw = raw_markets.get("HT")
        else:
            ft_raw = raw_markets
            ht_raw = None

        periods = {
            "FT": normalize_period_market_block(
                ft_raw,
                f"{title} FT",
            ),
            "HT": normalize_period_market_block(
                ht_raw,
                f"{title} HT",
            ),
        }

        if not period_has_market_data(
            periods["FT"]
        ):
            raise ValueError(
                f"{title} requires FT market data."
            )

        output.append({
            "key": key,
            "title": title,
            "timestamp": raw_book.get(
                "timestamp"
            ),
            "markets": periods,
        })

    return output


def default_candidate_label(
    match,
    period,
    market,
    selection,
    line,
    team=None,
):
    home = match["home"]
    away = match["away"]

    period_text = (
        "全場"
        if period == "FT"
        else "半場"
    )

    if market == "1X2":
        if selection == "HOME":
            return f"{home} {period_text}勝"

        if selection == "DRAW":
            return f"{period_text}和"

        return f"{away} {period_text}勝"

    if market == "AH":
        if selection == "HOME":
            return (
                f"{home} {period_text} "
                f"{float(line):+g}"
            )

        return (
            f"{away} {period_text} "
            f"{-float(line):+g}"
        )

    if market == "HHAD":
        if selection == "HOME":
            side = f"{home} 勝"
        elif selection == "DRAW":
            side = "和"
        else:
            side = f"{away} 勝"

        return (
            f"全場讓球主客和 "
            f"主隊{float(line):+g}：{side}"
        )

    side = (
        "大"
        if selection == "OVER"
        else "細"
    )

    if market == "TEAM_OU":
        team_name = (
            home
            if team == "HOME"
            else away
        )

        return (
            f"{team_name} {period_text}入球 "
            f"{side} {float(line):g}"
        )

    return (
        f"{period_text}入球 "
        f"{side} {float(line):g}"
    )


def normalize_hkjc_markets(
    raw_markets,
    match,
):
    if (
        not isinstance(raw_markets, list)
        or not raw_markets
    ):
        raise ValueError(
            "At least one HKJC candidate "
            "market is required."
        )

    output = []
    seen_ids = set()

    for index, raw in enumerate(
        raw_markets
    ):
        if not isinstance(raw, dict):
            raise ValueError(
                f"HKJC market #{index + 1} "
                "must be an object."
            )

        period = normalize_period(
            raw.get("period", "FT")
        )

        market = normalize_market(
            raw.get("market")
        )

        if (
            period == "HT"
            and market == "HHAD"
        ):
            raise ValueError(
                "HT Handicap HAD is not supported. "
                "Use HT AH."
            )

        selection = normalize_selection(
            market,
            raw.get("selection"),
        )

        team = None

        if market == "TEAM_OU":
            team = normalize_team(
                raw.get("team")
            )

        if market in {
            "AH",
            "OU",
            "TEAM_OU",
        }:
            line = validate_quarter_line(
                raw.get("line"),
                f"HKJC market #{index + 1} line",
            )
        elif market == "HHAD":
            line = validate_integer_line(
                raw.get("line"),
                f"HKJC market #{index + 1} handicap",
            )
        else:
            line = None

        market_id = str(
            raw.get(
                "id",
                f"M{index + 1:03d}",
            )
        ).strip()

        if market_id in seen_ids:
            raise ValueError(
                f"Duplicate HKJC market ID: "
                f"{market_id}"
            )

        seen_ids.add(market_id)

        odds = validate_odds(
            raw.get("odds"),
            f"HKJC {market_id}",
        )

        label = str(
            raw.get(
                "label",
                default_candidate_label(
                    match,
                    period,
                    market,
                    selection,
                    line,
                    team,
                ),
            )
        ).strip()

        output.append({
            "id": market_id,
            "label": label,
            "period": period,
            "market": market,
            "selection": selection,
            "line": line,
            "team": team,
            "odds": odds,
            "candidate_key": candidate_key(
                period,
                market,
                selection,
                line,
                team,
            ),
            "group_key": market_group_key(
                period,
                market,
                line,
                team,
            ),
            "family_key": market_family_key(
                period,
                market,
                team,
            ),
            "original_index": index,
        })

    return output


def normalize_support_thresholds(
    raw,
):
    if raw is None:
        raw = {}

    if not isinstance(raw, dict):
        raise ValueError(
            "support_thresholds must be an object."
        )

    output = {}

    for period in ["FT", "HT"]:
        section = raw.get(
            period,
            raw.get(period.lower(), {}),
        )

        if section is None:
            section = {}

        if not isinstance(section, dict):
            raise ValueError(
                f"{period} support thresholds "
                "must be an object."
            )

        output[period] = {
            "max_prior_spread": (
                validate_probability_threshold(
                    section.get(
                        "max_prior_spread"
                    ),
                    f"{period} maximum prior spread",
                )
            ),
            "max_feasible_width": (
                validate_probability_threshold(
                    section.get(
                        "max_feasible_width"
                    ),
                    f"{period} maximum feasible width",
                )
            ),
            "max_maxent_gap": (
                validate_probability_threshold(
                    section.get(
                        "max_maxent_gap"
                    ),
                    f"{period} maximum MaxEnt gap",
                )
            ),
            "max_hidden_line_absolute_error": (
                validate_probability_threshold(
                    section.get(
                        "max_hidden_line_absolute_error"
                    ),
                    (
                        f"{period} maximum hidden-line "
                        "absolute error"
                    ),
                )
            ),
        }

    return output


def validate_input_data(input_data):
    if not isinstance(input_data, dict):
        raise ValueError(
            "Input must be an object."
        )

    data = copy.deepcopy(
        input_data
    )

    match_raw = data.get(
        "match",
        {},
    )

    if not isinstance(match_raw, dict):
        raise ValueError(
            "match must be an object."
        )

    home = str(
        match_raw.get("home", "")
    ).strip()

    away = str(
        match_raw.get("away", "")
    ).strip()

    if not home or not away:
        raise ValueError(
            "Home and away teams are required."
        )

    if home.casefold() == away.casefold():
        raise ValueError(
            "Home and away teams cannot match."
        )

    match = {
        "name": str(
            match_raw.get(
                "name",
                f"{home} vs {away}",
            )
        ).strip(),
        "home": home,
        "away": away,
        "competition": str(
            match_raw.get(
                "competition",
                "",
            )
        ).strip(),
        "kickoff": str(
            match_raw.get(
                "kickoff",
                "",
            )
        ).strip(),
        "snapshot_time": str(
            match_raw.get(
                "snapshot_time",
                "",
            )
        ).strip(),
    }

    sharp_books = normalize_sharp_books(
        data
    )

    markets = normalize_hkjc_markets(
        data.get("hkjc_markets"),
        match,
    )

    settings_raw = data.get(
        "settings",
        {},
    )

    if not isinstance(settings_raw, dict):
        raise ValueError(
            "settings must be an object."
        )

    minimum_odds = validate_odds(
        settings_raw.get(
            "minimum_odds",
            1.50,
        ),
        "Minimum HKJC odds",
    )

    maximum_odds_raw = settings_raw.get(
        "maximum_odds"
    )

    maximum_odds = (
        None
        if maximum_odds_raw in {None, ""}
        else validate_odds(
            maximum_odds_raw,
            "Maximum HKJC odds",
        )
    )

    if (
        maximum_odds is not None
        and maximum_odds < minimum_odds
    ):
        raise ValueError(
            "Maximum odds cannot be below "
            "minimum odds."
        )

    maximum_recommendations = int(
        settings_raw.get(
            "max_recommendations",
            CONFIG[
                "DEFAULT_MAX_RECOMMENDATIONS"
            ],
        )
    )

    if not (
        1 <= maximum_recommendations <= 10
    ):
        raise ValueError(
            "Maximum recommendations must "
            "be between 1 and 10."
        )

    minimum_shortlist_hit = (
        validate_probability_threshold(
            settings_raw.get(
                "minimum_shortlist_hit_probability",
                settings_raw.get(
                    "minimum_official_hit_probability",
                    CONFIG[
                        "DEFAULT_MINIMUM_SHORTLIST_HIT"
                    ],
                ),
            ),
            "Minimum shortlist hit probability",
            allow_none=False,
        )
    )

    correct_score_count = int(
        settings_raw.get(
            "correct_score_count",
            CONFIG[
                "DEFAULT_CORRECT_SCORE_COUNT"
            ],
        )
    )

    if not (
        0 <= correct_score_count <= 5
    ):
        raise ValueError(
            "Correct-score count must "
            "be between 0 and 5."
        )

    devig_methods_raw = settings_raw.get(
        "devig_methods",
        CONFIG[
            "DEFAULT_DEVIG_METHODS"
        ],
    )

    if isinstance(devig_methods_raw, str):
        devig_methods_raw = [
            item.strip()
            for item
            in devig_methods_raw.split(",")
            if item.strip()
        ]

    devig_methods = tuple(
        dict.fromkeys(
            str(item).strip().upper()
            for item in devig_methods_raw
        )
    )

    if not devig_methods:
        raise ValueError(
            "At least one de-vig method "
            "is required."
        )

    for method in devig_methods:
        if method not in {
            "MULTIPLICATIVE",
            "POWER",
        }:
            raise ValueError(
                f"Unsupported de-vig method: "
                f"{method}"
            )

    forecast_priors_raw = settings_raw.get(
        "forecast_priors",
        PARAMETRIC_PRIORS,
    )

    if isinstance(forecast_priors_raw, str):
        forecast_priors_raw = [
            item.strip()
            for item
            in forecast_priors_raw.split(",")
            if item.strip()
        ]

    forecast_priors = tuple(
        dict.fromkeys(
            normalize_prior(item)
            for item in forecast_priors_raw
        )
    )

    if not forecast_priors:
        raise ValueError(
            "At least one forecasting prior "
            "is required."
        )

    if any(
        prior not in PARAMETRIC_PRIORS
        for prior in forecast_priors
    ):
        raise ValueError(
            "forecast_priors may contain only "
            "parametric forecasting priors."
        )

    audit_priors_raw = settings_raw.get(
        "audit_priors",
        AUDIT_PRIORS,
    )

    if isinstance(audit_priors_raw, str):
        audit_priors_raw = [
            item.strip()
            for item
            in audit_priors_raw.split(",")
            if item.strip()
        ]

    audit_priors = tuple(
        dict.fromkeys(
            normalize_prior(item)
            for item in audit_priors_raw
        )
    )

    if any(
        prior not in AUDIT_PRIORS
        for prior in audit_priors
    ):
        raise ValueError(
            "audit_priors may contain only "
            "audit priors."
        )

    primary_source = str(
        settings_raw.get(
            "primary_source",
            sharp_books[0]["key"],
        )
    ).strip().lower()

    available_keys = {
        book["key"]
        for book in sharp_books
    }

    if primary_source not in available_keys:
        raise ValueError(
            f"Primary source "
            f"{primary_source!r} "
            "was not supplied."
        )

    ev_rejection_floor_raw = (
        settings_raw.get(
            "ev_rejection_floor"
        )
    )

    if ev_rejection_floor_raw in {
        None,
        "",
    }:
        ev_rejection_floor = None
    else:
        ev_rejection_floor = float(
            ev_rejection_floor_raw
        )

        if (
            not math.isfinite(
                ev_rejection_floor
            )
            or ev_rejection_floor < -1.0
        ):
            raise ValueError(
                "Invalid EV rejection floor."
            )

    support_thresholds = (
        normalize_support_thresholds(
            settings_raw.get(
                "support_thresholds"
            )
        )
    )

    require_market_supported = bool(
        settings_raw.get(
            "require_market_supported_for_shortlist",
            False,
        )
    )

    features_raw = settings_raw.get(
        "features",
        {},
    )

    if not isinstance(features_raw, dict):
        features_raw = {}

    features = {
        "quality_gate": bool(
            features_raw.get(
                "quality_gate",
                True,
            )
        ),
        "stress_audit": bool(
            features_raw.get(
                "stress_audit",
                True,
            )
        ),
        "family_out_audit": bool(
            features_raw.get(
                "family_out_audit",
                True,
            )
        ),
        "adaptive_grids": bool(
            features_raw.get(
                "adaptive_grids",
                True,
            )
        ),
        "ht_ft_coherence": bool(
            features_raw.get(
                "ht_ft_coherence",
                True,
            )
        ),
        "feasible_bounds": bool(
            features_raw.get(
                "feasible_bounds",
                True,
            )
        ),
        "maxent_audit": bool(
            features_raw.get(
                "maxent_audit",
                True,
            )
        ),
    }

    if not features["maxent_audit"]:
        audit_priors = tuple(
            prior
            for prior in audit_priors
            if prior != "FLAT_GRID_MAXENT"
        )

    return {
        "match": match,
        "sharp_books": sharp_books,
        "hkjc_markets": markets,
        "settings": {
            "minimum_odds": minimum_odds,
            "maximum_odds": maximum_odds,
            "max_recommendations": (
                maximum_recommendations
            ),
            "minimum_shortlist_hit_probability": (
                minimum_shortlist_hit
            ),
            # Backward-compatible alias.
            "minimum_official_hit_probability": (
                minimum_shortlist_hit
            ),
            "correct_score_count": (
                correct_score_count
            ),
            "devig_methods": list(
                devig_methods
            ),
            "forecast_priors": list(
                forecast_priors
            ),
            "audit_priors": list(
                audit_priors
            ),
            "primary_source": (
                primary_source
            ),
            "ev_rejection_floor": (
                ev_rejection_floor
            ),
            "support_thresholds": (
                support_thresholds
            ),
            "require_market_supported_for_shortlist": (
                require_market_supported
            ),
            "features": features,
        },
    }


# ============================================================
# 5. Sharp-market constraints
# ============================================================

def build_book_constraints(
    book,
    period,
    devig_method,
):
    period = normalize_period(period)

    constraints = []
    markets = book["markets"][period]

    def add_constraint(
        market,
        selection,
        line,
        target,
        odds,
        team=None,
    ):
        constraints.append({
            "period": period,
            "market": market,
            "selection": selection,
            "line": line,
            "team": team,
            "group_key": market_group_key(
                period,
                market,
                line,
                team,
            ),
            "family_key": market_family_key(
                period,
                market,
                team,
            ),
            "candidate_key": candidate_key(
                period,
                market,
                selection,
                line,
                team,
            ),
            "quote_identity": quote_identity(
                period,
                market,
                selection,
                line,
                team,
            ),
            "target": float(target),
            "odds": float(odds),
            "source": book["key"],
            "source_title": book["title"],
            "devig_method": devig_method,
        })

    one_x_two = markets.get("1X2")

    if one_x_two is not None:
        odds = [
            one_x_two["home"],
            one_x_two["draw"],
            one_x_two["away"],
        ]

        probabilities = devig_probabilities(
            odds,
            devig_method,
        )

        for selection, target, price in zip(
            ["HOME", "DRAW", "AWAY"],
            probabilities,
            odds,
        ):
            add_constraint(
                "1X2",
                selection,
                None,
                target,
                price,
            )

    for row in markets.get("AH", []):
        odds = [
            row["home"],
            row["away"],
        ]

        probabilities = devig_probabilities(
            odds,
            devig_method,
        )

        add_constraint(
            "AH",
            "HOME",
            row["line"],
            probabilities[0],
            odds[0],
        )

        add_constraint(
            "AH",
            "AWAY",
            row["line"],
            probabilities[1],
            odds[1],
        )

    for row in markets.get("OU", []):
        odds = [
            row["over"],
            row["under"],
        ]

        probabilities = devig_probabilities(
            odds,
            devig_method,
        )

        add_constraint(
            "OU",
            "OVER",
            row["line"],
            probabilities[0],
            odds[0],
        )

        add_constraint(
            "OU",
            "UNDER",
            row["line"],
            probabilities[1],
            odds[1],
        )

    for row in markets.get("HHAD", []):
        odds = [
            row["home"],
            row["draw"],
            row["away"],
        ]

        probabilities = devig_probabilities(
            odds,
            devig_method,
        )

        for selection, target, price in zip(
            ["HOME", "DRAW", "AWAY"],
            probabilities,
            odds,
        ):
            add_constraint(
                "HHAD",
                selection,
                row["line"],
                target,
                price,
            )

    for row in markets.get(
        "TEAM_OU",
        [],
    ):
        odds = [
            row["over"],
            row["under"],
        ]

        probabilities = devig_probabilities(
            odds,
            devig_method,
        )

        add_constraint(
            "TEAM_OU",
            "OVER",
            row["line"],
            probabilities[0],
            odds[0],
            row["team"],
        )

        add_constraint(
            "TEAM_OU",
            "UNDER",
            row["line"],
            probabilities[1],
            odds[1],
            row["team"],
        )

    return constraints


def constraint_group_count(
    constraints,
):
    return len({
        item["group_key"]
        for item in constraints
    })


def constraint_family_count(
    constraints,
):
    return len({
        item["family_key"]
        for item in constraints
    })


def validate_constraint_subset(
    constraints,
):
    if (
        len(constraints)
        < CONFIG[
            "MINIMUM_CONSTRAINTS"
        ]
    ):
        return False

    if (
        constraint_family_count(
            constraints
        )
        < CONFIG[
            "MINIMUM_MARKET_FAMILIES"
        ]
    ):
        return False

    return True


def period_identification(
    constraints,
):
    families = {
        item["market"]
        for item in constraints
    }

    directional = bool(
        families
        & {
            "1X2",
            "AH",
            "HHAD",
        }
    )

    goal_level = bool(
        families
        & {
            "OU",
            "TEAM_OU",
        }
    )

    minimum_coverage = bool(
        directional
        and goal_level
        and validate_constraint_subset(
            constraints
        )
    )

    return {
        "families": sorted(families),
        "directional_information": directional,
        "goal_level_information": goal_level,
        "minimum_structural_coverage_met": (
            minimum_coverage
        ),
        # Compatibility only. This no longer means
        # candidate probability is tightly identified.
        "fully_identified": minimum_coverage,
        "warning": (
            None
            if minimum_coverage
            else (
                "Minimum directional and goal-level "
                "market coverage was not met."
            )
        ),
    }


# ============================================================
# 6. Prior distributions
# ============================================================

def independent_poisson_distribution(
    lambda_home,
    lambda_away,
    max_goals,
):
    values = np.arange(
        max_goals + 1
    )

    home_probability = poisson.pmf(
        values,
        lambda_home,
    )

    away_probability = poisson.pmf(
        values,
        lambda_away,
    )

    matrix = np.outer(
        home_probability,
        away_probability,
    )

    total = float(matrix.sum())

    if total <= TOL:
        raise ValueError(
            "Independent Poisson mass is invalid."
        )

    matrix /= total

    return matrix.ravel()


def legal_rho_bounds(
    lambda_home,
    lambda_away,
):
    epsilon = 1e-8

    lower = max(
        -0.30,
        -1.0 / lambda_home + epsilon,
        -1.0 / lambda_away + epsilon,
    )

    upper = min(
        0.30,
        (
            1.0
            / (
                lambda_home
                * lambda_away
            )
        )
        - epsilon,
        1.0 - epsilon,
    )

    if lower >= upper:
        raise ValueError(
            "No legal Dixon-Coles rho interval."
        )

    return lower, upper


def decode_rho(
    lambda_home,
    lambda_away,
    raw_parameter,
):
    lower, upper = legal_rho_bounds(
        lambda_home,
        lambda_away,
    )

    return (
        lower
        + sigmoid(raw_parameter)
        * (upper - lower)
    )


def dixon_coles_distribution(
    lambda_home,
    lambda_away,
    rho,
    max_goals,
):
    values = np.arange(
        max_goals + 1
    )

    home_probability = poisson.pmf(
        values,
        lambda_home,
    )

    away_probability = poisson.pmf(
        values,
        lambda_away,
    )

    matrix = np.outer(
        home_probability,
        away_probability,
    )

    tau_00 = (
        1.0
        - lambda_home
        * lambda_away
        * rho
    )

    tau_10 = (
        1.0
        + lambda_away * rho
    )

    tau_01 = (
        1.0
        + lambda_home * rho
    )

    tau_11 = 1.0 - rho

    if min(
        tau_00,
        tau_10,
        tau_01,
        tau_11,
    ) <= 0:
        raise ValueError(
            "Illegal Dixon-Coles "
            "low-score adjustment."
        )

    matrix[0, 0] *= tau_00
    matrix[1, 0] *= tau_10
    matrix[0, 1] *= tau_01
    matrix[1, 1] *= tau_11

    total = float(matrix.sum())

    if total <= TOL:
        raise ValueError(
            "Dixon-Coles mass is invalid."
        )

    matrix /= total

    return matrix.ravel()


def compoisson_marginal(
    rate,
    dispersion,
    max_goals,
):
    if rate <= 0.0:
        raise ValueError(
            "COM-Poisson rate must be positive."
        )

    if dispersion <= 0.0:
        raise ValueError(
            "COM-Poisson dispersion must "
            "be positive."
        )

    values = np.arange(
        max_goals + 1,
        dtype=float,
    )

    log_weights = (
        values * math.log(rate)
        - dispersion
        * gammaln(values + 1.0)
    )

    log_weights -= logsumexp(
        log_weights
    )

    probabilities = np.exp(
        log_weights
    )

    probabilities /= probabilities.sum()

    return probabilities


def compoisson_shared_distribution(
    lambda_home,
    lambda_away,
    dispersion,
    max_goals,
):
    home_probability = compoisson_marginal(
        lambda_home,
        dispersion,
        max_goals,
    )

    away_probability = compoisson_marginal(
        lambda_away,
        dispersion,
        max_goals,
    )

    matrix = np.outer(
        home_probability,
        away_probability,
    )

    matrix /= matrix.sum()

    return matrix.ravel()


def flat_grid_distribution(
    max_goals,
):
    state_count = (
        max_goals + 1
    ) ** 2

    return np.full(
        state_count,
        1.0 / state_count,
        dtype=float,
    )


# ============================================================
# 7. Constraint enrichment and prior fitting
# ============================================================

def enrich_constraints(
    constraints,
    max_goals,
):
    home_scores, away_scores = (
        score_arrays(max_goals)
    )

    enriched = []
    matrix_rows = []
    targets = []

    for constraint in constraints:
        a_coeff, b_coeff = (
            settlement_coefficients(
                home_scores,
                away_scores,
                constraint["market"],
                constraint["selection"],
                constraint["line"],
                constraint.get("team"),
            )
        )

        target = float(
            constraint["target"]
        )

        item = dict(
            constraint
        )

        item["a_coeff"] = a_coeff
        item["b_coeff"] = b_coeff

        enriched.append(item)

        # q = E[a] / (1 - E[b])
        # E[a + q b] = q
        matrix_rows.append(
            a_coeff
            + target * b_coeff
        )

        targets.append(target)

    return (
        enriched,
        np.vstack(
            matrix_rows
        ).astype(float),
        np.asarray(
            targets,
            dtype=float,
        ),
    )


def prior_residuals(
    probabilities,
    enriched,
):
    return np.asarray([
        (
            effective_fair_probability(
                probabilities,
                item["a_coeff"],
                item["b_coeff"],
            )
            - item["target"]
        )
        for item in enriched
    ])


def fit_parametric_prior(
    constraints,
    max_goals,
    prior_type,
):
    prior_type = normalize_prior(
        prior_type
    )

    if prior_type not in PARAMETRIC_PRIORS:
        raise ValueError(
            f"{prior_type} is not a "
            "parametric forecasting prior."
        )

    enriched, _, _ = enrich_constraints(
        constraints,
        max_goals,
    )

    starts = [
        (0.35, 0.25),
        (0.55, 0.45),
        (0.80, 0.60),
        (1.10, 1.10),
        (1.50, 0.90),
        (0.90, 1.50),
        (1.90, 1.20),
        (1.20, 1.90),
    ]

    min_rate = CONFIG[
        "MIN_GOAL_RATE"
    ]

    max_rate = CONFIG[
        "MAX_GOAL_RATE"
    ]

    results = []

    if prior_type == "INDEPENDENT_POISSON":
        lower_bounds = np.asarray([
            math.log(min_rate),
            math.log(min_rate),
        ])

        upper_bounds = np.asarray([
            math.log(max_rate),
            math.log(max_rate),
        ])

        def decode(parameters):
            lambda_home = math.exp(
                float(parameters[0])
            )

            lambda_away = math.exp(
                float(parameters[1])
            )

            probabilities = (
                independent_poisson_distribution(
                    lambda_home,
                    lambda_away,
                    max_goals,
                )
            )

            return {
                "lambda_home": lambda_home,
                "lambda_away": lambda_away,
                "rho": None,
                "dispersion": 1.0,
                "probabilities": probabilities,
            }

        parameter_starts = [
            np.asarray([
                math.log(home_start),
                math.log(away_start),
            ])
            for home_start, away_start
            in starts
        ]

    elif prior_type == "DIXON_COLES":
        lower_bounds = np.asarray([
            math.log(min_rate),
            math.log(min_rate),
            -10.0,
        ])

        upper_bounds = np.asarray([
            math.log(max_rate),
            math.log(max_rate),
            10.0,
        ])

        def decode(parameters):
            lambda_home = math.exp(
                float(parameters[0])
            )

            lambda_away = math.exp(
                float(parameters[1])
            )

            rho = decode_rho(
                lambda_home,
                lambda_away,
                parameters[2],
            )

            probabilities = (
                dixon_coles_distribution(
                    lambda_home,
                    lambda_away,
                    rho,
                    max_goals,
                )
            )

            return {
                "lambda_home": lambda_home,
                "lambda_away": lambda_away,
                "rho": rho,
                "dispersion": 1.0,
                "probabilities": probabilities,
            }

        parameter_starts = [
            np.asarray([
                math.log(home_start),
                math.log(away_start),
                0.0,
            ])
            for home_start, away_start
            in starts
        ]

    else:
        minimum_dispersion = CONFIG[
            "MIN_COM_DISPERSION"
        ]

        maximum_dispersion = CONFIG[
            "MAX_COM_DISPERSION"
        ]

        lower_bounds = np.asarray([
            math.log(min_rate),
            math.log(min_rate),
            math.log(minimum_dispersion),
        ])

        upper_bounds = np.asarray([
            math.log(max_rate),
            math.log(max_rate),
            math.log(maximum_dispersion),
        ])

        def decode(parameters):
            lambda_home = math.exp(
                float(parameters[0])
            )

            lambda_away = math.exp(
                float(parameters[1])
            )

            dispersion = math.exp(
                float(parameters[2])
            )

            probabilities = (
                compoisson_shared_distribution(
                    lambda_home,
                    lambda_away,
                    dispersion,
                    max_goals,
                )
            )

            return {
                "lambda_home": lambda_home,
                "lambda_away": lambda_away,
                "rho": None,
                "dispersion": dispersion,
                "probabilities": probabilities,
            }

        dispersion_starts = [
            0.70,
            1.00,
            1.35,
        ]

        parameter_starts = [
            np.asarray([
                math.log(home_start),
                math.log(away_start),
                math.log(dispersion_start),
            ])
            for home_start, away_start
            in starts
            for dispersion_start
            in dispersion_starts
        ]

    def residual_function(parameters):
        try:
            decoded = decode(
                parameters
            )

            return prior_residuals(
                decoded["probabilities"],
                enriched,
            )

        except Exception:
            return np.full(
                len(enriched),
                100.0,
                dtype=float,
            )

    for start in parameter_starts:
        try:
            result = least_squares(
                residual_function,
                x0=start,
                bounds=(
                    lower_bounds,
                    upper_bounds,
                ),
                method="trf",
                max_nfev=CONFIG[
                    "FIT_MAX_EVALUATIONS"
                ],
                ftol=1e-11,
                xtol=1e-11,
                gtol=1e-11,
            )
        except Exception:
            continue

        if np.all(
            np.isfinite(result.fun)
        ):
            results.append(result)

    if not results:
        raise RuntimeError(
            f"{prior_type} prior fitting failed."
        )

    best = min(
        results,
        key=lambda item: (
            float(
                np.max(
                    np.abs(item.fun)
                )
            ),
            float(
                np.mean(
                    item.fun ** 2
                )
            ),
        ),
    )

    decoded = decode(
        best.x
    )

    dispersion = decoded[
        "dispersion"
    ]

    dispersion_at_bound = False

    if prior_type == "COM_POISSON_SHARED":
        dispersion_at_bound = bool(
            math.isclose(
                dispersion,
                CONFIG[
                    "MIN_COM_DISPERSION"
                ],
                rel_tol=0.0,
                abs_tol=1e-4,
            )
            or math.isclose(
                dispersion,
                CONFIG[
                    "MAX_COM_DISPERSION"
                ],
                rel_tol=0.0,
                abs_tol=1e-4,
            )
        )

    return {
        "prior_type": prior_type,
        **decoded,
        "maximum_prior_residual": float(
            np.max(
                np.abs(best.fun)
            )
        ),
        "prior_rmse": float(
            math.sqrt(
                np.mean(
                    best.fun ** 2
                )
            )
        ),
        "optimizer_success": bool(
            best.success
        ),
        "optimizer_message": str(
            best.message
        ),
        "dispersion_at_bound": (
            dispersion_at_bound
        ),
    }


def construct_prior(
    constraints,
    max_goals,
    prior_type,
):
    prior_type = normalize_prior(
        prior_type
    )

    if prior_type == "FLAT_GRID_MAXENT":
        return {
            "prior_type": prior_type,
            "lambda_home": None,
            "lambda_away": None,
            "rho": None,
            "dispersion": None,
            "probabilities": (
                flat_grid_distribution(
                    max_goals
                )
            ),
            "maximum_prior_residual": None,
            "prior_rmse": None,
            "optimizer_success": True,
            "optimizer_message": (
                "No parametric fit required."
            ),
            "dispersion_at_bound": False,
        }

    return fit_parametric_prior(
        constraints,
        max_goals,
        prior_type,
    )


# ============================================================
# 8. Entropy reconstruction
# ============================================================

def solve_minimum_slack(
    matrix,
    targets,
):
    matrix = np.asarray(
        matrix,
        dtype=float,
    )

    targets = np.asarray(
        targets,
        dtype=float,
    )

    quote_count, state_count = (
        matrix.shape
    )

    objective = np.zeros(
        state_count + 1,
        dtype=float,
    )

    objective[-1] = 1.0

    positive = np.hstack([
        matrix,
        -np.ones(
            (quote_count, 1),
            dtype=float,
        ),
    ])

    negative = np.hstack([
        -matrix,
        -np.ones(
            (quote_count, 1),
            dtype=float,
        ),
    ])

    equality = np.zeros(
        (1, state_count + 1),
        dtype=float,
    )

    equality[
        0,
        :state_count
    ] = 1.0

    result = linprog(
        c=objective,
        A_ub=np.vstack([
            positive,
            negative,
        ]),
        b_ub=np.concatenate([
            targets,
            -targets,
        ]),
        A_eq=equality,
        b_eq=np.asarray(
            [1.0],
            dtype=float,
        ),
        bounds=(
            [(0.0, 1.0)]
            * state_count
            + [(0.0, None)]
        ),
        method="highs",
    )

    if not result.success:
        raise RuntimeError(
            "Minimum-slack market LP failed: "
            f"{result.message}"
        )

    fallback = np.asarray(
        result.x[:state_count],
        dtype=float,
    )

    fallback = np.maximum(
        fallback,
        0.0,
    )

    fallback /= fallback.sum()

    return {
        "minimum_slack": float(
            result.x[-1]
        ),
        "fallback_probabilities": fallback,
        "solver_message": str(
            result.message
        ),
    }


def entropy_projection(
    prior,
    matrix,
    targets,
):
    prior = np.asarray(
        prior,
        dtype=float,
    )

    prior = np.maximum(
        prior,
        1e-300,
    )

    prior /= prior.sum()

    matrix = np.asarray(
        matrix,
        dtype=float,
    )

    targets = np.asarray(
        targets,
        dtype=float,
    )

    slack_result = solve_minimum_slack(
        matrix,
        targets,
    )

    minimum_slack = slack_result[
        "minimum_slack"
    ]

    allowed_slack = (
        minimum_slack
        + CONFIG[
            "PROJECTION_BUFFER"
        ]
    )

    lower = targets - allowed_slack
    upper = targets + allowed_slack

    inequality_matrix = np.vstack([
        matrix,
        -matrix,
    ])

    inequality_bound = np.concatenate([
        upper,
        -lower,
    ])

    log_prior = np.log(prior)

    def objective_and_gradient(
        multipliers,
    ):
        transformed = (
            log_prior
            - inequality_matrix.T
            @ multipliers
        )

        log_normalizer = logsumexp(
            transformed
        )

        probabilities = np.exp(
            transformed
            - log_normalizer
        )

        objective = (
            log_normalizer
            + inequality_bound
            @ multipliers
        )

        gradient = (
            inequality_bound
            - inequality_matrix
            @ probabilities
        )

        return (
            float(objective),
            np.asarray(
                gradient,
                dtype=float,
            ),
        )

    initial = np.zeros(
        len(inequality_bound),
        dtype=float,
    )

    result = minimize(
        fun=lambda values: (
            objective_and_gradient(
                values
            )[0]
        ),
        x0=initial,
        jac=lambda values: (
            objective_and_gradient(
                values
            )[1]
        ),
        method="L-BFGS-B",
        bounds=[
            (0.0, None)
            for _ in initial
        ],
        options={
            "maxiter": CONFIG[
                "PROJECTION_MAX_ITERATIONS"
            ],
            "ftol": 1e-12,
            "gtol": 1e-9,
            "maxls": 40,
        },
    )

    transformed = (
        log_prior
        - inequality_matrix.T
        @ result.x
    )

    probabilities = np.exp(
        transformed
        - logsumexp(transformed)
    )

    probabilities /= probabilities.sum()

    residual = float(
        np.max(
            np.abs(
                matrix @ probabilities
                - targets
            )
        )
    )

    if (
        not result.success
        or np.any(
            ~np.isfinite(probabilities)
        )
        or residual
        > allowed_slack + 1e-6
    ):
        probabilities = (
            slack_result[
                "fallback_probabilities"
            ]
        )

        projection_method = (
            "MINIMUM_SLACK_LP_FALLBACK"
        )

        residual = float(
            np.max(
                np.abs(
                    matrix @ probabilities
                    - targets
                )
            )
        )
    else:
        projection_method = (
            "DUAL_RELATIVE_ENTROPY"
        )

    return {
        "probabilities": probabilities,
        "minimum_slack": minimum_slack,
        "allowed_slack": allowed_slack,
        "maximum_equation_residual": residual,
        "projection_method": projection_method,
        "optimizer_success": bool(
            result.success
        ),
        "optimizer_message": str(
            result.message
        ),
    }


def scenario_quality_status(model):
    if (
        model.get(
            "boundary_mass",
            1.0,
        )
        > CONFIG[
            "BOUNDARY_TOLERANCE"
        ]
    ):
        return "FAIL"

    if (
        model.get("projection_method")
        != "DUAL_RELATIVE_ENTROPY"
    ):
        return "CAUTION"

    if (
        model.get(
            "minimum_slack",
            1.0,
        )
        > CONFIG[
            "PROJECTION_BUFFER"
        ]
    ):
        return "CAUTION"

    if model.get(
        "dispersion_at_bound",
        False,
    ):
        return "CAUTION"

    return "PASS"


def build_market_model(
    constraints,
    max_goals,
    prior_type,
):
    if not validate_constraint_subset(
        constraints
    ):
        raise ValueError(
            "Insufficient independent "
            "market structure."
        )

    prior_fit = construct_prior(
        constraints,
        max_goals,
        prior_type,
    )

    enriched, matrix, targets = (
        enrich_constraints(
            constraints,
            max_goals,
        )
    )

    projection = entropy_projection(
        prior_fit["probabilities"],
        matrix,
        targets,
    )

    probabilities = projection[
        "probabilities"
    ]

    effective_residuals = [
        abs(
            effective_fair_probability(
                probabilities,
                item["a_coeff"],
                item["b_coeff"],
            )
            - item["target"]
        )
        for item in enriched
    ]

    boundary_mass = score_boundary_mass(
        probabilities,
        max_goals,
    )

    output = {
        "probabilities": probabilities,
        "prior_type": prior_fit[
            "prior_type"
        ],
        "prior_role": (
            "FORECAST"
            if prior_type in PARAMETRIC_PRIORS
            else "AUDIT_ONLY"
        ),
        "lambda_home": prior_fit[
            "lambda_home"
        ],
        "lambda_away": prior_fit[
            "lambda_away"
        ],
        "rho": prior_fit["rho"],
        "dispersion": prior_fit[
            "dispersion"
        ],
        "dispersion_at_bound": prior_fit[
            "dispersion_at_bound"
        ],
        "maximum_prior_residual": (
            prior_fit[
                "maximum_prior_residual"
            ]
        ),
        "prior_rmse": (
            prior_fit["prior_rmse"]
        ),
        "minimum_slack": projection[
            "minimum_slack"
        ],
        "allowed_slack": projection[
            "allowed_slack"
        ],
        "maximum_equation_residual": (
            projection[
                "maximum_equation_residual"
            ]
        ),
        "maximum_effective_residual": (
            max(effective_residuals)
            if effective_residuals
            else 0.0
        ),
        "projection_method": projection[
            "projection_method"
        ],
        "constraint_count": len(
            constraints
        ),
        "market_group_count": (
            constraint_group_count(
                constraints
            )
        ),
        "market_family_count": (
            constraint_family_count(
                constraints
            )
        ),
        "boundary_mass": boundary_mass,
        "max_goals": max_goals,
    }

    output["quality_status"] = (
        scenario_quality_status(
            output
        )
    )

    return output


# ============================================================
# 9. Adaptive grids and full scenarios
# ============================================================

def period_grid_limits(period):
    period = normalize_period(period)

    if period == "HT":
        return (
            CONFIG[
                "HT_INITIAL_MAX_GOALS"
            ],
            CONFIG[
                "HT_MAX_GOALS_SAFETY"
            ],
        )

    return (
        CONFIG[
            "FT_INITIAL_MAX_GOALS"
        ],
        CONFIG[
            "FT_MAX_GOALS_SAFETY"
        ],
    )


def configured_priors(data):
    return (
        list(
            data["settings"][
                "forecast_priors"
            ]
        )
        + list(
            data["settings"][
                "audit_priors"
            ]
        )
    )


def build_period_full_scenarios(
    data,
    period,
    constraint_sets,
    max_goals,
):
    scenarios = []
    errors = []

    for book in data["sharp_books"]:
        for method in data[
            "settings"
        ]["devig_methods"]:
            specification_key = (
                period,
                book["key"],
                method,
            )

            constraints = constraint_sets.get(
                specification_key,
                [],
            )

            if not constraints:
                continue

            for prior_type in configured_priors(
                data
            ):
                scenario_id = (
                    f"{period}|{book['key']}|"
                    f"{method}|{prior_type}|FULL"
                )

                try:
                    model = build_market_model(
                        constraints,
                        max_goals,
                        prior_type,
                    )

                    scenarios.append({
                        "id": scenario_id,
                        "period": period,
                        "source": book["key"],
                        "source_title": (
                            book["title"]
                        ),
                        "devig_method": method,
                        "scenario_type": (
                            "FULL_MARKET"
                        ),
                        "excluded_group": None,
                        "excluded_family": None,
                        **model,
                    })

                except Exception as error:
                    errors.append({
                        "scenario_id": scenario_id,
                        "period": period,
                        "source": book["key"],
                        "devig_method": method,
                        "prior_type": prior_type,
                        "error": str(error),
                    })

    return scenarios, errors


def adaptive_period_reconstruction(
    data,
    period,
    constraint_sets,
):
    initial, safety = period_grid_limits(
        period
    )

    features = data[
        "settings"
    ]["features"]

    forecast_priors = set(
        data["settings"][
            "forecast_priors"
        ]
    )

    max_goals = initial
    expansion_history = []

    while True:
        scenarios, errors = (
            build_period_full_scenarios(
                data,
                period,
                constraint_sets,
                max_goals,
            )
        )

        forecast_scenarios = [
            scenario
            for scenario in scenarios
            if scenario["prior_type"]
            in forecast_priors
        ]

        maximum_boundary = (
            max(
                scenario["boundary_mass"]
                for scenario
                in forecast_scenarios
            )
            if forecast_scenarios
            else None
        )

        audit_maximum_boundary = (
            max(
                scenario["boundary_mass"]
                for scenario in scenarios
                if scenario["prior_role"]
                == "AUDIT_ONLY"
            )
            if any(
                scenario["prior_role"]
                == "AUDIT_ONLY"
                for scenario in scenarios
            )
            else None
        )

        expansion_history.append({
            "max_goals": max_goals,
            "forecast_scenario_count": len(
                forecast_scenarios
            ),
            "audit_scenario_count": (
                len(scenarios)
                - len(forecast_scenarios)
            ),
            "maximum_forecast_boundary_mass": (
                maximum_boundary
            ),
            "maximum_audit_boundary_mass": (
                audit_maximum_boundary
            ),
        })

        if not forecast_scenarios:
            return {
                "max_goals": max_goals,
                "scenarios": scenarios,
                "errors": errors,
                "grid_complete": False,
                "expansion_history": (
                    expansion_history
                ),
            }

        if not features["adaptive_grids"]:
            grid_complete = bool(
                maximum_boundary
                <= CONFIG[
                    "BOUNDARY_TOLERANCE"
                ]
            )
            break

        if (
            maximum_boundary
            <= CONFIG[
                "BOUNDARY_TOLERANCE"
            ]
        ):
            grid_complete = True
            break

        if max_goals >= safety:
            grid_complete = False
            break

        max_goals = min(
            safety,
            max_goals
            + CONFIG[
                "GRID_EXPANSION_STEP"
            ],
        )

    if not grid_complete:
        for scenario in scenarios:
            if scenario[
                "prior_role"
            ] == "FORECAST":
                scenario[
                    "quality_status"
                ] = "FAIL"

    return {
        "max_goals": max_goals,
        "scenarios": scenarios,
        "errors": errors,
        "grid_complete": grid_complete,
        "expansion_history": (
            expansion_history
        ),
    }


# ============================================================
# 10. Target-line-out and family-out scenarios
# ============================================================

def direct_target_for_candidate(
    constraints,
    candidate,
):
    values = [
        float(item["target"])
        for item in constraints
        if item["candidate_key"]
        == candidate["candidate_key"]
    ]

    if not values:
        return None

    return float(
        np.median(values)
    )


def build_removed_scenarios(
    data,
    candidate,
    constraint_sets,
    full_scenario_map,
    max_goals,
    removal_type,
):
    scenarios = []
    errors = []

    for book in data["sharp_books"]:
        for method in data[
            "settings"
        ]["devig_methods"]:
            specification_key = (
                candidate["period"],
                book["key"],
                method,
            )

            constraints = constraint_sets.get(
                specification_key,
                [],
            )

            if not constraints:
                continue

            direct_target = (
                direct_target_for_candidate(
                    constraints,
                    candidate,
                )
            )

            if removal_type == "TARGET_GROUP":
                removal_key = candidate[
                    "group_key"
                ]

                present = any(
                    item["group_key"]
                    == removal_key
                    for item in constraints
                )

                reduced_constraints = [
                    item
                    for item in constraints
                    if item["group_key"]
                    != removal_key
                ]

                scenario_type = (
                    "TARGET_GROUP_OUT"
                )

            elif removal_type == "FAMILY":
                removal_key = candidate[
                    "family_key"
                ]

                present = any(
                    item["family_key"]
                    == removal_key
                    for item in constraints
                )

                reduced_constraints = [
                    item
                    for item in constraints
                    if item["family_key"]
                    != removal_key
                ]

                scenario_type = (
                    "WHOLE_FAMILY_OUT"
                )

            else:
                raise ValueError(
                    f"Unsupported removal type: "
                    f"{removal_type}"
                )

            # If the exact target group is unquoted,
            # the full model is already target-line-out.
            #
            # If an entire family is absent, family-out
            # is not testable. The full model must not be
            # misreported as a successful family-out test.
            if not present:
                if removal_type == "FAMILY":
                    errors.append({
                        "period": candidate[
                            "period"
                        ],
                        "source": book["key"],
                        "devig_method": method,
                        "reason": (
                            "FAMILY_NOT_PRESENT_"
                            "NOT_TESTABLE"
                        ),
                        "family": removal_key,
                    })
                    continue

                for prior_type in configured_priors(
                    data
                ):
                    full_scenario = (
                        full_scenario_map.get(
                            (
                                candidate["period"],
                                book["key"],
                                method,
                                prior_type,
                            )
                        )
                    )

                    if full_scenario is None:
                        errors.append({
                            "period": candidate[
                                "period"
                            ],
                            "source": book["key"],
                            "devig_method": method,
                            "prior_type": prior_type,
                            "reason": (
                                "FULL_MODEL_UNAVAILABLE"
                            ),
                        })
                        continue

                    scenarios.append({
                        **full_scenario,
                        "id": (
                            f"{candidate['period']}|"
                            f"{book['key']}|{method}|"
                            f"{prior_type}|UNQUOTED|"
                            f"{removal_key}"
                        ),
                        "scenario_type": (
                            "UNQUOTED_LINE_"
                            "RECONSTRUCTION"
                        ),
                        "excluded_group": None,
                        "excluded_family": None,
                        "direct_devig_target": None,
                        "_constraints": constraints,
                    })

                continue

            if not validate_constraint_subset(
                reduced_constraints
            ):
                errors.append({
                    "period": candidate[
                        "period"
                    ],
                    "source": book["key"],
                    "devig_method": method,
                    "reason": (
                        "INSUFFICIENT_MARKETS_"
                        "AFTER_REMOVAL"
                    ),
                    "removal_type": removal_type,
                    "removal_key": removal_key,
                })
                continue

            for prior_type in configured_priors(
                data
            ):
                scenario_id = (
                    f"{candidate['period']}|"
                    f"{book['key']}|{method}|"
                    f"{prior_type}|{scenario_type}|"
                    f"{removal_key}"
                )

                try:
                    model = build_market_model(
                        reduced_constraints,
                        max_goals,
                        prior_type,
                    )

                    scenarios.append({
                        "id": scenario_id,
                        "period": candidate[
                            "period"
                        ],
                        "source": book["key"],
                        "source_title": (
                            book["title"]
                        ),
                        "devig_method": method,
                        "scenario_type": (
                            scenario_type
                        ),
                        "excluded_group": (
                            removal_key
                            if removal_type
                            == "TARGET_GROUP"
                            else None
                        ),
                        "excluded_family": (
                            removal_key
                            if removal_type
                            == "FAMILY"
                            else None
                        ),
                        "direct_devig_target": (
                            direct_target
                        ),
                        "_constraints": (
                            reduced_constraints
                        ),
                        **model,
                    })

                except Exception as error:
                    errors.append({
                        "scenario_id": scenario_id,
                        "period": candidate[
                            "period"
                        ],
                        "source": book["key"],
                        "devig_method": method,
                        "prior_type": prior_type,
                        "reason": (
                            "REMOVED_MODEL_FAILED"
                        ),
                        "error": str(error),
                    })

    return scenarios, errors


# ============================================================
# 11. Feasible candidate probability bounds
# ============================================================

def event_probability_bounds(
    constraints,
    max_goals,
    event_mask,
):
    if not validate_constraint_subset(
        constraints
    ):
        return {
            "status": "NOT_TESTABLE",
            "minimum_probability": None,
            "maximum_probability": None,
            "width": None,
            "required_slack": None,
            "solver_message": (
                "Insufficient market structure."
            ),
        }

    _, matrix, targets = enrich_constraints(
        constraints,
        max_goals,
    )

    slack_result = solve_minimum_slack(
        matrix,
        targets,
    )

    required_slack = slack_result[
        "minimum_slack"
    ]

    allowed_slack = (
        required_slack
        + CONFIG[
            "LP_NUMERICAL_BUFFER"
        ]
    )

    state_count = matrix.shape[1]

    inequality_matrix = np.vstack([
        matrix,
        -matrix,
    ])

    inequality_bound = np.concatenate([
        targets + allowed_slack,
        -targets + allowed_slack,
    ])

    equality_matrix = np.ones(
        (1, state_count),
        dtype=float,
    )

    equality_bound = np.asarray(
        [1.0],
        dtype=float,
    )

    objective = np.asarray(
        event_mask,
        dtype=float,
    )

    minimum_result = linprog(
        c=objective,
        A_ub=inequality_matrix,
        b_ub=inequality_bound,
        A_eq=equality_matrix,
        b_eq=equality_bound,
        bounds=[
            (0.0, 1.0)
            for _ in range(state_count)
        ],
        method="highs",
    )

    maximum_result = linprog(
        c=-objective,
        A_ub=inequality_matrix,
        b_ub=inequality_bound,
        A_eq=equality_matrix,
        b_eq=equality_bound,
        bounds=[
            (0.0, 1.0)
            for _ in range(state_count)
        ],
        method="highs",
    )

    if (
        not minimum_result.success
        or not maximum_result.success
    ):
        messages = []

        if not minimum_result.success:
            messages.append(
                "MIN: "
                + str(minimum_result.message)
            )

        if not maximum_result.success:
            messages.append(
                "MAX: "
                + str(maximum_result.message)
            )

        return {
            "status": "FAILED",
            "minimum_probability": None,
            "maximum_probability": None,
            "width": None,
            "required_slack": (
                required_slack
            ),
            "solver_message": "; ".join(
                messages
            ),
        }

    minimum_probability = clamp(
        float(minimum_result.fun),
        0.0,
        1.0,
    )

    maximum_probability = clamp(
        -float(maximum_result.fun),
        0.0,
        1.0,
    )

    return {
        "status": "COMPLETED",
        "minimum_probability": (
            minimum_probability
        ),
        "maximum_probability": (
            maximum_probability
        ),
        "width": (
            maximum_probability
            - minimum_probability
        ),
        "required_slack": (
            required_slack
        ),
        "solver_message": (
            "MIN: "
            + str(minimum_result.message)
            + "; MAX: "
            + str(maximum_result.message)
        ),
    }


def build_feasible_bounds_audit(
    candidate_scenarios,
    max_goals,
    hit_mask,
):
    records = []
    processed = set()

    for scenario in candidate_scenarios:
        key = (
            scenario["source"],
            scenario["devig_method"],
        )

        if key in processed:
            continue

        processed.add(key)

        constraints = scenario.get(
            "_constraints"
        )

        if constraints is None:
            continue

        result = event_probability_bounds(
            constraints=constraints,
            max_goals=max_goals,
            event_mask=hit_mask,
        )

        records.append({
            "source": scenario["source"],
            "devig_method": (
                scenario["devig_method"]
            ),
            **result,
        })

    completed = [
        item
        for item in records
        if item["status"] == "COMPLETED"
    ]

    lower_values = [
        item["minimum_probability"]
        for item in completed
    ]

    upper_values = [
        item["maximum_probability"]
        for item in completed
    ]

    width_values = [
        item["width"]
        for item in completed
    ]

    return {
        "status": (
            "COMPLETED"
            if completed
            else "NOT_TESTABLE"
        ),
        "lower": summary_statistics(
            lower_values
        ),
        "upper": summary_statistics(
            upper_values
        ),
        "width": summary_statistics(
            width_values
        ),
        "overall_envelope": {
            "minimum": (
                min(lower_values)
                if lower_values
                else None
            ),
            "maximum": (
                max(upper_values)
                if upper_values
                else None
            ),
            "width": (
                max(upper_values)
                - min(lower_values)
                if lower_values
                and upper_values
                else None
            ),
        },
        "scenario_count": len(
            completed
        ),
        "records": records,
        "interpretation": (
            "This is a market-consistent feasible "
            "range, not a forecast confidence interval."
        ),
    }


# ============================================================
# 12. Structural stress
# ============================================================

def build_uncertainty_profiles(
    constraint_sets,
):
    profiles = {}

    for constraints in constraint_sets.values():
        for item in constraints:
            identity = item[
                "quote_identity"
            ]

            profile = profiles.setdefault(
                identity,
                {
                    "all": [],
                    "by_source": {},
                },
            )

            profile["all"].append(
                float(item["target"])
            )

            profile[
                "by_source"
            ].setdefault(
                item["source"],
                [],
            ).append(
                float(item["target"])
            )

    return profiles


def stress_target_interval(
    constraint,
    level,
    uncertainty_profiles,
):
    target = float(
        constraint["target"]
    )

    profile = uncertainty_profiles.get(
        constraint["quote_identity"],
        {},
    )

    if level == "LIGHT":
        values = [target]

    elif level == "MEDIUM":
        values = (
            profile
            .get("by_source", {})
            .get(
                constraint["source"],
                [target],
            )
        )

    elif level == "HEAVY":
        values = profile.get(
            "all",
            [target],
        )

    else:
        raise ValueError(
            f"Unsupported stress level: {level}"
        )

    lower = clamp(
        min(values),
        0.0,
        1.0,
    )

    upper = clamp(
        max(values),
        0.0,
        1.0,
    )

    return lower, upper


def robust_minimum_event_probability(
    constraints,
    max_goals,
    event_mask,
    level,
    uncertainty_profiles,
):
    if not validate_constraint_subset(
        constraints
    ):
        return {
            "status": "NOT_TESTABLE",
            "minimum_probability": None,
            "required_slack": None,
            "solver_message": (
                "Insufficient market structure."
            ),
        }

    home_scores, away_scores = (
        score_arrays(max_goals)
    )

    state_count = len(
        home_scores
    )

    upper_rows = []
    upper_bounds = []

    for constraint in constraints:
        a_coeff, b_coeff = (
            settlement_coefficients(
                home_scores,
                away_scores,
                constraint["market"],
                constraint["selection"],
                constraint["line"],
                constraint.get("team"),
            )
        )

        lower, upper = (
            stress_target_interval(
                constraint,
                level,
                uncertainty_profiles,
            )
        )

        # Effective probability <= upper.
        upper_rows.append(
            a_coeff
            + upper * b_coeff
        )
        upper_bounds.append(upper)

        # Effective probability >= lower.
        upper_rows.append(
            -(
                a_coeff
                + lower * b_coeff
            )
        )
        upper_bounds.append(-lower)

    inequality_matrix = np.asarray(
        upper_rows,
        dtype=float,
    )

    inequality_bound = np.asarray(
        upper_bounds,
        dtype=float,
    )

    slack_matrix = np.hstack([
        inequality_matrix,
        -np.ones(
            (
                len(inequality_bound),
                1,
            ),
            dtype=float,
        ),
    ])

    equality = np.zeros(
        (1, state_count + 1),
        dtype=float,
    )

    equality[0, :state_count] = 1.0

    slack_objective = np.zeros(
        state_count + 1,
        dtype=float,
    )

    slack_objective[-1] = 1.0

    slack_result = linprog(
        c=slack_objective,
        A_ub=slack_matrix,
        b_ub=inequality_bound,
        A_eq=equality,
        b_eq=np.asarray([1.0]),
        bounds=(
            [(0.0, 1.0)] * state_count
            + [(0.0, None)]
        ),
        method="highs",
    )

    if not slack_result.success:
        return {
            "status": "FAILED",
            "minimum_probability": None,
            "required_slack": None,
            "solver_message": str(
                slack_result.message
            ),
        }

    required_slack = float(
        slack_result.x[-1]
    )

    allowed_slack = (
        required_slack
        + CONFIG[
            "LP_NUMERICAL_BUFFER"
        ]
    )

    objective = np.asarray(
        event_mask,
        dtype=float,
    )

    result = linprog(
        c=objective,
        A_ub=inequality_matrix,
        b_ub=(
            inequality_bound
            + allowed_slack
        ),
        A_eq=np.ones(
            (1, state_count),
            dtype=float,
        ),
        b_eq=np.asarray([1.0]),
        bounds=[
            (0.0, 1.0)
            for _ in range(
                state_count
            )
        ],
        method="highs",
    )

    if not result.success:
        return {
            "status": "FAILED",
            "minimum_probability": None,
            "required_slack": required_slack,
            "solver_message": str(
                result.message
            ),
        }

    return {
        "status": "COMPLETED",
        "minimum_probability": float(
            result.fun
        ),
        "required_slack": required_slack,
        "solver_message": str(
            result.message
        ),
    }


def build_stress_audit(
    candidate_scenarios,
    max_goals,
    hit_mask,
    uncertainty_profiles,
):
    output = {}

    for level in [
        "LIGHT",
        "MEDIUM",
        "HEAVY",
    ]:
        records = []
        processed = set()

        for scenario in candidate_scenarios:
            key = (
                scenario["source"],
                scenario["devig_method"],
            )

            if key in processed:
                continue

            processed.add(key)

            constraints = scenario.get(
                "_constraints"
            )

            if constraints is None:
                continue

            record = (
                robust_minimum_event_probability(
                    constraints=constraints,
                    max_goals=max_goals,
                    event_mask=hit_mask,
                    level=level,
                    uncertainty_profiles=(
                        uncertainty_profiles
                    ),
                )
            )

            records.append({
                "source": scenario["source"],
                "devig_method": scenario[
                    "devig_method"
                ],
                **record,
            })

        probabilities = [
            item["minimum_probability"]
            for item in records
            if item[
                "minimum_probability"
            ] is not None
        ]

        output[level.lower()] = {
            "minimum_hit_probability": (
                min(probabilities)
                if probabilities
                else None
            ),
            "median_hit_probability": (
                float(
                    np.median(probabilities)
                )
                if probabilities
                else None
            ),
            "scenario_count": len(
                probabilities
            ),
            "records": records,
        }

    return output


# ============================================================
# 13. HT-FT coherence
# ============================================================

def minimum_ht_ft_violation(
    ht_probabilities,
    ht_max_goals,
    ft_probabilities,
    ft_max_goals,
):
    ht_probabilities = np.asarray(
        ht_probabilities,
        dtype=float,
    ).reshape(-1)

    ft_probabilities = np.asarray(
        ft_probabilities,
        dtype=float,
    ).reshape(-1)

    expected_ht_count = (
        int(ht_max_goals) + 1
    ) ** 2

    expected_ft_count = (
        int(ft_max_goals) + 1
    ) ** 2

    if (
        len(ht_probabilities)
        != expected_ht_count
        or len(ft_probabilities)
        != expected_ft_count
    ):
        return {
            "status": "FAILED",
            "minimum_violation_mass": None,
            "coherent": False,
            "solver_message": (
                "Probability-vector length does not "
                "match its score grid."
            ),
        }

    if (
        np.any(~np.isfinite(ht_probabilities))
        or np.any(~np.isfinite(ft_probabilities))
    ):
        return {
            "status": "FAILED",
            "minimum_violation_mass": None,
            "coherent": False,
            "solver_message": (
                "Non-finite probabilities."
            ),
        }

    negative_tolerance = 1e-12

    if (
        np.min(ht_probabilities)
        < -negative_tolerance
        or np.min(ft_probabilities)
        < -negative_tolerance
    ):
        return {
            "status": "FAILED",
            "minimum_violation_mass": None,
            "coherent": False,
            "solver_message": (
                "Materially negative probabilities."
            ),
        }

    ht_probabilities = np.maximum(
        ht_probabilities,
        0.0,
    )

    ft_probabilities = np.maximum(
        ft_probabilities,
        0.0,
    )

    ht_total = float(
        np.sum(ht_probabilities)
    )

    ft_total = float(
        np.sum(ft_probabilities)
    )

    if (
        ht_total <= TOL
        or ft_total <= TOL
    ):
        return {
            "status": "FAILED",
            "minimum_violation_mass": None,
            "coherent": False,
            "solver_message": (
                "HT or FT probability mass "
                "is invalid."
            ),
        }

    ht_probabilities /= ht_total
    ft_probabilities /= ft_total

    ht_home, ht_away = score_arrays(
        ht_max_goals
    )

    ft_home, ft_away = score_arrays(
        ft_max_goals
    )

    support_tolerance = 1e-14

    active_ht = np.flatnonzero(
        ht_probabilities
        > support_tolerance
    )

    active_ft = np.flatnonzero(
        ft_probabilities
        > support_tolerance
    )

    removed_ht_mass = float(
        1.0
        - np.sum(
            ht_probabilities[active_ht]
        )
    )

    removed_ft_mass = float(
        1.0
        - np.sum(
            ft_probabilities[active_ft]
        )
    )

    compatible_ht_indices = []
    compatible_ft_indices = []

    for ht_index in active_ht:
        compatible_ft = active_ft[
            (
                ft_home[active_ft]
                >= ht_home[ht_index]
            )
            & (
                ft_away[active_ft]
                >= ht_away[ht_index]
            )
        ]

        compatible_ht_indices.extend(
            [int(ht_index)]
            * len(compatible_ft)
        )

        compatible_ft_indices.extend(
            int(index)
            for index in compatible_ft
        )

    variable_count = len(
        compatible_ht_indices
    )

    tolerance = (
        CONFIG[
            "TAIL_TOLERANCE"
        ]
        + CONFIG[
            "LP_NUMERICAL_BUFFER"
        ]
        + removed_ht_mass
        + removed_ft_mass
    )

    if variable_count == 0:
        return {
            "status": "COMPLETED",
            "minimum_violation_mass": 1.0,
            "coherent": False,
            "tolerance": tolerance,
            "solver_message": (
                "No compatible HT-to-FT "
                "transitions exist."
            ),
        }

    ht_row_lookup = {
        int(state_index): row_index
        for row_index, state_index
        in enumerate(active_ht)
    }

    ft_row_lookup = {
        int(state_index): row_index
        for row_index, state_index
        in enumerate(active_ft)
    }

    row_count = (
        len(active_ht)
        + len(active_ft)
    )

    rows = []
    columns = []
    values = []

    for variable_index, (
        ht_index,
        ft_index,
    ) in enumerate(
        zip(
            compatible_ht_indices,
            compatible_ft_indices,
        )
    ):
        rows.append(
            ht_row_lookup[ht_index]
        )
        columns.append(
            variable_index
        )
        values.append(1.0)

        rows.append(
            len(active_ht)
            + ft_row_lookup[ft_index]
        )
        columns.append(
            variable_index
        )
        values.append(1.0)

    capacity_matrix = coo_matrix(
        (
            values,
            (rows, columns),
        ),
        shape=(
            row_count,
            variable_count,
        ),
    ).tocsr()

    capacity_bounds = np.concatenate([
        ht_probabilities[active_ht],
        ft_probabilities[active_ft],
    ])

    result = linprog(
        c=-np.ones(
            variable_count,
            dtype=float,
        ),
        A_ub=capacity_matrix,
        b_ub=capacity_bounds,
        bounds=(
            0.0,
            None,
        ),
        method="highs",
    )

    if not result.success:
        return {
            "status": "FAILED",
            "minimum_violation_mass": None,
            "coherent": False,
            "solver_message": str(
                result.message
            ),
        }

    compatible_mass = clamp(
        -float(result.fun),
        0.0,
        1.0,
    )

    violation_mass = clamp(
        1.0 - compatible_mass,
        0.0,
        1.0,
    )

    return {
        "status": "COMPLETED",
        "minimum_violation_mass": (
            violation_mass
        ),
        "coherent": bool(
            violation_mass <= tolerance
        ),
        "tolerance": tolerance,
        "solver_message": str(
            result.message
        ),
    }


def build_ht_ft_coherence_audit(
    data,
    period_results,
):
    ft_section = period_results.get(
        "FT",
        {}
    )

    ht_section = period_results.get(
        "HT",
        {}
    )

    forecast_priors = set(
        data["settings"][
            "forecast_priors"
        ]
    )

    ft_scenarios = [
        item
        for item in ft_section.get(
            "scenarios",
            [],
        )
        if item["prior_type"]
        in forecast_priors
    ]

    ht_scenarios = [
        item
        for item in ht_section.get(
            "scenarios",
            [],
        )
        if item["prior_type"]
        in forecast_priors
    ]

    if not ft_scenarios or not ht_scenarios:
        return {
            "status": "NOT_AVAILABLE",
            "reason": (
                "Both FT and HT forecasting "
                "reconstructions are required."
            ),
            "scenarios": [],
        }

    ft_map = {
        (
            item["source"],
            item["devig_method"],
            item["prior_type"],
        ): item
        for item in ft_scenarios
    }

    records = []

    for ht_scenario in ht_scenarios:
        key = (
            ht_scenario["source"],
            ht_scenario["devig_method"],
            ht_scenario["prior_type"],
        )

        ft_scenario = ft_map.get(
            key
        )

        if ft_scenario is None:
            continue

        result = minimum_ht_ft_violation(
            ht_probabilities=(
                ht_scenario[
                    "probabilities"
                ]
            ),
            ht_max_goals=(
                ht_section["max_goals"]
            ),
            ft_probabilities=(
                ft_scenario[
                    "probabilities"
                ]
            ),
            ft_max_goals=(
                ft_section["max_goals"]
            ),
        )

        records.append({
            "source": key[0],
            "devig_method": key[1],
            "prior_type": key[2],
            **result,
        })

    if not records:
        return {
            "status": "NOT_AVAILABLE",
            "reason": (
                "No matching FT and HT "
                "forecast scenarios."
            ),
            "scenarios": [],
        }

    if any(
        item["status"] == "FAILED"
        for item in records
    ):
        status = "FAIL"
    elif any(
        not item["coherent"]
        for item in records
    ):
        status = "FAIL"
    else:
        status = "PASS"

    primary_source = data[
        "settings"
    ]["primary_source"]

    required_pairs = {
        (method, prior)
        for method
        in data["settings"][
            "devig_methods"
        ]
        for prior
        in data["settings"][
            "forecast_priors"
        ]
    }

    primary_records = [
        item
        for item in records
        if item["source"]
        == primary_source
    ]

    completed_pairs = {
        (
            item["devig_method"],
            item["prior_type"],
        )
        for item in primary_records
        if item["status"]
        == "COMPLETED"
    }

    primary_complete = (
        completed_pairs
        == required_pairs
    )

    primary_pass = bool(
        primary_complete
        and all(
            item["coherent"]
            for item in primary_records
        )
    )

    return {
        "status": status,
        "primary_source_complete": (
            primary_complete
        ),
        "primary_source_pass": (
            primary_pass
        ),
        "scenarios": records,
    }


# ============================================================
# 14. Candidate evaluation helpers
# ============================================================

def candidate_price_status(
    conservative_ev,
    median_ev,
):
    if conservative_ev is None:
        return "UNKNOWN"

    if conservative_ev >= -TOL:
        return "FAIR_OR_BETTER"

    if (
        median_ev is not None
        and median_ev >= -TOL
    ):
        return "MIXED_PRICE"

    if conservative_ev <= CONFIG[
        "SEVERE_PRICE_EV_WARNING"
    ]:
        return "SEVERELY_UNDERPAID"

    if conservative_ev <= CONFIG[
        "POOR_PRICE_EV_WARNING"
    ]:
        return "POOR_PRICE"

    return "SLIGHTLY_UNDERPAID"


def summarize_scenario_metrics(
    scenario_results,
):
    probability = {
        field: summary_statistics(
            item[field]
            for item in scenario_results
        )
        for field in [
            "hit_probability",
            "nonloss_probability",
            "loss_probability",
            "full_win",
            "half_win",
            "push",
            "half_loss",
            "full_loss",
        ]
    }

    return {
        "probability": {
            "hit": probability[
                "hit_probability"
            ],
            "nonloss": probability[
                "nonloss_probability"
            ],
            "loss": probability[
                "loss_probability"
            ],
            "full_win": probability[
                "full_win"
            ],
            "half_win": probability[
                "half_win"
            ],
            "push": probability["push"],
            "half_loss": probability[
                "half_loss"
            ],
            "full_loss": probability[
                "full_loss"
            ],
        },
        "expected_return": (
            summary_statistics(
                item["expected_return"]
                for item in scenario_results
            )
        ),
        "fair_odds": summary_statistics(
            item["fair_odds"]
            for item in scenario_results
            if item["fair_odds"] is not None
        ),
        "effective_fair_probability": (
            summary_statistics(
                item[
                    "effective_fair_probability"
                ]
                for item in scenario_results
            )
        ),
        "fair_probability_discrepancy": (
            summary_statistics(
                item[
                    "fair_probability_discrepancy"
                ]
                for item in scenario_results
                if item[
                    "fair_probability_discrepancy"
                ] is not None
            )
        ),
        "absolute_fair_probability_discrepancy": (
            summary_statistics(
                abs(
                    item[
                        "fair_probability_discrepancy"
                    ]
                )
                for item in scenario_results
                if item[
                    "fair_probability_discrepancy"
                ] is not None
            )
        ),
    }


def evaluate_model_scenarios(
    candidate,
    scenarios,
    a_coeff,
    b_coeff,
):
    scenario_results = []

    for scenario in scenarios:
        metrics = candidate_metrics(
            probabilities=scenario[
                "probabilities"
            ],
            a_coeff=a_coeff,
            b_coeff=b_coeff,
            odds=candidate["odds"],
        )

        discrepancy = None

        if (
            scenario.get(
                "direct_devig_target"
            )
            is not None
        ):
            discrepancy = (
                metrics[
                    "effective_fair_probability"
                ]
                - scenario[
                    "direct_devig_target"
                ]
            )

        scenario_results.append({
            "scenario_id": scenario["id"],
            "period": scenario["period"],
            "source": scenario["source"],
            "source_title": scenario[
                "source_title"
            ],
            "devig_method": scenario[
                "devig_method"
            ],
            "prior_type": scenario[
                "prior_type"
            ],
            "prior_role": scenario[
                "prior_role"
            ],
            "scenario_type": scenario[
                "scenario_type"
            ],
            "excluded_group": scenario.get(
                "excluded_group"
            ),
            "excluded_family": scenario.get(
                "excluded_family"
            ),
            "direct_devig_target": (
                scenario.get(
                    "direct_devig_target"
                )
            ),
            "fair_probability_discrepancy": (
                discrepancy
            ),
            "model_quality": scenario[
                "quality_status"
            ],
            "model_diagnostics": {
                "lambda_home": scenario[
                    "lambda_home"
                ],
                "lambda_away": scenario[
                    "lambda_away"
                ],
                "rho": scenario["rho"],
                "dispersion": scenario[
                    "dispersion"
                ],
                "dispersion_at_bound": scenario[
                    "dispersion_at_bound"
                ],
                "minimum_slack": scenario[
                    "minimum_slack"
                ],
                "maximum_equation_residual": (
                    scenario[
                        "maximum_equation_residual"
                    ]
                ),
                "maximum_effective_residual": (
                    scenario[
                        "maximum_effective_residual"
                    ]
                ),
                "projection_method": (
                    scenario[
                        "projection_method"
                    ]
                ),
                "constraint_count": (
                    scenario[
                        "constraint_count"
                    ]
                ),
                "market_group_count": (
                    scenario[
                        "market_group_count"
                    ]
                ),
                "market_family_count": (
                    scenario[
                        "market_family_count"
                    ]
                ),
                "boundary_mass": scenario[
                    "boundary_mass"
                ],
            },
            **metrics,
        })

    return scenario_results


def build_prior_diagnostics(
    data,
    period,
    forecast_results,
    audit_results,
):
    required_priors = set(
        data["settings"][
            "forecast_priors"
        ]
    )

    grouped_forecasts = {}

    for result in forecast_results:
        key = (
            result["source"],
            result["devig_method"],
        )

        grouped_forecasts.setdefault(
            key,
            [],
        ).append(result)

    audit_map = {
        (
            result["source"],
            result["devig_method"],
            result["prior_type"],
        ): result
        for result in audit_results
    }

    records = []

    for key, values in grouped_forecasts.items():
        by_prior = {
            item["prior_type"]: item
            for item in values
        }

        available_priors = set(
            by_prior
        )

        complete = (
            available_priors
            == required_priors
        )

        hit_values = [
            item["hit_probability"]
            for item in values
        ]

        central_hit = (
            float(np.median(hit_values))
            if hit_values
            else None
        )

        prior_minimum = (
            min(hit_values)
            if hit_values
            else None
        )

        prior_maximum = (
            max(hit_values)
            if hit_values
            else None
        )

        prior_spread = (
            prior_maximum
            - prior_minimum
            if prior_minimum is not None
            else None
        )

        maxent_result = audit_map.get(
            (
                key[0],
                key[1],
                "FLAT_GRID_MAXENT",
            )
        )

        maxent_hit = (
            maxent_result[
                "hit_probability"
            ]
            if maxent_result is not None
            else None
        )

        maxent_gap = (
            abs(
                maxent_hit
                - central_hit
            )
            if (
                maxent_hit is not None
                and central_hit is not None
            )
            else None
        )

        records.append({
            "source": key[0],
            "devig_method": key[1],
            "complete": complete,
            "available_priors": sorted(
                available_priors
            ),
            "required_priors": sorted(
                required_priors
            ),
            "hit_by_prior": {
                prior: (
                    by_prior[prior][
                        "hit_probability"
                    ]
                    if prior in by_prior
                    else None
                )
                for prior in sorted(
                    required_priors
                )
            },
            "parametric_central_hit": (
                central_hit
            ),
            "parametric_minimum_hit": (
                prior_minimum
            ),
            "parametric_maximum_hit": (
                prior_maximum
            ),
            "prior_spread": (
                prior_spread
            ),
            "maxent_hit": maxent_hit,
            "maxent_absolute_gap": (
                maxent_gap
            ),
        })

    complete_records = [
        item
        for item in records
        if item["complete"]
    ]

    return {
        "status": (
            "COMPLETE"
            if records
            and len(complete_records)
            == len(records)
            else (
                "PARTIAL"
                if records
                else "NOT_AVAILABLE"
            )
        ),
        "parametric_central_hit": (
            summary_statistics(
                item[
                    "parametric_central_hit"
                ]
                for item in records
                if item[
                    "parametric_central_hit"
                ] is not None
            )
        ),
        "prior_spread": (
            summary_statistics(
                item["prior_spread"]
                for item in records
                if item["prior_spread"]
                is not None
            )
        ),
        "maxent_absolute_gap": (
            summary_statistics(
                item[
                    "maxent_absolute_gap"
                ]
                for item in records
                if item[
                    "maxent_absolute_gap"
                ] is not None
            )
        ),
        "records": records,
    }


def classify_market_support(
    thresholds,
    prior_diagnostics,
    feasible_bounds,
    forecast_summary,
):
    configured = {
        key: value
        for key, value in thresholds.items()
        if value is not None
    }

    if not configured:
        return {
            "status": "CALIBRATION_REQUIRED",
            "market_supported": None,
            "reasons": [
                "NO_CALIBRATED_SUPPORT_THRESHOLDS"
            ],
            "thresholds": thresholds,
            "note": (
                "Configure FT and HT thresholds "
                "from historical hidden-line "
                "validation before treating this "
                "label as a publication gate."
            ),
        }

    reasons = []

    if prior_diagnostics[
        "status"
    ] != "COMPLETE":
        reasons.append(
            "FORECAST_PRIOR_SET_INCOMPLETE"
        )

    prior_spread = prior_diagnostics[
        "prior_spread"
    ]["maximum"]

    feasible_width = feasible_bounds[
        "width"
    ]["maximum"]

    maxent_gap = prior_diagnostics[
        "maxent_absolute_gap"
    ]["maximum"]

    hidden_error = forecast_summary[
        "absolute_fair_probability_discrepancy"
    ]["maximum"]

    if (
        thresholds["max_prior_spread"]
        is not None
    ):
        if prior_spread is None:
            reasons.append(
                "PRIOR_SPREAD_NOT_AVAILABLE"
            )
        elif (
            prior_spread
            > thresholds[
                "max_prior_spread"
            ] + TOL
        ):
            reasons.append(
                "PRIOR_SPREAD_TOO_WIDE"
            )

    if (
        thresholds["max_feasible_width"]
        is not None
    ):
        if feasible_width is None:
            reasons.append(
                "FEASIBLE_WIDTH_NOT_AVAILABLE"
            )
        elif (
            feasible_width
            > thresholds[
                "max_feasible_width"
            ] + TOL
        ):
            reasons.append(
                "FEASIBLE_RANGE_TOO_WIDE"
            )

    if (
        thresholds["max_maxent_gap"]
        is not None
    ):
        if maxent_gap is None:
            reasons.append(
                "MAXENT_GAP_NOT_AVAILABLE"
            )
        elif (
            maxent_gap
            > thresholds[
                "max_maxent_gap"
            ] + TOL
        ):
            reasons.append(
                "MAXENT_GAP_TOO_WIDE"
            )

    if (
        thresholds[
            "max_hidden_line_absolute_error"
        ]
        is not None
        and hidden_error is not None
        and hidden_error
        > thresholds[
            "max_hidden_line_absolute_error"
        ] + TOL
    ):
        reasons.append(
            "HIDDEN_LINE_ERROR_TOO_LARGE"
        )

    market_supported = not reasons

    return {
        "status": (
            "MARKET_SUPPORTED"
            if market_supported
            else "NOT_MARKET_SUPPORTED"
        ),
        "market_supported": (
            market_supported
        ),
        "reasons": reasons,
        "thresholds": thresholds,
        "observed": {
            "maximum_prior_spread": (
                prior_spread
            ),
            "maximum_feasible_width": (
                feasible_width
            ),
            "maximum_maxent_gap": (
                maxent_gap
            ),
            "maximum_hidden_line_absolute_error": (
                hidden_error
            ),
        },
    }


# ============================================================
# 15. Candidate evaluation
# ============================================================

def evaluate_candidate(
    data,
    candidate,
    constraint_sets,
    full_scenario_map,
    period_results,
    uncertainty_profiles,
    coherence_audit,
):
    period = candidate[
        "period"
    ]

    max_goals = period_results[
        period
    ]["max_goals"]

    home_scores, away_scores = (
        score_arrays(max_goals)
    )

    a_coeff, b_coeff = (
        settlement_coefficients(
            home_scores,
            away_scores,
            candidate["market"],
            candidate["selection"],
            candidate["line"],
            candidate.get("team"),
        )
    )

    payout = (
        a_coeff * candidate["odds"]
        + b_coeff
    )

    profit = payout - 1.0

    hit_mask = profit > TOL
    nonloss_mask = profit >= -TOL
    miss_mask = profit < -TOL

    candidate_scenarios, errors = (
        build_removed_scenarios(
            data=data,
            candidate=candidate,
            constraint_sets=constraint_sets,
            full_scenario_map=(
                full_scenario_map
            ),
            max_goals=max_goals,
            removal_type="TARGET_GROUP",
        )
    )

    all_scenario_results = (
        evaluate_model_scenarios(
            candidate,
            candidate_scenarios,
            a_coeff,
            b_coeff,
        )
    )

    forecast_results = [
        item
        for item in all_scenario_results
        if item["prior_role"]
        == "FORECAST"
    ]

    audit_results = [
        item
        for item in all_scenario_results
        if item["prior_role"]
        == "AUDIT_ONLY"
    ]

    summaries = summarize_scenario_metrics(
        forecast_results
    )

    audit_summaries = (
        summarize_scenario_metrics(
            audit_results
        )
    )

    prior_diagnostics = (
        build_prior_diagnostics(
            data,
            period,
            forecast_results,
            audit_results,
        )
    )

    if data["settings"]["features"][
        "feasible_bounds"
    ]:
        feasible_bounds = (
            build_feasible_bounds_audit(
                candidate_scenarios,
                max_goals,
                hit_mask,
            )
        )
    else:
        feasible_bounds = {
            "status": "DISABLED",
            "lower": summary_statistics([]),
            "upper": summary_statistics([]),
            "width": summary_statistics([]),
            "overall_envelope": {
                "minimum": None,
                "maximum": None,
                "width": None,
            },
            "scenario_count": 0,
            "records": [],
        }

    support_assessment = (
        classify_market_support(
            thresholds=data[
                "settings"
            ][
                "support_thresholds"
            ][period],
            prior_diagnostics=(
                prior_diagnostics
            ),
            feasible_bounds=(
                feasible_bounds
            ),
            forecast_summary=(
                summaries
            ),
        )
    )

    family_scenarios = []
    family_errors = []

    if data["settings"]["features"][
        "family_out_audit"
    ]:
        family_scenarios, family_errors = (
            build_removed_scenarios(
                data=data,
                candidate=candidate,
                constraint_sets=(
                    constraint_sets
                ),
                full_scenario_map=(
                    full_scenario_map
                ),
                max_goals=max_goals,
                removal_type="FAMILY",
            )
        )

    family_all_results = (
        evaluate_model_scenarios(
            candidate,
            family_scenarios,
            a_coeff,
            b_coeff,
        )
    )

    family_results = [
        item
        for item in family_all_results
        if item["prior_role"]
        == "FORECAST"
    ]

    family_audit_results = [
        item
        for item in family_all_results
        if item["prior_role"]
        == "AUDIT_ONLY"
    ]

    family_hit = summary_statistics(
        item["hit_probability"]
        for item in family_results
    )

    threshold = data[
        "settings"
    ][
        "minimum_shortlist_hit_probability"
    ]

    if not family_results:
        family_status = "NOT_TESTABLE"
    elif (
        family_hit["minimum"] is not None
        and family_hit["minimum"]
        >= threshold - TOL
    ):
        family_status = "ROBUST"
    else:
        family_status = "FRAGILE"

    if data["settings"]["features"][
        "stress_audit"
    ]:
        stress_audit = build_stress_audit(
            candidate_scenarios=(
                candidate_scenarios
            ),
            max_goals=max_goals,
            hit_mask=hit_mask,
            uncertainty_profiles=(
                uncertainty_profiles
            ),
        )
    else:
        stress_audit = {
            "status": "DISABLED"
        }

    primary_source = data[
        "settings"
    ]["primary_source"]

    required_pairs = {
        (method, prior)
        for method
        in data["settings"][
            "devig_methods"
        ]
        for prior
        in data["settings"][
            "forecast_priors"
        ]
    }

    primary_results = [
        item
        for item in forecast_results
        if item["source"]
        == primary_source
    ]

    primary_pairs = {
        (
            item["devig_method"],
            item["prior_type"],
        )
        for item in primary_results
    }

    primary_complete = (
        primary_pairs
        == required_pairs
    )

    quality_values = {
        item["model_quality"]
        for item in forecast_results
    }

    if (
        not forecast_results
        or "FAIL" in quality_values
    ):
        model_quality = "FAIL"
    elif (
        "CAUTION" in quality_values
        or not primary_complete
    ):
        model_quality = "CAUTION"
    else:
        model_quality = "PASS"

    reasons = []

    minimum_odds = data[
        "settings"
    ]["minimum_odds"]

    maximum_odds = data[
        "settings"
    ]["maximum_odds"]

    if candidate["odds"] < minimum_odds:
        reasons.append(
            "BELOW_MINIMUM_ODDS"
        )

    if (
        maximum_odds is not None
        and candidate["odds"]
        > maximum_odds
    ):
        reasons.append(
            "ABOVE_MAXIMUM_ODDS"
        )

    if not forecast_results:
        reasons.append(
            "NO_VALID_FORECAST_RECONSTRUCTION"
        )

    if not primary_complete:
        reasons.append(
            "PRIMARY_SOURCE_FORECAST_"
            "SCENARIOS_INCOMPLETE"
        )

    if (
        data["settings"]["features"][
            "quality_gate"
        ]
        and model_quality == "FAIL"
    ):
        reasons.append(
            "MODEL_QUALITY_GATE_FAILED"
        )

    primary_constraints = []

    for method in data[
        "settings"
    ]["devig_methods"]:
        primary_constraints.extend(
            constraint_sets.get(
                (
                    period,
                    primary_source,
                    method,
                ),
                [],
            )
        )

    identification = (
        period_identification(
            primary_constraints
        )
    )

    identification[
        "candidate_feasible_bounds"
    ] = feasible_bounds

    if (
        period == "HT"
        and not identification[
            "minimum_structural_coverage_met"
        ]
    ):
        reasons.append(
            "HT_MINIMUM_STRUCTURAL_"
            "COVERAGE_NOT_MET"
        )

    if (
        period == "HT"
        and data["settings"]["features"][
            "ht_ft_coherence"
        ]
        and coherence_audit.get(
            "status"
        )
        not in {
            "NOT_AVAILABLE",
            "DISABLED",
            None,
        }
        and not coherence_audit.get(
            "primary_source_pass",
            False,
        )
    ):
        reasons.append(
            "HT_FT_COHERENCE_FAILED"
        )

    ev_summary = summaries[
        "expected_return"
    ]

    ev_floor = data[
        "settings"
    ]["ev_rejection_floor"]

    if (
        ev_floor is not None
        and ev_summary["minimum"]
        is not None
        and ev_summary["minimum"]
        < ev_floor - TOL
    ):
        reasons.append(
            "BELOW_OPTIONAL_EV_FLOOR"
        )

    if data["settings"][
        "require_market_supported_for_shortlist"
    ]:
        if support_assessment[
            "market_supported"
        ] is not True:
            reasons.append(
                "MARKET_SUPPORT_GATE_NOT_PASSED"
            )

    eligible = not reasons

    price_status = candidate_price_status(
        ev_summary["minimum"],
        ev_summary["median"],
    )

    hit_signature = np.packbits(
        hit_mask.astype(np.uint8)
    ).tobytes()

    nonloss_signature = np.packbits(
        nonloss_mask.astype(np.uint8)
    ).tobytes()

    return {
        "id": candidate["id"],
        "label": candidate["label"],
        "period": period,
        "market": candidate["market"],
        "selection": candidate[
            "selection"
        ],
        "line": candidate["line"],
        "team": candidate.get("team"),
        "hkjc_odds": candidate["odds"],
        "group_key": candidate[
            "group_key"
        ],
        "family_key": candidate[
            "family_key"
        ],
        "eligible": eligible,
        "exclusion_reasons": reasons,

        "shortlisted": False,
        "shortlist_rank": None,
        "shortlist_status": (
            "NOT_ASSESSED"
        ),
        "shortlist_exclusion_reasons": [],
        "conflicts_with": [],

        # Backward-compatible aliases.
        "official": False,
        "official_rank": None,
        "official_status": (
            "NOT_ASSESSED"
        ),
        "official_exclusion_reasons": [],

        "forecast_scenario_count": len(
            forecast_results
        ),
        "audit_scenario_count": len(
            audit_results
        ),
        "scenario_count": len(
            all_scenario_results
        ),
        "primary_source_complete": (
            primary_complete
        ),
        "identification": identification,
        "model_quality": {
            "status": model_quality,
            "primary_complete": (
                primary_complete
            ),
            "scenario_quality_values": (
                sorted(quality_values)
            ),
        },

        # Main summaries use forecasting priors only.
        **summaries,

        "prior_diagnostics": (
            prior_diagnostics
        ),
        "maxent_audit": {
            "status": (
                "COMPLETED"
                if audit_results
                else "NOT_AVAILABLE"
            ),
            **audit_summaries,
            "scenarios": audit_results,
        },
        "feasible_probability_bounds": (
            feasible_bounds
        ),
        "market_support": (
            support_assessment
        ),

        "price_status": price_status,
        "scenarios": forecast_results,
        "audit_scenarios": audit_results,
        "scenario_errors": errors,

        "family_out_audit": {
            "status": family_status,
            "family": candidate[
                "family_key"
            ],
            "probability": {
                "hit": family_hit,
            },
            "forecast_scenario_count": len(
                family_results
            ),
            "audit_scenario_count": len(
                family_audit_results
            ),
            "scenarios": family_results,
            "audit_scenarios": (
                family_audit_results
            ),
            "errors": family_errors,
        },

        "stress_audit": stress_audit,

        "_a_coeff": a_coeff,
        "_b_coeff": b_coeff,
        "_payout": payout,
        "_profit": profit,
        "_hit_mask": hit_mask,
        "_nonloss_mask": nonloss_mask,
        "_miss_mask": miss_mask,
        "_hit_signature": hit_signature,
        "_nonloss_signature": (
            nonloss_signature
        ),
        "_home_scores": home_scores,
        "_away_scores": away_scores,
        "_max_goals": max_goals,
        "_original_index": candidate[
            "original_index"
        ],
    }


# ============================================================
# 16. Engine shortlist
# ============================================================

def recommendation_sort_key(
    candidate,
):
    hit = candidate[
        "probability"
    ]["hit"]

    nonloss = candidate[
        "probability"
    ]["nonloss"]

    full_win = candidate[
        "probability"
    ]["full_win"]

    full_loss = candidate[
        "probability"
    ]["full_loss"]

    return (
        (
            hit["minimum"]
            if hit["minimum"] is not None
            else -1.0
        ),
        (
            hit["median"]
            if hit["median"] is not None
            else -1.0
        ),
        (
            nonloss["minimum"]
            if nonloss["minimum"] is not None
            else -1.0
        ),
        (
            full_win["minimum"]
            if full_win["minimum"] is not None
            else -1.0
        ),
        -(
            full_loss["maximum"]
            if full_loss["maximum"]
            is not None
            else 1.0
        ),
        -candidate[
            "_original_index"
        ],
    )


def cross_period_can_both_hit(
    first,
    second,
):
    if first["period"] == "HT":
        ht_candidate = first
        ft_candidate = second
    else:
        ht_candidate = second
        ft_candidate = first

    ht_indices = np.flatnonzero(
        ht_candidate["_hit_mask"]
    )

    ft_indices = np.flatnonzero(
        ft_candidate["_hit_mask"]
    )

    if (
        len(ht_indices) == 0
        or len(ft_indices) == 0
    ):
        return False

    ht_home = ht_candidate[
        "_home_scores"
    ][ht_indices]

    ht_away = ht_candidate[
        "_away_scores"
    ][ht_indices]

    ft_home = ft_candidate[
        "_home_scores"
    ][ft_indices]

    ft_away = ft_candidate[
        "_away_scores"
    ][ft_indices]

    for h_home, h_away in zip(
        ht_home,
        ht_away,
    ):
        if np.any(
            (ft_home >= h_home)
            & (ft_away >= h_away)
        ):
            return True

    return False


def candidates_can_both_hit(
    first,
    second,
):
    if first["period"] != second["period"]:
        return cross_period_can_both_hit(
            first,
            second,
        )

    first_mask = np.asarray(
        first["_hit_mask"],
        dtype=bool,
    )

    second_mask = np.asarray(
        second["_hit_mask"],
        dtype=bool,
    )

    if first_mask.shape != second_mask.shape:
        raise ValueError(
            "Candidate score grids do not match."
        )

    return bool(
        np.any(
            first_mask
            & second_mask
        )
    )


def candidate_conflict_reason(
    candidate,
    selected_candidate,
):
    if (
        candidate["group_key"]
        == selected_candidate["group_key"]
    ):
        return (
            "SAME_MARKET_GROUP_AS_"
            f"{selected_candidate['id']}"
        )

    if (
        candidate["period"]
        == selected_candidate["period"]
        and candidate["_hit_signature"]
        == selected_candidate[
            "_hit_signature"
        ]
    ):
        return (
            "SAME_HIT_EVENT_AS_"
            f"{selected_candidate['id']}"
        )

    if not candidates_can_both_hit(
        candidate,
        selected_candidate,
    ):
        return (
            "CONTRADICTS_"
            f"{selected_candidate['id']}"
        )

    return None


def choose_recommendations(
    evaluated_candidates,
    maximum_recommendations,
    minimum_hit_probability,
):
    for candidate in evaluated_candidates:
        candidate["shortlisted"] = False
        candidate["shortlist_rank"] = None
        candidate["shortlist_status"] = (
            "NOT_SELECTED"
        )
        candidate[
            "shortlist_exclusion_reasons"
        ] = []
        candidate["conflicts_with"] = []

        candidate["official"] = False
        candidate["official_rank"] = None
        candidate["official_status"] = (
            "NOT_SELECTED"
        )
        candidate[
            "official_exclusion_reasons"
        ] = []

    ordered = sorted(
        evaluated_candidates,
        key=recommendation_sort_key,
        reverse=True,
    )

    selected = []

    for candidate in ordered:
        shortlist_reasons = []

        if not candidate["eligible"]:
            shortlist_reasons.append(
                "BASE_CANDIDATE_INELIGIBLE"
            )

        conservative_hit = (
            candidate
            .get("probability", {})
            .get("hit", {})
            .get("minimum")
        )

        if conservative_hit is None:
            shortlist_reasons.append(
                "NO_CONSERVATIVE_HIT_PROBABILITY"
            )
        elif (
            conservative_hit
            < minimum_hit_probability - TOL
        ):
            shortlist_reasons.append(
                "BELOW_MINIMUM_SHORTLIST_"
                "HIT_PROBABILITY"
            )

        conflict_records = []

        if not shortlist_reasons:
            for existing in selected:
                conflict_reason = (
                    candidate_conflict_reason(
                        candidate,
                        existing,
                    )
                )

                if conflict_reason is not None:
                    conflict_records.append({
                        "selected_id": (
                            existing["id"]
                        ),
                        "selected_label": (
                            existing["label"]
                        ),
                        "reason": conflict_reason,
                    })

        if conflict_records:
            shortlist_reasons.append(
                "CONFLICTS_WITH_HIGHER_"
                "RANKED_SHORTLIST_PICK"
            )

            candidate["conflicts_with"] = (
                conflict_records
            )

        if shortlist_reasons:
            candidate[
                "shortlist_exclusion_reasons"
            ] = shortlist_reasons

            candidate["shortlist_status"] = (
                "REFERENCE_ONLY"
            )

            candidate[
                "official_exclusion_reasons"
            ] = list(shortlist_reasons)

            candidate["official_status"] = (
                "REFERENCE_ONLY"
            )
            continue

        if (
            len(selected)
            >= int(maximum_recommendations)
        ):
            reason = (
                "MAXIMUM_RECOMMENDATIONS_REACHED"
            )

            candidate[
                "shortlist_exclusion_reasons"
            ] = [reason]

            candidate["shortlist_status"] = (
                "REFERENCE_ONLY"
            )

            candidate[
                "official_exclusion_reasons"
            ] = [reason]

            candidate["official_status"] = (
                "REFERENCE_ONLY"
            )
            continue

        selected.append(candidate)

        candidate["shortlisted"] = True
        candidate["shortlist_status"] = (
            "ENGINE_SHORTLIST"
        )
        candidate["shortlist_rank"] = len(
            selected
        )

        # Backward-compatible aliases.
        candidate["official"] = True
        candidate["official_status"] = (
            "ENGINE_SHORTLIST"
        )
        candidate["official_rank"] = len(
            selected
        )

    return selected


# ============================================================
# 17. Recommendation-set analysis
# ============================================================

def empty_joint_metrics(
    status="NOT_AVAILABLE",
    reason=None,
):
    return {
        "status": status,
        "reason": reason,
        "scenario_count": 0,
        "all_hit_probability": (
            summary_statistics([])
        ),
        "at_least_one_hit_probability": (
            summary_statistics([])
        ),
        "all_miss_probability": (
            summary_statistics([])
        ),
        "pair_compatibility": [],
        "scenarios": [],
    }


def joint_recommendation_metrics(
    selected,
    period_results,
):
    if not selected:
        return empty_joint_metrics(
            "NO_SELECTIONS"
        )

    periods = {
        item["period"]
        for item in selected
    }

    if len(periods) != 1:
        output = empty_joint_metrics(
            status=(
                "NOT_AVAILABLE_CROSS_PERIOD"
            ),
            reason=(
                "No arbitrary HT-FT joint "
                "distribution is imposed."
            ),
        )

        output["pair_compatibility"] = [
            {
                "first_id": first["id"],
                "first_label": first["label"],
                "second_id": second["id"],
                "second_label": second["label"],
                "can_both_hit": (
                    candidates_can_both_hit(
                        first,
                        second,
                    )
                ),
            }
            for first_index, first
            in enumerate(selected)
            for second in selected[
                first_index + 1:
            ]
        ]

        return output

    period = next(iter(periods))

    full_scenarios = [
        scenario
        for scenario
        in (
            period_results
            .get(period, {})
            .get("scenarios", [])
        )
        if scenario["prior_role"]
        == "FORECAST"
    ]

    if not full_scenarios:
        return empty_joint_metrics(
            reason=(
                "No full forecasting scenarios "
                "available."
            )
        )

    scenario_records = []

    for scenario in full_scenarios:
        probabilities = scenario[
            "probabilities"
        ]

        hit_masks = [
            candidate["_hit_mask"]
            for candidate in selected
        ]

        miss_masks = [
            candidate["_miss_mask"]
            for candidate in selected
        ]

        all_hit_mask = np.logical_and.reduce(
            hit_masks
        )

        at_least_one_hit_mask = (
            np.logical_or.reduce(
                hit_masks
            )
        )

        all_miss_mask = np.logical_and.reduce(
            miss_masks
        )

        scenario_records.append({
            "scenario_id": scenario["id"],
            "prior_type": scenario[
                "prior_type"
            ],
            "all_hit_probability": float(
                probabilities
                @ all_hit_mask.astype(float)
            ),
            "at_least_one_hit_probability": (
                float(
                    probabilities
                    @ at_least_one_hit_mask.astype(
                        float
                    )
                )
            ),
            "all_miss_probability": float(
                probabilities
                @ all_miss_mask.astype(float)
            ),
        })

    pair_compatibility = []

    for first_index in range(
        len(selected)
    ):
        for second_index in range(
            first_index + 1,
            len(selected),
        ):
            first = selected[first_index]
            second = selected[second_index]

            pair_mask = (
                first["_hit_mask"]
                & second["_hit_mask"]
            )

            pair_probabilities = [
                float(
                    scenario["probabilities"]
                    @ pair_mask.astype(float)
                )
                for scenario
                in full_scenarios
            ]

            pair_compatibility.append({
                "first_id": first["id"],
                "first_label": first["label"],
                "second_id": second["id"],
                "second_label": second["label"],
                "can_both_hit": bool(
                    np.any(pair_mask)
                ),
                "joint_hit_probability": (
                    summary_statistics(
                        pair_probabilities
                    )
                ),
            })

    return {
        "status": "COMPLETED",
        "period": period,
        "scenario_count": len(
            scenario_records
        ),
        "all_hit_probability": (
            summary_statistics(
                item[
                    "all_hit_probability"
                ]
                for item in scenario_records
            )
        ),
        "at_least_one_hit_probability": (
            summary_statistics(
                item[
                    "at_least_one_hit_probability"
                ]
                for item in scenario_records
            )
        ),
        "all_miss_probability": (
            summary_statistics(
                item[
                    "all_miss_probability"
                ]
                for item in scenario_records
            )
        ),
        "pair_compatibility": (
            pair_compatibility
        ),
        "scenarios": scenario_records,
    }


# ============================================================
# 18. Correct-score reference
# ============================================================

def calculate_correct_scores(
    full_scenarios,
    max_goals,
    count,
):
    empty_output = {
        "recommendations": [],
        "combined_probability": (
            summary_statistics([])
        ),
        "warning": (
            "Correct scores are mutually exclusive "
            "and naturally have lower probabilities."
        ),
    }

    forecast_scenarios = [
        scenario
        for scenario in full_scenarios
        if scenario["prior_role"]
        == "FORECAST"
    ]

    if (
        count <= 0
        or not forecast_scenarios
    ):
        return empty_output

    probability_matrix = np.vstack([
        scenario["probabilities"]
        for scenario in forecast_scenarios
    ])

    minimum = np.min(
        probability_matrix,
        axis=0,
    )

    median = np.median(
        probability_matrix,
        axis=0,
    )

    maximum = np.max(
        probability_matrix,
        axis=0,
    )

    home_scores, away_scores = (
        score_arrays(max_goals)
    )

    records = []

    for index in range(
        len(home_scores)
    ):
        central_probability = float(
            median[index]
        )

        records.append({
            "home_goals": int(
                home_scores[index]
            ),
            "away_goals": int(
                away_scores[index]
            ),
            "score": (
                f"{int(home_scores[index])}"
                "-"
                f"{int(away_scores[index])}"
            ),
            "probability": {
                "minimum": float(
                    minimum[index]
                ),
                "median": (
                    central_probability
                ),
                "maximum": float(
                    maximum[index]
                ),
                "count": len(
                    forecast_scenarios
                ),
            },
            "central_fair_odds": (
                1.0 / central_probability
                if central_probability > TOL
                else None
            ),
            "period": "FT",
            "section": (
                "HIGH_RISK_CORRECT_"
                "SCORE_REFERENCE"
            ),
        })

    records.sort(
        key=lambda item: (
            item["probability"]["minimum"],
            item["probability"]["median"],
        ),
        reverse=True,
    )

    selected = records[
        :int(count)
    ]

    selected_indices = [
        (
            item["home_goals"]
            * (max_goals + 1)
            + item["away_goals"]
        )
        for item in selected
    ]

    combined_probabilities = [
        float(
            np.sum(
                scenario[
                    "probabilities"
                ][selected_indices]
            )
        )
        for scenario in forecast_scenarios
    ]

    return {
        "recommendations": selected,
        "combined_probability": (
            summary_statistics(
                combined_probabilities
            )
        ),
        "warning": (
            "Correct scores are mutually exclusive "
            "and naturally have lower probabilities."
        ),
    }


# ============================================================
# 19. Output helpers
# ============================================================

def public_candidate_record(
    candidate,
):
    return {
        key: value
        for key, value
        in candidate.items()
        if not key.startswith("_")
    }


def public_full_scenario_record(
    scenario,
):
    return {
        key: value
        for key, value
        in scenario.items()
        if (
            key != "probabilities"
            and not key.startswith("_")
        )
    }


def aggregate_engine_quality(
    period_results,
    coherence_audit,
):
    scenario_statuses = []

    for section in period_results.values():
        for scenario in section.get(
            "scenarios",
            [],
        ):
            if scenario[
                "prior_role"
            ] == "FORECAST":
                scenario_statuses.append(
                    scenario[
                        "quality_status"
                    ]
                )

        if (
            any(
                scenario[
                    "prior_role"
                ] == "FORECAST"
                for scenario in section.get(
                    "scenarios",
                    [],
                )
            )
            and not section.get(
                "grid_complete",
                False,
            )
        ):
            scenario_statuses.append(
                "FAIL"
            )

    if (
        "FAIL" in scenario_statuses
        or coherence_audit.get(
            "status"
        ) == "FAIL"
    ):
        status = "FAIL"
    elif "CAUTION" in scenario_statuses:
        status = "CAUTION"
    else:
        status = "PASS"

    return {
        "status": status,
        "forecast_scenario_statuses": (
            scenario_statuses
        ),
        "ht_ft_coherence_status": (
            coherence_audit.get("status")
        ),
        "audit_priors_affect_global_quality": (
            False
        ),
    }


# ============================================================
# 20. Main engine
# ============================================================

def run_engine(input_data):
    started = datetime.now(
        timezone.utc
    )

    data = validate_input_data(
        input_data
    )

    constraint_sets = {}

    for book in data["sharp_books"]:
        for period in ["FT", "HT"]:
            if not period_has_market_data(
                book["markets"][period]
            ):
                continue

            for method in data[
                "settings"
            ]["devig_methods"]:
                constraints = (
                    build_book_constraints(
                        book,
                        period,
                        method,
                    )
                )

                if constraints:
                    constraint_sets[
                        (
                            period,
                            book["key"],
                            method,
                        )
                    ] = constraints

    period_results = {}

    for period in ["FT", "HT"]:
        has_constraints = any(
            key[0] == period
            for key in constraint_sets
        )

        if not has_constraints:
            continue

        period_results[period] = (
            adaptive_period_reconstruction(
                data=data,
                period=period,
                constraint_sets=(
                    constraint_sets
                ),
            )
        )

    if (
        "FT" not in period_results
        or not any(
            scenario["prior_role"]
            == "FORECAST"
            for scenario
            in period_results[
                "FT"
            ]["scenarios"]
        )
    ):
        raise RuntimeError(
            "No valid FT forecasting "
            "reconstruction was produced."
        )

    full_scenario_map = {}

    for period, section in (
        period_results.items()
    ):
        for scenario in section[
            "scenarios"
        ]:
            full_scenario_map[
                (
                    period,
                    scenario["source"],
                    scenario[
                        "devig_method"
                    ],
                    scenario[
                        "prior_type"
                    ],
                )
            ] = scenario

    primary_source = data[
        "settings"
    ]["primary_source"]

    required_primary_pairs = {
        (method, prior)
        for method
        in data["settings"][
            "devig_methods"
        ]
        for prior
        in data["settings"][
            "forecast_priors"
        ]
    }

    primary_ft_pairs = {
        (
            scenario["devig_method"],
            scenario["prior_type"],
        )
        for scenario in period_results[
            "FT"
        ]["scenarios"]
        if (
            scenario["source"]
            == primary_source
            and scenario[
                "prior_role"
            ] == "FORECAST"
        )
    }

    if (
        primary_ft_pairs
        != required_primary_pairs
    ):
        raise RuntimeError(
            "The primary sharp source did not "
            "produce every configured FT "
            "forecast prior/de-vig scenario."
        )

    coherence_audit = (
        build_ht_ft_coherence_audit(
            data,
            period_results,
        )
        if data["settings"]["features"][
            "ht_ft_coherence"
        ]
        else {
            "status": "DISABLED",
            "scenarios": [],
        }
    )

    uncertainty_profiles = (
        build_uncertainty_profiles(
            constraint_sets
        )
    )

    evaluated_candidates = []

    for candidate in data[
        "hkjc_markets"
    ]:
        if candidate["period"] not in (
            period_results
        ):
            placeholder = {
                **candidate,
                "eligible": False,
                "exclusion_reasons": [
                    "PERIOD_MODEL_UNAVAILABLE"
                ],

                "shortlisted": False,
                "shortlist_rank": None,
                "shortlist_status": (
                    "REFERENCE_ONLY"
                ),
                "shortlist_exclusion_reasons": [],

                "official": False,
                "official_rank": None,
                "official_status": (
                    "REFERENCE_ONLY"
                ),
                "official_exclusion_reasons": [],

                "conflicts_with": [],
                "scenario_count": 0,
                "forecast_scenario_count": 0,
                "audit_scenario_count": 0,
                "primary_source_complete": False,
                "identification": {
                    "fully_identified": False,
                    "minimum_structural_coverage_met": (
                        False
                    ),
                },
                "model_quality": {
                    "status": "FAIL",
                },
                "probability": {
                    field: summary_statistics([])
                    for field in [
                        "hit",
                        "nonloss",
                        "loss",
                        "full_win",
                        "half_win",
                        "push",
                        "half_loss",
                        "full_loss",
                    ]
                },
                "effective_fair_probability": (
                    summary_statistics([])
                ),
                "fair_odds": (
                    summary_statistics([])
                ),
                "expected_return": (
                    summary_statistics([])
                ),
                "price_status": "UNKNOWN",
                "fair_probability_discrepancy": (
                    summary_statistics([])
                ),
                "absolute_fair_probability_discrepancy": (
                    summary_statistics([])
                ),
                "prior_diagnostics": {
                    "status": "NOT_AVAILABLE",
                    "prior_spread": (
                        summary_statistics([])
                    ),
                },
                "maxent_audit": {
                    "status": "NOT_AVAILABLE",
                },
                "feasible_probability_bounds": {
                    "status": "NOT_TESTABLE",
                    "lower": summary_statistics([]),
                    "upper": summary_statistics([]),
                    "width": summary_statistics([]),
                },
                "market_support": {
                    "status": "NOT_AVAILABLE",
                    "market_supported": None,
                },
                "scenarios": [],
                "audit_scenarios": [],
                "scenario_errors": [],
                "family_out_audit": {
                    "status": "NOT_TESTABLE",
                },
                "stress_audit": {
                    "status": "NOT_TESTABLE",
                },

                "_original_index": candidate[
                    "original_index"
                ],
                "_hit_mask": np.zeros(
                    1,
                    dtype=bool,
                ),
                "_nonloss_mask": np.zeros(
                    1,
                    dtype=bool,
                ),
                "_miss_mask": np.ones(
                    1,
                    dtype=bool,
                ),
                "_hit_signature": b"",
                "_nonloss_signature": b"",
                "_home_scores": np.zeros(1),
                "_away_scores": np.zeros(1),
            }

            evaluated_candidates.append(
                placeholder
            )
            continue

        evaluated_candidates.append(
            evaluate_candidate(
                data=data,
                candidate=candidate,
                constraint_sets=(
                    constraint_sets
                ),
                full_scenario_map=(
                    full_scenario_map
                ),
                period_results=(
                    period_results
                ),
                uncertainty_profiles=(
                    uncertainty_profiles
                ),
                coherence_audit=(
                    coherence_audit
                ),
            )
        )

    selected = choose_recommendations(
        evaluated_candidates=(
            evaluated_candidates
        ),
        maximum_recommendations=(
            data["settings"][
                "max_recommendations"
            ]
        ),
        minimum_hit_probability=(
            data["settings"][
                "minimum_shortlist_hit_probability"
            ]
        ),
    )

    recommendations = []

    for candidate in selected:
        public = public_candidate_record(
            candidate
        )

        public["rank"] = candidate[
            "shortlist_rank"
        ]

        public["ranking_basis"] = (
            "Conservative positive-return "
            "probability across configured "
            "parametric forecasting priors, "
            "sources and de-vig methods."
        )

        public["publication_status"] = (
            "MANUAL_REVIEW_REQUIRED"
        )

        recommendations.append(public)

    joint_metrics = (
        joint_recommendation_metrics(
            selected=selected,
            period_results=period_results,
        )
    )

    ft_section = period_results[
        "FT"
    ]

    correct_scores = (
        calculate_correct_scores(
            full_scenarios=(
                ft_section["scenarios"]
            ),
            max_goals=(
                ft_section["max_goals"]
            ),
            count=data["settings"][
                "correct_score_count"
            ],
        )
    )

    engine_quality = (
        aggregate_engine_quality(
            period_results,
            coherence_audit,
        )
    )

    finished = datetime.now(
        timezone.utc
    )

    all_candidate_records = [
        public_candidate_record(
            candidate
        )
        for candidate
        in sorted(
            evaluated_candidates,
            key=recommendation_sort_key,
            reverse=True,
        )
    ]

    public_periods = {}

    for period, section in (
        period_results.items()
    ):
        forecast_scenarios = [
            scenario
            for scenario in section[
                "scenarios"
            ]
            if scenario[
                "prior_role"
            ] == "FORECAST"
        ]

        audit_scenarios = [
            scenario
            for scenario in section[
                "scenarios"
            ]
            if scenario[
                "prior_role"
            ] == "AUDIT_ONLY"
        ]

        public_periods[period] = {
            "max_goals_per_team": (
                section["max_goals"]
            ),
            "state_count": (
                (
                    section["max_goals"]
                    + 1
                ) ** 2
            ),
            "forecast_scenario_count": len(
                forecast_scenarios
            ),
            "audit_scenario_count": len(
                audit_scenarios
            ),
            "grid_complete": section[
                "grid_complete"
            ],
            "grid_expansion_history": (
                section[
                    "expansion_history"
                ]
            ),
            "forecast_scenarios": [
                public_full_scenario_record(
                    scenario
                )
                for scenario
                in forecast_scenarios
            ],
            "audit_scenarios": [
                public_full_scenario_record(
                    scenario
                )
                for scenario
                in audit_scenarios
            ],
            "full_scenario_errors": (
                section["errors"]
            ),
        }

    output = {
        "engine": {
            "name": ENGINE_NAME,
            "version": ENGINE_VERSION,
            "generated_at_utc": (
                finished.isoformat()
            ),
        },
        "status": "COMPLETED",
        "match": data["match"],
        "settings": data["settings"],

        "methodology": {
            "historical_team_data_used": False,
            "arbitrary_model_weights_used": False,
            "arbitrary_ht_ft_lambda_ratio_used": (
                False
            ),
            "ev_used_for_ranking": False,
            "kelly_used": False,

            "forecast_priors": (
                data["settings"][
                    "forecast_priors"
                ]
            ),
            "audit_priors": (
                data["settings"][
                    "audit_priors"
                ]
            ),

            "forecast_aggregation": (
                "Main probability summaries use "
                "only configured parametric priors. "
                "No prior receives an arbitrary "
                "numerical weight."
            ),

            "maxent_role": (
                "Audit only. It is excluded from "
                "the central forecast, conservative "
                "forecast minimum and shortlist "
                "ranking."
            ),

            "feasible_range_role": (
                "Linear-programming minimum and "
                "maximum candidate hit probability "
                "compatible with remaining market "
                "constraints. It is not a forecast "
                "confidence interval."
            ),

            "market_support_role": (
                "MARKET_SUPPORTED is produced only "
                "when calibrated FT or HT thresholds "
                "are supplied. No default thresholds "
                "are guessed."
            ),

            "ranking_objective": (
                "CONSERVATIVE_PARAMETRIC_"
                "HIT_PROBABILITY"
            ),

            "shortlist_not_publication": True,
            "manual_review_required": True,

            "minimum_shortlist_hit_probability": (
                data["settings"][
                    "minimum_shortlist_hit_probability"
                ]
            ),

            "maximum_recommendations_is_ceiling": (
                True
            ),

            "contradictory_shortlist_picks_blocked": (
                True
            ),

            "target_market_exclusion": {
                "1X2": (
                    "Remove the complete period "
                    "1X2 market."
                ),
                "AH": (
                    "Remove both sides of the exact "
                    "period AH line."
                ),
                "OU": (
                    "Remove both sides of the exact "
                    "period O/U line."
                ),
                "HHAD": (
                    "Remove all three outcomes of "
                    "the exact FT Handicap HAD."
                ),
                "TEAM_OU": (
                    "Remove both sides of the exact "
                    "team-total line."
                ),
            },

            "whole_family_out": {
                "1X2": (
                    "Remove the complete period "
                    "1X2 family."
                ),
                "AH": (
                    "Remove the complete period "
                    "AH ladder."
                ),
                "OU": (
                    "Remove the complete period "
                    "O/U ladder."
                ),
                "HHAD": (
                    "Remove the complete FT "
                    "Handicap HAD family."
                ),
                "TEAM_OU": (
                    "Remove the selected team's "
                    "complete period team-total "
                    "ladder."
                ),
                "absent_family_behavior": (
                    "NOT_TESTABLE"
                ),
            },

            "stress_definition": {
                "light": (
                    "Exact supplied de-vig target "
                    "structure with only necessary "
                    "feasibility slack."
                ),
                "medium": (
                    "Within-source envelope across "
                    "configured de-vig methods."
                ),
                "heavy": (
                    "Across-source and across-de-vig "
                    "target envelope."
                ),
                "automatic_rejection": False,
            },

            "ht_architecture": (
                "Independent HT reconstruction "
                "under every configured forecasting "
                "prior using direct HT markets."
            ),

            "ht_ft_coherence": (
                "Prior-matched optimal-transport "
                "audit enforcing FT goals not below "
                "HT goals."
            ),

            "prior_roles": {
                "INDEPENDENT_POISSON": (
                    "Independent equidispersed "
                    "goal-count baseline."
                ),
                "DIXON_COLES": (
                    "Poisson model with low-score "
                    "adjustment for 0-0, 1-0, 0-1 "
                    "and 1-1."
                ),
                "COM_POISSON_SHARED": (
                    "Independent team counts with "
                    "one shared dispersion parameter."
                ),
                "FLAT_GRID_MAXENT": (
                    "Grid-dependent assumption-light "
                    "audit prior only."
                ),
            },

            "projection_role": (
                "Relative-entropy reconciliation "
                "with de-vigged sharp-market "
                "constraints."
            ),

            "hit_definition": (
                "Gross payout greater than stake. "
                "Full wins and half wins are hits; "
                "pushes are not hits."
            ),

            "ev_role": (
                "Price warning and optional hard "
                "rejection floor only."
            ),

            "all_lines_retained": True,
        },

        "model_quality": engine_quality,
        "ht_ft_coherence": coherence_audit,

        "model": {
            # Backward-compatible FT summary,
            # containing forecasting scenarios only.
            "max_goals_per_team": (
                ft_section["max_goals"]
            ),
            "state_count": (
                (
                    ft_section["max_goals"]
                    + 1
                ) ** 2
            ),
            "full_scenario_count": len([
                scenario
                for scenario
                in ft_section["scenarios"]
                if scenario[
                    "prior_role"
                ] == "FORECAST"
            ]),
            "full_scenarios": [
                public_full_scenario_record(
                    scenario
                )
                for scenario
                in ft_section["scenarios"]
                if scenario[
                    "prior_role"
                ] == "FORECAST"
            ],
            "audit_scenarios": [
                public_full_scenario_record(
                    scenario
                )
                for scenario
                in ft_section["scenarios"]
                if scenario[
                    "prior_role"
                ] == "AUDIT_ONLY"
            ],
            "full_scenario_errors": (
                ft_section["errors"]
            ),

            "periods": public_periods,
        },

        "engine_shortlist": {
            "minimum_hit_probability": (
                data["settings"][
                    "minimum_shortlist_hit_probability"
                ]
            ),
            "maximum_recommendations": (
                data["settings"][
                    "max_recommendations"
                ]
            ),
            "shortlist_count": len(
                recommendations
            ),
            "manual_publication_review_required": (
                True
            ),
            "market_support_gate_required": (
                data["settings"][
                    "require_market_supported_for_shortlist"
                ]
            ),
            "contradiction_filter_used": True,
            "same_hit_event_filter_used": True,
            "stress_used_as_hard_gate": False,
            "family_out_used_as_hard_gate": False,
            "maxent_used_for_ranking": False,
            "ev_used_for_ranking": False,
        },

        # Backward-compatible section name.
        "official_selection": {
            "meaning": (
                "ENGINE_SHORTLIST_NOT_FINAL_"
                "PUBLICATION"
            ),
            "minimum_hit_probability": (
                data["settings"][
                    "minimum_shortlist_hit_probability"
                ]
            ),
            "maximum_recommendations": (
                data["settings"][
                    "max_recommendations"
                ]
            ),
            "official_count": len(
                recommendations
            ),
            "contradiction_filter_used": True,
            "same_hit_event_filter_used": True,
            "stress_used_as_hard_gate": False,
            "family_out_used_as_hard_gate": False,
        },

        "recommendations": recommendations,
        "recommendation_set": joint_metrics,
        "correct_scores": correct_scores,

        "candidate_markets": (
            all_candidate_records
        ),

        "excluded_markets": [
            {
                "id": candidate["id"],
                "label": candidate["label"],
                "base_exclusion_reasons": (
                    candidate[
                        "exclusion_reasons"
                    ]
                ),
                "shortlist_exclusion_reasons": (
                    candidate[
                        "shortlist_exclusion_reasons"
                    ]
                ),
                "official_exclusion_reasons": (
                    candidate[
                        "official_exclusion_reasons"
                    ]
                ),
                "conflicts_with": (
                    candidate[
                        "conflicts_with"
                    ]
                ),
            }
            for candidate
            in evaluated_candidates
            if not candidate["shortlisted"]
        ],

        "runtime": {
            "total_seconds": (
                finished - started
            ).total_seconds(),
        },

        "input_snapshot": data,
    }

    return to_builtin(output)
