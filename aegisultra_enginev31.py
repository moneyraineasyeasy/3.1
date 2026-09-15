"""
AEGIS ULTRA V3.1

Save as:
    aegisultra_enginev31.py

Required neighbouring file:
    aegisultra_enginev3.py

Dependencies:
    numpy
    scipy

Public entry point:
    run_engine(input_data)

IMPORTANT
---------
This is an executable module built on V3's unchanged utilities.
It does not monkey-patch V3.

No claim of empirically improved forecasting accuracy is made.
Numerical validity, market consistency and outcome calibration
are separate concepts.

Key changes
-----------
1. No minimum-slack LP distribution is used as a forecast.
2. Failed prior fits / entropy projections remain unavailable.
3. Explicit LP feasibility checks and entropy duality-gap checks.
4. Adaptive grids for each distinct retained constraint subset.
5. Full-market, target-group-out, equivalent-settlement-out and
   family-out probabilities are separate.
6. Candidate-level raw-versus-projected diagnostics.
7. Bounds/stress calculations do not depend on a successful prior.
8. Explicit missing/failed/unstable audit status.
9. A configurable reference forecast is separate from scenario minima.
10. Optional historical logit-sigmoid calibration helpers.
11. Match-weighted outcome evaluation and validation-row export.

Defaults
--------
- Forecast mode: TARGET_GROUP_OUT, preserving V3's basic architecture.
- Ranking: MINIMUM, preserving V3's conservative ranking objective.
- Central/reference prior: DIXON_COLES when configured.
- Exact minimum-slack reconciliation retained.
- Prior fitting uses V3-style outcome weighting unless explicitly changed.
- Calibration is NOT invented or automatically approved.
- No automatic adjustment against unders or receiving handicaps.

New input settings
------------------
settings["v31"] = {
    "forecast_mode": "TARGET_GROUP_OUT",  # or "FULL_MARKET"
    "ranking": "MINIMUM",                # CENTRAL or CALIBRATED
    "central_prior": "DIXON_COLES",
    "central_devig_method": "MULTIPLICATIVE",
    "prior_fit_weighting": "OUTCOME",     # GROUP or FAMILY_GROUP
    "require_complete_forecasts": True,
    "equivalent_settlement_audit": True,
    "calibrators": {}
}

Calibration workflow
--------------------
1. Save pre-match engine output.
2. After settlement, call make_validation_rows().
3. Fit a calibrator on EARLIER rows using fit_hit_calibrator().
4. Evaluate on separate LATER rows using evaluate_validation_rows().
5. Only after review, set:
       artifact["approved"] = True
       artifact["validation_end"] = "<UTC timestamp>"
6. Supply the artifact under:
       settings["v31"]["calibrators"][artifact["group"]]

Approval is a user declaration, not proof of calibration quality.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from typing import Any

import numpy as np
from scipy.optimize import least_squares, linprog, minimize
from scipy.sparse import coo_matrix
from scipy.special import expit, logsumexp

import aegisultra_enginev3 as v3


ENGINE_NAME = "Aegis Ultra Numerically Audited Reconstruction Engine"
ENGINE_VERSION = "3.1.0"

# Numerical tolerances, NOT empirically calibrated market-error thresholds.
NUMERICS = {
    "LP_SOLVER_TOLERANCE": 1e-9,
    "LP_ACCEPTANCE_TOLERANCE": 5e-9,
    "PROJECTION_BUFFER": 1e-8,
    "PROJECTION_ACCEPTANCE_TOLERANCE": 5e-9,
    "EFFECTIVE_NUMERICAL_TOLERANCE": 1e-7,
    "DUALITY_GAP_TOLERANCE": 1e-6,
    "PROJECTION_MAX_ITERATIONS": 4000,
    "FIT_MAX_EVALUATIONS": 800,
    "BOUNDARY_TOLERANCE": 1e-8,
    "GRID_STABILITY_TOLERANCE": 1e-7,
    "MIN_NONPUSH_MASS": 1e-8,
    "PROBABILITY_MASS_TOLERANCE": 1e-8,
    "MAX_EQUIVALENCE_PROBE_GOALS": 128,
}

LP_OPTIONS = {
    "primal_feasibility_tolerance": NUMERICS["LP_SOLVER_TOLERANCE"],
    "dual_feasibility_tolerance": NUMERICS["LP_SOLVER_TOLERANCE"],
    "ipm_optimality_tolerance": 1e-10,
}

PARAMETRIC_PRIORS = v3.PARAMETRIC_PRIORS
AUDIT_PRIORS = v3.AUDIT_PRIORS

TOL = v3.TOL


# ============================================================
# 1. Generic helpers
# ============================================================

def _utc(value):
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, str) and value.strip():
        result = datetime.fromisoformat(
            value.strip().replace("Z", "+00:00")
        )
    else:
        raise ValueError("A timezone-aware timestamp is required.")

    if result.tzinfo is None:
        raise ValueError("Timestamp must include its timezone.")

    return result.astimezone(timezone.utc)


def _stats(values):
    return v3.summary_statistics(
        value for value in values if value is not None
    )


def _finite_float(value, name):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc

    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")

    return result


def _public(value):
    """Recursively remove internal arrays and private keys."""
    if isinstance(value, dict):
        return {
            key: _public(item)
            for key, item in value.items()
            if not key.startswith("_")
            and key != "probabilities"
        }

    if isinstance(value, (list, tuple)):
        return [_public(item) for item in value]

    return v3.to_builtin(value)


def _normalise_probability_vector(values):
    p = np.asarray(values, dtype=float).reshape(-1).copy()

    if not len(p) or not np.all(np.isfinite(p)):
        raise ValueError("Invalid probability vector.")

    if float(p.min()) < -NUMERICS["LP_ACCEPTANCE_TOLERANCE"]:
        raise ValueError("Materially negative probability.")

    if abs(float(p.sum()) - 1.0) > NUMERICS["PROBABILITY_MASS_TOLERANCE"]:
        raise ValueError("Probability vector does not sum to one.")

    p = np.maximum(p, 0.0)
    p /= p.sum()
    return p


def _pad_probability_vector(p, old_max, new_max):
    if old_max > new_max:
        raise ValueError("Cannot pad to a smaller grid.")

    matrix = np.asarray(p, dtype=float).reshape(
        old_max + 1, old_max + 1
    )
    output = np.zeros((new_max + 1, new_max + 1), dtype=float)
    output[:old_max + 1, :old_max + 1] = matrix
    return output.ravel()


def _total_variation(first, second):
    return float(
        0.5 * np.abs(
            np.asarray(first) - np.asarray(second)
        ).sum()
    )


def _coverage(records, expected):
    completed = sum(
        record.get("status") == "COMPLETED"
        for record in records
    )

    if expected > 0 and completed == expected:
        status = "COMPLETED"
    elif completed:
        status = "INCOMPLETE"
    elif any(
        record.get("status") in {"FAILED", "UNSTABLE"}
        for record in records
    ):
        status = "FAILED"
    else:
        status = "NOT_TESTABLE"

    return {
        "status": status,
        "expected_count": expected,
        "completed_count": completed,
        "complete": expected > 0 and completed == expected,
    }


def _calibration_group(candidate):
    period = candidate["period"]
    market = candidate["market"]

    if market == "AH":
        line = float(candidate["line"])
        selected_handicap = (
            line if candidate["selection"] == "HOME" else -line
        )
        direction = (
            "RECEIVING" if selected_handicap > 0
            else "GIVING" if selected_handicap < 0
            else "LEVEL"
        )
        return f"{period}:AH:{direction}"

    return f"{period}:{market}:{candidate['selection']}"


# ============================================================
# 2. Input and configuration
# ============================================================

def _normalise_input(input_data):
    data = v3.validate_input_data(input_data)
    original_settings = input_data.get("settings", {})
    raw = original_settings.get("v31", {})

    if not isinstance(raw, dict):
        raise ValueError("settings.v31 must be an object.")

    forecast_priors = data["settings"]["forecast_priors"]
    devig_methods = data["settings"]["devig_methods"]

    options = {
        "forecast_mode": str(
            raw.get("forecast_mode", "TARGET_GROUP_OUT")
        ).upper(),
        "ranking": str(
            raw.get("ranking", "MINIMUM")
        ).upper(),
        "central_prior": v3.normalize_prior(
            raw.get(
                "central_prior",
                "DIXON_COLES"
                if "DIXON_COLES" in forecast_priors
                else forecast_priors[0],
            )
        ),
        "central_devig_method": str(
            raw.get(
                "central_devig_method",
                "MULTIPLICATIVE"
                if "MULTIPLICATIVE" in devig_methods
                else devig_methods[0],
            )
        ).upper(),
        "prior_fit_weighting": str(
            raw.get("prior_fit_weighting", "OUTCOME")
        ).upper(),
        "require_complete_forecasts": raw.get(
            "require_complete_forecasts", True
        ),
        "equivalent_settlement_audit": raw.get(
            "equivalent_settlement_audit", True
        ),
        "calibrators": copy.deepcopy(raw.get("calibrators", {})),
    }

    if options["forecast_mode"] not in {
        "FULL_MARKET", "TARGET_GROUP_OUT"
    }:
        raise ValueError("Unsupported v31 forecast_mode.")

    if options["ranking"] not in {
        "MINIMUM", "CENTRAL", "CALIBRATED"
    }:
        raise ValueError("Unsupported v31 ranking.")

    if options["prior_fit_weighting"] not in {
        "OUTCOME", "GROUP", "FAMILY_GROUP"
    }:
        raise ValueError("Unsupported prior_fit_weighting.")

    if options["central_prior"] not in forecast_priors:
        raise ValueError("central_prior must be a configured forecast prior.")

    if options["central_devig_method"] not in devig_methods:
        raise ValueError("central_devig_method must be configured.")

    for name in (
        "require_complete_forecasts",
        "equivalent_settlement_audit",
    ):
        if not isinstance(options[name], bool):
            raise ValueError(f"settings.v31.{name} must be boolean.")

    if not isinstance(options["calibrators"], dict):
        raise ValueError("settings.v31.calibrators must be an object.")

    # Preserve an explicit event identifier when provided.
    match_id = input_data.get("match", {}).get("id")
    if match_id is not None:
        data["match"]["id"] = str(match_id)

    data["settings"]["v31"] = options

    forecast_specification = {
        "version": ENGINE_VERSION,
        "forecast_mode": options["forecast_mode"],
        "central_prior": options["central_prior"],
        "central_devig_method": options["central_devig_method"],
        "prior_fit_weighting": options["prior_fit_weighting"],
        "forecast_priors": forecast_priors,
        "devig_methods": devig_methods,
        "primary_source": data["settings"]["primary_source"],
        "sources": sorted(book["key"] for book in data["sharp_books"]),
        "adaptive_grids": data["settings"]["features"]["adaptive_grids"],
        "numerics": NUMERICS,
    }

    pipeline_id = hashlib.sha256(
        json.dumps(
            forecast_specification,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:24]

    data["settings"]["v31"]["pipeline_id"] = pipeline_id
    return data


# ============================================================
# 3. Checked linear programming
# ============================================================

def _checked_lp(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None):
    c = np.asarray(c, dtype=float)

    result = linprog(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=(0.0, None),
        method="highs",
        options=LP_OPTIONS,
    )

    if (
        not result.success
        or result.x is None
        or not np.all(np.isfinite(result.x))
        or not math.isfinite(float(result.fun))
    ):
        raise RuntimeError(f"LP failed: {result.message}")

    x = np.asarray(result.x, dtype=float)
    violations = [max(0.0, -float(x.min()))]

    if A_ub is not None:
        violations.append(
            max(
                0.0,
                float(np.max(A_ub @ x - np.asarray(b_ub))),
            )
        )

    if A_eq is not None:
        violations.append(
            float(
                np.max(np.abs(A_eq @ x - np.asarray(b_eq)))
            )
        )

    residual = max(violations)

    if residual > NUMERICS["LP_ACCEPTANCE_TOLERANCE"]:
        raise RuntimeError(
            f"LP returned an unacceptable primal residual: {residual:g}"
        )

    return result


def _make_system(constraints, max_goals, intervals=None):
    """
    Build exact-target or interval market constraints.

    MIN_NONPUSH_MASS is a numerical safeguard against undefined
    effective probabilities. Bounds are conditional on this floor
    and the finite score grid.
    """
    if not v3.validate_constraint_subset(constraints):
        raise ValueError("Insufficient market-family structure.")

    enriched, _, targets = v3.enrich_constraints(
        constraints, max_goals
    )

    a = np.vstack([item["a_coeff"] for item in enriched])
    b = np.vstack([item["b_coeff"] for item in enriched])

    if intervals is None:
        lower = targets.copy()
        upper = targets.copy()
    else:
        lower = np.asarray([item[0] for item in intervals], dtype=float)
        upper = np.asarray([item[1] for item in intervals], dtype=float)

    if (
        lower.shape != targets.shape
        or upper.shape != targets.shape
        or not np.all(np.isfinite(lower))
        or not np.all(np.isfinite(upper))
        or np.any(lower < 0)
        or np.any(upper > 1)
        or np.any(lower > upper)
    ):
        raise ValueError("Invalid target intervals.")

    market_G = np.vstack([
        a + upper[:, None] * b,
        -(a + lower[:, None] * b),
    ])
    market_h = np.concatenate([upper, -lower])

    push_rows = b[np.any(b != 0.0, axis=1)]
    push_h = np.full(
        len(push_rows),
        1.0 - NUMERICS["MIN_NONPUSH_MASS"],
    )

    state_count = a.shape[1]

    # Slack relaxes market equations, NOT the nonpush safeguard.
    augmented_G = np.vstack([
        np.column_stack([
            market_G, -np.ones(len(market_h))
        ]),
        np.column_stack([
            push_rows, np.zeros(len(push_rows))
        ]),
    ])
    augmented_h = np.concatenate([market_h, push_h])

    equality = np.zeros((1, state_count + 1))
    equality[0, :state_count] = 1.0

    objective = np.zeros(state_count + 1)
    objective[-1] = 1.0

    result = _checked_lp(
        objective,
        augmented_G,
        augmented_h,
        equality,
        np.array([1.0]),
    )

    witness = _normalise_probability_vector(
        result.x[:state_count]
    )

    if (
        len(push_rows)
        and np.max(push_rows @ witness - push_h)
        > NUMERICS["LP_ACCEPTANCE_TOLERANCE"]
    ):
        raise RuntimeError("Normalised LP witness violates nonpush safeguard.")

    realised_slack = max(
        0.0,
        float(np.max(market_G @ witness - market_h)),
    )

    minimum_slack = max(
        0.0, float(result.x[-1]), realised_slack
    )

    allowed_slack = (
        minimum_slack + NUMERICS["PROJECTION_BUFFER"]
    )

    G = np.vstack([market_G, push_rows])
    h = np.concatenate([
        market_h + allowed_slack,
        push_h,
    ])

    return {
        "max_goals": max_goals,
        "constraints": constraints,
        "a": a,
        "b": b,
        "targets": targets,
        "lower": lower,
        "upper": upper,
        "G": G,
        "h": h,
        "minimum_slack": minimum_slack,
        "allowed_slack": allowed_slack,
        "state_count": state_count,
        "solver_message": str(result.message),
        # The LP witness is deliberately not returned as a forecast.
    }


# ============================================================
# 4. Prior fitting
# ============================================================

def _fit_weights(constraints, mode):
    if mode == "OUTCOME":
        return np.ones(len(constraints))

    group_counts = Counter(
        item["group_key"] for item in constraints
    )

    weights = np.array([
        1.0 / math.sqrt(group_counts[item["group_key"]])
        for item in constraints
    ])

    if mode == "FAMILY_GROUP":
        family_groups = {}
        for item in constraints:
            family_groups.setdefault(
                item["family_key"], set()
            ).add(item["group_key"])

        weights /= np.array([
            math.sqrt(len(family_groups[item["family_key"]]))
            for item in constraints
        ])

    return weights


def _fit_prior(system, prior_type, weighting):
    k = system["max_goals"]
    constraints = system["constraints"]

    if prior_type == "FLAT_GRID_MAXENT":
        return {
            "probabilities": v3.flat_grid_distribution(k),
            "lambda_home": None,
            "lambda_away": None,
            "rho": None,
            "dispersion": None,
            "optimizer_success": True,
            "optimizer_message": "No parametric fit required.",
            "maximum_prior_residual": None,
            "prior_rmse": None,
            "parameters_at_bound": False,
        }

    min_rate = v3.CONFIG["MIN_GOAL_RATE"]
    max_rate = v3.CONFIG["MAX_GOAL_RATE"]

    lower = [math.log(min_rate), math.log(min_rate)]
    upper = [math.log(max_rate), math.log(max_rate)]

    if prior_type == "DIXON_COLES":
        lower.append(-10.0)
        upper.append(10.0)
    elif prior_type == "COM_POISSON_SHARED":
        lower.append(math.log(v3.CONFIG["MIN_COM_DISPERSION"]))
        upper.append(math.log(v3.CONFIG["MAX_COM_DISPERSION"]))

    lower = np.asarray(lower)
    upper = np.asarray(upper)

    def decode(parameters):
        home_rate, away_rate = np.exp(parameters[:2])
        rho = None
        dispersion = 1.0

        if prior_type == "INDEPENDENT_POISSON":
            p = v3.independent_poisson_distribution(
                home_rate, away_rate, k
            )
        elif prior_type == "DIXON_COLES":
            rho = v3.decode_rho(
                home_rate, away_rate, parameters[2]
            )
            p = v3.dixon_coles_distribution(
                home_rate, away_rate, rho, k
            )
        elif prior_type == "COM_POISSON_SHARED":
            dispersion = math.exp(float(parameters[2]))
            p = v3.compoisson_shared_distribution(
                home_rate, away_rate, dispersion, k
            )
        else:
            raise ValueError(f"Unsupported prior: {prior_type}")

        return (
            _normalise_probability_vector(p),
            float(home_rate),
            float(away_rate),
            rho,
            dispersion,
        )

    weights = _fit_weights(constraints, weighting)

    def unweighted_residuals(p):
        denominator = 1.0 - system["b"] @ p
        if np.any(denominator <= 0.0):
            raise ValueError("Undefined prior effective probability.")

        return (
            system["a"] @ p / denominator
            - system["targets"]
        )

    def residuals(parameters):
        # Exceptions are not turned into constant fake residuals.
        p, *_ = decode(parameters)
        return weights * unweighted_residuals(p)

    rate_starts = [
        (0.35, 0.25), (0.55, 0.45),
        (0.80, 0.60), (1.10, 1.10),
        (1.50, 0.90), (0.90, 1.50),
        (1.90, 1.20), (1.20, 1.90),
    ]

    starts = []
    for home, away in rate_starts:
        base = [math.log(home), math.log(away)]

        if prior_type == "DIXON_COLES":
            starts.append(base + [0.0])
        elif prior_type == "COM_POISSON_SHARED":
            starts.extend(
                base + [math.log(dispersion)]
                for dispersion in (0.70, 1.00, 1.35)
            )
        else:
            starts.append(base)

    successful = []
    failures = []

    for start in starts:
        try:
            result = least_squares(
                residuals,
                np.asarray(start),
                bounds=(lower, upper),
                max_nfev=NUMERICS["FIT_MAX_EVALUATIONS"],
                ftol=1e-11,
                xtol=1e-11,
                gtol=1e-11,
            )

            if (
                result.success
                and np.all(np.isfinite(result.fun))
                and np.all(np.isfinite(result.x))
            ):
                successful.append(result)
            else:
                failures.append(str(result.message))
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            failures.append(str(exc))

    if not successful:
        raise RuntimeError(
            f"{prior_type}: no successful fit. "
            + "; ".join(failures[:3])
        )

    # Select by the actual least-squares objective being fitted.
    best = min(successful, key=lambda result: float(result.cost))
    p, home_rate, away_rate, rho, dispersion = decode(best.x)
    raw_residual = unweighted_residuals(p)

    at_bound = bool(
        np.any(np.abs(best.x - lower) <= 1e-4)
        or np.any(np.abs(best.x - upper) <= 1e-4)
    )

    return {
        "probabilities": p,
        "lambda_home": home_rate,
        "lambda_away": away_rate,
        "rho": rho,
        "dispersion": dispersion,
        "optimizer_success": True,
        "optimizer_message": str(best.message),
        "successful_start_count": len(successful),
        "failed_start_count": len(failures),
        "maximum_prior_residual": float(np.max(np.abs(raw_residual))),
        "prior_rmse": float(np.sqrt(np.mean(raw_residual ** 2))),
        "parameters_at_bound": at_bound,
    }


# ============================================================
# 5. Entropy projection: no forecasting LP fallback
# ============================================================

def _entropy_projection(prior, system):
    p0 = _normalise_probability_vector(prior)
    p0 = np.maximum(p0, 1e-300)
    p0 /= p0.sum()

    log_prior = np.log(p0)
    G, h = system["G"], system["h"]

    def objective_and_gradient(multipliers):
        z = log_prior - G.T @ multipliers
        normalizer = logsumexp(z)
        p = np.exp(z - normalizer)

        objective = normalizer + h @ multipliers
        gradient = h - G @ p

        return float(objective), gradient

    start = np.zeros(len(h))
    last_error = "Projection not attempted."

    # Retry the same entropy optimisation; never substitute the LP witness.
    for _ in range(2):
        result = minimize(
            objective_and_gradient,
            start,
            jac=True,
            method="L-BFGS-B",
            bounds=[(0.0, None)] * len(start),
            options={
                "maxiter": NUMERICS["PROJECTION_MAX_ITERATIONS"],
                "ftol": 1e-14,
                "gtol": 1e-10,
                "maxls": 50,
            },
        )

        if result.x is None or not np.all(np.isfinite(result.x)):
            last_error = "Non-finite entropy multipliers."
            break

        start = np.asarray(result.x)
        z = log_prior - G.T @ start
        log_p = z - logsumexp(z)
        p = np.exp(log_p)

        if not np.all(np.isfinite(p)):
            last_error = "Non-finite projected probabilities."
            continue

        p = _normalise_probability_vector(p)

        violation = max(0.0, float(np.max(G @ p - h)))
        kl = float(np.sum(p * (log_p - log_prior)))
        dual_objective = objective_and_gradient(start)[0]
        gap = kl + dual_objective

        denominator = 1.0 - system["b"] @ p

        if np.any(
            denominator < NUMERICS["MIN_NONPUSH_MASS"] / 2.0
        ):
            last_error = "Near-all-push effective probability."
            continue

        effective = system["a"] @ p / denominator

        # A market-equation slack s permits effective error s/(1-E[b]).
        permitted = system["allowed_slack"] / denominator
        effective_excess = max(
            0.0,
            float(np.max(effective - system["upper"] - permitted)),
            float(np.max(system["lower"] - effective - permitted)),
        )

        accepted = (
            result.success
            and violation
            <= NUMERICS["PROJECTION_ACCEPTANCE_TOLERANCE"]
            and abs(gap) <= NUMERICS["DUALITY_GAP_TOLERANCE"]
            and effective_excess
            <= NUMERICS["EFFECTIVE_NUMERICAL_TOLERANCE"]
        )

        if accepted:
            return {
                "probabilities": p,
                "projection_method": "DUAL_RELATIVE_ENTROPY",
                "optimizer_success": True,
                "optimizer_message": str(result.message),
                "maximum_constraint_violation": violation,
                "maximum_effective_residual": float(
                    np.max(np.abs(effective - system["targets"]))
                ),
                "minimum_nonpush_mass": float(denominator.min()),
                "duality_gap": gap,
                "relative_entropy": kl,
            }

        last_error = (
            f"{result.message}; constraint_violation={violation:g}; "
            f"duality_gap={gap:g}; effective_excess={effective_excess:g}"
        )

    raise RuntimeError(
        "Entropy projection unavailable; no forecasting LP fallback. "
        + last_error
    )


# ============================================================
# 6. Cached subset reconstruction and adaptive grids
# ============================================================

class ReconstructionCache:
    def __init__(self, data, probe_goals):
        self.data = data
        self.probe_goals = probe_goals
        self.cache = {}

    @staticmethod
    def _key(period, constraints):
        return (
            period,
            tuple(
                (
                    item["source"],
                    item["devig_method"],
                    item["group_key"],
                    item["selection"],
                    float(item["target"]),
                )
                for item in constraints
            ),
        )

    def get(self, period, constraints):
        key = self._key(period, constraints)
        if key not in self.cache:
            self.cache[key] = self._build(period, constraints)
        return self.cache[key]

    def _build(self, period, constraints):
        if not v3.validate_constraint_subset(constraints):
            return {
                "status": "NOT_TESTABLE",
                "reason": "INSUFFICIENT_MARKET_STRUCTURE",
                "models": {},
                "history": [],
                "_constraints": constraints,
                "_system": None,
                "_previous_system": None,
            }

        initial, safety = v3.period_grid_limits(period)
        step = v3.CONFIG["GRID_EXPANSION_STEP"]
        adaptive = self.data["settings"]["features"]["adaptive_grids"]
        forecast_priors = self.data["settings"]["forecast_priors"]
        all_priors = (
            forecast_priors + self.data["settings"]["audit_priors"]
        )
        weighting = self.data["settings"]["v31"]["prior_fit_weighting"]
        probe = self.probe_goals[period]

        previous_models = {}
        previous_system = None
        current_system = None
        history = []
        models = {}
        last_system_error = None

        k = initial

        while True:
            try:
                current_system = _make_system(constraints, k)
            except (ValueError, RuntimeError) as exc:
                last_system_error = str(exc)
                models = {
                    prior: {
                        "status": "FAILED",
                        "reason": last_system_error,
                        "prior_type": prior,
                        "usable": False,
                    }
                    for prior in all_priors
                }
                history.append({
                    "max_goals": k,
                    "system_error": last_system_error,
                })

                if not adaptive or k >= safety:
                    break

                k = min(k + step, safety)
                continue

            models = {}

            for prior in all_priors:
                role = (
                    "FORECAST"
                    if prior in PARAMETRIC_PRIORS
                    else "AUDIT_ONLY"
                )

                try:
                    fit = _fit_prior(current_system, prior, weighting)
                    projection = _entropy_projection(
                        fit["probabilities"], current_system
                    )

                    raw = fit["probabilities"]
                    projected = projection["probabilities"]

                    raw_boundary = v3.score_boundary_mass(raw, k)
                    boundary = v3.score_boundary_mass(projected, k)

                    padded_raw = _pad_probability_vector(raw, k, probe)
                    padded = _pad_probability_vector(projected, k, probe)

                    previous = previous_models.get(prior, {})
                    previous_p = previous.get("probabilities")

                    grid_change = (
                        _total_variation(padded, previous_p)
                        if previous_p is not None
                        else None
                    )

                    boundary_ok = (
                        boundary <= NUMERICS["BOUNDARY_TOLERANCE"]
                        and (
                            role == "AUDIT_ONLY"
                            or raw_boundary <= NUMERICS["BOUNDARY_TOLERANCE"]
                        )
                    )

                    stable = (
                        grid_change is not None
                        and grid_change
                        <= NUMERICS["GRID_STABILITY_TOLERANCE"]
                    )

                    # If adaptation is disabled, no stability claim is made.
                    usable = boundary_ok and stable

                    home, away = v3.score_arrays(k)

                    caution = (
                        current_system["minimum_slack"]
                        > NUMERICS["PROJECTION_BUFFER"]
                        or fit["parameters_at_bound"]
                    )

                    models[prior] = {
                        "status": "COMPLETED" if usable else "UNSTABLE",
                        "usable": usable,
                        "prior_type": prior,
                        "prior_role": role,
                        "quality_status": (
                            "FAIL" if not usable
                            else "CAUTION" if caution
                            else "PASS"
                        ),
                        "probabilities": padded,
                        "_prior_probabilities": padded_raw,
                        "max_goals": probe,
                        "reconstruction_max_goals": k,
                        "lambda_home": fit["lambda_home"],
                        "lambda_away": fit["lambda_away"],
                        "rho": fit["rho"],
                        "dispersion": fit["dispersion"],
                        "parameters_at_bound": fit["parameters_at_bound"],
                        "prior_optimizer_success": fit["optimizer_success"],
                        "prior_optimizer_message": fit["optimizer_message"],
                        "maximum_prior_residual": fit["maximum_prior_residual"],
                        "prior_rmse": fit["prior_rmse"],
                        "prior_expected_home_goals": float(raw @ home),
                        "prior_expected_away_goals": float(raw @ away),
                        "projected_expected_home_goals": float(projected @ home),
                        "projected_expected_away_goals": float(projected @ away),
                        "raw_boundary_mass": raw_boundary,
                        "boundary_mass": boundary,
                        "grid_total_variation_change": grid_change,
                        "grid_stable": stable,
                        "minimum_slack": current_system["minimum_slack"],
                        "allowed_slack": current_system["allowed_slack"],
                        "constraint_count": len(constraints),
                        "market_group_count": v3.constraint_group_count(
                            constraints
                        ),
                        "market_family_count": v3.constraint_family_count(
                            constraints
                        ),
                        "structural_coverage": v3.period_identification(
                            constraints
                        ),
                        **{
                            name: value
                            for name, value in projection.items()
                            if name != "probabilities"
                        },
                    }

                except (ValueError, RuntimeError, FloatingPointError) as exc:
                    models[prior] = {
                        "status": "FAILED",
                        "usable": False,
                        "prior_type": prior,
                        "prior_role": role,
                        "quality_status": "FAIL",
                        "reason": str(exc),
                        "reconstruction_max_goals": k,
                    }

            available_forecasts = [
                models[prior]
                for prior in forecast_priors
                if "probabilities" in models[prior]
            ]

            history.append({
                "max_goals": k,
                "minimum_slack": current_system["minimum_slack"],
                "usable_forecast_count": sum(
                    models[prior].get("usable", False)
                    for prior in forecast_priors
                ),
                "forecast_errors": {
                    prior: models[prior].get("reason")
                    for prior in forecast_priors
                    if models[prior]["status"] == "FAILED"
                },
            })

            # Audit priors do not force forecasting grids to grow.
            # Missing priors remain explicitly missing.
            all_available_stable = (
                bool(available_forecasts)
                and all(model["usable"] for model in available_forecasts)
            )

            if all_available_stable or not adaptive or k >= safety:
                break

            # A second-grid check is mandatory before claiming stability.
            previous_models = models
            previous_system = current_system
            k = min(k + step, safety)

        complete = all(
            models.get(prior, {}).get("usable", False)
            for prior in forecast_priors
        )

        # If final system construction failed, do not use an earlier
        # system as if it belonged to the final failed grid.
        if current_system is not None and current_system["max_goals"] != k:
            current_system = None

        return {
            "status": "COMPLETED" if complete else "INCOMPLETE",
            "forecast_complete": complete,
            "reason": last_system_error if current_system is None else None,
            "models": models,
            "history": history,
            "max_goals": k,
            "_constraints": constraints,
            "_system": current_system,
            "_previous_system": previous_system,
        }


# ============================================================
# 7. Market-removal specifications
# ============================================================

def _settlement_signature(a, b):
    # Settlement coefficients are exact multiples of 0.5 here.
    return (
        np.asarray(a, dtype=np.float64).tobytes(),
        np.asarray(b, dtype=np.float64).tobytes(),
    )


def _equivalent_groups(constraints, candidate, probe_goals):
    home, away = v3.score_arrays(probe_goals)

    hidden = [
        item for item in constraints
        if item["group_key"] == candidate["group_key"]
    ]

    if not hidden:
        hidden = [candidate]

    signatures = set()

    for item in hidden:
        a, b = v3.settlement_coefficients(
            home, away,
            item["market"], item["selection"],
            item.get("line"), item.get("team"),
        )
        signatures.add(_settlement_signature(a, b))
        signatures.add(_settlement_signature(1.0 - a - b, b))

    groups = {candidate["group_key"]}

    for item in constraints:
        a, b = v3.settlement_coefficients(
            home, away,
            item["market"], item["selection"],
            item.get("line"), item.get("team"),
        )

        if _settlement_signature(a, b) in signatures:
            groups.add(item["group_key"])

    return groups


def _retained_constraints(constraints, candidate, mode, probe_goals):
    direct_target = v3.direct_target_for_candidate(
        constraints, candidate
    )

    if mode == "FULL_MARKET":
        return {
            "status": "AVAILABLE",
            "constraints": constraints,
            "removed_groups": [],
            "direct_target": direct_target,
            "scenario_type": mode,
        }

    if mode == "FAMILY_OUT":
        present = any(
            item["family_key"] == candidate["family_key"]
            for item in constraints
        )
        if not present:
            return {
                "status": "NOT_TESTABLE",
                "reason": "FAMILY_NOT_PRESENT",
            }

        reduced = [
            item for item in constraints
            if item["family_key"] != candidate["family_key"]
        ]

    elif mode == "TARGET_GROUP_OUT":
        reduced = [
            item for item in constraints
            if item["group_key"] != candidate["group_key"]
        ]

    elif mode == "EQUIVALENT_SETTLEMENT_OUT":
        removed_groups = _equivalent_groups(
            constraints, candidate, probe_goals
        )
        reduced = [
            item for item in constraints
            if item["group_key"] not in removed_groups
        ]

    else:
        raise ValueError(f"Unsupported reconstruction mode: {mode}")

    removed = sorted(
        {item["group_key"] for item in constraints}
        - {item["group_key"] for item in reduced}
    )

    return {
        "status": "AVAILABLE",
        "constraints": reduced,
        "removed_groups": removed,
        "direct_target": direct_target,
        "scenario_type": (
            "UNQUOTED_LINE_RECONSTRUCTION"
            if mode == "TARGET_GROUP_OUT" and not removed
            else mode
        ),
    }


# ============================================================
# 8. Forecast evaluation
# ============================================================

def _evaluate_mode(data, candidate, mode, constraint_sets, cache):
    period = candidate["period"]
    forecast_priors = data["settings"]["forecast_priors"]
    all_priors = forecast_priors + data["settings"]["audit_priors"]

    records = []
    scenarios = []
    audit_scenarios = []
    subset_entries = []

    for book in data["sharp_books"]:
        for method in data["settings"]["devig_methods"]:
            key = (period, book["key"], method)
            constraints = constraint_sets.get(key)

            if not constraints:
                continue

            specification = _retained_constraints(
                constraints,
                candidate,
                mode,
                cache.probe_goals[period],
            )

            entry = {
                "source": book["key"],
                "devig_method": method,
                "mode": mode,
                "specification": specification,
                "_bundle": None,
            }

            if specification["status"] == "AVAILABLE":
                entry["_bundle"] = cache.get(
                    period, specification["constraints"]
                )

            subset_entries.append(entry)

            for prior in all_priors:
                role = (
                    "FORECAST"
                    if prior in forecast_priors
                    else "AUDIT_ONLY"
                )

                base = {
                    "source": book["key"],
                    "source_title": book["title"],
                    "devig_method": method,
                    "prior_type": prior,
                    "prior_role": role,
                    "period": period,
                    "mode": mode,
                }

                if specification["status"] != "AVAILABLE":
                    records.append({
                        **base,
                        "status": "NOT_TESTABLE",
                        "reason": specification["reason"],
                    })
                    continue

                bundle = entry["_bundle"]
                model = bundle["models"].get(prior)

                if model is None or not model.get("usable", False):
                    records.append({
                        **base,
                        "status": (
                            model.get("status", "FAILED")
                            if model else bundle["status"]
                        ),
                        "reason": (
                            model.get("reason", "GRID_NOT_VERIFIED")
                            if model else bundle.get("reason")
                        ),
                        "diagnostics": _public(model or {}),
                    })
                    continue

                home, away = v3.score_arrays(model["max_goals"])
                a, b = v3.settlement_coefficients(
                    home, away,
                    candidate["market"], candidate["selection"],
                    candidate["line"], candidate.get("team"),
                )

                metrics = v3.candidate_metrics(
                    model["probabilities"], a, b, candidate["odds"]
                )
                raw_metrics = v3.candidate_metrics(
                    model["_prior_probabilities"],
                    a, b, candidate["odds"],
                )

                target = specification["direct_target"]
                discrepancy = (
                    metrics["effective_fair_probability"] - target
                    if target is not None else None
                )

                scenario = {
                    **base,
                    "id": (
                        f"{period}|{book['key']}|{method}|{prior}|"
                        f"{mode}|{candidate['id']}"
                    ),
                    "scenario_id": (
                        f"{period}|{book['key']}|{method}|{prior}|"
                        f"{mode}|{candidate['id']}"
                    ),
                    "status": "COMPLETED",
                    "scenario_type": specification["scenario_type"],
                    "removed_groups": specification["removed_groups"],
                    "direct_devig_target": target,
                    "fair_probability_discrepancy": discrepancy,
                    "model_quality": model["quality_status"],
                    "model_diagnostics": _public(model),
                    "raw_prior_hit": raw_metrics["hit_probability"],
                    "projected_hit": metrics["hit_probability"],
                    "projection_change": (
                        metrics["hit_probability"]
                        - raw_metrics["hit_probability"]
                    ),
                    **metrics,
                }

                records.append({
                    **base,
                    "status": "COMPLETED",
                    "quality": model["quality_status"],
                })

                if role == "FORECAST":
                    scenarios.append(scenario)
                else:
                    audit_scenarios.append(scenario)

    forecast_records = [
        item for item in records if item["prior_role"] == "FORECAST"
    ]
    coverage = _coverage(forecast_records, len(forecast_records))

    reference = next(
        (
            item for item in scenarios
            if item["source"] == data["settings"]["primary_source"]
            and item["devig_method"]
            == data["settings"]["v31"]["central_devig_method"]
            and item["prior_type"]
            == data["settings"]["v31"]["central_prior"]
        ),
        None,
    )

    return {
        **coverage,
        "mode": mode,
        "central_probability": (
            reference["hit_probability"] if reference else None
        ),
        "central_scenario_id": (
            reference["scenario_id"] if reference else None
        ),
        "central_definition": "CONFIGURED_REFERENCE_SCENARIO",
        "scenarios": scenarios,
        "audit_scenarios": audit_scenarios,
        "records": records,
        **v3.summarize_scenario_metrics(scenarios),
        "_subset_entries": subset_entries,
    }


# ============================================================
# 9. Candidate bounds and stress
# ============================================================

def _bounds_on_system(system, candidate):
    home, away = v3.score_arrays(system["max_goals"])
    a, b = v3.settlement_coefficients(
        home, away,
        candidate["market"], candidate["selection"],
        candidate["line"], candidate.get("team"),
    )

    event = (
        a * candidate["odds"] + b - 1.0 > TOL
    ).astype(float)

    equality = np.ones((1, system["state_count"]))

    minimum = _checked_lp(
        event, system["G"], system["h"],
        equality, np.array([1.0]),
    )
    maximum = _checked_lp(
        -event, system["G"], system["h"],
        equality, np.array([1.0]),
    )

    low = float(minimum.fun)
    high = -float(maximum.fun)

    if low > high + NUMERICS["LP_ACCEPTANCE_TOLERANCE"]:
        raise RuntimeError("LP minimum exceeds LP maximum.")

    low = float(np.clip(low, 0.0, 1.0))
    high = float(np.clip(high, 0.0, 1.0))

    return {
        "minimum_probability": low,
        "maximum_probability": high,
        "width": max(0.0, high - low),
        "required_slack": system["minimum_slack"],
        "max_goals": system["max_goals"],
    }


def _candidate_bounds(mode_result, candidate, profiles=None, level=None):
    records = []

    for entry in mode_result["_subset_entries"]:
        base = {
            "source": entry["source"],
            "devig_method": entry["devig_method"],
        }
        bundle = entry["_bundle"]

        if bundle is None or bundle["_system"] is None:
            records.append({
                **base,
                "status": "NOT_TESTABLE",
                "reason": "NO_RETAINED_MARKET_SYSTEM",
            })
            continue

        current = bundle["_system"]
        previous = bundle["_previous_system"]

        if previous is None:
            records.append({
                **base,
                "status": "UNSTABLE",
                "reason": "NO_SECOND_GRID_COMPARISON",
            })
            continue

        try:
            if level is not None:
                constraints = bundle["_constraints"]
                intervals = [
                    v3.stress_target_interval(item, level, profiles)
                    for item in constraints
                ]

                current = _make_system(
                    constraints, current["max_goals"], intervals
                )
                previous = _make_system(
                    constraints, previous["max_goals"], intervals
                )

            now = _bounds_on_system(current, candidate)
            before = _bounds_on_system(previous, candidate)

            change = max(
                abs(
                    now["minimum_probability"]
                    - before["minimum_probability"]
                ),
                abs(
                    now["maximum_probability"]
                    - before["maximum_probability"]
                ),
                abs(now["required_slack"] - before["required_slack"]),
            )

            stable = (
                change <= NUMERICS["GRID_STABILITY_TOLERANCE"]
            )

            records.append({
                **base,
                **now,
                "status": "COMPLETED" if stable else "UNSTABLE",
                "grid_change": change,
                "previous_grid": before,
            })

        except (ValueError, RuntimeError) as exc:
            records.append({
                **base,
                "status": "FAILED",
                "reason": str(exc),
            })

    valid = [
        item for item in records if item["status"] == "COMPLETED"
    ]

    lows = [item["minimum_probability"] for item in valid]
    highs = [item["maximum_probability"] for item in valid]
    widths = [item["width"] for item in valid]

    return {
        **_coverage(records, len(records)),
        "lower": _stats(lows),
        "upper": _stats(highs),
        "width": _stats(widths),
        "overall_envelope": {
            "minimum": min(lows) if lows else None,
            "maximum": max(highs) if highs else None,
            "width": (
                max(highs) - min(lows) if lows and highs else None
            ),
        },
        "records": records,
        "minimum_hit_probability": min(lows) if lows else None,
        "conditioning": {
            "finite_grid": True,
            "minimum_nonpush_mass": NUMERICS["MIN_NONPUSH_MASS"],
            "minimum_slack_relaxation": True,
            "second_grid_stability_check": True,
        },
        "interpretation": (
            "Finite-grid market-feasible bounds, not a confidence interval. "
            "Stability on two tested grids is not an infinite-support proof."
        ),
    }


# ============================================================
# 10. Prior / MaxEnt diagnostics and support classification
# ============================================================

def _prior_diagnostics(data, mode_result):
    required = set(data["settings"]["forecast_priors"])
    grouped = {}

    for item in mode_result["scenarios"]:
        key = (item["source"], item["devig_method"])
        grouped.setdefault(key, {})[item["prior_type"]] = item

    audit_map = {
        (item["source"], item["devig_method"]): item
        for item in mode_result["audit_scenarios"]
        if item["prior_type"] == "FLAT_GRID_MAXENT"
    }

    records = []

    for entry in mode_result["_subset_entries"]:
        key = (entry["source"], entry["devig_method"])
        forecasts = grouped.get(key, {})
        complete = set(forecasts) == required

        values = [
            item["hit_probability"] for item in forecasts.values()
        ]
        raw_values = [
            item["raw_prior_hit"] for item in forecasts.values()
        ]

        maxent = audit_map.get(key)
        central = float(np.median(values)) if values else None

        records.append({
            "source": key[0],
            "devig_method": key[1],
            "status": "COMPLETED" if complete else "INCOMPLETE",
            "available_priors": sorted(forecasts),
            "required_priors": sorted(required),
            "prior_spread": (
                max(values) - min(values) if complete else None
            ),
            "raw_prior_spread": (
                max(raw_values) - min(raw_values) if complete else None
            ),
            "maxent_status": (
                "COMPLETED" if maxent is not None else "NOT_AVAILABLE"
            ),
            "maxent_absolute_gap": (
                abs(maxent["hit_probability"] - central)
                if maxent is not None and complete else None
            ),
        })

    coverage = _coverage(records, len(records))

    return {
        **coverage,
        "prior_spread": _stats(item["prior_spread"] for item in records),
        "raw_prior_spread": _stats(
            item["raw_prior_spread"] for item in records
        ),
        "maxent_absolute_gap": _stats(
            item["maxent_absolute_gap"] for item in records
        ),
        "maxent_complete": bool(records) and all(
            item["maxent_status"] == "COMPLETED"
            for item in records
        ),
        "records": records,
    }


def _market_support(thresholds, diagnostics, bounds, hidden_result):
    configured = {
        key: value for key, value in thresholds.items()
        if value is not None
    }

    if not configured:
        return {
            "status": "CALIBRATION_REQUIRED",
            "market_supported": None,
            "reasons": ["NO_USER_SUPPLIED_SUPPORT_THRESHOLDS"],
            "thresholds": thresholds,
        }

    reasons = []

    if not diagnostics["complete"]:
        reasons.append("FORECAST_PRIOR_SET_INCOMPLETE")

    observed = {
        "max_prior_spread": diagnostics["prior_spread"]["maximum"],
        # V3.1 uses the overall envelope, including cross-source differences.
        "max_feasible_width": bounds.get(
            "overall_envelope", {}
        ).get("width"),
        "max_maxent_gap": diagnostics["maxent_absolute_gap"]["maximum"],
        "max_hidden_line_absolute_error": hidden_result[
            "absolute_fair_probability_discrepancy"
        ]["maximum"],
    }

    prerequisites = {
        "max_prior_spread": diagnostics["complete"],
        "max_feasible_width": bounds.get("complete", False),
        "max_maxent_gap": diagnostics["maxent_complete"],
        "max_hidden_line_absolute_error": (
            hidden_result["complete"]
            and bool(hidden_result["scenarios"])
            and all(
                item["direct_devig_target"] is not None
                for item in hidden_result["scenarios"]
            )
        ),
    }

    for name, threshold in configured.items():
        value = observed[name]
        if not prerequisites[name] or value is None:
            reasons.append(f"{name.upper()}_NOT_TESTABLE")
        elif value > threshold + TOL:
            reasons.append(f"{name.upper()}_EXCEEDED")

    return {
        "status": (
            "MARKET_SUPPORTED" if not reasons
            else "NOT_MARKET_SUPPORTED"
        ),
        "market_supported": not reasons,
        "reasons": reasons,
        "thresholds": thresholds,
        "observed": observed,
        "note": (
            "User-supplied diagnostic thresholds are not evidence "
            "of outcome probability calibration."
        ),
    }


# ============================================================
# 11. Optional historical calibration
# ============================================================

def _apply_calibrator(data, candidate, probability):
    group = _calibration_group(candidate)
    artifact = data["settings"]["v31"]["calibrators"].get(group)

    result = {
        "status": "NOT_CONFIGURED",
        "group": group,
        "probability": None,
    }

    if artifact is None:
        return result

    try:
        if probability is None:
            raise ValueError("Central probability unavailable.")

        if artifact.get("method") != "LOGIT_SIGMOID":
            raise ValueError("Unsupported calibration method.")

        if artifact.get("approved") is not True:
            raise ValueError("Calibrator has not been approved.")

        if artifact.get("group") != group:
            raise ValueError("Calibration group mismatch.")

        if artifact.get("pipeline_id") != data["settings"]["v31"]["pipeline_id"]:
            raise ValueError("Calibration pipeline mismatch.")

        training_end = _utc(artifact["training_end"])
        validation_end = _utc(artifact["validation_end"])
        prediction_time = _utc(data["match"]["snapshot_time"])

        if not training_end < validation_end < prediction_time:
            raise ValueError(
                "Require training_end < validation_end < prediction_time."
            )

        slope = _finite_float(artifact["slope"], "Calibration slope")
        intercept = _finite_float(
            artifact["intercept"], "Calibration intercept"
        )

        if slope < 0.0:
            raise ValueError("Calibration slope must be non-negative.")

        p = float(np.clip(probability, 1e-12, 1.0 - 1e-12))
        logit = math.log(p / (1.0 - p))
        calibrated = float(expit(slope * logit + intercept))

        return {
            "status": "APPLIED",
            "group": group,
            "probability": calibrated,
            "method": "LOGIT_SIGMOID",
            "training_end": training_end.isoformat(),
            "validation_end": validation_end.isoformat(),
            "note": (
                "Candidate hit calibration only; it does not "
                "recalibrate the joint score distribution."
            ),
        }

    except (KeyError, TypeError, ValueError) as exc:
        return {
            **result,
            "status": "REJECTED",
            "reason": str(exc),
        }


def fit_hit_calibrator(rows):
    """
    Fit sigmoid(a * logit(p) + b), with a >= 0.

    This is logistic recalibration on logit probabilities, not a call
    to sklearn's raw-score Platt implementation.

    Input rows must:
    - belong to one calibration group and pipeline;
    - contain valid central_probability and hit;
    - contain prediction_time and settled_at;
    - come only from the chosen historical TRAINING period.

    Returns an UNAPPROVED artifact. Independent validation is still needed.
    """
    rows = list(rows)

    if len(rows) < 2:
        raise ValueError("At least two training rows are required.")

    groups = {row["group"] for row in rows}
    pipelines = {row["pipeline_id"] for row in rows}

    if len(groups) != 1 or len(pipelines) != 1:
        raise ValueError("Use one calibration group and pipeline per fit.")

    probabilities = []
    labels = []
    times = []

    for row in rows:
        probability = _finite_float(
            row["central_probability"], "central_probability"
        )
        label = row["hit"]

        if not 0.0 <= probability <= 1.0 or label not in (0, 1):
            raise ValueError("Invalid probability or hit label.")

        prediction_time = _utc(row["prediction_time"])
        settled_at = _utc(row["settled_at"])

        if prediction_time >= settled_at:
            raise ValueError("Prediction must precede settlement.")

        probabilities.append(probability)
        labels.append(int(label))
        times.append(settled_at)

    y = np.asarray(labels, dtype=float)

    if len(np.unique(y)) != 2:
        raise ValueError("Training data must contain hits and non-hits.")

    p = np.clip(np.asarray(probabilities), 1e-12, 1.0 - 1e-12)
    x = np.log(p / (1.0 - p))

    if float(np.std(x)) <= 1e-12:
        raise ValueError("Insufficient variation in forecast probabilities.")

    counts = Counter(row["match_id"] for row in rows)
    weights = np.array([
        1.0 / counts[row["match_id"]] for row in rows
    ])
    weights /= weights.sum()

    def objective(parameters):
        z = parameters[0] * x + parameters[1]
        fitted = expit(z)

        loss = float(
            weights @ (np.logaddexp(0.0, z) - y * z)
        )
        error = weights * (fitted - y)
        gradient = np.array([error @ x, error.sum()])
        return loss, gradient

    fit = minimize(
        objective,
        np.array([1.0, 0.0]),
        jac=True,
        method="L-BFGS-B",
        bounds=[(0.0, None), (None, None)],
        options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-9},
    )

    if not fit.success or not np.all(np.isfinite(fit.x)):
        raise RuntimeError(f"Calibration fit failed: {fit.message}")

    return {
        "method": "LOGIT_SIGMOID",
        "group": next(iter(groups)),
        "pipeline_id": next(iter(pipelines)),
        "slope": float(fit.x[0]),
        "intercept": float(fit.x[1]),
        "training_end": max(times).isoformat(),
        "training_row_count": len(rows),
        "training_match_count": len(counts),
        "training_match_ids": sorted(str(key) for key in counts),
        "approved": False,
        "validation_end": None,
        "status": "FITTED_NOT_VALIDATED",
    }


def evaluate_validation_rows(rows, calibrator=None, bins=10):
    """
    Evaluate central probabilities, optionally after calibration.

    Each match gets equal total weight within the supplied rows.
    Bin counts and match counts are reported.
    No independence-based confidence intervals are invented.
    """
    rows = list(rows)

    if not rows:
        raise ValueError("No validation rows.")

    if not isinstance(bins, int) or bins < 2:
        raise ValueError("bins must be an integer >= 2.")

    training_ids = set()

    if calibrator is not None:
        training_ids = set(calibrator.get("training_match_ids", []))
        training_end = _utc(calibrator["training_end"])

    predictions = []
    labels = []

    for row in rows:
        probability = _finite_float(
            row["central_probability"], "central_probability"
        )

        if not 0.0 <= probability <= 1.0 or row["hit"] not in (0, 1):
            raise ValueError("Invalid validation row.")

        if calibrator is not None:
            if row["pipeline_id"] != calibrator["pipeline_id"]:
                raise ValueError("Validation pipeline mismatch.")
            if row["group"] != calibrator["group"]:
                raise ValueError("Validation group mismatch.")
            if str(row["match_id"]) in training_ids:
                raise ValueError("A training match appears in validation.")
            if _utc(row["prediction_time"]) <= training_end:
                raise ValueError("Validation must follow calibration training.")

            p = float(np.clip(probability, 1e-12, 1.0 - 1e-12))
            probability = float(expit(
                calibrator["slope"] * math.log(p / (1.0 - p))
                + calibrator["intercept"]
            ))

        predictions.append(probability)
        labels.append(int(row["hit"]))

    p = np.asarray(predictions)
    y = np.asarray(labels, dtype=float)

    counts = Counter(row["match_id"] for row in rows)
    weights = np.array([
        1.0 / counts[row["match_id"]] for row in rows
    ])
    weights /= weights.sum()

    clipped = np.clip(p, 1e-12, 1.0 - 1e-12)
    brier = float(weights @ ((p - y) ** 2))
    log_loss = float(
        -weights @ (
            y * np.log(clipped)
            + (1.0 - y) * np.log1p(-clipped)
        )
    )

    assignments = np.minimum((p * bins).astype(int), bins - 1)
    reliability = []

    for index in range(bins):
        mask = assignments == index

        if not np.any(mask):
            continue

        local_weights = weights[mask]
        local_weights /= local_weights.sum()

        matching = [
            row for row, selected in zip(rows, mask) if selected
        ]

        reliability.append({
            "bin_lower": index / bins,
            "bin_upper": (index + 1) / bins,
            "row_count": int(mask.sum()),
            "match_count": len({
                row["match_id"] for row in matching
            }),
            "mean_prediction": float(local_weights @ p[mask]),
            "observed_hit_rate": float(local_weights @ y[mask]),
        })

    return {
        "row_count": len(rows),
        "match_count": len(counts),
        "brier_score": brier,
        "log_loss": log_loss,
        "reliability": reliability,
        "calibration_applied": calibrator is not None,
        "validation_end": max(
            _utc(row["settled_at"]) for row in rows
        ).isoformat(),
        "warning": (
            "Proper scores assess more than calibration. "
            "Inspect reliability and subgroup sample sizes too."
        ),
    }


# ============================================================
# 12. HT-FT coherence with checked LP
# ============================================================

def _transport_check(ht, ft):
    ht_p = _normalise_probability_vector(ht["probabilities"])
    ft_p = _normalise_probability_vector(ft["probabilities"])

    hh, ha = v3.score_arrays(ht["max_goals"])
    fh, fa = v3.score_arrays(ft["max_goals"])

    active_ht = np.flatnonzero(ht_p > 1e-14)
    active_ft = np.flatnonzero(ft_p > 1e-14)

    source_indices = []
    destination_indices = []

    for i in active_ht:
        compatible = active_ft[
            (fh[active_ft] >= hh[i])
            & (fa[active_ft] >= ha[i])
        ]
        source_indices.extend([int(i)] * len(compatible))
        destination_indices.extend(int(j) for j in compatible)

    if not source_indices:
        return {
            "status": "COMPLETED",
            "coherent": False,
            "minimum_violation_mass": 1.0,
        }

    source_indices = np.asarray(source_indices)
    destination_indices = np.asarray(destination_indices)
    n = len(source_indices)

    rows = np.concatenate([
        source_indices,
        len(ht_p) + destination_indices,
    ])
    columns = np.concatenate([np.arange(n), np.arange(n)])

    capacity = coo_matrix(
        (np.ones(2 * n), (rows, columns)),
        shape=(len(ht_p) + len(ft_p), n),
    ).tocsr()

    capacities = np.concatenate([ht_p, ft_p])

    result = _checked_lp(
        -np.ones(n),
        capacity,
        capacities,
    )

    flow = np.maximum(np.asarray(result.x), 0.0)
    original_mass = float(flow.sum())

    # Conservatively repair any tiny capacity excess before reporting
    # a physically feasible compatible-flow mass.
    used = np.asarray(capacity @ flow)
    ratios = np.ones_like(capacities)
    positive = used > 0
    ratios[positive] = np.minimum(
        1.0, capacities[positive] / used[positive]
    )

    flow *= np.minimum(
        ratios[source_indices],
        ratios[len(ht_p) + destination_indices],
    )

    compatible_mass = float(flow.sum())
    repair_mass = max(0.0, original_mass - compatible_mass)
    violation = max(0.0, 1.0 - compatible_mass)

    tolerance = (
        2.0 * NUMERICS["PROJECTION_BUFFER"]
        + float(ht_p[ht_p <= 1e-14].sum())
        + float(ft_p[ft_p <= 1e-14].sum())
    )

    reliable = repair_mass <= NUMERICS["PROJECTION_BUFFER"]

    return {
        "status": "COMPLETED" if reliable else "FAILED",
        "coherent": reliable and violation <= tolerance,
        "minimum_violation_mass": violation,
        "numerical_repair_mass": repair_mass,
        "tolerance": tolerance,
        "solver_message": str(result.message),
    }


def _coherence_audit(data, full_models, constraint_sets):
    if not data["settings"]["features"]["ht_ft_coherence"]:
        return {"status": "DISABLED", "scenarios": []}

    records = []

    for book in data["sharp_books"]:
        for method in data["settings"]["devig_methods"]:
            if ("HT", book["key"], method) not in constraint_sets:
                continue

            for prior in data["settings"]["forecast_priors"]:
                key = (book["key"], method, prior)
                ht = full_models.get(("HT",) + key)
                ft = full_models.get(("FT",) + key)

                base = {
                    "source": book["key"],
                    "devig_method": method,
                    "prior_type": prior,
                }

                if ht is None or ft is None:
                    records.append({
                        **base,
                        "status": "FAILED",
                        "coherent": False,
                        "reason": "MATCHED_VALID_FORECAST_UNAVAILABLE",
                    })
                    continue

                try:
                    records.append({
                        **base,
                        **_transport_check(ht, ft),
                    })
                except (ValueError, RuntimeError) as exc:
                    records.append({
                        **base,
                        "status": "FAILED",
                        "coherent": False,
                        "reason": str(exc),
                    })

    if not records:
        return {"status": "NOT_AVAILABLE", "scenarios": []}

    primary = [
        record for record in records
        if record["source"] == data["settings"]["primary_source"]
    ]

    required = (
        len(data["settings"]["forecast_priors"])
        * len(data["settings"]["devig_methods"])
    )

    primary_complete = (
        len(primary) == required
        and all(item["status"] == "COMPLETED" for item in primary)
    )

    passed = all(
        item["status"] == "COMPLETED" and item["coherent"]
        for item in records
    )

    return {
        "status": "PASS" if passed else "FAIL",
        "primary_source_complete": primary_complete,
        "primary_source_pass": (
            primary_complete
            and all(item["coherent"] for item in primary)
        ),
        "scenarios": records,
        "scope": "FULL_MARKET_PRIOR_MATCHED_DISTRIBUTIONS",
        "note": (
            "This does not certify candidate-specific removed-market "
            "distributions as HT-FT coherent."
        ),
    }


# ============================================================
# 13. Candidate construction and shortlist
# ============================================================

def _build_candidate(
    data, candidate, constraint_sets, cache, profiles, coherence
):
    full = _evaluate_mode(
        data, candidate, "FULL_MARKET", constraint_sets, cache
    )
    hidden = _evaluate_mode(
        data, candidate, "TARGET_GROUP_OUT", constraint_sets, cache
    )

    options = data["settings"]["v31"]
    main = (
        full if options["forecast_mode"] == "FULL_MARKET"
        else hidden
    )

    equivalent = (
        _evaluate_mode(
            data, candidate, "EQUIVALENT_SETTLEMENT_OUT",
            constraint_sets, cache,
        )
        if options["equivalent_settlement_audit"]
        else {"status": "DISABLED"}
    )

    family = (
        _evaluate_mode(
            data, candidate, "FAMILY_OUT", constraint_sets, cache
        )
        if data["settings"]["features"]["family_out_audit"]
        else {"status": "DISABLED"}
    )

    diagnostics = _prior_diagnostics(data, main)

    bounds = (
        _candidate_bounds(main, candidate)
        if data["settings"]["features"]["feasible_bounds"]
        else {
            "status": "DISABLED",
            "complete": False,
            "overall_envelope": {"minimum": None, "maximum": None, "width": None},
        }
    )

    if data["settings"]["features"]["stress_audit"]:
        stress = {
            level.lower(): _candidate_bounds(
                main, candidate, profiles, level
            )
            for level in ("LIGHT", "MEDIUM", "HEAVY")
        }
    else:
        stress = {"status": "DISABLED"}

    support = _market_support(
        data["settings"]["support_thresholds"][candidate["period"]],
        diagnostics,
        bounds,
        hidden,
    )

    calibration = _apply_calibrator(
        data, candidate, main["central_probability"]
    )

    reasons = []
    forecasts = main["scenarios"]

    primary = [
        record for record in main["records"]
        if record["prior_role"] == "FORECAST"
        and record["source"] == data["settings"]["primary_source"]
    ]

    expected_primary = (
        len(data["settings"]["forecast_priors"])
        * len(data["settings"]["devig_methods"])
    )

    primary_complete = (
        len(primary) == expected_primary
        and all(record["status"] == "COMPLETED" for record in primary)
    )

    if not forecasts:
        reasons.append("NO_VALID_FORECAST")
    if not primary_complete:
        reasons.append("PRIMARY_FORECASTS_INCOMPLETE")
    if options["require_complete_forecasts"] and not main["complete"]:
        reasons.append("SUPPLIED_FORECAST_SCENARIOS_INCOMPLETE")

    if candidate["odds"] < data["settings"]["minimum_odds"]:
        reasons.append("BELOW_MINIMUM_ODDS")

    maximum_odds = data["settings"]["maximum_odds"]
    if maximum_odds is not None and candidate["odds"] > maximum_odds:
        reasons.append("ABOVE_MAXIMUM_ODDS")

    # Inspect retained constraints, not the original full-market block.
    if candidate["period"] == "HT":
        primary_entries = [
            entry for entry in main["_subset_entries"]
            if entry["source"] == data["settings"]["primary_source"]
        ]

        if not primary_entries or any(
            entry["specification"]["status"] != "AVAILABLE"
            or not v3.period_identification(
                entry["specification"].get("constraints", [])
            )["minimum_structural_coverage_met"]
            for entry in primary_entries
        ):
            reasons.append("HT_RETAINED_MARKET_COVERAGE_INSUFFICIENT")

        if data["settings"]["features"]["ht_ft_coherence"]:
            if coherence.get("primary_source_pass") is not True:
                reasons.append("HT_FT_COHERENCE_NOT_PASSED")

    ev_floor = data["settings"]["ev_rejection_floor"]
    conservative_ev = main["expected_return"]["minimum"]

    if (
        ev_floor is not None
        and conservative_ev is not None
        and conservative_ev < ev_floor - TOL
    ):
        reasons.append("BELOW_OPTIONAL_EV_FLOOR")

    if data["settings"]["require_market_supported_for_shortlist"]:
        if support["market_supported"] is not True:
            reasons.append("MARKET_SUPPORT_GATE_NOT_PASSED")

    if options["ranking"] == "CENTRAL":
        ranking_probability = main["central_probability"]
    elif options["ranking"] == "CALIBRATED":
        ranking_probability = calibration["probability"]
    else:
        ranking_probability = main["probability"]["hit"]["minimum"]

    if ranking_probability is None:
        reasons.append("RANKING_PROBABILITY_UNAVAILABLE")

    minimum_hit = data["settings"]["minimum_shortlist_hit_probability"]
    if (
        ranking_probability is not None
        and ranking_probability < minimum_hit - TOL
    ):
        reasons.append("BELOW_MINIMUM_SHORTLIST_HIT")

    quality = (
        "FAIL" if not forecasts
        else "CAUTION" if (
            not main["complete"]
            or any(item["model_quality"] != "PASS" for item in forecasts)
        )
        else "PASS"
    )

    home, away = v3.score_arrays(
        cache.probe_goals[candidate["period"]]
    )
    a, b = v3.settlement_coefficients(
        home, away,
        candidate["market"], candidate["selection"],
        candidate["line"], candidate.get("team"),
    )
    profit = a * candidate["odds"] + b - 1.0
    hit_mask = profit > TOL

    family_assessment = "NOT_TESTABLE"
    if family.get("complete"):
        family_minimum = family["probability"]["hit"]["minimum"]
        family_assessment = (
            "ROBUST"
            if family_minimum is not None and family_minimum >= minimum_hit
            else "FRAGILE"
        )

    return {
        "id": candidate["id"],
        "label": candidate["label"],
        "period": candidate["period"],
        "market": candidate["market"],
        "selection": candidate["selection"],
        "line": candidate["line"],
        "team": candidate.get("team"),
        "hkjc_odds": candidate["odds"],
        "group_key": candidate["group_key"],
        "family_key": candidate["family_key"],
        "eligible": not reasons,
        "exclusion_reasons": reasons,
        "shortlisted": False,
        "shortlist_rank": None,
        "shortlist_status": "REFERENCE_ONLY",
        "shortlist_exclusion_reasons": [],
        "conflicts_with": [],
        "publication_status": "MANUAL_REVIEW_REQUIRED",
        "forecast_mode": options["forecast_mode"],
        "ranking_basis": options["ranking"],
        "ranking_probability": ranking_probability,
        "central_probability": main["central_probability"],
        "central_scenario_id": main["central_scenario_id"],
        "calibrated_probability": calibration["probability"],
        "calibration": calibration,
        "scenario_minimum": main["probability"]["hit"]["minimum"],
        "scenario_median": main["probability"]["hit"]["median"],
        "forecast_scenario_count": len(forecasts),
        "primary_source_complete": primary_complete,
        "forecast_complete": main["complete"],
        "model_quality": {"status": quality},
        **v3.summarize_scenario_metrics(forecasts),
        "price_status": v3.candidate_price_status(
            conservative_ev, main["expected_return"]["median"]
        ),
        "scenarios": forecasts,
        "prior_diagnostics": diagnostics,
        "feasible_probability_bounds": bounds,
        "market_support": support,
        "stress_audit": stress,
        "full_market_forecast": full,
        "target_line_out_audit": hidden,
        "equivalent_settlement_out_audit": equivalent,
        "family_out_audit": {
            **family,
            "robustness_assessment": family_assessment,
        },
        "_original_index": candidate["original_index"],
        "_hit_mask": hit_mask,
        "_miss_mask": profit < -TOL,
        "_nonloss_mask": profit >= -TOL,
        "_hit_signature": np.packbits(hit_mask).tobytes(),
        "_home_scores": home,
        "_away_scores": away,
        "_max_goals": cache.probe_goals[candidate["period"]],
    }


def _ranking_key(candidate):
    probability = candidate["ranking_probability"]
    return (
        probability if probability is not None else -1.0,
        -candidate["_original_index"],
    )


def _choose_shortlist(candidates, maximum):
    selected = []

    for candidate in sorted(candidates, key=_ranking_key, reverse=True):
        if not candidate["eligible"]:
            candidate["shortlist_exclusion_reasons"] = list(
                candidate["exclusion_reasons"]
            )
            continue

        conflicts = []

        for existing in selected:
            reason = v3.candidate_conflict_reason(candidate, existing)
            if reason is not None:
                conflicts.append({
                    "selected_id": existing["id"],
                    "reason": reason,
                })

        if conflicts:
            candidate["conflicts_with"] = conflicts
            candidate["shortlist_exclusion_reasons"] = [
                "CONFLICTS_WITH_HIGHER_RANKED_PICK"
            ]
            continue

        if len(selected) >= maximum:
            candidate["shortlist_exclusion_reasons"] = [
                "MAXIMUM_RECOMMENDATIONS_REACHED"
            ]
            continue

        selected.append(candidate)
        candidate["shortlisted"] = True
        candidate["shortlist_rank"] = len(selected)
        candidate["shortlist_status"] = "ENGINE_SHORTLIST"

    return selected


# ============================================================
# 14. Main engine
# ============================================================

def run_engine(input_data):
    started = datetime.now(timezone.utc)
    data = _normalise_input(input_data)

    constraint_sets = {}

    for book in data["sharp_books"]:
        for period in ("FT", "HT"):
            if not v3.period_has_market_data(book["markets"][period]):
                continue

            for method in data["settings"]["devig_methods"]:
                constraint_sets[(period, book["key"], method)] = (
                    v3.build_book_constraints(book, period, method)
                )

    # Larger diagnostic grids prevent simple equivalence checks from
    # declaring all extreme-line events identical on a too-small grid.
    probe_goals = {}

    for period in ("FT", "HT"):
        _, safety = v3.period_grid_limits(period)

        lines = [
            abs(float(item["line"]))
            for key, constraints in constraint_sets.items()
            if key[0] == period
            for item in constraints
            if item["line"] is not None
        ]
        lines.extend(
            abs(float(candidate["line"]))
            for candidate in data["hkjc_markets"]
            if candidate["period"] == period
            and candidate["line"] is not None
        )

        probe = max(safety, int(math.ceil(max(lines, default=0.0))) + 2)

        if probe > NUMERICS["MAX_EQUIVALENCE_PROBE_GOALS"]:
            raise ValueError(
                "Input lines exceed the supported equivalence-probe limit."
            )

        probe_goals[period] = probe

    cache = ReconstructionCache(data, probe_goals)
    full_models = {}
    period_results = {}
    full_records = []

    for period in ("FT", "HT"):
        scenarios = []

        for key, constraints in constraint_sets.items():
            if key[0] != period:
                continue

            _, source, method = key
            bundle = cache.get(period, constraints)

            full_records.append({
                "period": period,
                "source": source,
                "devig_method": method,
                "status": bundle["status"],
                "grid_history": bundle["history"],
                "models": _public(bundle["models"]),
            })

            for prior, model in bundle["models"].items():
                if (
                    prior not in data["settings"]["forecast_priors"]
                    or not model.get("usable", False)
                ):
                    continue

                scenario = {
                    **model,
                    "id": f"{period}|{source}|{method}|{prior}|FULL",
                    "source": source,
                    "devig_method": method,
                    "period": period,
                }

                full_models[(period, source, method, prior)] = scenario
                scenarios.append(scenario)

        if any(key[0] == period for key in constraint_sets):
            period_results[period] = {
                "max_goals": probe_goals[period],
                "scenarios": scenarios,
            }

    coherence = _coherence_audit(data, full_models, constraint_sets)
    profiles = v3.build_uncertainty_profiles(constraint_sets)

    candidates = [
        _build_candidate(
            data, candidate, constraint_sets, cache, profiles, coherence
        )
        for candidate in data["hkjc_markets"]
    ]

    selected = _choose_shortlist(
        candidates, data["settings"]["max_recommendations"]
    )

    # These remain full-market, uncalibrated distribution calculations.
    joint = v3.joint_recommendation_metrics(
        selected, period_results
    )
    joint["probability_basis"] = "VALID_FULL_MARKET_SCORE_DISTRIBUTIONS"
    joint["candidate_hit_calibration_applied"] = False

    ft_scenarios = period_results.get("FT", {}).get("scenarios", [])

    correct_scores = v3.calculate_correct_scores(
        ft_scenarios,
        probe_goals["FT"],
        data["settings"]["correct_score_count"],
    )
    correct_scores["probability_basis"] = (
        "VALID_FULL_MARKET_SCORE_DISTRIBUTIONS"
    )

    all_full_complete = bool(full_records) and all(
        record["status"] == "COMPLETED" for record in full_records
    )

    full_quality_values = [
        model["quality_status"]
        for model in full_models.values()
    ]

    global_quality = (
        "FAIL" if not ft_scenarios
        else "CAUTION" if (
            not all_full_complete
            or "CAUTION" in full_quality_values
            or coherence.get("status") == "FAIL"
        )
        else "PASS"
    )

    finished = datetime.now(timezone.utc)

    output = {
        "engine": {
            "name": ENGINE_NAME,
            "version": ENGINE_VERSION,
            "generated_at_utc": finished.isoformat(),
            "implementation_dependency": "aegisultra_enginev3.py",
        },
        "status": (
            "COMPLETED"
            if all_full_complete and all(
                item["forecast_complete"] for item in candidates
            )
            else "COMPLETED_WITH_DIAGNOSTICS"
        ),
        "match": data["match"],
        "settings": data["settings"],
        "pipeline_id": data["settings"]["v31"]["pipeline_id"],
        "model_quality": {
            "status": global_quality,
            "full_forecast_complete": all_full_complete,
            "meaning": (
                "Numerical/model diagnostic status, not validated "
                "real-world forecasting accuracy."
            ),
        },
        "methodology": {
            "forecast_mode": data["settings"]["v31"]["forecast_mode"],
            "ranking": data["settings"]["v31"]["ranking"],
            "central_probability": "CONFIGURED_REFERENCE_SCENARIO",
            "scenario_minimum": (
                "Minimum over available valid forecasting scenarios; "
                "not a confidence bound."
            ),
            "forecasting_lp_fallback_used": False,
            "failed_forecasts_excluded": True,
            "exact_minimum_slack_projection": True,
            "experimental_interval_forecasting_enabled": False,
            "new_priors_added": False,
            "prior_fit_weighting": (
                data["settings"]["v31"]["prior_fit_weighting"]
            ),
            "equivalent_settlement_audit": (
                "Removes directly identical/complementary settlement "
                "groups on the diagnostic grid. This does not remove "
                "every possible algebraic implication of other markets."
            ),
            "grid_check": (
                "Boundary and consecutive-grid total-variation checks; "
                "not a proof about infinite-support probabilities."
            ),
            "minimum_nonpush_mass": NUMERICS["MIN_NONPUSH_MASS"],
            "calibration": (
                "Optional, externally reviewed historical artifact. "
                "Never fabricated from current-match prices."
            ),
            "manual_publication_review_required": True,
            "automatic_under_or_receiving_handicap_penalty": False,
            "ev_used_for_ranking": False,
        },
        "model": {
            "full_reconstruction_records": full_records,
            "periods": {
                period: {
                    "diagnostic_grid_max_goals": section["max_goals"],
                    "valid_full_forecast_count": len(section["scenarios"]),
                    "full_scenarios": _public(section["scenarios"]),
                }
                for period, section in period_results.items()
            },
        },
        "ht_ft_coherence": coherence,
        "engine_shortlist": {
            "shortlist_count": len(selected),
            "maximum_recommendations": (
                data["settings"]["max_recommendations"]
            ),
            "ranking": data["settings"]["v31"]["ranking"],
            "minimum_hit_probability": (
                data["settings"]["minimum_shortlist_hit_probability"]
            ),
            "manual_publication_review_required": True,
        },
        "recommendations": [
            _public(candidate) for candidate in selected
        ],
        "candidate_markets": [
            _public(candidate)
            for candidate in sorted(
                candidates, key=_ranking_key, reverse=True
            )
        ],
        "excluded_markets": [
            {
                "id": candidate["id"],
                "label": candidate["label"],
                "reasons": candidate["shortlist_exclusion_reasons"],
                "conflicts_with": candidate["conflicts_with"],
            }
            for candidate in candidates if not candidate["shortlisted"]
        ],
        "recommendation_set": joint,
        "correct_scores": correct_scores,
        "runtime": {
            "total_seconds": (finished - started).total_seconds(),
            "distinct_constraint_subsets": len(cache.cache),
        },
        "input_snapshot": data,
    }

    return v3.to_builtin(output)


# ============================================================
# 15. Outcome logging
# ============================================================

def make_validation_rows(engine_output, outcome):
    """
    Export all candidates, not just selected/published candidates.

    Example outcome:
        {
            "settled_at": "2026-01-02T22:00:00+00:00",
            "FT": {"home": 2, "away": 1},
            "HT": {"home": 0, "away": 0},
        }

    The caller is responsible for supplying genuine pre-match snapshots
    and correct settlement data.
    """
    match = engine_output["match"]
    prediction_time = _utc(match["snapshot_time"])
    kickoff = _utc(match["kickoff"])
    settled_at = _utc(outcome["settled_at"])

    if not prediction_time < kickoff < settled_at:
        raise ValueError(
            "Require pre-match prediction_time < kickoff < settled_at."
        )

    match_id = match.get("id")

    if not match_id:
        identity = {
            "home": match["home"],
            "away": match["away"],
            "competition": match.get("competition", ""),
            "kickoff": kickoff.isoformat(),
        }
        match_id = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode("utf-8")
        ).hexdigest()[:24]

    rows = []

    for candidate in engine_output["candidate_markets"]:
        period_score = outcome.get(candidate["period"])

        if period_score is None:
            continue

        home = _finite_float(period_score["home"], "Home score")
        away = _finite_float(period_score["away"], "Away score")

        if (
            home < 0 or away < 0
            or not home.is_integer()
            or not away.is_integer()
        ):
            raise ValueError("Scores must be non-negative integers.")

        a, b = v3.settlement_coefficients(
            np.array([home]),
            np.array([away]),
            candidate["market"],
            candidate["selection"],
            candidate["line"],
            candidate.get("team"),
        )

        profit = float(
            a[0] * candidate["hkjc_odds"] + b[0] - 1.0
        )

        categories = v3.classify_settlement(a, b)
        settlement = next(
            name for name, mask in categories.items() if bool(mask[0])
        )

        rows.append({
            "match_id": str(match_id),
            "candidate_id": candidate["id"],
            "pipeline_id": engine_output["pipeline_id"],
            "group": _calibration_group(candidate),
            "period": candidate["period"],
            "market": candidate["market"],
            "selection": candidate["selection"],
            "line": candidate["line"],
            "team": candidate.get("team"),
            "prediction_time": prediction_time.isoformat(),
            "kickoff": kickoff.isoformat(),
            "settled_at": settled_at.isoformat(),
            "central_probability": candidate["central_probability"],
            "calibrated_probability": candidate["calibrated_probability"],
            "scenario_minimum": candidate["scenario_minimum"],
            "full_market_probability": candidate[
                "full_market_forecast"
            ]["central_probability"],
            "target_line_out_probability": candidate[
                "target_line_out_audit"
            ]["central_probability"],
            "hit": int(profit > TOL),
            "settlement": settlement,
            "actual_profit_per_unit": profit,
            "eligible": candidate["eligible"],
            "shortlisted": candidate["shortlisted"],
            "forecast_complete": candidate["forecast_complete"],
        })

    return rows


# ============================================================
# 16. Small deterministic smoke tests
# ============================================================

def run_smoke_tests():
    """
    These are basic checks, not a production test suite or backtest.
    No forecasting-accuracy claim follows from passing them.
    """
    home = np.array([0.0, 1.0, 2.0])
    away = np.zeros(3)

    # Under 1.25: full win at 0, half win at 1, full loss at 2.
    a, b = v3.settlement_coefficients(
        home, away, "OU", "UNDER", 1.25
    )
    np.testing.assert_allclose(a, [1.0, 0.5, 0.0])
    np.testing.assert_allclose(b, [0.0, 0.5, 0.0])

    # Under 0.75: half loss at exactly one goal.
    a, b = v3.settlement_coefficients(
        np.array([1.0]), np.array([0.0]),
        "OU", "UNDER", 0.75,
    )
    np.testing.assert_allclose(a, [0.0])
    np.testing.assert_allclose(b, [0.5])

    # Away receiving +0.25 is represented by HOME line -0.25.
    a, b = v3.settlement_coefficients(
        np.array([0.0]), np.array([0.0]),
        "AH", "AWAY", -0.25,
    )
    np.testing.assert_allclose(a, [0.5])
    np.testing.assert_allclose(b, [0.5])

    # 1X2 HOME and AH HOME -0.5 have identical settlements.
    h, a_score = v3.score_arrays(5)
    first = v3.settlement_coefficients(
        h, a_score, "1X2", "HOME"
    )
    second = v3.settlement_coefficients(
        h, a_score, "AH", "HOME", -0.5
    )
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])

    # Checked simplex LP.
    lp = _checked_lp(
        np.array([0.0, 1.0]),
        A_eq=np.ones((1, 2)),
        b_eq=np.array([1.0]),
    )
    assert abs(float(lp.fun)) <= NUMERICS["LP_ACCEPTANCE_TOLERANCE"]

    # Padding preserves mass and coordinates.
    p = np.zeros(9)
    p[1 * 3 + 2] = 1.0
    padded = _pad_probability_vector(p, 2, 4)
    assert padded[1 * 5 + 2] == 1.0
    assert float(padded.sum()) == 1.0

    return {
        "status": "PASSED",
        "scope": "BASIC_SETTLEMENT_LP_AND_PADDING_SMOKE_TESTS",
        "forecast_accuracy_tested": False,
    }


if __name__ == "__main__":
    print(json.dumps(run_smoke_tests(), indent=2))
