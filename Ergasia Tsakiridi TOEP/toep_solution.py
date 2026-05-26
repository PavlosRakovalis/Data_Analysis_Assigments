from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import gamma
from scipy.stats import binom, chi2, norm


BASE_DIR = Path(__file__).resolve().parent
PARAMETERS_FILE = BASE_DIR / "parameters.csv"
DATA_FILE = BASE_DIR / "data.csv"


def load_parameters() -> dict[tuple[str, str], float]:
    params = pd.read_csv(PARAMETERS_FILE)
    return {
        (str(row["Meros"]).strip(), str(row["Parametros"]).strip()): float(row["Timi"])
        for _, row in params.iterrows()
    }


def pa_single(p: float, n: int, c: int) -> float:
    return float(binom.cdf(c, n, p))


def aoq_single(p: float, lot_size: int, n: int, c: int) -> float:
    return p * pa_single(p, n, c) * (lot_size - n) / lot_size


def ati_single(p: float, lot_size: int, n: int, c: int) -> float:
    return n + (lot_size - n) * (1.0 - pa_single(p, n, c))


def escaped_defectives_single(p: float, lot_size: int, n: int, c: int) -> float:
    return p * pa_single(p, n, c) * (lot_size - n)


def maximize_on_unit_interval(func) -> tuple[float, float]:
    result = minimize_scalar(lambda x: -func(x), bounds=(0.0, 1.0), method="bounded")
    candidates = [
        (0.0, func(0.0)),
        (1.0, func(1.0)),
        (float(result.x), func(float(result.x))),
    ]
    return max(candidates, key=lambda item: item[1])


def aoql_single(lot_size: int, n: int, c: int) -> tuple[float, float]:
    return maximize_on_unit_interval(lambda p: aoq_single(p, lot_size, n, c))


def find_single_plan(
    lot_size: int,
    p1: float,
    alpha: float,
    beta: float,
    aoql_limit: float,
) -> dict[str, float]:
    p2 = 7.0 * p1

    for n in range(1, lot_size + 1):
        candidates = []
        for c in range(n + 1):
            pa_p1 = pa_single(p1, n, c)
            pa_p2 = pa_single(p2, n, c)
            if pa_p1 < 1.0 - alpha or pa_p2 > beta:
                continue

            p_at_aoql, max_aoq = aoql_single(lot_size, n, c)
            if max_aoq <= aoql_limit:
                candidates.append(
                    {
                        "n": n,
                        "c": c,
                        "Pa_p1": pa_p1,
                        "producer_risk": 1.0 - pa_p1,
                        "Pa_p2": pa_p2,
                        "customer_risk": pa_p2,
                        "AOQL": max_aoq,
                        "p_at_AOQL": p_at_aoql,
                        "ATI_p1": ati_single(p1, lot_size, n, c),
                        "ATI_p2": ati_single(p2, lot_size, n, c),
                    }
                )

        if candidates:
            return sorted(candidates, key=lambda item: (item["n"], item["c"]))[0]

    raise RuntimeError("No feasible single sampling plan was found.")


def double_components(p: float, m: int, c2: int) -> tuple[float, float, float]:
    pa_first = float(binom.pmf(0, m, p))
    pa_second = 0.0
    continuation_probability = 0.0

    for d1 in range(1, min(c2, m) + 1):
        prob_d1 = float(binom.pmf(d1, m, p))
        continuation_probability += prob_d1
        pa_second += prob_d1 * float(binom.cdf(c2 - d1, m, p))

    return pa_first, pa_second, continuation_probability


def pa_double(p: float, m: int, c2: int) -> float:
    pa_first, pa_second, _ = double_components(p, m, c2)
    return pa_first + pa_second


def asn_double(p: float, m: int, c2: int) -> float:
    _, _, continuation_probability = double_components(p, m, c2)
    return m + m * continuation_probability


def aoq_double(p: float, lot_size: int, m: int, c2: int) -> float:
    pa_first, pa_second, _ = double_components(p, m, c2)
    outgoing_after_first = (lot_size - m) * pa_first
    outgoing_after_second = (lot_size - 2 * m) * pa_second
    return p * (outgoing_after_first + outgoing_after_second) / lot_size


def ati_double(p: float, lot_size: int, m: int, c2: int) -> float:
    pa_first, pa_second, _ = double_components(p, m, c2)
    return m * pa_first + 2 * m * pa_second + lot_size * (1.0 - pa_first - pa_second)


def aoql_double(lot_size: int, m: int, c2: int) -> tuple[float, float]:
    return maximize_on_unit_interval(lambda p: aoq_double(p, lot_size, m, c2))


def find_double_plan(
    lot_size: int,
    single_n: int,
    p1: float,
    alpha: float,
    beta: float,
    aoql_limit: float,
) -> dict[str, float]:
    p2 = 7.0 * p1
    min_m = single_n // 2 + 1

    for m in range(min_m, lot_size // 2 + 1):
        candidates = []
        for c2 in range(1, 2 * m + 1):
            pa_p1 = pa_double(p1, m, c2)
            pa_p2 = pa_double(p2, m, c2)
            if pa_p1 < 1.0 - alpha or pa_p2 > beta:
                continue

            p_at_aoql, max_aoq = aoql_double(lot_size, m, c2)
            if max_aoq <= aoql_limit:
                candidates.append(
                    {
                        "n1": m,
                        "n2": m,
                        "c1": 0,
                        "c2": c2,
                        "Pa_p1": pa_p1,
                        "producer_risk": 1.0 - pa_p1,
                        "Pa_p2": pa_p2,
                        "customer_risk": pa_p2,
                        "AOQL": max_aoq,
                        "p_at_AOQL": p_at_aoql,
                        "ASN_p1": asn_double(p1, m, c2),
                        "ASN_p2": asn_double(p2, m, c2),
                        "ATI_p1": ati_double(p1, lot_size, m, c2),
                        "ATI_p2": ati_double(p2, lot_size, m, c2),
                    }
                )

        if candidates:
            return sorted(candidates, key=lambda item: (item["n1"], item["c2"]))[0]

    raise RuntimeError("No feasible double sampling plan was found.")


def average_total_cost_single(
    lot_size: int,
    inspection_cost: float,
    escaped_defective_cost: float,
    n: int,
    c: int,
    p_low: float,
    p_high: float,
) -> float:
    def total_cost_at_p(p: float) -> float:
        inspection = inspection_cost * ati_single(p, lot_size, n, c)
        escaped = escaped_defective_cost * escaped_defectives_single(p, lot_size, n, c)
        return inspection + escaped

    integral, _ = quad(total_cost_at_p, p_low, p_high, epsabs=1e-9, epsrel=1e-9)
    return integral / (p_high - p_low)


def find_cost_optimal_single_plan(
    lot_size: int,
    inspection_cost: float,
    escaped_defective_cost: float,
    p_low: float,
    p_high: float,
    max_c: int = 1,
) -> tuple[dict[str, float], pd.DataFrame]:
    rows = []
    for n in range(1, lot_size + 1):
        for c in range(min(max_c, n) + 1):
            avg_cost = average_total_cost_single(
                lot_size,
                inspection_cost,
                escaped_defective_cost,
                n,
                c,
                p_low,
                p_high,
            )
            rows.append({"n": n, "c": c, "average_total_cost": avg_cost})

    costs = pd.DataFrame(rows).sort_values(["average_total_cost", "n", "c"]).reset_index(drop=True)
    return costs.iloc[0].to_dict(), costs


def c4_factor(sample_size: int) -> float:
    return math.sqrt(2.0 / (sample_size - 1)) * gamma(sample_size / 2) / gamma((sample_size - 1) / 2)


def xbar_power(
    mu0: float,
    sigma0: float,
    sample_size: int,
    k: float,
    delta: float,
    sigma_multiplier: float,
) -> float:
    lcl = mu0 - k * sigma0 / math.sqrt(sample_size)
    ucl = mu0 + k * sigma0 / math.sqrt(sample_size)
    shifted_mu = mu0 + delta
    shifted_sigma = sigma_multiplier * sigma0
    shifted_se = shifted_sigma / math.sqrt(sample_size)
    return float(norm.cdf(lcl, shifted_mu, shifted_se) + 1.0 - norm.cdf(ucl, shifted_mu, shifted_se))


def s_chart_power(
    lcl_s: float,
    ucl_s: float,
    sigma0: float,
    sample_size: int,
    sigma_multiplier: float,
) -> float:
    nu = sample_size - 1
    shifted_sigma = sigma_multiplier * sigma0
    lower_probability = chi2.cdf(nu * (lcl_s / shifted_sigma) ** 2, nu)
    upper_probability = 1.0 - chi2.cdf(nu * (ucl_s / shifted_sigma) ** 2, nu)
    return float(lower_probability + upper_probability)


def solve_part_b(params: dict[tuple[str, str], float]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mu0 = 9.0
    sample_size = 5
    production_rate = 100.0
    sampling_interval_hours = 1.0

    spec_half_width = params[("B", "A")]
    sigma = params[("B", "sigma")]
    poor_quality_cost = params[("B", "K")]
    delta = params[("B", "Delta")]
    k = params[("B", "k")]

    c4 = c4_factor(sample_size)
    sigma_s = sigma * math.sqrt(1.0 - c4**2)

    xbar_lcl = mu0 - k * sigma / math.sqrt(sample_size)
    xbar_ucl = mu0 + k * sigma / math.sqrt(sample_size)
    s_cl = c4 * sigma
    s_lcl = max(0.0, s_cl - k * sigma_s)
    s_ucl = s_cl + k * sigma_s

    nu = sample_size - 1
    alpha_xbar = 2.0 * (1.0 - norm.cdf(k))
    alpha_s = s_chart_power(s_lcl, s_ucl, sigma, sample_size, 1.0)
    alpha_either = 1.0 - (1.0 - alpha_xbar) * (1.0 - alpha_s)

    xbar_power_mean_shift_only = xbar_power(mu0, sigma, sample_size, k, delta, 1.0)
    xbar_power_mean_shift_and_sigma_increase = xbar_power(mu0, sigma, sample_size, k, delta, 1.2)
    s_power_sigma_increase_only = s_chart_power(s_lcl, s_ucl, sigma, sample_size, 1.2)

    shifted_defective_probability = float(
        norm.cdf(mu0 - spec_half_width, mu0 + delta, sigma)
        + 1.0
        - norm.cdf(mu0 + spec_half_width, mu0 + delta, sigma)
    )
    detection_probability = xbar_power_mean_shift_only
    arl_after_shift = 1.0 / detection_probability
    average_hours_to_signal = sampling_interval_hours * arl_after_shift
    average_hours_to_signal_random_shift = sampling_interval_hours * (arl_after_shift - 0.5)
    poor_quality_cost_until_detection = (
        shifted_defective_probability
        * production_rate
        * poor_quality_cost
        * average_hours_to_signal
    )
    poor_quality_cost_until_detection_random_shift = (
        shifted_defective_probability
        * production_rate
        * poor_quality_cost
        * average_hours_to_signal_random_shift
    )

    limits = pd.DataFrame(
        [
            {
                "chart": "Xbar",
                "LCL": xbar_lcl,
                "CL": mu0,
                "UCL": xbar_ucl,
            },
            {
                "chart": "S",
                "LCL": s_lcl,
                "CL": s_cl,
                "UCL": s_ucl,
            },
        ]
    )

    probabilities = pd.DataFrame(
        [
            {"quantity": "alpha_xbar_chart", "value": alpha_xbar},
            {"quantity": "alpha_s_chart", "value": alpha_s},
            {"quantity": "alpha_either_chart", "value": alpha_either},
            {
                "quantity": "xbar_power_mean_shift_sigma_unchanged",
                "value": xbar_power_mean_shift_only,
            },
            {
                "quantity": "xbar_power_mean_shift_sigma_increased_20_percent",
                "value": xbar_power_mean_shift_and_sigma_increase,
            },
            {
                "quantity": "s_power_sigma_increased_20_percent_mean_unchanged",
                "value": s_power_sigma_increase_only,
            },
            {
                "quantity": "s_power_sigma_increased_20_percent_with_mean_shift",
                "value": s_power_sigma_increase_only,
            },
            {"quantity": "defective_probability_after_mean_shift", "value": shifted_defective_probability},
            {"quantity": "arl_xbar_after_mean_shift", "value": arl_after_shift},
            {"quantity": "average_hours_to_signal_shift_just_after_sample", "value": average_hours_to_signal},
            {
                "quantity": "average_hours_to_signal_random_shift_in_interval",
                "value": average_hours_to_signal_random_shift,
            },
            {
                "quantity": "poor_quality_cost_until_detection_shift_just_after_sample",
                "value": poor_quality_cost_until_detection,
            },
            {
                "quantity": "poor_quality_cost_until_detection_random_shift_in_interval",
                "value": poor_quality_cost_until_detection_random_shift,
            },
        ]
    )

    samples = pd.read_csv(DATA_FILE)
    measurement_columns = [f"x{i}" for i in range(1, sample_size + 1)]
    samples["xbar"] = samples[measurement_columns].mean(axis=1)
    samples["s"] = samples[measurement_columns].std(axis=1, ddof=1)
    samples["xbar_out_of_control"] = (samples["xbar"] < xbar_lcl) | (samples["xbar"] > xbar_ucl)
    samples["s_out_of_control"] = (samples["s"] < s_lcl) | (samples["s"] > s_ucl)
    samples["any_out_of_control"] = samples["xbar_out_of_control"] | samples["s_out_of_control"]

    return limits, probabilities, samples


def save_acceptance_plots(
    lot_size: int,
    p1: float,
    alpha: float,
    beta: float,
    single_plan: dict[str, float],
    double_plan: dict[str, float],
) -> None:
    p2 = 7.0 * p1
    p_grid = np.linspace(0.0, 0.12, 600)

    single_pa = [pa_single(p, int(single_plan["n"]), int(single_plan["c"])) for p in p_grid]
    double_pa = [pa_double(p, int(double_plan["n1"]), int(double_plan["c2"])) for p in p_grid]
    single_ati = [ati_single(p, lot_size, int(single_plan["n"]), int(single_plan["c"])) for p in p_grid]
    double_ati = [ati_double(p, lot_size, int(double_plan["n1"]), int(double_plan["c2"])) for p in p_grid]
    single_aoq = [aoq_single(p, lot_size, int(single_plan["n"]), int(single_plan["c"])) for p in p_grid]
    double_aoq = [aoq_double(p, lot_size, int(double_plan["n1"]), int(double_plan["c2"])) for p in p_grid]

    plt.figure(figsize=(9, 5.5))
    plt.plot(p_grid, single_pa, label=f"Simple: n={int(single_plan['n'])}, c={int(single_plan['c'])}", linewidth=2)
    plt.plot(
        p_grid,
        double_pa,
        label=f"Double: n1=n2={int(double_plan['n1'])}, c1=0, c2={int(double_plan['c2'])}",
        linewidth=2,
    )
    plt.axvline(p1, color="black", linestyle="--", linewidth=1, label="p1")
    plt.axvline(p2, color="gray", linestyle="--", linewidth=1, label="p2=7p1")
    plt.axhline(1.0 - alpha, color="tab:green", linestyle=":", linewidth=1, label="1-alpha")
    plt.axhline(beta, color="tab:red", linestyle=":", linewidth=1, label="beta")
    plt.xlabel("Fraction defective p")
    plt.ylabel("Probability of acceptance")
    plt.title("Operating characteristic curves")
    plt.ylim(-0.02, 1.02)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(BASE_DIR / "acceptance_oc_curves.png", dpi=200)
    plt.close()

    plt.figure(figsize=(9, 5.5))
    plt.plot(p_grid, single_ati, label="Simple sampling", linewidth=2)
    plt.plot(p_grid, double_ati, label="Double sampling", linewidth=2)
    plt.axvline(p1, color="black", linestyle="--", linewidth=1, label="p1")
    plt.axvline(p2, color="gray", linestyle="--", linewidth=1, label="p2=7p1")
    plt.xlabel("Fraction defective p")
    plt.ylabel("ATI(p)")
    plt.title("Average total inspection curves")
    plt.ylim(0, lot_size * 1.05)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(BASE_DIR / "acceptance_ati_curves.png", dpi=200)
    plt.close()

    plt.figure(figsize=(9, 5.5))
    plt.plot(p_grid, single_aoq, label="Simple sampling", linewidth=2)
    plt.plot(p_grid, double_aoq, label="Double sampling", linewidth=2)
    plt.axhline(3.0 * p1, color="tab:red", linestyle="--", linewidth=1, label="3p1 limit")
    plt.xlabel("Fraction defective p")
    plt.ylabel("AOQ(p)")
    plt.title("Average outgoing quality curves")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(BASE_DIR / "acceptance_aoq_curves.png", dpi=200)
    plt.close()


def save_control_chart_plot(limits: pd.DataFrame, samples: pd.DataFrame) -> None:
    x_limits = limits[limits["chart"] == "Xbar"].iloc[0]
    s_limits = limits[limits["chart"] == "S"].iloc[0]

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(samples["Deigma"], samples["xbar"], marker="o", linewidth=1.8)
    axes[0].axhline(x_limits["CL"], color="black", linewidth=1, label="CL")
    axes[0].axhline(x_limits["UCL"], color="tab:red", linestyle="--", linewidth=1, label="UCL/LCL")
    axes[0].axhline(x_limits["LCL"], color="tab:red", linestyle="--", linewidth=1)
    out_x = samples[samples["xbar_out_of_control"]]
    axes[0].scatter(out_x["Deigma"], out_x["xbar"], color="tab:red", zorder=3)
    axes[0].set_ylabel("Sample mean")
    axes[0].set_title("Xbar chart")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(samples["Deigma"], samples["s"], marker="o", linewidth=1.8)
    axes[1].axhline(s_limits["CL"], color="black", linewidth=1, label="CL")
    axes[1].axhline(s_limits["UCL"], color="tab:red", linestyle="--", linewidth=1, label="UCL/LCL")
    axes[1].axhline(s_limits["LCL"], color="tab:red", linestyle="--", linewidth=1)
    out_s = samples[samples["s_out_of_control"]]
    axes[1].scatter(out_s["Deigma"], out_s["s"], color="tab:red", zorder=3)
    axes[1].set_xlabel("Sample")
    axes[1].set_ylabel("Sample standard deviation")
    axes[1].set_title("S chart")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(BASE_DIR / "control_charts_samples.png", dpi=200)
    plt.close(fig)


def main() -> None:
    params = load_parameters()

    lot_size = int(params[("A", "N")])
    inspection_cost = params[("A", "Ci")]
    escaped_defective_cost = params[("A", "Cd")]
    p1 = params[("A", "ASP")]
    alpha = params[("A", "alpha")]
    beta = params[("A", "beta")]
    p_low = params[("A", "p")]
    p_high = params[("A", "q")]
    p2 = 7.0 * p1
    aoql_limit = 3.0 * p1

    single_plan = find_single_plan(lot_size, p1, alpha, beta, aoql_limit)
    double_plan = find_double_plan(lot_size, int(single_plan["n"]), p1, alpha, beta, aoql_limit)

    optimal_cost_plan, all_costs = find_cost_optimal_single_plan(
        lot_size,
        inspection_cost,
        escaped_defective_cost,
        p_low,
        p_high,
        max_c=1,
    )
    q1_average_cost = average_total_cost_single(
        lot_size,
        inspection_cost,
        escaped_defective_cost,
        int(single_plan["n"]),
        int(single_plan["c"]),
        p_low,
        p_high,
    )

    acceptance_summary = pd.DataFrame(
        [
            {
                "question": "A1",
                "plan": "simple_min_n",
                "N": lot_size,
                "n": int(single_plan["n"]),
                "c": int(single_plan["c"]),
                "p1": p1,
                "p2": p2,
                **{key: value for key, value in single_plan.items() if key not in {"n", "c"}},
            },
            {
                "question": "A2",
                "plan": "double_min_n1_equal_samples",
                "N": lot_size,
                "n1": int(double_plan["n1"]),
                "n2": int(double_plan["n2"]),
                "c1": int(double_plan["c1"]),
                "c2": int(double_plan["c2"]),
                "p1": p1,
                "p2": p2,
                **{key: value for key, value in double_plan.items() if key not in {"n1", "n2", "c1", "c2"}},
            },
        ]
    )
    acceptance_summary.to_csv(BASE_DIR / "acceptance_sampling_summary.csv", index=False)

    cost_comparison = pd.DataFrame(
        [
            {
                "scheme": "cost_optimal_simple_c_le_1",
                "n": int(optimal_cost_plan["n"]),
                "c": int(optimal_cost_plan["c"]),
                "average_total_cost": optimal_cost_plan["average_total_cost"],
                "difference_vs_optimal": 0.0,
                "percent_higher_than_optimal": 0.0,
            },
            {
                "scheme": "A1_simple_plan",
                "n": int(single_plan["n"]),
                "c": int(single_plan["c"]),
                "average_total_cost": q1_average_cost,
                "difference_vs_optimal": q1_average_cost - optimal_cost_plan["average_total_cost"],
                "percent_higher_than_optimal": 100.0
                * (q1_average_cost / optimal_cost_plan["average_total_cost"] - 1.0),
            },
        ]
    )
    cost_comparison.to_csv(BASE_DIR / "a4_cost_comparison.csv", index=False)
    all_costs.head(25).to_csv(BASE_DIR / "a4_top_cost_plans.csv", index=False)

    save_acceptance_plots(lot_size, p1, alpha, beta, single_plan, double_plan)

    limits, probabilities, sample_results = solve_part_b(params)
    limits.to_csv(BASE_DIR / "control_chart_limits.csv", index=False)
    probabilities.to_csv(BASE_DIR / "control_chart_probabilities.csv", index=False)
    sample_results.to_csv(BASE_DIR / "sample_control_results.csv", index=False)
    save_control_chart_plot(limits, sample_results)

    print("\nPart A")
    print("Single sampling plan:", {k: single_plan[k] for k in ["n", "c", "Pa_p1", "Pa_p2", "AOQL", "ATI_p1"]})
    print(
        "Double sampling plan:",
        {k: double_plan[k] for k in ["n1", "n2", "c1", "c2", "Pa_p1", "Pa_p2", "AOQL", "ASN_p1", "ATI_p1"]},
    )
    print("Cost-optimal simple plan with c<=1:", optimal_cost_plan)
    print("A1 plan average total cost:", q1_average_cost)

    print("\nPart B")
    print(limits.to_string(index=False))
    print(probabilities.to_string(index=False))
    print(sample_results[["Deigma", "xbar", "s", "xbar_out_of_control", "s_out_of_control"]].to_string(index=False))


if __name__ == "__main__":
    main()
