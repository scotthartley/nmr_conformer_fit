import argparse
import csv
import math

import numpy as np
from scipy import stats
from scipy.optimize import least_squares


def read_csv(filepath: str) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    """Parse a CSV file containing NMR shieldings and experimental shifts.

    Expected format:
        Column 0:       proton label
        Columns 1..k-1: shielding values for each conformer (headers = conformer names)
        Column k:       empty (blank separator)
        Column k+1:     experimental chemical shift

    Returns:
        labels:          list of proton label strings, length l
        conformer_names: list of conformer name strings, length m
        shieldings:      np.ndarray of shape (l, m)
        exp_shifts:      np.ndarray of shape (l,)
    """
    with open(filepath, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)

    # Detect blank separator column: first empty cell after column 0
    blank_col = None
    for i, cell in enumerate(header):
        if i > 0 and cell.strip() == "":
            blank_col = i
            break
    if blank_col is None:
        raise ValueError(
            "Could not find blank separator column in CSV header. "
            "Expected an empty column between shielding columns and experimental shifts."
        )

    conformer_names = [h.strip() for h in header[1:blank_col]]
    m = len(conformer_names)
    if m == 0:
        raise ValueError("No conformer columns found before the blank separator column.")

    labels: list[str] = []
    shielding_rows: list[list[float]] = []
    exp_shift_list: list[float] = []

    with open(filepath, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row_num, row in enumerate(reader, start=2):
            if not any(cell.strip() for cell in row):
                continue  # skip blank rows
            labels.append(row[0].strip())
            shielding_rows.append([float(row[i]) for i in range(1, blank_col)])
            exp_shift_list.append(float(row[blank_col + 1]))

    shieldings = np.array(shielding_rows)   # shape (l, m)
    exp_shifts = np.array(exp_shift_list)   # shape (l,)
    return labels, conformer_names, shieldings, exp_shifts


def scaled_shifts(
    weights: np.ndarray,
    shieldings: np.ndarray,
    exp_shifts: np.ndarray,
) -> np.ndarray:
    """Compute predicted chemical shifts as a weighted average over conformers.

    Args:
        weights:    (m,) weight vector; weights should sum to 1 and each be >= 0
        shieldings: (l, m) array of isotropic shieldings; rows = protons, cols = conformers
        exp_shifts: (l,) array of experimental chemical shifts (unused here; reserved
                    for future slope/intercept linear correction)

    Returns:
        (l,) array of predicted chemical shifts
    """
    net_isotropic_shieldings = shieldings @ weights

    reg = stats.linregress(exp_shifts, net_isotropic_shieldings)

    return (net_isotropic_shieldings - reg.intercept) / reg.slope


def constrained_weights(
    weights: np.ndarray,
    fix_weight: tuple[int, float]
) -> np.ndarray:
    """Convert unconstrained weights to weights that must sum to 1. Based on
    the softmax function.

    Args:
        weights: (m,) weight vector
        fix_weight: tuple of (index, weight) to hold fixed

    Returns:
        (m,) array of normalized weights
    """
    if fix_weight:
        weights = np.delete(weights, fix_weight[0])
        weight_sum = 1.0 - fix_weight[1]
        e_x = np.exp(weights - np.max(weights))
        new_weights = weight_sum * e_x / e_x.sum()
        new_weights = np.insert(new_weights, fix_weight[0], fix_weight[1])
    else:
        weight_sum = 1.0
        e_x = np.exp(weights - np.max(weights))
        new_weights = weight_sum * e_x / e_x.sum()

    return new_weights


def residuals(
    weights: np.ndarray,
    shieldings: np.ndarray,
    exp_shifts: np.ndarray,
    fix_weight: tuple[int, float] = None
) -> np.ndarray:
    """Compute per-proton residuals (predicted - experimental).

    Args:
        weights:    (m,) weight vector
        shieldings: (l, m) array of isotropic shieldings
        exp_shifts: (l,) array of experimental chemical shifts
        fix_weight: tuple of (index, weight) to hold fixed

    Returns:
        (l,) array of residuals (predicted_shift - exp_shift) for each proton
    """
    pred_shifts = scaled_shifts(
        constrained_weights(weights, fix_weight),
        shieldings,
        exp_shifts
    )

    return pred_shifts - exp_shifts


def optimize(
    shieldings: np.ndarray,
    exp_shifts: np.ndarray,
    fix_weight: tuple[int, float] = None
) -> np.ndarray:
    """Find the conformer weight vector that minimizes the sum of squared residuals.

    Args:
        shieldings: (l, m) array of isotropic shieldings
        exp_shifts: (l,) array of experimental chemical shifts
        fix_weight: tuple of (index, weight) to hold fixed

    Returns:
        (m,) array of optimised conformer weights
    """
    num_conformers = shieldings.shape[1]
    initial_guess = np.full(num_conformers, 1.0)

    results = least_squares(
        residuals,
        initial_guess,
        args = (shieldings, exp_shifts, fix_weight)
    )

    return constrained_weights(results.x, fix_weight)


def bootstrap(
    shieldings: np.ndarray,
    exp_shifts: np.ndarray,
    residuals: np.ndarray,
    conf_int: float = 0.95,
    iterations: int = 1000
) -> list[tuple[float, float]]:
    """Determine the uncertainty in the weights using a non-parametric
    bootstrapping method. Runs iterations simulations on datasets that
    have been perturbed by the residuals, storing the weights. For each
    weight, the results are ranked and the bottom and top values of the
    confidence interval selected.

    Args:
        shieldings: (l, m) array of isotropic shieldings
        exp_shifts: (l,) array of experimental chemical shifts
        residuals: (l,) array of residuals from the optimization
        conf_int: confidence interval for the analysis
        iterations: number of bootstrapping iterations to run
    """
    num_weights = shieldings.shape[1]
    num_shifts = len(exp_shifts)
    interval = math.ceil((iterations - conf_int * iterations)/2)


    all_weights = []
    for n in range(iterations):
        random_residuals = np.random.choice(residuals, num_shifts)
        perturbed_shifts = exp_shifts + random_residuals
        weights = optimize(shieldings, perturbed_shifts)
        all_weights.append(weights)
    all_weights_arr = np.array(all_weights)

    conf_ints = []
    for i in range(num_weights):
        variation = all_weights_arr[:, i]
        variation.sort()
        conf_ints.append((variation[interval], variation[iterations - interval]))

    return conf_ints


def output(
    labels: list[str],
    conformer_names: list[str],
    shieldings: np.ndarray,
    exp_shifts: np.ndarray,
    ci: float = 0.95,
    boot: int = None,
    fixed_weight_analysis: bool = False
) -> None:
    """Run optimization, print a summary table, run bootstrapping
    analysis, print per-conformer weights, assess effect of variation of
    parameters.
    """
    weights = optimize(shieldings, exp_shifts)
    pred = scaled_shifts(weights, shieldings, exp_shifts)
    resid = exp_shifts - pred

    col_w = max(len(lbl) for lbl in labels)
    header = f"{'Proton':<{col_w}}  {'Exp shift':>10}  {'Pred shift':>10}  {'Residual':>10}"
    print(header)
    print("-" * len(header))
    for lbl, exp, p, r in zip(labels, exp_shifts, pred, resid):
        print(f"{lbl:<{col_w}}  {exp:>10.4f}  {p:>10.4f}  {r:>10.4f}")

    print()
    print("Optimized conformer weights:")
    name_w = max(len(n) for n in conformer_names)

    # Determine errors by bootstrapping
    if boot:
        conf_ints = bootstrap(shieldings, exp_shifts, resid, conf_int = ci, iterations=boot)
        bot_label = f"bottom {(1-ci)/2*100:.1f}%:"
        top_label = f"top {100-(1-ci)/2*100:.1f}%:"
        for name, w, rng in zip(conformer_names, weights, conf_ints):
            print(f"  {name:<{name_w}}  {w:.6f}   {bot_label} {rng[0]:.6f}   {top_label} {rng[1]:.6f}")
    else:
        for name, w in zip(conformer_names, weights):
            print(f"  {name:<{name_w}}  {w:.6f}")

    # Assess variation in SSR for each parameter
    if fixed_weight_analysis:
        for index in range(len(weights)):
            print()
            print(f"Variation in {conformer_names[index]}:")
            for val in np.linspace(0.0, 1.0, 21):
                new_weights = optimize(
                    shieldings,
                    exp_shifts,
                    fix_weight = (index, val)
                )
                new_pred = scaled_shifts(new_weights, shieldings, exp_shifts)
                new_residuals = new_pred - exp_shifts
                ss_residuals = np.sum(new_residuals**2)
                print(f"{val:.6f} {ss_residuals:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit conformer populations to NMR chemical shifts."
    )
    parser.add_argument("csv_file", help="Path to input CSV file")
    parser.add_argument("-c", "--conf", help="Confidence interval", type=float, default=0.95)
    parser.add_argument("-b", "--boot", help="Bootstrap iterations", type=int)
    parser.add_argument("-v", "--fixedweight", help="Variation of parameter analysis")
    args = parser.parse_args()

    labels, conformer_names, shieldings, exp_shifts = read_csv(args.csv_file)
    output(
        labels,
        conformer_names,
        shieldings,
        exp_shifts,
        args.conf,
        args.boot,
        args.fixedweight
    )


if __name__ == "__main__":
    main()
