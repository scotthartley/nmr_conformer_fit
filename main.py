import argparse
import csv

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
        (1,) array of normalized weights
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


def output(
    labels: list[str],
    conformer_names: list[str],
    shieldings: np.ndarray,
    exp_shifts: np.ndarray,
) -> None:
    """Run optimization, print a summary table, print per-conformer weights,
    assess uncertainties.
    """
    weights = optimize(shieldings, exp_shifts)
    pred = scaled_shifts(weights, shieldings, exp_shifts)
    resid = pred - exp_shifts

    col_w = max(len(lbl) for lbl in labels)
    header = f"{'Proton':<{col_w}}  {'Exp shift':>10}  {'Pred shift':>10}  {'Residual':>10}"
    print(header)
    print("-" * len(header))
    for lbl, exp, p, r in zip(labels, exp_shifts, pred, resid):
        print(f"{lbl:<{col_w}}  {exp:>10.4f}  {p:>10.4f}  {r:>10.4f}")

    print()
    print("Optimised conformer weights:")
    name_w = max(len(n) for n in conformer_names)
    for name, w in zip(conformer_names, weights):
        print(f"  {name:<{name_w}}  {w:.6f}")

    # Assess errors
    for index in range(len(weights)):
        print(f"Variation in {labels[index]}:")
        for val in np.linspace(0.0, 1.0, 11):
            new_weights = optimize(
                shieldings,
                exp_shifts,
                fix_weight = (index, val)
            )
            new_pred = scaled_shifts(new_weights, shieldings, exp_shifts)
            new_residuals = new_pred - exp_shifts
            ss_residuals = np.sum(new_residuals**2)
            print(f"{val} {ss_residuals}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit conformer populations to NMR chemical shifts."
    )
    parser.add_argument("csv_file", help="Path to input CSV file")
    args = parser.parse_args()

    labels, conformer_names, shieldings, exp_shifts = read_csv(args.csv_file)
    output(labels, conformer_names, shieldings, exp_shifts)


if __name__ == "__main__":
    main()
