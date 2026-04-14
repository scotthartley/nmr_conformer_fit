# nmr_conformer_fit

A command-line tool for fitting conformer populations to NMR chemical shifts. Given computed isotropic shieldings for each conformer of a molecule and a set of experimental <sup>1</sup>H chemical shifts, the tool finds the conformer weight vector (population distribution) that minimizes the sum of squared residuals. Uncertainty in the fitted weights can be estimated by non-parametric bootstrapping.

## Method

Predicted chemical shifts are computed as a population-weighted average of computed isotropic shieldings across conformers, then linearly scaled against the experimental shifts (slope/intercept correction via linear regression). The optimization minimizes the sum of squared residuals between predicted and experimental shifts using `scipy.optimize.least_squares` with a softmax reparameterization to enforce the constraints that all weights are non-negative and sum to 1.

## Installation

### Standalone binary (no Python required)

Pre-built binaries are available on the [Releases](../../releases) page. Download the binary for your platform, make it executable, and run it directly:

```bash
chmod +x nmr_conformer_fit
./nmr_conformer_fit input.csv
```

### From source

Requires Python 3.11+ and [uv](https://github.com/astral-sh/uv) (or pip).

```bash
git clone https://github.com/scotthartley/nmr_conformer_fit.git
cd nmr_conformer_fit
uv sync
uv run main.py input.csv
```

## Input format

The input is a CSV file with the following structure:

```
label, conformer_1, conformer_2, ..., conformer_n, , exp_shift
H1,    25.1,        26.3,                           , 7.21
H2,    31.4,        29.8,                           , 6.85
H3,    18.7,        20.1,                           , 5.40
```

- **Column 0**: proton label
- **Columns 1 to n**: computed isotropic shielding values, one column per conformer (column headers are the conformer names)
- **Blank column**: an empty column acts as a separator
- **Last column**: experimental <sup>1</sup>H chemical shift (ppm)

See `test.csv` for a minimal example.

## Usage

```
nmr_conformer_fit [-h] [-c CONF] [-b BOOT] [-v] [-w FILENAME] csv_file
```

### Arguments

| Argument | Description | Default |
|---|---|---|
| `csv_file` | Path to input CSV file | *(required)* |
| `-c`, `--conf` | Confidence interval for bootstrapping | `0.95` |
| `-b`, `--boot` | Number of bootstrap iterations | *(off)* |
| `-v`, `--fixedweight` | Run fixed-weight variation analysis | *(off)* |
| `-w`, `--write` | Write output to a file instead of stdout | *(off)* |

### Examples

Basic fit:
```bash
nmr_conformer_fit input.csv
```

Fit with 95% confidence intervals from 2000 bootstrap iterations:
```bash
nmr_conformer_fit input.csv --boot 2000
```

Fit with 90% confidence intervals:
```bash
nmr_conformer_fit input.csv --boot 1000 --conf 0.90
```

Fit with fixed-weight variation analysis and write results to a file:
```bash
nmr_conformer_fit input.csv --fixedweight --boot 1000 --write results.txt
```

## Output

The tool prints a table of experimental vs. predicted chemical shifts and residuals, the sum of squared residuals, and the optimized conformer weights. With `--boot`, confidence interval bounds are appended to each weight. With `--fixedweight`, a grid of SSR values is printed for each conformer weight varied from 0 to 1 in steps of 0.05.

```
Proton    Exp shift   Pred shift     Residual
-------------------------------------------------
H1           7.2100       7.2100      -0.0000
H2           6.8500       6.8500      -0.0000
H3           5.4000       5.4000       0.0000

Sum squared residuals: 0.000000

Optimized conformer weights:
  conf_A  0.523847   bottom 2.5%: 0.412000   top 97.5%: 0.635000
  conf_B  0.476153   bottom 2.5%: 0.365000   top 97.5%: 0.588000
```

## Building from source

The standalone binary is built with [PyInstaller](https://pyinstaller.org):

```bash
uv sync --group dev
pyinstaller nmr_conformer_fit.spec
```

The binary is written to `dist/nmr_conformer_fit`.

## Dependencies

- [NumPy](https://numpy.org/) >= 2.4
- [SciPy](https://scipy.org/) >= 1.17

## Development

Scientific code (e.g., fitting, statistics) was written by SH. I/O and packaging code was written by Claude Code. See the git commit history for details.

## License

MIT — see [LICENSE](LICENSE).
