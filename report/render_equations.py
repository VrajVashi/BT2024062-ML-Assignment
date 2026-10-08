import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

eq_dir = Path("results/equations")
eq_dir.mkdir(parents=True, exist_ok=True)

# Using mathtext compatible syntax with Computer Modern font styling
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'STIXGeneral'

eqs = {
    'eq1': r"$\hat{y} = \beta_0 + \sum_{1 \leq |\alpha| \leq d} \beta_\alpha x^\alpha$",
    'eq2': r"$\binom{p+d}{d} - 1$",
    'eq3': r"$\sum_{i=1}^n (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^m \beta_j^2$",
    'eq4': r"$\mathrm{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2$",
    'eq5': r"$R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}$",
    'ridge_set': r"$\{0.001,\, 0.01,\, 0.1,\, 1,\, 10,\, 100,\, 1000\}$",
}

for name, tex in eqs.items():
    fig = plt.figure(figsize=(4, 0.45), dpi=300)
    fig.text(0.5, 0.5, tex, fontsize=11, ha='center', va='center')
    fig.savefig(eq_dir / f"{name}.png", dpi=300, bbox_inches='tight', transparent=True, pad_inches=0.03)
    plt.close(fig)
    print(f"Rendered {name}")
