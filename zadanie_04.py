# -*- coding: utf-8 -*-
"""Lekcja 16, zadanie 4: Korelacja Pearsona.

Skrypt: nie wymaga importowania innych zadań.
Używa wyłącznie NumPy, Pandas (gdy wymagane) i biblioteki standardowej.
Wykresy są tekstowe (ASCII/Unicode), zgodnie z lekcją.
Uruchomienie: python zadanie_XX.py
"""

import math
import numpy as np


def make_synthetic_flowers(seed=42):
    """Kanoniczny generator kwiatów z lekcji: 150 wierszy, 4 cechy, A/B/C."""
    rng = np.random.default_rng(seed)
    n_per = 50
    means = np.array([
        [5.0, 3.4, 1.5, 0.25],
        [5.9, 2.8, 4.3, 1.30],
        [6.6, 3.0, 5.6, 2.00],
    ])
    std = np.array([0.35, 0.35, 0.45, 0.30])
    parts = [rng.normal(loc=means[i], scale=std, size=(n_per, 4))
             for i in range(3)]
    return np.vstack(parts), np.repeat(['A', 'B', 'C'], n_per)


def pearson_corr(x, y):
    """Współczynnik Pearsona ze wzoru na kowariancję i wariancje."""
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    dx, dy = a - a.mean(), b - b.mean()
    denom = np.sqrt(np.dot(dx, dx) * np.dot(dy, dy))
    if denom == 0:
        raise ValueError('Nie da się wyznaczyć korelacji dla stałej zmiennej.')
    return float(np.dot(dx, dy) / denom)

def corr_pvalue_approx(r, n):
    """Dwustronne przybliżenie normalne ze statystyki t (dla dużych n)."""
    t = r * math.sqrt((n - 2) / max(1.0 - r*r, 1e-15))
    # erfc unika utraty precyzji dla bardzo małych p-value.
    return math.erfc(abs(t) / math.sqrt(2.0))


def ascii_scatter(x, y, title='Wykres punktowy', rows=20, cols=60, regression=True):
    """ASCII scatter z opcjonalną prostą regresji liniowej."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    dx, dy = max(xmax - xmin, 1e-12), max(ymax - ymin, 1e-12)
    grid = [[' '] * cols for _ in range(rows)]
    slope, intercept = np.polyfit(x, y, 1)
    if regression:
        for xx in np.linspace(xmin, xmax, cols * 3):
            yy = slope * xx + intercept
            if ymin <= yy <= ymax:
                c = int(np.clip((xx - xmin) / dx * (cols - 1), 0, cols - 1))
                r = rows - 1 - int(np.clip((yy - ymin) / dy * (rows - 1), 0, rows - 1))
                grid[r][c] = '·'
    for xx, yy in zip(x, y):
        c = int(np.clip((xx - xmin) / dx * (cols - 1), 0, cols - 1))
        r = rows - 1 - int(np.clip((yy - ymin) / dy * (rows - 1), 0, rows - 1))
        grid[r][c] = '●' if grid[r][c] == ' ' or grid[r][c] == '·' else '#'
    print(f'\n{title} (● dane; # wiele punktów; · regresja)')
    print(f'  y = {slope:+.3f} x {intercept:+.3f}')
    for line in grid:
        print('  |' + ''.join(line) + '|')
    print('  +' + '-' * cols + '+')
    print(f'  x: [{xmin:.2f}; {xmax:.2f}]  y: [{ymin:.2f}; {ymax:.2f}]')

def ascii_scatter_mini(x, y, title='Scatter'):
    ascii_scatter(x, y, title=title, rows=10, cols=36, regression=False)


def main():
    X, species = make_synthetic_flowers()
    x, y = X[:, 2], X[:, 3]  # petal_length i petal_width
    r = pearson_corr(x, y)
    p = corr_pvalue_approx(r, len(x))
    slope = r * y.std(ddof=0) / x.std(ddof=0)
    intercept = y.mean() - slope * x.mean()
    direction = 'dodatnia' if r >= 0 else 'ujemna'
    strength = 'silna' if abs(r) > 0.7 else 'umiarkowana' if abs(r) > 0.4 else 'słaba'
    print('ZADANIE 4 - Korelacja petal_length vs petal_width')
    print(f'Pearson r = {r:+.5f}, p-value (aproks.) = {p:.4g}')
    print(f'Prosta regresji: petal_width = {slope:.4f} * petal_length {intercept:+.4f}')
    print(f'Interpretacja: korelacja {strength}, {direction}.')
    ascii_scatter(x, y, title='petal_length vs petal_width', rows=20, cols=60)


if __name__ == "__main__":
    main()
