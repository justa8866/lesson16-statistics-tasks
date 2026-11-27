# -*- coding: utf-8 -*-
"""Lekcja 16, zadanie 11: Korelacja Pearson vs Spearman.

Skrypt: nie wymaga importowania innych zadań.
Używa wyłącznie NumPy, Pandas (gdy wymagane) i biblioteki standardowej.
Wykresy są tekstowe (ASCII/Unicode), zgodnie z lekcją.
Uruchomienie: python zadanie_XX.py
"""

import math
import numpy as np


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


def ranks(x):
    """Rangi 1..n przez dwukrotne argsort (brak remisów w tych danych)."""
    return np.argsort(np.argsort(x)) + 1

def spearman_corr(x, y):
    """Korelacja Pearsona między rangami."""
    return pearson_corr(ranks(x), ranks(y))


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
    rng = np.random.default_rng(seed=42)
    n = 100
    x = np.linspace(0, 10, n)
    linear = 2*x + rng.normal(0, 1, n)
    quadratic = x*x + rng.normal(0, 5, n)
    with_outliers = 2*x + rng.normal(0, 1, n)
    with_outliers[[0, 1, 2, 3, 4]] = [80, -70, 90, -90, 110]
    datasets = [
        ('Liniowa', x, linear),
        ('Kwadratowa', x, quadratic),
        ('5 outlierów', x, with_outliers),
    ]
    rows = []
    print('ZADANIE 11 - Porównanie korelacji Pearsona i Spearmana')
    for label, xx, yy in datasets:
        pearson = pearson_corr(xx, yy)
        spearman = spearman_corr(xx, yy)
        rows.append((label, pearson, spearman, spearman - pearson))
        ascii_scatter_mini(xx, yy, title=label)
    print('\nPorównanie współczynników:')
    print(f'{"Typ danych":<17} {"Pearson":>10} {"Spearman":>10} {"różnica ρ-r":>13}')
    for name, pearson, spearman, diff in rows:
        print(f'{name:<17} {pearson:+10.4f} {spearman:+10.4f} {diff:+13.4f}')
    print('Interpretacja: Spearman, oparty na rangach, opisuje zależności monotoniczne,')
    print('a Pearson liniowe. Wartości skrajne wpływają silniej na Pearsona.')


if __name__ == "__main__":
    main()
