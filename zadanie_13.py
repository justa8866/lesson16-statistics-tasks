# -*- coding: utf-8 -*-
"""Lekcja 16, zadanie 13: Symulacja Central Limit Theorem.

Skrypt: nie wymaga importowania innych zadań.
Używa wyłącznie NumPy, Pandas (gdy wymagane) i biblioteki standardowej.
Wykresy są tekstowe (ASCII/Unicode), zgodnie z lekcją.
Uruchomienie: python zadanie_XX.py
"""

import math
import numpy as np


def jarque_bera_manual(data):
    """JB = n/6 * [skew^2 + excess_kurtosis^2/4], p z chi2(2)."""
    x = np.asarray(data, dtype=float)
    mean = x.mean()
    std = x.std(ddof=0)
    if std <= 0:
        raise ValueError('Test JB wymaga zmiennej o niezerowym odchyleniu.')
    skew = np.mean(((x - mean) / std) ** 3)
    kurt = np.mean(((x - mean) / std) ** 4) - 3.0
    jb = len(x) / 6.0 * (skew**2 + kurt**2 / 4.0)
    return jb, math.exp(-jb / 2.0), skew, kurt


def ascii_histogram(data, bins=20, height=10, label='Histogram'):
    """Histogram rysowany wyłącznie w terminalu (bez matplotlib)."""
    data = np.asarray(data, dtype=float)
    counts, edges = np.histogram(data, bins=bins)
    peak = max(int(counts.max()), 1)
    heights = np.rint(counts / peak * height).astype(int)
    print(f'\n{label} (n={len(data)}, maksymalna liczność={peak}):')
    for level in range(height, 0, -1):
        print('  |' + ''.join('█' if v >= level else ' ' for v in heights) + '|')
    print('  +' + '-' * bins + '+')
    print(f'   min={edges[0]:.2f}, max={edges[-1]:.2f}')
    return counts, edges


def normal_cdf(x, mu=0.0, sigma=1.0):
    """Dystrybuanta N(mu, sigma): 0.5 * (1 + erf(z/sqrt(2)))."""
    if sigma <= 0:
        raise ValueError('sigma musi być dodatnia')
    z = (np.asarray(x, dtype=float) - mu) / (sigma * math.sqrt(2.0))
    values = 0.5 * (1.0 + np.vectorize(math.erf, otypes=[float])(z))
    return float(values) if values.ndim == 0 else values

def normal_pdf(x, mu=0.0, sigma=1.0):
    """Gęstość rozkładu normalnego."""
    z = (np.asarray(x, dtype=float) - mu) / sigma
    return np.exp(-0.5 * z**2) / (sigma * np.sqrt(2.0 * np.pi))

def normal_ppf(p, iterations=65):
    """Odwrotna dystrybuanta N(0,1) obliczana przez bisekcję."""
    p_array = np.asarray(p, dtype=float)
    if np.any((p_array <= 0) | (p_array >= 1)):
        raise ValueError('Wszystkie wartości p muszą należeć do (0,1)')
    lo = np.full_like(p_array, -9.0, dtype=float)
    hi = np.full_like(p_array, 9.0, dtype=float)
    for _ in range(iterations):
        mid = (lo + hi) / 2.0
        lo = np.where(normal_cdf(mid) < p_array, mid, lo)
        hi = np.where(normal_cdf(mid) >= p_array, mid, hi)
    result = (lo + hi) / 2.0
    return float(result) if result.ndim == 0 else result


def ascii_qq_plot(data, name='Q-Q plot', rows=12, cols=50):
    """Empiryczne kwantyle z-score vs kwantyle teoretyczne N(0,1)."""
    ordered = np.sort(np.asarray(data, dtype=float))
    n = len(ordered)
    empirical = (ordered - ordered.mean()) / ordered.std(ddof=0)
    p = (np.arange(1, n + 1) - 0.5) / n
    theoretical = normal_ppf(p)
    xmin, xmax = float(theoretical.min()), float(theoretical.max())
    ymin = float(min(empirical.min(), theoretical.min()))
    ymax = float(max(empirical.max(), theoretical.max()))
    grid = [[' '] * cols for _ in range(rows)]

    def put(x, y, marker, overwrite=True):
        col = int(np.clip(round((x - xmin) / max(xmax - xmin, 1e-12) * (cols - 1)), 0, cols - 1))
        row = int(np.clip(round((ymax - y) / max(ymax - ymin, 1e-12) * (rows - 1)), 0, rows - 1))
        if overwrite or grid[row][col] == ' ':
            grid[row][col] = marker

    for x in np.linspace(xmin, xmax, 180):
        put(x, x, '·', overwrite=False)
    for x, y in zip(theoretical, empirical):
        put(float(x), float(y), '*')
    print(f'\n{name}: * = dane, · = linia idealna y=x')
    for row in grid:
        print('  |' + ''.join(row) + '|')
    print('  +' + '-' * cols + '+')
    print(f'  Teoretyczne kwantyle od {xmin:+.2f} do {xmax:+.2f}')


def main():
    rng = np.random.default_rng(seed=42)
    samples = rng.uniform(0, 10, size=(10000, 30))
    means = samples.mean(axis=1)
    mu_theory = 5.0
    std_theory = math.sqrt(100 / 12) / math.sqrt(30)
    jb, p, skew, kurt = jarque_bera_manual(means)
    print('ZADANIE 13 - Centralne Twierdzenie Graniczne (CLT)')
    print(f'10 000 średnich z prób rozmiaru n=30 z U(0,10)')
    print(f'Średnia teoretyczna={mu_theory:.4f}, empiryczna={means.mean():.4f}')
    print(f'Std teoretyczne={std_theory:.4f}, empiryczne={means.std(ddof=1):.4f}')
    print(f'Jarque-Bera: JB={jb:.4f}, p={p:.5f}, skośność={skew:+.4f}, kurtoza={kurt:+.4f}')
    print('Interpretacja JB:', 'odrzucamy ścisłą normalność' if p < 0.05
          else 'nie ma podstaw do odrzucenia normalności')
    ascii_histogram(means, bins=30, height=10, label='Rozkład średnich z U(0,10)')
    ascii_qq_plot(means, name='CLT: Q-Q średnich')
    print('CLT: mimo jednostajnego rozkładu obserwacji, średnie z dostatecznie')
    print('licznych niezależnych prób dążą rozkładem do normalnego.')
    print('Przy dużym n test JB może wykryć nawet drobne odchylenia od idealnej normalności.')


if __name__ == "__main__":
    main()
