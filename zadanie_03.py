# -*- coding: utf-8 -*-
"""Lekcja 16, zadanie 3: Analiza rozkładu kwiatów.

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
    X, species = make_synthetic_flowers(seed=42)
    sepal_length = X[:, 0]
    print('ZADANIE 3 - Normalność cechy sepal_length')
    print('Hipoteza H0: dane mają rozkład normalny; alfa = 0.05.')
    for name, data in [('Cała populacja', sepal_length)] + [
        (f'Gatunek {letter}', sepal_length[species == letter])
        for letter in ['A', 'B', 'C']
    ]:
        jb, p, skew, kurt = jarque_bera_manual(data)
        verdict = 'odrzucamy H0' if p < 0.05 else 'nie odrzucamy H0'
        print(f'{name:<17}: n={len(data):3d}, JB={jb:8.4f}, p={p:.6f}; {verdict}')
        print(f'  Skośność={skew:+.3f}; kurtoza nadmiarowa={kurt:+.3f}')
    ascii_histogram(sepal_length, bins=20, height=10, label='sepal_length - wszystkie gatunki')
    ascii_qq_plot(sepal_length, name='Q-Q: cała populacja')
    ascii_qq_plot(sepal_length[species == 'A'], name='Q-Q: gatunek A')
    print('Wniosek: mieszanina kilku gatunków może nie być normalna,')
    print('choć każda grupa była generowana osobno z rozkładu normalnego.')
    print('p >= 0.05 nie dowodzi normalności; oznacza brak podstaw do jej odrzucenia.')


if __name__ == "__main__":
    main()
