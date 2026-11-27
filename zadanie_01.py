# -*- coding: utf-8 -*-
"""Lekcja 16, zadanie 1: Generowanie rozkładu normalnego.

Skrypt: nie wymaga importowania innych zadań.
Używa wyłącznie NumPy, Pandas (gdy wymagane) i biblioteki standardowej.
Wykresy są tekstowe (ASCII/Unicode), zgodnie z lekcją.
Uruchomienie: python zadanie_XX.py
"""

import math
import numpy as np


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


def main():
    rng = np.random.default_rng(seed=42)
    x = rng.normal(loc=50, scale=10, size=1000)
    fraction = ((x >= 40) & (x <= 60)).mean()
    theoretical = normal_cdf(60, 50, 10) - normal_cdf(40, 50, 10)
    print('ZADANIE 1 - Rozkład normalny N(50, 10)')
    print(f'Próbek: {len(x)}, średnia: {x.mean():.3f}, odchylenie: {x.std(ddof=1):.3f}')
    print(f'W przedziale [40,60]: {fraction*100:.2f}% empirycznie')
    print(f'Teoretycznie w zakresie ±1σ: {theoretical*100:.2f}%')
    print(f'Różnica: {(fraction-theoretical)*100:+.2f} punktów procentowych')
    ascii_histogram(x, bins=20, height=10, label='N(50,10)')
    print('  Oznaczenie przedziału: --[μ-σ=40]====[μ=50]====[μ+σ=60]--')
    print('Wniosek: wynik z 1000 losowań jest przybliżony i nie musi wynosić dokładnie 68,27%.')


if __name__ == "__main__":
    main()
