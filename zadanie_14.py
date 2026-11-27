# -*- coding: utf-8 -*-
"""Lekcja 16, zadanie 14: Bootstrapping - przedziały ufności.

Skrypt: nie wymaga importowania innych zadań.
Używa wyłącznie NumPy, Pandas (gdy wymagane) i biblioteki standardowej.
Wykresy są tekstowe (ASCII/Unicode), zgodnie z lekcją.
Uruchomienie: python zadanie_XX.py
"""

import math
import os
import numpy as np
import pandas as pd


TITANIC_URL = (
    'https://raw.githubusercontent.com/datasciencedojo/'
    'datasets/master/titanic.csv'
)

def load_titanic():
    """Pobiera Titanic; opcjonalnie TITANIC_CSV wskazuje lokalny plik CSV."""
    source = os.environ.get('TITANIC_CSV', TITANIC_URL)
    try:
        df = pd.read_csv(source)
    except Exception as exc:
        raise SystemExit(
            'Nie można otworzyć danych Titanic. Połącz się z Internetem '
            'lub ustaw zmienną TITANIC_CSV na ścieżkę do pobranego titanic.csv. '
            f'Szczegóły: {exc}'
        ) from exc
    required = {'Age', 'Fare', 'Survived', 'Pclass'}
    missing = required.difference(df.columns)
    if missing:
        raise SystemExit(f'Brak kolumn w Titanic CSV: {sorted(missing)}')
    return df


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
    df = load_titanic()
    ages = df['Age'].dropna().to_numpy(dtype=float)
    rng = np.random.default_rng(seed=42)
    n = len(ages)
    indices = rng.integers(0, n, size=(10000, n))
    boot_means = ages[indices].mean(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    mean, std = float(ages.mean()), float(ages.std(ddof=1))
    normal_lo = mean - 1.96 * std / math.sqrt(n)
    normal_hi = mean + 1.96 * std / math.sqrt(n)
    print('ZADANIE 14 - 95% przedział ufności średniego wieku (Titanic)')
    print(f'n={n}; średnia wieku = {mean:.4f} lat')
    print(f'Bootstrap 10000 razy: [{lo:.4f}, {hi:.4f}]')
    print(f'Normalny przedział ufności: [{normal_lo:.4f}, {normal_hi:.4f}]')
    print(f'Różnica dolnych krańców: {lo-normal_lo:+.4f} roku')
    print(f'Różnica górnych krańców: {hi-normal_hi:+.4f} roku')
    ascii_histogram(boot_means, bins=30, height=10, label='Rozkład średnich bootstrap')
    width = 46
    line = ['═'] * (width + 1)
    center = int(round((mean - lo) / max(hi - lo, 1e-12) * width))
    line[max(0, min(width, center))] = '●'
    print('\n95% CI (bootstrap):')
    print(f'  {lo:.2f} ├{"".join(line)}┤ {hi:.2f}')
    print(f'  ● = średnia próby ({mean:.2f})')


if __name__ == "__main__":
    main()
