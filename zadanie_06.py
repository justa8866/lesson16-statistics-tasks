# -*- coding: utf-8 -*-
"""Lekcja 16, zadanie 6: Test t jednej próbki.

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


def main():
    df = load_titanic()
    ages = df['Age'].dropna().to_numpy(dtype=float)
    mu0 = 30.0
    n = len(ages)
    mean = float(ages.mean())
    s = float(ages.std(ddof=1))
    t_stat = (mean - mu0) / (s / math.sqrt(n))
    p_value = 2.0 * (1.0 - normal_cdf(abs(t_stat), 0, 1))
    print('ZADANIE 6 - Czy średni wiek pasażera Titanica wynosi 30 lat?')
    print('H0: μ = 30; H1: μ != 30; poziom istotności 0,05')
    print(f'n={n}, średnia={mean:.3f}, odch. std.={s:.3f}')
    print(f't={t_stat:+.5f}, p-value≈{p_value:.6f} (aproksymacja normalna)')
    print('Wniosek:', 'odrzucamy H0.' if p_value < 0.05 else 'nie odrzucamy H0.')

    width = 54
    low, high = min(mean, mu0) - 3, max(mean, mu0) + 3
    position = lambda v: min(width - 1, max(0, round((v - low) / (high - low) * (width - 1))))
    line = ['─'] * width
    line[position(mean)] = 'x'
    line[position(mu0)] = '0' if position(mu0) != position(mean) else '*'
    print('\n  ' + '├' + ''.join(line) + '┤')
    print(f'  x = średnia próby ({mean:.2f}), 0 = H0 ({mu0:.2f}); * = wspólna pozycja')


if __name__ == "__main__":
    main()
