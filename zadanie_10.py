# -*- coding: utf-8 -*-
"""Lekcja 16, zadanie 10: Wykrywanie outliers metodą z-score.

Skrypt: nie wymaga importowania innych zadań.
Używa wyłącznie NumPy, Pandas (gdy wymagane) i biblioteki standardowej.
Wykresy są tekstowe (ASCII/Unicode), zgodnie z lekcją.
Uruchomienie: python zadanie_XX.py
"""

import math
import numpy as np
import pandas as pd


def make_synthetic_wine(n_samples=1600, seed=42):
    """Generator 11 cech + quality z lekcji 16, sekcja 5."""
    rng = np.random.default_rng(seed)
    alcohol = rng.normal(10.4, 1.1, size=n_samples).clip(8.4, 14.9)
    volatile_acidity = rng.gamma(2.0, 0.25, size=n_samples).clip(0.1, 1.6)
    sulphates = rng.gamma(5.0, 0.13, size=n_samples).clip(0.33, 2.0)
    citric_acid = rng.uniform(0.0, 0.75, size=n_samples)
    residual_sugar = rng.gamma(2.0, 1.2, size=n_samples).clip(0.9, 15.0)
    chlorides = rng.gamma(3.0, 0.03, size=n_samples).clip(0.012, 0.61)
    free_sulfur_dioxide = rng.gamma(3.0, 5.0, size=n_samples).clip(1, 72)
    total_sulfur_dioxide = free_sulfur_dioxide * rng.uniform(2.0, 4.5, size=n_samples)
    density = rng.normal(0.9967, 0.0019, size=n_samples)
    ph = rng.normal(3.31, 0.154, size=n_samples).clip(2.74, 4.01)
    fixed_acidity = rng.gamma(7.0, 1.2, size=n_samples).clip(4.6, 15.9)
    quality_raw = (5.0 + 0.32 * (alcohol - 10.4)
                   - 1.15 * (volatile_acidity - 0.5)
                   + 0.57 * (sulphates - 0.65)
                   + 0.05 * (citric_acid - 0.3)
                   + rng.normal(0, 0.5, size=n_samples))
    quality = np.round(np.clip(quality_raw, 3, 8)).astype(int)
    return pd.DataFrame({
        'fixed acidity': fixed_acidity, 'volatile acidity': volatile_acidity,
        'citric acid': citric_acid, 'residual sugar': residual_sugar,
        'chlorides': chlorides, 'free sulfur dioxide': free_sulfur_dioxide,
        'total sulfur dioxide': total_sulfur_dioxide,
        'density': density, 'pH': ph, 'sulphates': sulphates,
        'alcohol': alcohol, 'quality': quality,
    })


def ascii_boxplot(data_list, labels, width=52):
    """Wykresy pudełkowe: - wąsy (min/max), = IQR, | kwartyle, * mediana."""
    arrays = [np.asarray(d, dtype=float) for d in data_list]
    all_values = np.concatenate(arrays)
    low, high = float(all_values.min()), float(all_values.max())
    span = max(high - low, 1e-12)
    print('\nASCII-boxplot (- min/max; | Q1/Q3; = IQR; * mediana):')
    for data, label in zip(arrays, labels):
        q1, median, q3 = np.percentile(data, [25, 50, 75])
        positions = [int(round((v - low) / span * (width - 1)))
                     for v in (data.min(), q1, median, q3, data.max())]
        imin, iq1, imed, iq3, imax = positions
        line = [' '] * width
        for i in range(imin, imax + 1):
            line[i] = '-'
        for i in range(iq1, iq3 + 1):
            line[i] = '='
        line[iq1], line[iq3], line[imed] = '|', '|', '*'
        print(f'  {label:<16} |{"".join(line)}| śr.={data.mean():.3f} (n={len(data)})')
    print(f'  Wspólna skala: {low:.2f} - {high:.2f}')

def ascii_boxplot_two(a, b, label_a='A', label_b='B'):
    ascii_boxplot([a, b], [label_a, label_b])


def main():
    df = make_synthetic_wine(n_samples=1600, seed=42)
    alcohol = df['alcohol'].to_numpy(dtype=float)
    z_score = (alcohol - alcohol.mean()) / alcohol.std(ddof=1)
    outlier_mask = abs(z_score) > 3.0
    before = alcohol
    after = alcohol[~outlier_mask]
    print('ZADANIE 10 - Outliers w zawartości alkoholu')
    print(f'Wykryto {outlier_mask.sum()} outlierów na {len(alcohol)} próbek')
    for i in np.flatnonzero(outlier_mask):
        print(f'  indeks={i:4d}, alcohol={alcohol[i]:.3f}, z={z_score[i]:+.3f}')
    print('\nStatystyki przed i po usunięciu:')
    print(f'{"statystyka":<18} {"przed":>12} {"po":>12}')
    stats = {
        'mean': (before.mean(), after.mean()),
        'median': (np.median(before), np.median(after)),
        'std (ddof=1)': (before.std(ddof=1), after.std(ddof=1)),
    }
    for label, (a, b) in stats.items():
        print(f'{label:<18} {a:>12.4f} {b:>12.4f}')
    ascii_boxplot([before, after], ['Przed', 'Po'])
    print('Interpretacja: z-score mierzy odległość od średniej w jednostkach odchylenia.')
    print('Zerowa liczba outlierów jest również prawidłowym wynikiem symulacji.')


if __name__ == "__main__":
    main()
