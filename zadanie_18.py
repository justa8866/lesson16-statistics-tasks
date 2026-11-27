# -*- coding: utf-8 -*-
"""Lekcja 16, zadanie 18: Multiple testing correction (Bonferroni).

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


def main():
    df = make_synthetic_wine(n_samples=1600, seed=42)
    cols = [col for col in df.columns if col != 'quality']
    X = df[cols].to_numpy(dtype=float)
    y = df['quality'].to_numpy(dtype=float)
    n = len(y)
    alpha = 0.05
    corrected_alpha = alpha / len(cols)
    print('ZADANIE 18 - Korelacja quality z każdą z 11 cech')
    print(f'Liczba testów={len(cols)}; alfa={alpha}; Bonferroni alfa_corr={corrected_alpha:.6f}')
    print(f'  {"Cecha":<23} {"Pearson r":>10} {"p-value":>13} {"przed":>8} {"po":>8}')
    before = after = 0
    for i, name in enumerate(cols):
        r = pearson_corr(X[:, i], y)
        p_value = corr_pvalue_approx(r, n)
        is_before = p_value < alpha
        is_after = p_value < corrected_alpha
        before += is_before
        after += is_after
        print(f'  {name:<23} {r:+10.4f} {p_value:>13.3g} '
              f'{"TAK" if is_before else "NIE":>8} {"TAK" if is_after else "NIE":>8}')
    print(f'Istotne przed korektą: {before}, po korekcie: {after}')
    print('Bonferroni ogranicza ryzyko przynajmniej jednego błędu typu I')
    print('w rodzinie 11 testów: family-wise error rate (FWER) <= 0.05.')
    print('Dla dużych n p-value korelacji to przybliżenie z lekcji, nie test dokładny.')


if __name__ == "__main__":
    main()
