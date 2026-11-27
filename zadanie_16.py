# -*- coding: utf-8 -*-
"""Lekcja 16, zadanie 16: Multikolinearność przez ranking korelacji.

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


def ascii_heatmap(C, labels):
    """Macierz korelacji jako tabela liczb i znaków ASCII/Unicode."""
    labels = [str(x) for x in labels]
    short = [name[:11] for name in labels]
    print('\nMacierz korelacji (█ silna dodatnia; ▓ średnia; · około 0; @ silna ujemna):')
    print(' ' * 14 + ''.join(f'{name:>12}' for name in short))
    for i, name in enumerate(short):
        cells = []
        for r in C[i]:
            mark = ('█' if r >= 0.7 else '▓' if r >= 0.4 else
                    '▒' if r >= 0.2 else '·' if r > -0.2 else
                    '-' if r > -0.4 else '=' if r > -0.7 else '@')
            cells.append(f'{mark}{r:+.2f}'.rjust(12))
        print(f'{name:>12}  ' + ''.join(cells))


def main():
    df = make_synthetic_wine(n_samples=1600, seed=42)
    selected = [col for col in df.columns if col != 'quality']
    original = list(selected)
    C = np.corrcoef(df[selected].to_numpy().T)
    print('ZADANIE 16 - Greedy correlation pruning na winach')
    print(f'Początkowo {len(selected)} cech; próg |r| > 0.7')
    ascii_heatmap(C, selected)
    removed = []
    threshold = 0.7
    while len(selected) > 1:
        C = np.corrcoef(df[selected].to_numpy().T)
        absolute = np.abs(C)
        np.fill_diagonal(absolute, 0.0)
        i, j = np.unravel_index(np.argmax(absolute), absolute.shape)
        strength = absolute[i, j]
        if strength <= threshold:
            break
        mean_i = absolute[i].sum() / (len(selected)-1)
        mean_j = absolute[j].sum() / (len(selected)-1)
        removed_index = i if mean_i >= mean_j else j
        dropped = selected[removed_index]
        print(f'Para: {selected[i]} / {selected[j]}: |r|={strength:.4f}; '
              f'średnie |r|={mean_i:.4f}/{mean_j:.4f}; usuwam: {dropped}')
        removed.append(dropped)
        selected.pop(removed_index)
    C_final = np.corrcoef(df[selected].to_numpy().T)
    upper = np.abs(C_final[np.triu_indices(len(selected), 1)])
    max_r = float(upper.max()) if len(upper) else 0.0
    print(f'\nUsunięte cechy ({len(removed)}): {removed}')
    print(f'Pozostałe cechy ({len(selected)}): {selected}')
    print(f'Największa korelacja poza diagonalą po redukcji: |r|={max_r:.4f}')
    assert max_r <= threshold + 1e-12, 'Nie usunięto wszystkich par > próg!'
    ascii_heatmap(C_final, selected)
    print('Interpretacja: liczba usuniętych cech zależy od faktycznej macierzy.')
    print('Dane syntetyczne nie muszą dać 2-4 usuniętych cech podanych orientacyjnie w PDF.')


if __name__ == "__main__":
    main()
