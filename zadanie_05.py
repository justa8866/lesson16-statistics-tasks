# -*- coding: utf-8 -*-
"""Lekcja 16, zadanie 5: Macierz korelacji.

Skrypt: nie wymaga importowania innych zadań.
Używa wyłącznie NumPy, Pandas (gdy wymagane) i biblioteki standardowej.
Wykresy są tekstowe (ASCII/Unicode), zgodnie z lekcją.
Uruchomienie: python zadanie_XX.py
"""

import math
import numpy as np
import pandas as pd


def make_california_housing(n_samples=2000, seed=42):
    """Syntetyczny generator California Housing z lekcji (bez sklearn)."""
    rng = np.random.default_rng(seed)
    med_inc = rng.gamma(shape=2.5, scale=1.6, size=n_samples) + 0.5
    house_age = rng.uniform(1, 52, size=n_samples)
    ave_rooms = 3 + 0.8 * med_inc + rng.normal(0, 0.8, size=n_samples)
    ave_bedrms = 1 + 0.1 * ave_rooms + rng.normal(0, 0.1, size=n_samples)
    population = rng.gamma(shape=2.0, scale=700, size=n_samples)
    ave_occup = 2.5 + rng.normal(0, 0.6, size=n_samples)
    latitude = rng.uniform(32.5, 42.0, size=n_samples)
    longitude = rng.uniform(-124.5, -114.0, size=n_samples)
    house_value = (0.45 * med_inc - 0.01 * house_age + 0.03 * ave_rooms
                   - 0.02 * ave_bedrms - 0.0001 * population
                   + 0.12 * (latitude - 36) - 0.05 * (longitude + 119)
                   + rng.normal(0, 0.4, size=n_samples))
    house_value = np.clip(house_value, 0.15, 5.0)
    return pd.DataFrame({
        'MedInc': med_inc, 'HouseAge': house_age, 'AveRooms': ave_rooms,
        'AveBedrms': ave_bedrms, 'Population': population,
        'AveOccup': ave_occup, 'Latitude': latitude, 'Longitude': longitude,
        'MedHouseVal': house_value,
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
    df = make_california_housing(n_samples=2000)
    cols = list(df.columns)
    C = np.corrcoef(df.values.T)
    print('ZADANIE 5 - Macierz korelacji California Housing')
    print(f'Zbiór danych: {df.shape[0]} obserwacji, {df.shape[1]} kolumn')
    ascii_heatmap(C, cols)
    features = [(i, name) for i, name in enumerate(cols) if name != 'MedHouseVal']
    pairs = sorted(
        [(abs(C[i, j]), C[i, j], a, b)
         for pos, (i, a) in enumerate(features)
         for j, b in features[pos + 1:]],
        reverse=True,
    )
    print('\nTOP 3 pary cech poza targetem (według |r|):')
    for rank, (_, r, a, b) in enumerate(pairs[:3], start=1):
        print(f'{rank}. {a:<15} i {b:<15} r={r:+.4f}')
    print('Interpretacja: AveRooms zależy w generatorze od MedInc,')
    print('zaś AveBedrms zależy od AveRooms; takie powiązania zwiększają korelację.')
    print('Latitude i Longitude są w tej wersji generatora losowane niezależnie,')
    print('dlatego nie należy zakładać, że muszą mieć wysoką korelację.')


if __name__ == "__main__":
    main()
