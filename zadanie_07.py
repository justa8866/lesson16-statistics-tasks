# -*- coding: utf-8 -*-
"""Lekcja 16, zadanie 7: Test t dwóch prób.

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


def permutation_pvalue(a, b, n_perm=2000, seed=42):
    """Dwustronny test permutacyjny różnicy średnich."""
    a, b = np.asarray(a), np.asarray(b)
    merged = np.concatenate([a, b])
    n_a = len(a)
    observed = abs(a.mean() - b.mean())
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(n_perm):
        indices = rng.permutation(len(merged))
        diff = abs(merged[indices[:n_a]].mean() - merged[indices[n_a:]].mean())
        extreme += diff >= observed
    # Poprawka +1: unikamy fałszywie dokładnego p=0 w Monte Carlo.
    return (extreme + 1) / (n_perm + 1)


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
    df = load_titanic()
    age_survived = df.loc[df['Survived'] == 1, 'Age'].dropna().to_numpy(dtype=float)
    age_died = df.loc[df['Survived'] == 0, 'Age'].dropna().to_numpy(dtype=float)
    n1, n2 = len(age_survived), len(age_died)
    m1, m2 = float(age_survived.mean()), float(age_died.mean())
    v1, v2 = float(age_survived.var(ddof=1)), float(age_died.var(ddof=1))
    se = math.sqrt(v1 / n1 + v2 / n2)
    t_stat = (m1 - m2) / se   # test Welcha bez założenia równości wariancji
    p_normal = math.erfc(abs(t_stat) / math.sqrt(2.0))
    p_perm = permutation_pvalue(age_survived, age_died, n_perm=2000, seed=42)
    pooled_sd = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    cohen_d = (m1 - m2) / pooled_sd
    effect = 'mały' if abs(cohen_d) < 0.2 else 'średni' if abs(cohen_d) < 0.8 else 'duży'
    print('ZADANIE 7 - Czy średni wiek różnił się między grupami?')
    print(f'Przeżyli: n={n1}, średni wiek={m1:.3f}; zmarli: n={n2}, średni wiek={m2:.3f}')
    print(f'Różnica średnich (przeżyli - zmarli): {m1-m2:+.3f} lat')
    print(f'Welch t={t_stat:+.4f}, p≈{p_normal:.6f} (duża próba, normalna aproks.)')
    print(f'Permutacje (2000): p≈{p_perm:.6f}')
    print(f"Cohen's d={cohen_d:+.4f}; efekt {effect}")
    print('Wniosek:', 'różnica istotna statystycznie.' if p_perm < 0.05
          else 'nie wykazano istotności statystycznej.')
    ascii_boxplot_two(age_survived, age_died, 'Przeżyli', 'Zmarli')


if __name__ == "__main__":
    main()
