# -*- coding: utf-8 -*-
"""Lekcja 16, zadanie 12: ANOVA ręcznie - porównanie wielu grup.

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


def anova_manual(groups):
    """Jednoczynnikowa ANOVA: F=MSB/MSW, bez scipy."""
    k = len(groups)
    n = sum(len(g) for g in groups)
    grand_mean = np.concatenate(groups).mean()
    ssb = sum(len(g)*(g.mean()-grand_mean)**2 for g in groups)
    ssw = sum(np.sum((g-g.mean())**2) for g in groups)
    msb, msw = ssb/(k-1), ssw/(n-k)
    return msb/msw, ssb, ssw, msb, msw


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
    X, species = make_synthetic_flowers()
    groups = {letter: X[species == letter, 2] for letter in ['A', 'B', 'C']}
    F, SSB, SSW, MSB, MSW = anova_manual(list(groups.values()))
    print('ZADANIE 12 - ANOVA długości płatków, gatunki A/B/C')
    print('H0: μA = μB = μC; H1: przynajmniej jedna średnia jest inna')
    for name, data in groups.items():
        print(f'{name}: n={len(data)}, średnia={data.mean():.4f}, std={data.std(ddof=1):.4f}')
    print(f'SSB={SSB:.4f}, SSW={SSW:.4f}, MSB={MSB:.4f}, MSW={MSW:.4f}')
    print(f'F = MSB/MSW = {F:.4f}')
    rng = np.random.default_rng(42)
    all_values = X[:, 2]
    sizes = [len(g) for g in groups.values()]
    extreme = 0
    n_perm = 3000
    for _ in range(n_perm):
        perm = rng.permutation(all_values)
        cut1, cut2 = sizes[0], sizes[0] + sizes[1]
        f_perm = anova_manual([perm[:cut1], perm[cut1:cut2], perm[cut2:]])[0]
        extreme += f_perm >= F
    p = (extreme + 1) / (n_perm + 1)
    print(f'p-value z {n_perm} permutacji ≈ {p:.6f}')
    if p < 0.05:
        print('ANOVA istotna: testy parami z korektą Bonferroniego:')
        alpha_corr = 0.05 / 3
        for i, (a, b) in enumerate([('A', 'B'), ('A', 'C'), ('B', 'C')]):
            p_pair = permutation_pvalue(groups[a], groups[b], n_perm=3000, seed=100+i)
            result = 'ISTOTNA różnica' if p_pair < alpha_corr else 'brak istotności'
            print(f'  {a} vs {b}: p={p_pair:.6f}, próg={alpha_corr:.5f}: {result}')
            ascii_boxplot_two(groups[a], groups[b], a, b)
    else:
        print('Nie stwierdzono istotnych różnic między średnimi.')


if __name__ == "__main__":
    main()
