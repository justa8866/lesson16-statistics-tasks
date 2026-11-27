# -*- coding: utf-8 -*-
"""Lekcja 16, zadanie 17: Permutation test.

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


def ascii_histogram_with_marker(data, observed, label, bins=55, height=10):
    """Histogram permutacji i pionowa linia w wartości obserwowanej."""
    data = np.asarray(data, dtype=float)
    lo = min(float(data.min()), float(observed))
    hi = max(float(data.max()), float(observed))
    pad = (hi - lo) * 0.04
    lo -= pad
    hi += pad
    counts, edges = np.histogram(data, bins=bins, range=(lo, hi))
    scaled = np.rint(counts / max(counts.max(), 1) * height).astype(int)
    col_obs = int(np.clip(np.searchsorted(edges, observed, side='right') - 1, 0, bins - 1))
    print(f'\n{label} (█ permutacje; │ obserwowana różnica):')
    for h in range(height, 0, -1):
        chars = [('█' if v >= h else ' ') for v in scaled]
        chars[col_obs] = '◆' if chars[col_obs] == '█' else '│'
        print('  |' + ''.join(chars) + '|')
    print('  +' + '-' * bins + '+')
    print(f'  lewy koniec={lo:+.2f}; prawy koniec={hi:+.2f}; obserwacja={observed:+.4f}')


def main():
    X, labels = make_synthetic_flowers()
    A = X[labels == 'A', 2]
    B = X[labels == 'B', 2]
    diff_obs = float(A.mean() - B.mean())
    merged = np.concatenate([A, B])
    n_a = len(A)
    rng = np.random.default_rng(seed=42)
    n_perm = 10000
    diffs = np.empty(n_perm)
    for i in range(n_perm):
        shuffled = rng.permutation(merged)
        diffs[i] = shuffled[:n_a].mean() - shuffled[n_a:].mean()
    p_perm_raw = float((np.abs(diffs) >= abs(diff_obs)).mean())
    p_perm_corrected = (np.count_nonzero(np.abs(diffs) >= abs(diff_obs)) + 1)/(n_perm+1)
    v_a, v_b = A.var(ddof=1), B.var(ddof=1)
    se = math.sqrt(v_a/len(A) + v_b/len(B))
    t_stat = diff_obs / se
    p_welch_approx = math.erfc(abs(t_stat)/math.sqrt(2.0))
    welch_df = (v_a/len(A) + v_b/len(B))**2 / (
        (v_a/len(A))**2/(len(A)-1) + (v_b/len(B))**2/(len(B)-1))
    print('ZADANIE 17 - Test permutacyjny długości płatków A vs B')
    print(f'Średnia A={A.mean():.4f}, B={B.mean():.4f}; A-B={diff_obs:+.4f} cm')
    print(f'Permutation test ({n_perm}): p={p_perm_raw:.6f} (zgodnie z definicją zadania)')
    print(f'Praktyczne oszacowanie Monte Carlo z poprawką +1: p≈{p_perm_corrected:.6f}')
    print(f'Welch: t={t_stat:+.5f}, df≈{welch_df:.2f}, p≈{p_welch_approx:.3g}')
    print('Uwaga: p dla Welcha jest tu aproksymowane rozkładem normalnym, jak w lekcji.')
    ascii_histogram_with_marker(diffs, observed=diff_obs, label='Rozkład różnic przy H0')
    print('Wniosek:', 'odrzucamy H0 (różnica istotna).' if p_perm_corrected < 0.05
          else 'nie ma podstaw do odrzucenia H0.')
    print('Test permutacyjny nie wymaga normalności danych; jest użyteczny')
    print('zwłaszcza przy niewielkich próbach, jeśli obserwacje są wymienialne pod H0.')


if __name__ == "__main__":
    main()
