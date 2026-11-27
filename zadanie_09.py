# -*- coding: utf-8 -*-
"""Lekcja 16, zadanie 9: Rozkład dwumianowy - symulacja.

Skrypt: nie wymaga importowania innych zadań.
Używa wyłącznie NumPy, Pandas (gdy wymagane) i biblioteki standardowej.
Wykresy są tekstowe (ASCII/Unicode), zgodnie z lekcją.
Uruchomienie: python zadanie_XX.py
"""

import math
import numpy as np


def binom_pmf(k, n, p):
    """Rozkład dwumianowy P(X=k) = C(n,k)p^k(1-p)^(n-k)."""
    return math.comb(n, k) * p**k * (1.0 - p)**(n-k)

def ascii_barplot_pmf(xs, ps, title, symbol='█', width=40):
    print(f'\n{title}')
    max_p = max(float(np.max(ps)), 1e-12)
    for k, prob in zip(xs, ps):
        width_at_k = int(round(prob / max_p * width))
        print(f'  {int(k):2d}: P={prob:.5f} |{symbol*width_at_k}')


def main():
    rng = np.random.default_rng(seed=42)
    n, p, experiments = 20, 0.5, 1000
    sim = rng.binomial(n=n, p=p, size=experiments)
    frequencies = np.bincount(sim, minlength=n + 1)
    empirical = frequencies / experiments
    theoretical = np.array([binom_pmf(k, n, p) for k in range(n + 1)])
    print('ZADANIE 9 - Rzut 20 monetami, 1000 eksperymentów')
    print('Histogram empiryczny (█):')
    ascii_barplot_pmf(np.arange(n + 1), empirical, 'Próba: 1000 eksperymentów', '█')
    print('\nRozkład teoretyczny (·):')
    ascii_barplot_pmf(np.arange(n + 1), theoretical, 'Rozkład B(20, 0.5)', '·')
    print(f'\nŚrednia empiryczna: {sim.mean():.4f}')
    print(f'Średnia teoretyczna np: {n*p:.4f}')
    print(f'Suma prawdopodobieństw teoretycznych: {theoretical.sum():.8f}')
    print(f'Największa bezwzględna różnica częstości: {np.max(abs(empirical-theoretical)):.4f}')
    print('Interpretacja: przy 1000 losowań histogram przybliża teoretyczny PMF,')
    print('ale liczności w poszczególnych punktach podlegają losowym wahaniom.')


if __name__ == "__main__":
    main()
