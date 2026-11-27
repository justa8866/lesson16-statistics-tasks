# -*- coding: utf-8 -*-
"""Lekcja 16, zadanie 2: Obliczanie prawdopodobieństw (ręcznie).

Skrypt: nie wymaga importowania innych zadań.
Używa wyłącznie NumPy, Pandas (gdy wymagane) i biblioteki standardowej.
Wykresy są tekstowe (ASCII/Unicode), zgodnie z lekcją.
Uruchomienie: python zadanie_XX.py
"""

import math
import numpy as np


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
    mu, sigma = 100, 15
    p_low = normal_cdf(85, mu, sigma)
    p_high = 1 - normal_cdf(120, mu, sigma)
    p_middle = normal_cdf(110, mu, sigma) - normal_cdf(90, mu, sigma)
    print('ZADANIE 2 - Prawdopodobieństwa X ~ N(100, 15)')
    print(f'P(X < 85) = {p_low:.6f} = {p_low*100:.2f}%')
    print(f'P(X > 120) = {p_high:.6f} = {p_high*100:.2f}%')
    print(f'P(90 < X < 110) = {p_middle:.6f} = {p_middle*100:.2f}%')
    print(f'Interpretacja: około {p_low*100:.1f}% wartości jest mniejszych od 85.')
    print(f'Interpretacja: około {p_high*100:.1f}% wartości przekracza 120.')
    print(f'Interpretacja: około {p_middle*100:.1f}% wartości leży między 90 i 110.')
    print('W rozkładzie ciągłym < oraz <= dają te same prawdopodobieństwa.')


if __name__ == "__main__":
    main()
