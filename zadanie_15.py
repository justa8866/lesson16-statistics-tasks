# -*- coding: utf-8 -*-
"""Lekcja 16, zadanie 15: Power analysis dla A/B testu (ręcznie).

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
    p1, p2 = 0.10, 0.12
    alpha, power = 0.05, 0.80
    z_alpha = normal_ppf(1.0 - alpha / 2.0)
    z_beta = normal_ppf(power)
    n_formula = (z_alpha + z_beta)**2 * (p1*(1-p1) + p2*(1-p2)) / (p1-p2)**2
    n_required = math.ceil(n_formula)
    print('ZADANIE 15 - Planowanie wielkości grup w teście A/B')
    print(f'p1={p1:.2f}, p2={p2:.2f}, alfa={alpha:.2f}, moc docelowa={power:.0%}')
    print(f'Z_(1-alfa/2)={z_alpha:.5f}, Z_mocy={z_beta:.5f}')
    print(f'Wynik wzoru: n={n_formula:.3f}; zaokrąglamy w górę: {n_required} osób w grupie')
    rng = np.random.default_rng(seed=42)
    n_tests = 1000
    successes_a = rng.binomial(n=n_required, p=p1, size=n_tests)
    successes_b = rng.binomial(n=n_required, p=p2, size=n_tests)
    rate_a, rate_b = successes_a/n_required, successes_b/n_required
    p_pool = (successes_a + successes_b) / (2*n_required)
    se = np.sqrt(p_pool * (1-p_pool) * (2/n_required))
    z = (rate_b - rate_a) / se
    # W teście dwustronnym p < alfa odpowiada |z| > z_(1-alfa/2).
    significant = np.abs(z) > z_alpha
    actual_power = float(significant.mean())
    print(f'Symulacja {n_tests} testów: wykryto efekt w {significant.sum()} przypadkach')
    print(f'Empiryczna moc = {actual_power:.3f} ({actual_power*100:.1f}%)')
    print('Wniosek: wzrost konwersji o tylko 2 p.p. jest niewielki względem')
    print('wariancji wyników binarnych, więc potrzeba tysięcy użytkowników w każdej grupie.')
    print('Uwaga: wynik formuły może nieznacznie różnić się od ~3835 podanych w PDF')
    print('przez dokładniejsze wartości kwantyli i zaokrąglenie w górę.')


if __name__ == "__main__":
    main()
