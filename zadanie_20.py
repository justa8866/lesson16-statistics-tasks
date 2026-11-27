# -*- coding: utf-8 -*-
"""Lekcja 16, zadanie 20: Bayesian A/B test.

Skrypt: nie wymaga importowania innych zadań.
Używa wyłącznie NumPy, Pandas (gdy wymagane) i biblioteki standardowej.
Wykresy są tekstowe (ASCII/Unicode), zgodnie z lekcją.
Uruchomienie: python zadanie_XX.py
"""

import math
import numpy as np


def ascii_posterior_overlay(samples_a, samples_b, bins=48, height=12):
    """Dwa nakładające się histogramy: █ A, · B, ▒ oba."""
    joint = np.concatenate([samples_a, samples_b])
    edges = np.linspace(joint.min(), joint.max(), bins + 1)
    ca = np.histogram(samples_a, bins=edges)[0]
    cb = np.histogram(samples_b, bins=edges)[0]
    maximum = max(int(ca.max()), int(cb.max()), 1)
    sa = np.rint(ca / maximum * height).astype(int)
    sb = np.rint(cb / maximum * height).astype(int)
    print('\nNałożone rozkłady posterior: █ = A, · = B, ▒ = oba')
    for level in range(height, 0, -1):
        line = []
        for a, b in zip(sa, sb):
            line.append('▒' if a >= level and b >= level else
                        '█' if a >= level else '·' if b >= level else ' ')
        print('  |' + ''.join(line) + '|')
    print('  +' + '-' * bins + '+')
    print(f'  p: {edges[0]:.3f} .. {edges[-1]:.3f}; posteriory mogą się nakładać')


def main():
    rng = np.random.default_rng(seed=42)
    n = 500
    conv_a = rng.binomial(n=1, p=0.10, size=n)
    conv_b = rng.binomial(n=1, p=0.12, size=n)
    successes_a = int(conv_a.sum())
    successes_b = int(conv_b.sum())
    rate_a, rate_b = successes_a/n, successes_b/n
    # Beta(1,1) + obserwacje Bernoulliego => posterior Beta(1+S, 1+n-S).
    alpha_a, beta_a = 1 + successes_a, 1 + n-successes_a
    alpha_b, beta_b = 1 + successes_b, 1 + n-successes_b
    posterior_a = rng.beta(alpha_a, beta_a, size=10000)
    posterior_b = rng.beta(alpha_b, beta_b, size=10000)
    probability_b_better = float((posterior_b > posterior_a).mean())
    pooled = (successes_a + successes_b)/(2*n)
    se_pool = math.sqrt(pooled * (1-pooled) * (2/n))
    z = (rate_b - rate_a)/se_pool
    p_two_sided = math.erfc(abs(z)/math.sqrt(2.0))
    print('ZADANIE 20 - Bayesowski test A/B, prior Beta(1,1)')
    print(f'Wersja A: {successes_a}/{n} konwersji = {rate_a:.2%}')
    print(f'Wersja B: {successes_b}/{n} konwersji = {rate_b:.2%}')
    print(f'Posterior A: Beta({alpha_a}, {beta_a})')
    print(f'Posterior B: Beta({alpha_b}, {beta_b})')
    print(f'P(p_B > p_A | dane) ≈ {probability_b_better:.4f} ({probability_b_better:.1%})')
    print(f'Klasyczny test proporcji: z={z:+.4f}, p-value={p_two_sided:.6f}')
    print('Decyzja przy alfa=0.05:', 'istotna różnica' if p_two_sided < 0.05
          else 'brak statystycznie istotnej różnicy')
    ascii_posterior_overlay(posterior_a, posterior_b)
    print('Bayes: interpretujemy prawdopodobieństwo przewagi B po zaobserwowaniu danych.')
    print('Podejście częstościowe: p-value mierzy zgodność danych z H0,')
    print('nie jest prawdopodobieństwem prawdziwości H0 ani tego, że B jest lepsze.')


if __name__ == "__main__":
    main()
