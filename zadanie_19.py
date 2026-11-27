# -*- coding: utf-8 -*-
"""Lekcja 16, zadanie 19: Analiza rozkładów z dopasowaniem (ręczny K-S test).

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


def ascii_histogram(data, bins=20, height=10, label='Histogram'):
    """Histogram rysowany wyłącznie w terminalu (bez matplotlib)."""
    data = np.asarray(data, dtype=float)
    counts, edges = np.histogram(data, bins=bins)
    peak = max(int(counts.max()), 1)
    heights = np.rint(counts / peak * height).astype(int)
    print(f'\n{label} (n={len(data)}, maksymalna liczność={peak}):')
    for level in range(height, 0, -1):
        print('  |' + ''.join('█' if v >= level else ' ' for v in heights) + '|')
    print('  +' + '-' * bins + '+')
    print(f'   min={edges[0]:.2f}, max={edges[-1]:.2f}')
    return counts, edges


def ascii_pdf_comparison(x, pdf_func, title, bins=24, width=45):
    """W każdym koszyku słupek danych i punkt oczekiwanej liczności z PDF."""
    counts, edges = np.histogram(x, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    expected = np.asarray(pdf_func(centers)) * len(x) * np.diff(edges)
    scale = max(float(counts.max()), float(expected.max()), 1e-12)
    print(f'\nPorównanie rozkładu: {title}')
    for k, (count, exp_count) in enumerate(zip(counts, expected)):
        bar_len = int(round(count / scale * width))
        mark_idx = int(np.clip(round(exp_count / scale * width), 0, width - 1))
        line = [' '] * width
        for i in range(bar_len):
            line[i] = '█'
        line[mark_idx] = '◆' if line[mark_idx] == '█' else '·'
        print(f'  {centers[k]:7.1f}: |{"".join(line)}| obs={count:3d}, model≈{exp_count:6.1f}')


def main():
    df = load_titanic()
    fares = df['Fare'].dropna().to_numpy(dtype=float)
    fares = fares[fares > 0]
    if len(fares) < 10:
        raise SystemExit('Za mało dodatnich cen biletów do wykonania analizy.')
    x = np.sort(fares)
    n = len(x)
    mu, sigma = float(x.mean()), float(x.std(ddof=1))
    log_x = np.log(x)
    mu_log, sigma_log = float(log_x.mean()), float(log_x.std(ddof=1))
    lam = 1.0 / mu
    distributions = {
        'Normalny': (
            normal_cdf(x, mu, sigma),
            lambda t: normal_pdf(t, mu, sigma),
            f'mu={mu:.3f}, sigma={sigma:.3f}',
        ),
        'Lognormalny': (
            normal_cdf(np.log(x), mu_log, sigma_log),
            lambda t: normal_pdf(np.log(np.maximum(t, 1e-12)), mu_log, sigma_log)/np.maximum(t, 1e-12),
            f'mu_log={mu_log:.3f}, sigma_log={sigma_log:.3f}',
        ),
        'Wykładniczy': (
            1.0 - np.exp(-lam * x),
            lambda t: lam * np.exp(-lam * np.maximum(t, 0)),
            f'lambda={lam:.5f}',
        ),
    }
    rows = []
    print('ZADANIE 19 - MLE i test Kolmogorova-Smirnova dla Fare > 0')
    print(f'n={n}, min={x.min():.2f}, mediana={np.median(x):.2f}, max={x.max():.2f}')
    for name, (cdf, pdf, params) in distributions.items():
        # Dokładna empiryczna statystyka dwustronna: prawa i lewa granica ECDF.
        D_plus = float(np.max(np.arange(1, n+1)/n - cdf))
        D_minus = float(np.max(cdf - np.arange(n)/n))
        D = max(D_plus, D_minus)
        p_approx = min(1.0, 2.0 * math.exp(-2.0*n*D*D))
        rows.append((name, D, p_approx, pdf))
        print(f'  {name:<13}: {params}')
    print('\nPorównanie dopasowań (najmniejsze D = najlepsze):')
    print(f'{"Rozkład":<14} {"Statystyka D":>15} {"p-value (aproks.)":>20}')
    for name, D, p, _ in sorted(rows, key=lambda a: a[1]):
        print(f'{name:<14} {D:>15.5f} {p:>20.5g}')
    winner = min(rows, key=lambda item: item[1])
    print(f'Najlepiej według D: {winner[0]} (D={winner[1]:.5f})')
    print('Uwaga: p-value jest jedynie orientacyjne: klasyczny test K-S nie')
    print('uwzględnia estymacji parametrów na tych samych danych.')
    ascii_histogram(x, bins=24, height=8, label='Fare - histogram empiryczny')
    for name, _, _, pdf in rows:
        ascii_pdf_comparison(x, pdf, name, bins=24)
    print('Histogram z nakładką: █ = empiryczna liczność, · = poziom oczekiwany z PDF.')


if __name__ == "__main__":
    main()
