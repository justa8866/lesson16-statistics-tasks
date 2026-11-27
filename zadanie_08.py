# -*- coding: utf-8 -*-
"""Lekcja 16, zadanie 8: Test chi-kwadrat.

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


def chi2_statistic(observed):
    """Statystyka chi-kwadrat i tablica oczekiwanych liczności."""
    O = np.asarray(observed, dtype=float)
    E = np.outer(O.sum(axis=1), O.sum(axis=0)) / O.sum()
    if np.any(E <= 0):
        raise ValueError('Każda oczekiwana liczność musi być dodatnia.')
    return float(np.sum((O - E) ** 2 / E)), E

def chi2_monte_carlo_pvalue(observed, n_sim=3000, seed=42):
    """Test permutacyjny niezależności, z zachowaniem obu marginesów tabeli."""
    observed = np.asarray(observed, dtype=int)
    rows, cols = observed.shape
    row_counts = observed.sum(axis=1)
    col_counts = observed.sum(axis=0)
    labels = np.repeat(np.arange(cols), col_counts)
    expected = np.outer(row_counts, col_counts) / observed.sum()
    actual, _ = chi2_statistic(observed)
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(n_sim):
        permuted = rng.permutation(labels)
        sim = np.zeros_like(observed)
        start = 0
        for row, n in enumerate(row_counts):
            sim[row] = np.bincount(permuted[start:start + n], minlength=cols)
            start += n
        stat = np.sum((sim - expected) ** 2 / expected)
        extreme += stat >= actual - 1e-12
    return (extreme + 1) / (n_sim + 1)


def ascii_stacked_bars(table, width=50):
    """Procentowe słupki: ░ nie przeżył (0); █ przeżył (1)."""
    fractions = table.div(table.sum(axis=1), axis=0)
    print('\nPrzeżywalność według klasy (░ = 0, █ = 1):')
    for label, row in fractions.iterrows():
        death = float(row.get(0, 0.0))
        survived = float(row.get(1, 0.0))
        n_death = int(round(death * width))
        bar = '░' * n_death + '█' * (width - n_death)
        print(f'  Klasa {label}: |{bar}|  nie: {death*100:5.1f}%  tak: {survived*100:5.1f}%')


def main():
    df = load_titanic()
    table = pd.crosstab(df['Pclass'], df['Survived'])
    observed = table.to_numpy(dtype=int)
    chi2, expected = chi2_statistic(observed)
    p_value = chi2_monte_carlo_pvalue(observed, n_sim=3000, seed=42)
    n = int(observed.sum())
    dfree = (observed.shape[0] - 1) * (observed.shape[1] - 1)
    cramers_v = math.sqrt(chi2 / (n * min(observed.shape[0] - 1, observed.shape[1] - 1)))
    print('ZADANIE 8 - Zależność przeżycia od klasy podróży')
    print('H0: klasa i przeżycie są niezależne; H1: zależne')
    print('Tabela obserwowana:')
    print(table)
    print('\nOczekiwane liczności przy H0:')
    print(pd.DataFrame(expected, index=table.index, columns=table.columns).round(2))
    print(f'chi²={chi2:.4f}; stopnie swobody={dfree}')
    print(f'p-value Monte Carlo (3000)≈{p_value:.6f}')
    print(f"Cramer's V={cramers_v:.4f}")
    print('Wniosek:', 'istnieje statystyczna zależność.' if p_value < 0.05
          else 'nie wykazano statystycznej zależności.')
    ascii_stacked_bars(table)


if __name__ == "__main__":
    main()
