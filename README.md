# Lekcja 16 — Statystyka i testowanie hipotez

Praca domowa z Data Science: 20 zadań w Pythonie dotyczących rozkładów prawdopodobieństwa, korelacji, testowania hipotez i analizy eksperymentów A/B.

## Uruchomienie

Wymagane są Python 3 oraz biblioteki NumPy i Pandas.

```bash
git clone https://github.com/justa8866/lesson16-statistics-tasks.git
cd lesson16-statistics-tasks
python -m pip install -r requirements.txt
python zadanie_01.py
```

Każdy plik `zadanie_XX.py` można uruchomić osobno. Wyniki i wykresy tekstowe są wyświetlane w terminalu.

## Spis zadań

| Plik | Temat |
| --- | --- |
| [zadanie_01.py](zadanie_01.py) | Generowanie rozkładu normalnego |
| [zadanie_02.py](zadanie_02.py) | Obliczanie prawdopodobieństw |
| [zadanie_03.py](zadanie_03.py) | Analiza rozkładu kwiatów |
| [zadanie_04.py](zadanie_04.py) | Korelacja Pearsona |
| [zadanie_05.py](zadanie_05.py) | Macierz korelacji |
| [zadanie_06.py](zadanie_06.py) | Test t jednej próbki |
| [zadanie_07.py](zadanie_07.py) | Test t dwóch prób |
| [zadanie_08.py](zadanie_08.py) | Test chi-kwadrat |
| [zadanie_09.py](zadanie_09.py) | Symulacja rozkładu dwumianowego |
| [zadanie_10.py](zadanie_10.py) | Wykrywanie wartości odstających metodą z-score |
| [zadanie_11.py](zadanie_11.py) | Porównanie korelacji Pearsona i Spearmana |
| [zadanie_12.py](zadanie_12.py) | ANOVA — porównanie wielu grup |
| [zadanie_13.py](zadanie_13.py) | Symulacja centralnego twierdzenia granicznego |
| [zadanie_14.py](zadanie_14.py) | Bootstrap i przedziały ufności |
| [zadanie_15.py](zadanie_15.py) | Analiza mocy i wielkość próby w teście A/B |
| [zadanie_16.py](zadanie_16.py) | Multikolinearność i ranking korelacji |
| [zadanie_17.py](zadanie_17.py) | Test permutacyjny |
| [zadanie_18.py](zadanie_18.py) | Korekta Bonferroniego dla wielu testów |
| [zadanie_19.py](zadanie_19.py) | Dopasowanie rozkładów i test Kołmogorowa–Smirnowa |
| [zadanie_20.py](zadanie_20.py) | Bayesowski test A/B |

## Dane

Zadania 6, 7, 8, 14 i 19 domyślnie pobierają zbiór [Titanic](https://github.com/datasciencedojo/datasets/blob/master/titanic.csv), więc wymagają połączenia z Internetem. Można też wskazać lokalny plik za pomocą zmiennej środowiskowej `TITANIC_CSV`, np. w PowerShell:

```powershell
$env:TITANIC_CSV = "C:\dane\titanic.csv"
python zadanie_06.py
```
