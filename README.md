# Prognoza dziennej sprzedaży na 28 dni

Projekt prognozuje **łączną dzienną sprzedaż** (`Units Sold`) sumowaną po 5 sklepach i 20 produktach na 28 dni do przodu.
Dane obejmują okres 1.01.2022 – 30.01.2024. Modele uczą się na danych do 2.01.2024, a ostatnie 28 dni (3–30.01.2024)
służą wyłącznie jako zbiór egzaminacyjny.

Porównujemy trzy proste prognozy odniesienia (naive, seasonal naive, średnia z 28 dni) z dwoma eksperymentami:

1. **ARIMAX** (regresja na zmiennych zewnętrznych z „pamięcią” błędów) oraz jego wariant sezonowy SARIMAX,
2. **ensemble** trzech różnych modeli: ARIMAX, globalny **LightGBM** (uczony na 100 seriach sklep × produkt) i **Ridge**,
   w dwóch wariantach (zwykła i ważona średnia).

Bonus: osobne prognozy dla trzech najlepiej sprzedających się produktów. Całość ma aplikację webową (Streamlit).

> Wszystkie wyniki na zbiorze egzaminacyjnym dotyczą wariantu ***oracle***: modele znają z góry prawdziwe wartości
> zmiennych zewnętrznych (epidemia, promocje, pogoda) dla dni prognozy.

## Wymagania i instalacja

Projekt był uruchamiany na Pythonie 3.12. Zależności z wersjami są w `requirements.txt` (pandas, numpy, scikit-learn,
statsmodels, lightgbm, streamlit, plotly, matplotlib).

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Na Debianie i Ubuntu do `venv` może być potrzebny pakiet systemowy: `sudo apt install python3-venv`.

## Dane

Plik `demand_forecasting.csv` **nie jest
częścią repozytorium**. Umieść go w katalogu głównym projektu (ścieżka jest w `config.py`). Bez niego:
  - nie da się ponownie policzyć modeli, a testy oparte na danych są pomijane,
  - aplikacja i testy wyników działają, bo korzystają z gotowych plików w `outputs/` (zawierają tylko agregaty i prognozy,
    bez surowych wierszy).

    [Link do pliku csv](https://www.kaggle.com/datasets/raminhuseyn/demand-forecasting-dataset)


## Co wpływa na sprzedaż i które zmienne wybraliśmy

Liczby poniżej policzył `python -m src.explore` wyłącznie na **danych treningowych** (732 dni, do 2.01.2024).

| Obserwacja | Wynik |
|---|---|
| **Epidemia** | średnia dzienna sprzedaż spada z 9 599 do 6 084 szt. (−36,6%). To największy pojedynczy efekt: sama flaga epidemii wyjaśnia 80,9% zmienności dziennej sprzedaży (R² = 0,809) |
| **Promocje** | dzień, w którym wszystkie produkty są w promocji, daje ok. +2 122 szt. (współczynnik przy udziale produktów w promocji, od 0 do 1) |
| **Pogoda** | gdyby wszystkie sklepy miały daną pogodę zamiast pochmurnej: słonecznie +774 szt., deszcz −721 szt., śnieg −1 355 szt. |
| **Rytm tygodniowy** | średnie sprzedaże w poszczególne dni tygodnia różnią się o 1,9% (o 2,8% po wyłączeniu dni epidemii): rytmu tygodniowego nie ma. Potwierdza to SARIMAX, który z sezonowością tygodniową wypadł w walidacji nieco gorzej niż ARIMAX |
| **Razem** | regresja liniowa na epidemii, promocjach i pogodzie ma R² = 0,906. To dopasowanie w próbie treningowej, a nie wynik prognozy |

### Zmienne użyte w modelach

| Zmienna | Jak jest zapisana | Gdzie |
|---|---|---|
| Epidemia | flaga 0/1 dla dnia (w danych to jedna wartość dla wszystkich sklepów i produktów) | ARIMAX, SARIMAX, Ridge, LightGBM |
| Promocje | udział produktów w promocji danego dnia (modele dzienne); flaga promocji wiersza (LightGBM) | ARIMAX, SARIMAX, Ridge, LightGBM |
| Pogoda | udziały sklepów ze słońcem, deszczem i śniegiem (modele dzienne; pochmurno jest kategorią odniesienia, żeby udziały nie sumowały się do stałej); pogoda sklepu (LightGBM) | ARIMAX, SARIMAX, Ridge, LightGBM |
| Sklep, produkt, kategoria, region | identyfikatory serii | tylko LightGBM |
| Historia sprzedaży serii | opóźnienie co najmniej 28 dni: sprzedaż sprzed 28 dni oraz średnie z 7 i 28 dni liczone od tego dnia | tylko LightGBM |

Baseline'y korzystają wyłącznie z historii sprzedaży z treningu. Zmienne zewnętrzne dla dni egzaminacyjnych podajemy
modelom w prawdziwej wartości. To wariant ***oracle***: pokazuje, jak dobry może być model, **jeśli** te informacje znamy
z góry. Promocje firma planuje z wyprzedzeniem, ale pogody i epidemii na kilka tygodni nie znamy, dlatego aplikacja ma
osobny scenariusz długości epidemii. Cechy sprzedażowe LightGBM mają opóźnienie co najmniej 28 dni, więc nie potrzeba
prognozy rekurencyjnej.

### Czego nie używamy i dlaczego

| Kolumna | Dlaczego odpada |
|---|---|
| `Demand`, `Inventory Level`, `Units Ordered` | **wyciek danych**: nie znamy ich z wyprzedzeniem. Test pilnuje, że nie ma ich wśród cech |
| `Competitor Pricing` | ceny konkurencji nie są znane z góry |
| `Discount` (rabat) | powiela promocję (korelacja dzienna 0,97, VIF 16, w regresji nieistotny, p = 0,66); usunięty ze wszystkich modeli, patrz [Zmiany po pierwszym pomiarze](#zmiany-po-pierwszym-pomiarze-na-egzaminie) |
| `Price` | korelacja z epidemią −0,91, a współczynnik przy sprzedaży dodatni (wyższa cena, wyższa sprzedaż). To znak, że cena podąża za popytem, więc prawdziwa cena z okresu egzaminacyjnego mogłaby ukrywać informację o sprzedaży. Modeli z ceną nie sprawdzaliśmy |
| `Seasonality` | etykieta pory roku wynikająca wprost z miesiąca. Zamiast niej Ridge dostał płynny cykl roczny (fale sin/cos dnia roku) |
| dzień tygodnia | rytmu tygodniowego w danych nie ma (różnice poniżej 3%), SARIMAX z sezonowością s=7 wypadł nieco gorzej niż ARIMAX |

## Metodologia

- **Trening:** 1.01.2022 – 2.01.2024 (732 dni). **Zbiór egzaminacyjny:** 3–30.01.2024 (28 dni), z czego pierwsze 6 dni
  (3–8.01) wypada w epidemii, a pozostałe 22 nie.
- **Parametry i wagi ensemble wybrano walidacją kroczącą (walk-forward) na treningu:** 6 kolejnych okien testowych po
  28 dni na końcu treningu (19.07.2023 – 2.01.2024), z rozszerzającym się oknem uczenia.
- **Baseline'y liczone uczciwie**, wyłącznie z treningu: naive to ostatni dzień treningu powtórzony 28 razy, seasonal
  naive to ostatnie 7 dni powtórzone 4 razy, średnia to średnia z ostatnich 28 dni.
- **Brak wycieku danych pilnują testy:** po pomnożeniu sprzedaży z okresu egzaminacyjnego (a w walidacji: z dni po foldzie)
  przez 1000 nie zmienia się żadna cecha, żaden baseline ani żadna prognoza.
- **Metryki:** MAE, RMSE, MAPE i bias (plus = model zawyża), z jednej wspólnej funkcji, liczone razem oraz osobno dla dni
  z epidemią i bez.
- **Egzamin liczono trzy razy** (wybór modeli w ciemno, cykl roczny w Ridge, usunięcie rabatu). Szczegóły w
  [Ograniczeniach](#ograniczenia).

### Modele

<!-- BEGIN:models -->
- **ARIMAX:** rząd (p,d,q) = (0, 1, 1), bez sezonowości, ze zmiennymi zewnętrznymi.
- **SARIMAX (wariant porównawczy):** rząd (0, 1, 1) z sezonowością (1, 0, 0, 7).
- **Ridge:** `alpha` = 30 na standaryzowanych zmiennych zewnętrznych plus cykl roczny (3 pary fal sin/cos dnia roku).
- **LightGBM (globalny):** `num_leaves` = 15, `n_estimators` = 200, `min_child_samples` = 20, `learning_rate` = 0,05, `random_state` = 42.
- **Ensemble:** zwykła średnia (po 1/3) oraz średnia ważona z wagami ARIMAX 0,25, LightGBM 0,10, Ridge 0,65.
<!-- END:models -->

## Wyniki na zbiorze egzaminacyjnym (wariant oracle)

<!-- BEGIN:summary -->
- Najlepszy baseline: **średnia 28 dni** (MAE 1 053,5, MAPE 14,4%).
- **Eksperyment 1, ARIMAX:** MAE 405,8, MAPE 5,1%, czyli błąd o 61,5% mniejszy niż najlepszego baseline'u. Wariant sezonowy SARIMAX: MAE 409,3.
- **Eksperyment 2, ensemble:** zwykła średnia MAE 389,6, ważona MAE 383,5. Najlepszy pojedynczy model: **Ridge** (MAE 374,0, MAPE 4,7%).
- Sześć modeli mieści się w MAE od 374,0 do 409,3 (różnica 9,4%), czyli błąd jest o 61,2%–64,5% mniejszy niż najlepszego baseline'u.
- Bias (plus = zawyża) sześciu modeli: od +220,7 do +300,6 szt. dziennie; wszystkie sześć modeli zawyża prognozę.
- MAPE Ridge (regresja z hamulcem) to 4,7%, a ARIMAX 5,1%.
<!-- END:summary -->

Wszystkie modele i oba warianty ensemble, posortowane rosnąco po MAE (szt. dziennie; pełna tabela: `outputs/comparison.csv`):

<!-- BEGIN:comparison -->
| Model | Grupa | MAE | RMSE | MAPE | Bias |
|---|---|---:|---:|---:|---:|
| Ridge | eksperyment 2 | 374,0 | 448,4 | 4,7% | +220,7 |
| Ensemble ważona średnia | eksperyment 2 | 383,5 | 459,3 | 4,8% | +244,4 |
| Ensemble zwykła średnia | eksperyment 2 | 389,6 | 467,0 | 4,9% | +263,4 |
| LightGBM | eksperyment 2 | 390,7 | 474,6 | 4,9% | +276,3 |
| ARIMAX | eksperyment 1 | 405,8 | 490,4 | 5,1% | +293,2 |
| SARIMAX | eksperyment 1 (porównawczy) | 409,3 | 495,3 | 5,1% | +300,6 |
| średnia 28 dni | baseline | 1 053,5 | 1 417,9 | 14,4% | +225,6 |
| naive | baseline | 2 460,1 | 2 766,5 | 27,0% | −2 386,2 |
| seasonal naive | baseline | 2 595,4 | 2 945,9 | 28,5% | −2 539,8 |
<!-- END:comparison -->

Osobno dla dni z epidemią (6 dni) i bez epidemii (22 dni):

<!-- BEGIN:epidemic -->
| Model | MAE epidemia | MAPE epidemia | Bias epidemia | MAE bez epidemii | MAPE bez epidemii | Bias bez epidemii |
|---|---:|---:|---:|---:|---:|---:|
| Ridge | 442,3 | 7,2% | +118,4 | 355,4 | 4,0% | +248,6 |
| Ensemble ważona średnia | 437,8 | 7,2% | +136,7 | 368,6 | 4,2% | +273,8 |
| Ensemble zwykła średnia | 423,7 | 6,9% | +136,7 | 380,3 | 4,3% | +298,0 |
| LightGBM | 382,5 | 6,3% | +88,0 | 393,0 | 4,5% | +327,7 |
| ARIMAX | 449,6 | 7,4% | +203,7 | 393,9 | 4,5% | +317,6 |
| SARIMAX | 450,3 | 7,4% | +206,1 | 398,1 | 4,5% | +326,4 |
| średnia 28 dni | 2 583,7 | 42,8% | +2 583,7 | 636,2 | 6,6% | −417,5 |
| naive | 372,8 | 6,1% | −28,2 | 3 029,3 | 32,8% | −3 029,3 |
| seasonal naive | 466,7 | 7,5% | −207,3 | 3 175,9 | 34,2% | −3 175,9 |
<!-- END:epidemic -->

Jak to czytać:

- **Wszystkie modele są znacznie lepsze od baseline'ów.** Naive i seasonal naive są dobre tylko w dniach epidemii, bo
  powtarzają poziom z ostatnich dni treningu. Po jej końcu mylą się najbardziej. Średnia z 28 dni „rozmywa” epidemię
  i dlatego jest najlepszym baseline'em.
- **Sześć modeli mieści się w wąskim paśmie błędu, a ich kolejność na jednym 28-dniowym egzaminie nie jest
  rozstrzygająca.** W walidacji na treningu kolejność była inna (tabela niżej).
- **Ensemble nie pobił najlepszego składnika (Ridge).** Ważona średnia jest lepsza od zwykłej, ale wagi dobrano na tych
  samych foldach, na których mierzymy walidację, więc jej przewaga w walidacji jest zawyżona. Składniki robią podobne
  błędy (korelacja błędów ARIMAX i Ridge powyżej 0,9), dlatego uśrednianie zyskuje niewiele.
- **Modele uzupełniają się w czasie:** LightGBM jest najlepszy w dniach epidemii, Ridge poza nią.
- **Wszystkie modele zawyżają prognozę na styczeń.** Wyjaśnia to sezonowość roczna: styczeń ma ujemny efekt
  (−122 szt./dzień), a modele znają ją słabo (Ridge dostał cykl roczny, ale na styczniu nie pomogło).

Błąd w walidacji kroczącej (na treningu) i na egzaminie:

<!-- BEGIN:validation -->
| Model | MAE w walidacji kroczącej (trening) | MAE na egzaminie |
|---|---:|---:|
| Ensemble ważona średnia | 298,7 | 383,5 |
| Ridge | 301,7 | 374,0 |
| Ensemble zwykła średnia | 310,1 | 389,6 |
| ARIMAX | 311,8 | 405,8 |
| LightGBM | 385,4 | 390,7 |
<!-- END:validation -->

## Prognozy dla trzech najlepiej sprzedających się produktów

Wybrane produkty (najwyższa średnia dzienna sprzedaż **w okresie treningowym**): <!-- BEGIN:bonus_products -->
**P0007**, **P0013**, **P0004**
<!-- END:bonus_products -->

Treść zadania nie precyzuje okresu, z którego liczyć średnią. Wybraliśmy trening, bo egzamin nie może wpływać na żadną
decyzję. Z całości danych trójka byłaby inna (P0007, P0004, P0009), ale miejsca 2–4 dzielą ułamki sztuki dziennie, więc to
praktycznie remis:

<!-- BEGIN:wybor_produktow -->
| Produkt | Średnia dzienna (trening) | Pozycja | Średnia dzienna (całość danych) | Pozycja |
|---|---:|---:|---:|---:|
| P0007 | 502,7 | 1 | 499,6 | 1 |
| P0013 | 497,4 | 2 | 495,7 | 4 |
| P0004 | 496,9 | 3 | 496,9 | 2 |
| P0009 | 495,7 | 4 | 495,8 | 3 |
| P0002 | 475,8 | 5 | 472,8 | 5 |
<!-- END:wybor_produktow -->

Prognoza produktu to suma prognoz globalnego LightGBM po 5 sklepach. Baseline'y liczone są z treningu danego produktu.

<!-- BEGIN:bonus -->
| Produkt | Model | MAE | RMSE | MAPE | Bias |
|---|---|---:|---:|---:|---:|
| P0007 | LightGBM | 101,5 | 121,0 | 26,8% | +70,3 |
| P0007 | średnia 28 dni | 113,0 | 132,5 | 31,9% | +43,1 |
| P0007 | seasonal naive | 135,9 | 179,4 | 27,6% | −128,0 |
| P0007 | naive | 149,4 | 190,0 | 30,7% | −142,9 |
| P0013 | LightGBM | 69,7 | 87,6 | 16,2% | +29,6 |
| P0013 | średnia 28 dni | 100,7 | 125,2 | 22,8% | −7,3 |
| P0013 | naive | 161,9 | 202,4 | 31,6% | −159,2 |
| P0013 | seasonal naive | 173,0 | 217,3 | 34,3% | −169,8 |
| P0004 | LightGBM | 60,7 | 77,9 | 15,6% | +0,6 |
| P0004 | średnia 28 dni | 103,6 | 131,0 | 27,4% | −11,7 |
| P0004 | naive | 190,5 | 218,8 | 37,4% | −175,6 |
| P0004 | seasonal naive | 207,8 | 234,1 | 42,5% | −189,6 |
<!-- END:bonus -->

- LightGBM ma najniższy błąd dla wszystkich trzech produktów, ale przewaga jest nierówna (mała dla P0007, gdzie poza
  epidemią lepsza jest średnia z 28 dni).
- Błąd względny jest większy niż dla sumy, bo pojedynczy produkt jest dużo bardziej zaszumiony niż suma 100 serii.
- Parametry LightGBM dobrano na sumie dziennej, a nie na produktach.


## Pasmo niepewności i scenariusze epidemii

**Pasmo 80%** to prognoza przesunięta o kwantyle 10% i 90% błędów z walidacji kroczącej (jedno pasmo o stałej szerokości na
model; baseline'y i SARIMAX bez pasma):

<!-- BEGIN:coverage -->
| Model | Pokrycie na egzaminie (deklarowane 80%) | Średnia szerokość pasma, szt. |
|---|---:|---:|
| ARIMAX | 75% | 1 046 |
| LightGBM | 75% | 1 217 |
| Ridge | 68% | 987 |
| Ensemble zwykła średnia | 71% | 1 038 |
| Ensemble ważona średnia | 71% | 1 013 |
<!-- END:coverage -->

Pasmo mieści rzeczywistą sprzedaż w mniejszej części dni niż deklarowane 80%. Różnica to jeden–dwa dni z 28, więc mieści się
w zwykłym rozrzucie tak małej próby (dni są ponadto zależne). Pasmo nie zna też biasu widocznego na egzaminie, więc leży
nieco za wysoko. Nie stroiliśmy go pod wynik egzaminu.

**Scenariusze epidemii** odpowiadają na pytanie „co, jeśli epidemia potrwa jeszcze N dni od 3.01.2024?”. Zmieniamy tylko flagę
epidemii w dniach prognozy (reszta zmiennych prawdziwa), a modele są te same. Dla N = 6, czyli faktycznej długości, prognozy
są identyczne z egzaminacyjnymi. Suma prognozy z 28 dni (szt.):

<!-- BEGIN:scenarios -->
| Model | N = 0 | N = 6 (faktycznie) | N = 28 | Zmiana sumy na każdy dzień epidemii |
|---|---:|---:|---:|---:|
| ARIMAX | 266 121 | 245 992 | 172 184 | −3 355 |
| Ensemble ważona średnia | 264 867 | 244 625 | 170 706 | −3 363 |
| Ensemble zwykła średnia | 265 623 | 245 157 | 171 122 | −3 375 |
| LightGBM | 266 637 | 245 519 | 171 106 | −3 412 |
| Ridge | 264 112 | 243 961 | 170 076 | −3 358 |
<!-- END:scenarios -->

Każdy dodatkowy dzień epidemii obniża sumę o podobną wielkość we wszystkich modelach. Niepewność co do długości epidemii jest
znacznie większa niż różnice między modelami, dlatego wynik z tabeli wyżej dotyczy sytuacji, w której długość epidemii znamy.

## Zmiany po pierwszym pomiarze na egzaminie

**Cykl roczny w Ridge (druga próba).** Kontrola (`python -m src.audit`) wykazała silną sezonowość roczną, której zmienne
zewnętrzne nie tłumaczą: średni błąd regresji treningowej waha się od −286 szt./dzień we wrześniu do +500 w sierpniu.
Ridge dostał więc fale sin/cos dnia roku (3 pary, `alpha` = 30, wybrane walidacją kroczącą). Ostatnia kolumna to kontrolny
styczeń rok wcześniej (uczenie do 2.01.2023, test 3–30.01.2023):

<!-- BEGIN:cykl_roczny -->
| Wariant | MAE w walidacji (trening) | MAE na egzaminie | MAE w kontrolnym styczniu |
|---|---:|---:|---:|
| Ridge bez cyklu rocznego | 380,6 | 372,0 | 288,5 |
| Ridge z cyklem rocznym (3 pary fal) | 301,7 | 374,0 | 286,7 |
<!-- END:cykl_roczny -->

W walidacji zysk jest duży (−20% MAE), ale pochodzi z miesięcy lipiec–grudzień. Na styczniu, czyli w okresie takim jak
egzamin, cykl roczny nic nie dał. W LightGBM go nie wdrożyliśmy: w walidacji pomagał, ale w foldzie grudniowym
i kontrolnym styczniu pogarszał wynik. Dane mają tylko dwa pełne cykle roczne, więc sezonowość jest szacowana z dwóch
powtórzeń.

**Usunięcie rabatu (trzecia próba).** Rabat jest funkcją promocji (bez promocji 0–10%, z promocją 10–25%), więc po
agregacji do dnia obie cechy są niemal tożsame (korelacja 0,97, VIF 16,1), a współczynnik rabatu był nieistotny
(p = 0,66) i miał nonsensowny ujemny znak. Usunęliśmy go ze wszystkich modeli i ponownie dobraliśmy parametry
walidacją kroczącą. **Żaden wybór ustawień się nie zmienił**, zmieniły się tylko wagi ensemble (0,25 / 0,10 / 0,65
zamiast 0,35 / 0,10 / 0,55). Wyniki na egzaminie przesunęły się o mniej niż 1,5% w obie strony (Ridge 377,2 → 374,0,
ARIMAX 400,2 → 405,8), więc powodem zmiany była poprawność doboru cech, a nie zysk w błędzie. Efekt promocji w regresji
wynosi teraz +2 122 szt.


## Ograniczenia

- **Wariant oracle.** Wyniki zakładają znajomość epidemii, promocji i pogody z góry. W praktyce znamy promocje, ale nie
  epidemię ani pogodę na kilka tygodni, więc realny błąd byłby większy. Scenariusze pokazują skalę tej niepewności.
- **Bias.** Wszystkie modele zawyżają prognozę na styczeń 2024 (styczeń ma ujemny efekt sezonowy). Ensemble go nie usuwa,
  bo składniki mylą się w tę samą stronę.
- **Egzamin liczony trzy razy.** Pierwszy raz na modelach dobranych w ciemno, drugi po dołożeniu cyklu rocznego do Ridge,
  trzeci po usunięciu rabatu. Każda decyzja zapadła wyłącznie na walidacji kroczącej na treningu, ale po zobaczeniu
  wcześniejszego wyniku egzaminacyjnego. Tabele w tym pliku pokazują więc **trzecią próbę**. Jedyne liczby z czystego,
  ślepego pomiaru to baseline'y. Przy kolejnej zmianie modeli trzeba wyznaczyć nowy okres testowy.
- **Jeden okres egzaminacyjny** (28 dni, z krótkim okresem epidemii na początku). Różnice rzędu kilku procent między
  modelami nie są rozstrzygające.
- **Pasmo niepewności** pochodzi z walidacji bez stycznia, ma stałą szerokość i nie zna biasu z egzaminu.
- **Prognozy dla produktów** pochodzą z modelu dobranego na sumie, bez własnej walidacji na poziomie produktu.


## Uruchomienie

```bash
python -m src.run_all                 # cały pipeline od zera (ok. 1 min) → outputs/ + tabele w dokumentach
streamlit run app/streamlit_app.py    # aplikacja na http://localhost:8501
python -m unittest discover -v        # testy (te wymagające danych są pomijane, gdy nie ma CSV)
```

Wyniki są deterministyczne (`random_state=42`). Każdy krok można uruchomić osobno (`python -m src.<nazwa>`), kolejność
i zależności widać w `src/run_all.py`. Parametry modeli w `config.py` wyszły z przeszukiwania siatek walidacją kroczącą,
które nie wchodzi do `run_all` (`python -m src.arimax`, `.ridge`, `.lgbm`, `.ensemble`); wynik wpisuje się ręcznie do
`config.py`.

## Aplikacja webowa

Aplikacja (Streamlit + Plotly) nie potrzebuje pliku z danymi: niczego nie uczy, tylko wczytuje gotowe pliki z `outputs/`
i wywołuje wspólną funkcję metryk.

- **Wykres:** rzeczywista sprzedaż, prognoza modelu, pasmo 80%, baseline (średnia z 28 dni) i dni z epidemią.
- **Metryki:** wybrany model obok baseline'ów, także osobno dla dni z epidemią i bez.
- **Fragmentatory:** widok (suma albo jeden z trzech produktów), model, zakres dat i scenariusz długości epidemii.
- W nagłówku widnieje oznaczenie **wariant oracle**.

## Kontener i wdrożenie

`Dockerfile` buduje obraz z kodem i gotowymi wynikami (bez pliku z danymi i bez bibliotek do uczenia,
`requirements-app.txt`):

```bash
docker build -t forecast-app .
docker run --rm -p 127.0.0.1:8501:8501 forecast-app     # http://localhost:8501
```

