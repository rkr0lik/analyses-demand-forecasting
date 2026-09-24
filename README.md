# Prognoza dziennej sprzedaży na 28 dni

## O co chodzi

Sieć ma 5 sklepów i 20 produktów. Chcemy przewidzieć, ile sztuk sprzeda **łącznie każdego dnia** (kolumna `Units Sold`)
przez następne 4 tygodnie, czyli 28 dni.

Dane obejmują okres od 1.01.2022 do 30.01.2024. Wszystko do 2.01.2024 to materiał, na którym modele się uczą
(**trening**). Ostatnie 28 dni (3–30.01.2024) odkładamy na bok i używamy dopiero na sam koniec, jak **egzaminu**: modele
przewidują te dni, a my porównujemy prognozę z tym, co naprawdę się sprzedało.

Co ze sobą porównujemy:

- **Proste metody odniesienia** (w kodzie: *baseline'y*). To nie są modele, tylko zdroworozsądkowe zgadywanie:
  - *naive*: „będzie tak jak ostatniego dnia”,
  - *seasonal naive*: „będzie tak jak w ten sam dzień tygodnia tydzień wcześniej”,
  - *średnia 28 dni*: „będzie tyle, ile średnio w ostatnich 4 tygodniach”.

  Jeśli model nie jest od nich wyraźnie lepszy, nie warto go używać.
- **Eksperyment 1: ARIMAX.** Model statystyczny, który łączy dwie rzeczy: wpływ czynników z zewnątrz (epidemia, promocje,
  pogoda) i to, że dzisiejsza sprzedaż zwykle przypomina wczorajszą. SARIMAX to ten sam model z dodanym rytmem tygodniowym.
- **Eksperyment 2: połączenie modeli** (*ensemble*). Uśredniamy prognozy trzech różnych modeli w nadziei, że ich błędy
  częściowo się zniosą:
  - **ARIMAX** (jak wyżej),
  - **Ridge**: zwykła regresja liniowa („każdy czynnik dodaje albo odejmuje określoną liczbę sztuk”) z „hamulcem”, który nie
    pozwala jej za bardzo dopasować się do przypadkowych wahań w danych,
  - **LightGBM**: model uczenia maszynowego zbudowany z wielu drzew decyzyjnych, czyli ciągów prostych pytań w rodzaju „czy
    jest promocja?”, „czy pada?”. Prognozuje każdą parę sklep × produkt (100 osobnych serii), a wyniki sumujemy.

  Połączenie liczymy na dwa sposoby: jako zwykłą średnią i jako średnią ważoną, w której lepszy model ma więcej do powiedzenia.

Dodatkowo robimy osobne prognozy dla trzech najlepiej sprzedających się produktów. Wyniki można obejrzeć w aplikacji
webowej (Streamlit).

> **Ważne zastrzeżenie.** Na egzaminie modele dostają **prawdziwe** informacje o epidemii, promocjach i pogodzie w dniach,
> które prognozują. To tak, jakby znać z góry bezbłędną prognozę pogody na miesiąc. Wyniki pokazują więc, jak dobre mogą być
> modele w najlepszym razie, a w praktyce błąd byłby większy. W projekcie nazywamy to wariantem ***oracle*** („wyrocznia”).

## Najważniejsze wyniki

<!-- BEGIN:summary -->
- W dniach egzaminu sieć sprzedawała średnio **8 492 szt. dziennie**. Na tym tle trzeba czytać błędy poniżej.
- Najlepsza z prostych metod (średnia z ostatnich 28 dni) myli się średnio o **1 054 szt. dziennie** (14,4%).
- Nasze modele mylą się średnio o **374–409 szt. dziennie** (4,7%–5,1%), czyli o 61%–64% mniej niż najlepsza prosta metoda.
- **Eksperyment 1 (ARIMAX):** 406 szt. dziennie (5,1%). Wersja z rytmem tygodniowym (SARIMAX): 409 szt.
- **Eksperyment 2 (połączenie modeli):** zwykła średnia 390 szt., średnia ważona 383 szt. Najlepszy okazał się jednak pojedynczy model **Ridge** (374 szt., 4,7%).
- Najlepszy i najsłabszy model dzieli tylko 9%, więc jeden egzamin nie wystarcza, żeby uczciwie wskazać zwycięzcę.
- Wszystkie modele **zawyżają** prognozę na styczeń 2024: średnio o 221–301 szt. dziennie za dużo.
<!-- END:summary -->

Szczegóły w części [Wyniki na egzaminie](#wyniki-na-egzaminie).

## Słowniczek

Jak mierzymy błąd prognozy (wszystkie miary liczy jedna wspólna funkcja w `src/metrics.py`):

| Miara | Co znaczy | Lepiej, gdy |
|---|---|---|
| **MAE** | o ile sztuk prognoza myli się przeciętnie w ciągu dnia, bez względu na to, czy w górę, czy w dół. Główna miara w projekcie | mniej |
| **RMSE** | podobnie jak MAE, ale mocniej karze duże pomyłki: jeden dzień z pomyłką o 1000 szt. waży więcej niż dziesięć dni po 100 szt. | mniej |
| **MAPE** | przeciętna pomyłka w procentach rzeczywistej sprzedaży | mniej |
| **Bias** | czy model systematycznie przesadza w jedną stronę. Plus: prognozy są średnio za wysokie, minus: za niskie | bliżej zera |

Inne pojęcia, które pojawiają się niżej:

- **Walidacja krocząca** (*walk-forward*): próbne egzaminy na danych treningowych. Model uczy się na danych do pewnego dnia,
  prognozuje kolejne 28 dni, potem okno przesuwa się o 28 dni dalej i wszystko się powtarza. Na tej podstawie wybieramy
  ustawienia modeli, żeby nie zaglądać do prawdziwego egzaminu.
- **Wyciek danych**: sytuacja, w której model przypadkiem dostaje informację, jakiej w chwili prognozowania by nie miał
  (np. przyszłą sprzedaż). Wyniki wyglądają wtedy świetnie, ale są oszukane.
- **R²**: jaką część wahań sprzedaży da się wytłumaczyć danym czynnikiem. 0 oznacza nic, 1 oznacza wszystko.
- **Korelacja**: jak mocno dwie wielkości zmieniają się razem. Wartość bliska 1 lub −1 oznacza, że to praktycznie ta sama
  informacja.
- **Pasmo niepewności 80%**: przedział wokół prognozy, w którym rzeczywista sprzedaż powinna się znaleźć w 8 dniach na 10.

## Wymagania i instalacja

Projekt działa na Pythonie 3.12. Biblioteki z wersjami są w `requirements.txt` (pandas, numpy, scikit-learn, statsmodels,
lightgbm, streamlit, plotly, matplotlib).

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Na Debianie i Ubuntu może być potrzebny dodatkowy pakiet: `sudo apt install python3-venv`.

## Dane

Pliku `demand_forecasting.csv` **nie ma w repozytorium**. Można go pobrać z
[Kaggle](https://www.kaggle.com/datasets/raminhuseyn/demand-forecasting-dataset) i wrzucić do głównego katalogu projektu
(ścieżka jest w `config.py`).

Bez tego pliku:

- nie da się od nowa policzyć modeli, a testy, które potrzebują danych, są pomijane,
- aplikacja i testy wyników działają normalnie, bo korzystają z gotowych plików w `outputs/` (są tam tylko sumy
  i prognozy, bez surowych danych).

## Co wpływa na sprzedaż

Zanim zbudowaliśmy modele, sprawdziliśmy, co w ogóle rusza sprzedażą. Liczby poniżej policzył `python -m src.explore`
wyłącznie na **danych treningowych** (732 dni, do 2.01.2024).

| Czynnik | Co wyszło |
|---|---|
| **Epidemia** | w dniach epidemii sieć sprzedaje średnio 6 084 zamiast 9 599 szt. dziennie, czyli o 36,6% mniej. To zdecydowanie najważniejszy czynnik: sama informacja „czy jest epidemia” tłumaczy ok. 81% wahań dziennej sprzedaży (R² = 0,809) |
| **Promocje** | gdyby w promocji były wszystkie produkty naraz, sprzedaż byłaby wyższa o ok. 2 122 szt. dziennie niż bez żadnej promocji. Zwykle w promocji jest tylko część produktów, więc efekt jest proporcjonalnie mniejszy |
| **Pogoda** | w porównaniu z dniem, gdy we wszystkich sklepach jest pochmurno: słońce daje +774 szt., deszcz −721 szt., śnieg −1 355 szt. |
| **Dzień tygodnia** | dni tygodnia prawie się nie różnią (o 1,9%, a po pominięciu epidemii o 2,8%). W tych danych nie ma czegoś takiego jak „mocna sobota”. Potwierdza to SARIMAX: z dodanym rytmem tygodniowym wypadł w próbnych egzaminach trochę gorzej niż ARIMAX bez niego |
| **Wszystko razem** | epidemia, promocje i pogoda razem tłumaczą ok. 91% wahań sprzedaży (R² = 0,906). Uwaga: to dopasowanie do danych, które model już widział, a nie sprawdzian prognozy |

### Czego używają modele

| Informacja | W jakiej postaci | Które modele |
|---|---|---|
| Epidemia | 0 lub 1 dla każdego dnia (w danych epidemia jest ta sama dla wszystkich sklepów i produktów) | wszystkie |
| Promocje | jaka część produktów jest danego dnia w promocji; LightGBM widzi promocję każdego produktu osobno | wszystkie |
| Pogoda | w jakiej części sklepów jest słońce, deszcz, śnieg (pochmurno to punkt odniesienia, bo cztery udziały zawsze sumowałyby się do 1); LightGBM widzi pogodę każdego sklepu osobno | wszystkie |
| Sklep, produkt, kategoria, region | który to sklep i produkt | tylko LightGBM |
| Wcześniejsza sprzedaż | sprzedaż sprzed 28 dni oraz średnie z 7 i 28 dni liczone od tamtego dnia | tylko LightGBM |

Proste metody odniesienia korzystają tylko z historii sprzedaży z okresu treningowego.

Dlaczego LightGBM patrzy na sprzedaż sprzed **co najmniej 28 dni**? Bo prognozujemy 28 dni do przodu. Gdyby model
potrzebował wczorajszej sprzedaży, musiałby w dniach prognozy używać własnych wcześniejszych prognoz zamiast prawdziwych
liczb, a błędy by się wtedy nawarstwiały.

Jak wspomnieliśmy wyżej, na egzaminie modele znają prawdziwą epidemię, promocje i pogodę w prognozowanych dniach. Promocje
firma planuje z wyprzedzeniem, więc to realistyczne. Pogody i epidemii na kilka tygodni naprzód nikt nie zna, dlatego
w aplikacji można sprawdzić różne scenariusze długości epidemii.

### Czego nie używamy i dlaczego

| Kolumna | Dlaczego jej nie używamy |
|---|---|
| `Demand`, `Inventory Level`, `Units Ordered` | to **wyciek danych**: popytu, stanu magazynu i zamówień nie znamy z wyprzedzeniem, a są mocno powiązane ze sprzedażą. Test sprawdza, że żaden model ich nie dostaje |
| `Competitor Pricing` | cen konkurencji nie znamy z góry |
| `Discount` (rabat) | powtarza tę samą informację co promocja (korelacja 0,97) i nic nie wnosi do modelu. Usunęliśmy go, patrz [Zmiany po pierwszym pomiarze](#zmiany-po-pierwszym-pomiarze-na-egzaminie) |
| `Price` | cena idzie w górę razem ze sprzedażą (w epidemii spada), czyli to raczej sprzedaż wpływa na cenę niż odwrotnie. Prawdziwa cena z okresu egzaminu mogłaby więc „podpowiadać” modelowi wynik. Modeli z ceną nie sprawdzaliśmy |
| `Seasonality` | pora roku wynika wprost z miesiąca. Zamiast niej Ridge dostał gładką krzywą roczną (fale sin/cos) |
| dzień tygodnia | rytmu tygodniowego w danych nie ma (różnice poniżej 3%) |

## Jak sprawdzaliśmy, że wynik jest uczciwy

- **Podział danych.** Trening: 1.01.2022 – 2.01.2024 (732 dni). Egzamin: 3–30.01.2024 (28 dni). Pierwsze 6 dni egzaminu
  (3–8.01) przypada jeszcze na epidemię, pozostałe 22 już nie.
- **Ustawienia modeli wybierały próbne egzaminy, a nie prawdziwy.** Zrobiliśmy 6 próbnych egzaminów po 28 dni na końcu
  okresu treningowego (19.07.2023 – 2.01.2024). Przed każdym model uczy się na wszystkich wcześniejszych danych.
  Tak samo dobraliśmy wagi w średniej ważonej.
- **Proste metody widzą tylko trening.** *Naive* powtarza 28 razy ostatni dzień treningu, *seasonal naive* powtarza
  4 razy ostatni tydzień, a *średnia* to średnia z ostatnich 28 dni treningu.
- **Wyciek danych wyłapują testy.** Testy mnożą sprzedaż z okresu egzaminu (a w próbnych egzaminach: z dni po danym
  oknie) przez 1000 i sprawdzają, że nie zmienia się ani żadna informacja podawana modelom, ani żadna prognoza.
  Gdyby model „podglądał” przyszłość, test by to wychwycił.
- **Błąd liczymy łącznie, a także osobno dla dni z epidemią i bez niej.**
- **Egzamin liczyliśmy trzy razy.** Pierwszy raz na modelach dobranych w ciemno, potem po dwóch poprawkach. Dlaczego to
  ważne, piszemy w [Ograniczeniach](#ograniczenia).

### Ustawienia modeli

<!-- BEGIN:models -->
- **ARIMAX:** ustawienie (p, d, q) = (0, 1, 1). W praktyce model przewiduje zmianę sprzedaży względem poprzedniego dnia, poprawia się o swoją wczorajszą pomyłkę i dodaje wpływ epidemii, promocji i pogody. Nie ma rytmu tygodniowego.
- **SARIMAX (dla porównania):** to samo ustawienie (0, 1, 1) plus rytm tygodniowy (1, 0, 0, 7), czyli sprzedaż zależy też od tego samego dnia tydzień wcześniej.
- **Ridge:** siła „hamulca” `alpha` = 30. Czynniki są sprowadzone do wspólnej skali. Do tego dochodzi cykl roczny, czyli gładka krzywa powtarzająca się co rok (3 pary fal sin/cos).
- **LightGBM:** 200 drzew decyzyjnych (`n_estimators`), każde z najwyżej 15 końcowymi odpowiedziami (`num_leaves`), a każda odpowiedź opiera się na co najmniej 20 przykładach (`min_child_samples`). Tempo uczenia 0,05 (`learning_rate`). Stałe ziarno losowości (`random_state` = 42) sprawia, że każde uruchomienie daje ten sam wynik.
- **Połączenie modeli (ensemble):** zwykła średnia (każdy model po 1/3) oraz średnia ważona: Ridge 0,65, ARIMAX 0,25, LightGBM 0,10.
<!-- END:models -->

## Wyniki na egzaminie

Wszystkie modele i proste metody, od najlepszej do najsłabszej (błędy w sztukach dziennie, pełna tabela w
`outputs/comparison.csv`):

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

Ta sama tabela z podziałem na dni z epidemią (6 dni) i bez niej (22 dni):

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

Co z tego wynika:

- **Każdy model jest dużo lepszy od prostych metod.** *Naive* i *seasonal naive* dobrze radzą sobie tylko w dniach
  epidemii, bo powtarzają niską sprzedaż z końca treningu. Kiedy epidemia się kończy, mylą się najbardziej. Średnia
  z 28 dni miesza dni z epidemią i bez, więc wypada najlepiej z trzech prostych metod.
- **Modele wypadły bardzo podobnie i z jednego 28-dniowego egzaminu nie da się ustalić, który jest najlepszy.**
  W próbnych egzaminach kolejność była inna (tabela niżej).
- **Połączenie modeli nie pokonało najlepszego z nich (Ridge).** Średnia ważona jest lepsza od zwykłej, ale wagi dobieraliśmy
  na tych samych próbnych egzaminach, na których potem mierzymy wynik, więc jej przewaga w próbach jest trochę
  przesadzona. Uśrednianie pomaga, gdy modele mylą się w różny sposób. Tu mylą się bardzo podobnie (korelacja błędów
  ARIMAX i Ridge przekracza 0,9), więc niewiele to daje.
- **Modele sprawdzają się w różnych momentach:** LightGBM najlepiej radzi sobie w dniach epidemii, a Ridge po niej.
- **Wszystkie modele przeszacowują styczeń.** Styczeń jest w tych danych słabszym miesiącem (ok. 122 szt. dziennie mniej,
  niż wynikałoby z epidemii, promocji i pogody), a modele słabo to uwzględniają. Ridge dostał krzywą roczną, ale na
  styczeń nie pomogła.

Porównanie błędu na próbnych egzaminach i na prawdziwym (MAE, szt. dziennie):

<!-- BEGIN:validation -->
| Model | MAE w walidacji kroczącej (trening) | MAE na egzaminie |
|---|---:|---:|
| Ensemble ważona średnia | 298,7 | 383,5 |
| Ridge | 301,7 | 374,0 |
| Ensemble zwykła średnia | 310,1 | 389,6 |
| ARIMAX | 311,8 | 405,8 |
| LightGBM | 385,4 | 390,7 |
<!-- END:validation -->

Na próbnych egzaminach najlepsza była średnia ważona, na prawdziwym Ridge. To kolejny znak, że różnice między modelami są
zbyt małe, żeby wskazać zwycięzcę.

## Prognozy dla trzech najlepiej sprzedających się produktów

Wybrane produkty (najwyższa średnia dzienna sprzedaż **w okresie treningowym**): <!-- BEGIN:bonus_products -->
**P0007**, **P0013**, **P0004**
<!-- END:bonus_products -->

Treść zadania nie mówi, z jakiego okresu liczyć średnią. Wybraliśmy okres treningowy, bo nic z egzaminu nie może wpływać
na nasze decyzje. Gdyby liczyć z całych danych, trójka byłaby inna (P0007, P0004, P0009), ale miejsca od 2. do 4. dzielą
ułamki sztuki dziennie, więc to praktycznie remis:

<!-- BEGIN:wybor_produktow -->
| Produkt | Średnia dzienna (trening) | Pozycja | Średnia dzienna (całość danych) | Pozycja |
|---|---:|---:|---:|---:|
| P0007 | 502,7 | 1 | 499,6 | 1 |
| P0013 | 497,4 | 2 | 495,7 | 4 |
| P0004 | 496,9 | 3 | 496,9 | 2 |
| P0009 | 495,7 | 4 | 495,8 | 3 |
| P0002 | 475,8 | 5 | 472,8 | 5 |
<!-- END:wybor_produktow -->

Prognoza dla produktu to suma prognoz LightGBM z 5 sklepów. Proste metody liczymy z historii danego produktu w okresie
treningowym.

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

- LightGBM ma najmniejszy błąd dla każdego z trzech produktów, ale przewaga jest różna. Dla P0007 jest niewielka, a po
  epidemii średnia z 28 dni radzi sobie tam nawet lepiej.
- Błąd w procentach jest tu dużo większy niż dla całej sieci. Sprzedaż jednego produktu mocno skacze z dnia na dzień,
  a w sumie 100 serii te skoki w dużej części się znoszą.
- Ustawienia LightGBM dobieraliśmy pod łączną sprzedaż, a nie pod pojedyncze produkty.

## Jak bardzo można ufać prognozie

### Pasmo niepewności

Wokół prognozy rysujemy **pasmo 80%**: przedział, w którym rzeczywista sprzedaż powinna się znaleźć w 8 dniach na 10.
Szerokość pasma wzięliśmy z pomyłek na próbnych egzaminach: pomijamy 10% najbardziej zaniżonych i 10% najbardziej
zawyżonych prognoz, a resztę wyznacza pasmo. Każdy model ma pasmo stałej szerokości (proste metody i SARIMAX go nie mają).

<!-- BEGIN:coverage -->
| Model | Pokrycie na egzaminie (deklarowane 80%) | Średnia szerokość pasma, szt. |
|---|---:|---:|
| ARIMAX | 75% | 1 046 |
| LightGBM | 75% | 1 217 |
| Ridge | 68% | 987 |
| Ensemble zwykła średnia | 71% | 1 038 |
| Ensemble ważona średnia | 71% | 1 013 |
<!-- END:coverage -->

Rzeczywista sprzedaż mieściła się w paśmie trochę rzadziej niż w 80% dni. Różnica to jeden–dwa dni z 28, więc przy tak
krótkim okresie może to być przypadek (tym bardziej że sąsiednie dni są do siebie podobne). Pasmo nie wie też, że modele
przeszacowują styczeń, więc leży nieco za wysoko. Nie poprawialiśmy go pod wynik egzaminu.

### Co, jeśli epidemia potrwa dłużej

Scenariusze odpowiadają na pytanie „co, jeśli od 3.01.2024 epidemia potrwa jeszcze N dni?”. Zmieniamy tylko informację
o epidemii w prognozowanych dniach, a wszystko inne (w tym same modele) zostaje bez zmian. Dla N = 6, czyli tyle, ile
epidemia trwała naprawdę, prognozy są identyczne z egzaminacyjnymi. Łączna prognozowana sprzedaż z 28 dni (szt.):

<!-- BEGIN:scenarios -->
| Model | N = 0 | N = 6 (faktycznie) | N = 28 | Zmiana sumy na każdy dzień epidemii |
|---|---:|---:|---:|---:|
| ARIMAX | 266 121 | 245 992 | 172 184 | −3 355 |
| Ensemble ważona średnia | 264 867 | 244 625 | 170 706 | −3 363 |
| Ensemble zwykła średnia | 265 623 | 245 157 | 171 122 | −3 375 |
| LightGBM | 266 637 | 245 519 | 171 106 | −3 412 |
| Ridge | 264 112 | 243 961 | 170 076 | −3 358 |
<!-- END:scenarios -->

Każdy dodatkowy dzień epidemii obniża łączną sprzedaż mniej więcej o tyle samo we wszystkich modelach. Niepewność co do
tego, jak długo potrwa epidemia, jest dużo większa niż różnice między modelami. Wyniki egzaminu dotyczą więc sytuacji,
w której długość epidemii znamy.

## Zmiany po pierwszym pomiarze na egzaminie

**Krzywa roczna w Ridge (druga próba).** Kontrola (`python -m src.audit`) pokazała, że sprzedaż zmienia się w ciągu roku
w sposób, którego epidemia, promocje i pogoda nie tłumaczą. Prosta regresja na tych czynnikach myliła się w zależności od
miesiąca przeciętnie o 286 szt. dziennie w jedną stronę (wrzesień) do 500 szt. dziennie w drugą (sierpień). Dlatego Ridge dostał gładką krzywą powtarzającą
się co rok (3 pary fal sin/cos, `alpha` = 30, wybrane na próbnych egzaminach). Ostatnia kolumna to dodatkowy sprawdzian na
styczniu rok wcześniej (nauka do 2.01.2023, prognoza na 3–30.01.2023):

<!-- BEGIN:cykl_roczny -->
| Wariant | MAE w walidacji (trening) | MAE na egzaminie | MAE w kontrolnym styczniu |
|---|---:|---:|---:|
| Ridge bez cyklu rocznego | 380,6 | 372,0 | 288,5 |
| Ridge z cyklem rocznym (3 pary fal) | 301,7 | 374,0 | 286,7 |
<!-- END:cykl_roczny -->

Na próbnych egzaminach zysk jest duży (błąd mniejszy o 20%), ale bierze się z miesięcy od lipca do grudnia. Na styczniu,
czyli w okresie takim jak egzamin, krzywa roczna nic nie dała. W LightGBM jej nie dodaliśmy: na próbnych egzaminach
ogólnie pomagała, ale w grudniu i w kontrolnym styczniu pogarszała wynik. Dane obejmują tylko dwa pełne lata, więc
wzór roczny jest oszacowany zaledwie z dwóch powtórzeń.

**Usunięcie rabatu (trzecia próba).** Rabat wynika wprost z promocji (bez promocji wynosi 0–10%, z promocją 10–25%), więc po
zsumowaniu do poziomu dnia to prawie ta sama informacja (korelacja 0,97). Model nie potrafił rozdzielić wpływu jednego
i drugiego, a wpływ rabatu wychodził nieistotny (p = 0,66) i do tego ujemny, co nie ma sensu. Usunęliśmy rabat ze
wszystkich modeli i od nowa dobraliśmy ustawienia na próbnych egzaminach. **Żadne ustawienie się nie zmieniło**, zmieniły
się tylko wagi w średniej ważonej (0,25 / 0,10 / 0,65 zamiast 0,35 / 0,10 / 0,55). Wyniki egzaminu przesunęły się o mniej
niż 1,5% w jedną lub drugą stronę (Ridge 377,2 → 374,0, ARIMAX 400,2 → 405,8). Rabat usunęliśmy więc dlatego, że tak jest
poprawnie, a nie dlatego, że poprawiło to wynik. Wpływ promocji wynosi teraz +2 122 szt. dziennie.

## Ograniczenia

- **Modele znały przyszłe epidemię, promocje i pogodę.** W praktyce znamy z góry promocje, ale nie epidemię ani pogodę na
  kilka tygodni, więc prawdziwy błąd byłby większy. Scenariusze epidemii pokazują, o ile.
- **Modele przeszacowują styczeń 2024.** Połączenie modeli tego nie naprawia, bo wszystkie mylą się w tę samą stronę.
- **Egzamin był liczony trzy razy.** Pierwszy raz na modelach dobranych w ciemno, drugi po dodaniu krzywej rocznej do
  Ridge, trzeci po usunięciu rabatu. Każdą decyzję podjęliśmy na podstawie próbnych egzaminów, ale już po zobaczeniu
  wcześniejszego wyniku. To trochę jak z uczniem, który zna już pytania z egzaminu: nawet jeśli uczy się uczciwie, trudno
  wykluczyć, że podświadomie przygotował się pod te konkretne pytania. Tabele pokazują więc **trzecie podejście**, a jedyne
  wyniki z pierwszego, „ślepego” podejścia to wyniki prostych metod. Przy kolejnej zmianie modeli trzeba sprawdzać je na
  nowym okresie.
- **Był tylko jeden egzamin**: 28 dni, w tym kilka dni epidemii na początku. Różnice między modelami rzędu kilku procent
  nie przesądzają, który jest lepszy.
- **Pasmo niepewności** powstało z próbnych egzaminów, w których nie było stycznia, ma stałą szerokość i nie uwzględnia
  tego, że modele przeszacowują styczeń.
- **Prognozy dla produktów** pochodzą z modelu dobranego pod łączną sprzedaż i nie przeszły osobnych próbnych egzaminów
  na poziomie produktu.

## Uruchomienie

```bash
python -m src.run_all                 # wszystko od zera (ok. 1 min) → pliki w outputs/ + tabele w tym pliku
streamlit run app/streamlit_app.py    # aplikacja na http://localhost:8501
python -m unittest discover -v        # testy (te, które potrzebują danych, są pomijane, gdy nie ma pliku CSV)
```

Wyniki są za każdym razem takie same (stałe ziarno losowości `random_state=42`). Każdy krok można uruchomić osobno
(`python -m src.<nazwa>`), a ich kolejność widać w `src/run_all.py`.

Ustawienia modeli w `config.py` wybrało przeszukiwanie wielu kombinacji na próbnych egzaminach. Nie jest ono częścią
`run_all`, bo trwa dłużej. Uruchamia się je osobno (`python -m src.arimax`, `.ridge`, `.lgbm`, `.ensemble`), a wynik
wpisuje ręcznie do `config.py`.

## Aplikacja webowa

Aplikacja (Streamlit + Plotly) nie potrzebuje pliku z danymi. Niczego nie uczy, tylko wczytuje gotowe wyniki z `outputs/`.

- **Wykres:** rzeczywista sprzedaż, prognoza wybranego modelu, pasmo 80%, średnia z 28 dni dla porównania i zaznaczone dni
  epidemii.
- **Błędy:** wybrany model obok prostych metod, także osobno dla dni z epidemią i bez niej.
- **Filtry:** cała sieć albo jeden z trzech produktów, model, zakres dat i scenariusz długości epidemii.
- W nagłówku jest przypomnienie, że to **wariant oracle** (modele znały przyszłą epidemię, promocje i pogodę).

## Kontener i wdrożenie

`Dockerfile` buduje obraz z kodem i gotowymi wynikami. Nie ma w nim pliku z danymi ani bibliotek potrzebnych do uczenia
modeli (tylko to, co w `requirements-app.txt`):

```bash
docker build -t forecast-app .
docker run --rm -p 127.0.0.1:8501:8501 forecast-app     # http://localhost:8501
```
