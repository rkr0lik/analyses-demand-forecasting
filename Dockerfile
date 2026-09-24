# Obraz z aplikacją Streamlit (prognoza dziennej sprzedaży na 28 dni).
# Aplikacja niczego nie uczy: czyta gotowe wyniki z outputs/, więc nie potrzebuje pliku z danymi
# ani bibliotek do uczenia modeli (statsmodels, scikit-learn, lightgbm) i obraz jest dzięki temu mały.
#
#   docker build -t forecast-app .
#   docker run --rm -p 127.0.0.1:8501:8501 forecast-app     # potem: http://localhost:8501
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Użytkownik bez uprawnień administratora.
RUN useradd --create-home --uid 10001 appuser

WORKDIR /srv/forecast

# Zależności najpierw: dzięki temu ta warstwa jest w cache, dopóki requirements-app.txt się nie zmieni.
COPY requirements-app.txt ./
RUN pip install -r requirements-app.txt

# Kod aplikacji i gotowe wyniki (lista plików ogranicza się do tego, co dopuszcza .dockerignore).
COPY --chown=appuser:appuser config.py ./
COPY --chown=appuser:appuser src ./src
COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser .streamlit ./.streamlit
COPY --chown=appuser:appuser outputs ./outputs

USER appuser

EXPOSE 8501

# Curl nie ma w obrazie slim, więc zdrowie sprawdzamy Pythonem.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4)"

# W kontenerze nasłuchujemy na wszystkich interfejsach; na zewnątrz port wystawia docker run / compose.
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
