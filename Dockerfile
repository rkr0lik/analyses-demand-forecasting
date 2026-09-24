# Obraz z aplikacją Streamlit. Aplikacja czyta gotowe wyniki z outputs/, więc w obrazie nie ma pliku
# z danymi ani bibliotek do uczenia modeli (statsmodels, scikit-learn, lightgbm).
#
#   docker build -t forecast-app .
#   docker run --rm -p 127.0.0.1:8501:8501 forecast-app     # potem: http://localhost:8501
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Aplikacja działa na zwykłym użytkowniku, nie na root.
RUN useradd --create-home --uid 10001 appuser

WORKDIR /srv/forecast

# Zależności instalujemy przed kopiowaniem kodu, żeby ta warstwa zostawała w cache, dopóki nie zmieni się requirements-app.txt.
COPY requirements-app.txt ./
RUN pip install -r requirements-app.txt

# Kod i gotowe wyniki. To, co może się tu dostać, ogranicza .dockerignore.
COPY --chown=appuser:appuser config.py ./
COPY --chown=appuser:appuser src ./src
COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser .streamlit ./.streamlit
COPY --chown=appuser:appuser outputs ./outputs

USER appuser

EXPOSE 8501

# W obrazie slim nie ma curla, więc healthcheck robimy Pythonem.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4)"

# 0.0.0.0, żeby port był dostępny spoza kontenera (wystawia go docker run -p).
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
