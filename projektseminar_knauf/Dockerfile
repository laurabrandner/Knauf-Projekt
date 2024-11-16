# Basis-Image
FROM python:3.8-slim

# Arbeitsverzeichnis festlegen
WORKDIR /app

# Anforderungen kopieren und installieren
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App-Dateien kopieren
COPY app/ .

# Port für Streamlit freigeben
EXPOSE 8501

# Startbefehl für das Dashboard
CMD ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
