# Knauf Projekt

## Beschreibung

Das Ziel dieses Projekts war die Entwicklung eines maschinellen Lernmodells, das die Bruchlast von Gipsfaserplatten basierend auf Sensordaten vorhersagen kann, sowie die Erstellung eines Dashboards zur Visualisierung der Ergebnisse und weiterer Analysen. Damit sollte ein innovativer Ansatz geschaffen werden, der die manuellen Qualitätsprüfungen, die alle zwei Stunden in Form von Labortests durchgeführt werden, ergänzt oder sogar ersetzt, um den Produktionsprozess effizienter, ressourcenschonender und nachhaltiger zu gestalten.

Im Rahmen des Projekts haben wir an drei zentralen Bereichen gearbeitet. Für diese Bereiche haben wir jeweils detaillierte Dokumentationen erstellt und hochgeladen:
1. EDA und Feature Engineering
- Dokumentation: Doku von Marius
- Notebook: Colab-Datei
2. ML-Modell
- Dokumentation: Doku von Bene
3. Dashboard und IT-Architektur
- Dokumentation: Doku von Laura

Des Weiteren wird eine Dokumentation zur Analyse der wirtschaftlichen Auswirkungen bereitgestellt. Dieser Abschnitt enthält eine detaillierte Kosten-Nutzen-Analyse, die die ökonomischen Vorteile und Effizienzgewinne des Projekts aufzeigt.

Die Anwendungen werden mithilfe von Docker bereitgestellt:

Voraussetzungen:

- **Docker Desktop** für Windows installiert

## Run Docker
```bash
docker build -t knauf_ml:latest .
docker run -p 8501:8501 knauf_ml:latest
docker-compose up --build
```