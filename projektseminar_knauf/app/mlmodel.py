import sqlite3
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import randint
import joblib
import os
import numpy as np

def preprocess_data(data, imputer=None, feature_names=None):
    # Konvertieren der 'timeutc' Spalte in einen Unix-Zeitstempel
    if 'timeutc' in data.columns:
        data['timeutc'] = pd.to_datetime(data['timeutc']).astype('int64') // 10**9

    # Drop 'Breaking_Load' und 'timeutc', falls vorhanden, da sie nicht für die Feature-Verarbeitung benötigt werden
    additional_columns = ['timeutc', 'Breaking_Load']
    additional_data = data[additional_columns].copy() if all(col in data.columns for col in additional_columns) else None
    data = data.drop(columns=additional_columns, errors='ignore')

    # Sicherstellen, dass alle Feature-Spalten vorhanden sind
    if feature_names is not None:
        data = data.reindex(columns=feature_names)

    # Imputation (falls noch kein Imputer übergeben wurde, wird einer erstellt)
    if imputer is None:
        imputer = KNNImputer(n_neighbors=5)
        data_imputed = imputer.fit_transform(data)
    else:
        data_imputed = imputer.transform(data)

    # Rückgabe der imputierten Daten als DataFrame mit den Originalspalten
    return pd.DataFrame(data_imputed, columns=data.columns), imputer, additional_data

def train_and_evaluate(data, feature_names=None):
    # Sicherstellen, dass `Breaking_Load` als Zielvariable vorhanden ist
    if 'Breaking_Load' not in data.columns:
        raise ValueError("The expected target column 'Breaking_Load' was not found in the data.")

    # Trennen der Zielspalte und Vorverarbeitung der Features
    y = data['Breaking_Load'].values
    data = data.drop(columns=['Breaking_Load'])  # Zielspalte aus den Feature-Daten entfernen

    # Vorverarbeitung der Features
    data, imputer, additional_data = preprocess_data(data, feature_names=feature_names)

    # Skalierung der Features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)

    # Aufteilen der Daten in Trainings- und Testsets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Definition der Hyperparameter-Suchstrategie
    param_dist = {
        'n_estimators': randint(50, 200),
        'max_depth': randint(10, 50),
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 10),
        'max_features': ['sqrt', 'log2', None]
    }

    print("Starte Randomized Search für Hyperparameter-Tuning...")
    rand_search = RandomizedSearchCV(
        RandomForestRegressor(random_state=42),
        param_distributions=param_dist,
        n_iter=20,
        cv=5,
        scoring='neg_mean_squared_error',
        random_state=42,
        verbose=1
    )
    rand_search.fit(X_train, y_train)
    best_rf_model = rand_search.best_estimator_

    print("Modelltraining abgeschlossen. Bewertung des Modells...")

    # Modellbewertung
    y_rf_pred = best_rf_model.predict(X_test)
    mse_rf = mean_squared_error(y_test, y_rf_pred)
    r2_rf = r2_score(y_test, y_rf_pred)
    print(f"Random Forest - MSE: {mse_rf:.2f}, R²: {r2_rf:.2f}")

    print("Speichere Modell und zugehörige Komponenten...")
    # Save model, imputer, scaler, and feature names
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    try:
        
        joblib.dump(imputer, os.path.join(os.path.dirname(__file__), 'imputer.joblib'))
        joblib.dump(scaler, os.path.join(os.path.dirname(__file__), 'scaler.joblib'))
        joblib.dump(list(data.columns), os.path.join(os.path.dirname(__file__), 'feature_names.joblib'))
        joblib.dump(best_rf_model, os.path.join(os.path.dirname(__file__), 'rf_model.joblib'))
        print("Model files saved successfully.")
    except Exception as e:
        print("Error saving model files:", e)
        return

    return best_rf_model, imputer, scaler, data.columns, additional_data

def load_data_from_sqlite(db_path, query):
    # Verbindung zur SQLite-Datenbank herstellen
    conn = sqlite3.connect(db_path)
    # SQL-Abfrage ausführen und die Daten in ein Pandas DataFrame laden
    data = pd.read_sql_query(query, conn)
    # Verbindung schließen
    conn.close()
    return data

def get_latest_timestamp(data):
    if 'timeutc' in data.columns:
        return data['timeutc'].max()
    else:
        return None

# Initiales Laden der Daten und Training des Modells
db_path = os.path.join(os.path.dirname(__file__), "database", "mldb.db")
query = 'SELECT * FROM top_30_features'
data = load_data_from_sqlite(db_path, query)

# Initiales Training des Modells
rf_model, imputer, scaler, X_columns, additional_data = train_and_evaluate(data)

# Letzten Zeitstempel speichern, um neue Daten zu erkennen
latest_timestamp = get_latest_timestamp(data)

# ================================
# Retraining-Block (vorerst auskommentiert)
# ================================
# while True:
#     time.sleep(30)  # Warte 30 Sekunden zwischen den Überprüfungen
#     print("Prüfe auf neue Daten in der Datenbank...")
#
#     try:
#         new_data = load_data_from_sqlite(db_path, query)
#
#         if new_data.empty:
#             print("Keine neuen Daten gefunden.")
#             continue
#
#         new_latest_timestamp = get_latest_timestamp(new_data)
#
#         if new_latest_timestamp is None:
#             print("Keine 'timeutc'-Spalte in den neuen Daten gefunden.")
#             continue
#
#         if new_latest_timestamp > latest_timestamp:
#             print("Neue Daten gefunden. Retraining wird durchgeführt...")
#             latest_timestamp = new_latest_timestamp
#             rf_model, imputer, scaler, X_columns, additional_data = train_and_evaluate(new_data, feature_names=X_columns)
#         else:
#             print("Keine neuen Daten zum Retrainieren.")
#     except Exception as e:
#         print(f"Fehler beim Abrufen der Daten: {e}")
