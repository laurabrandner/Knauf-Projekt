import streamlit as st 
import pandas as pd
import sqlite3  
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import re  
import shap  
import mlmodel
import joblib
import matplotlib.pyplot as plt
import functools
import os
import numpy as np

st.set_page_config(layout="wide")

# Konstanten für minimale Bruchlasten und Benchmark-Werte für die vier Plattendicken
MIN_BREAKING_LOADS = {9.5: 600, 12.5: 900, 15: 1125, 18: 1500}
BENCHMARK_VALUES = {9.5: 639.85, 12.5: 993.08, 15: 1514.44, 18: 1982.00}

# Farbkonstanten
RED_COLOR = '#d62728'  # Konsistentes Rot für alle roten Elemente

# Daten aus der SQLite-Datenbank laden und vorbereiten
def load_data():
    conn = sqlite3.connect(os.path.join(os.path.dirname(__file__), "database", "mldb.db"))
    query = "SELECT * FROM testdaten_vorhersagen_30features"
    df = pd.read_sql_query(query, conn).rename(columns={
        'timeutc': 'timestamp',
        'Thickness board': 'thickness_board',
        'Product_group': 'product_group',
        'Predicted': 'predicted_breaking_load',
        'Actual':'actual_breaking_load'
    })
    conn.close()
    
    # "timestamp" in datetime umwandeln und zusätzliche Spalten berechnen    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['min_breaking_load'] = df['thickness_board'].map(MIN_BREAKING_LOADS)
    df['percentage_deviation'] = ((df['predicted_breaking_load'] - df['min_breaking_load']) / df['min_breaking_load']) * 100
    
    return df

# ML-Modell und Preprocessing Objekte via joblib laden
rf_model = joblib.load(os.path.join(os.path.dirname(__file__), 'rf_model.joblib'))
imputer = joblib.load(os.path.join(os.path.dirname(__file__), 'imputer.joblib'))
scaler = joblib.load(os.path.join(os.path.dirname(__file__), 'scaler.joblib'))
feature_names = joblib.load(os.path.join(os.path.dirname(__file__), 'feature_names.joblib'))

# Funktion zur Erstellung des Gauge Charts zur Anzeige der aktuellen Bruchlast
def create_gauge_chart(value, min_load):
    upper_threshold = min_load * 1.05
    max_value = min_load * 1.5
    number_color = RED_COLOR if value < min_load else "orange" if min_load <= value < upper_threshold else "green"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={'font': {'color': number_color}},
        gauge={
            'axis': {'range': [None, max_value], 'tickwidth': 0.75, 'tickcolor': "darkblue"},
            'bar': {'color': "black", 'thickness': 0.15},
            'steps': [
                {'range': [0, min_load], 'color': RED_COLOR},
                {'range': [min_load, upper_threshold], 'color': "orange"},
                {'range': [upper_threshold, max_value], 'color': "green"}
            ],
            'threshold': {'line': {'color': "black", 'width': 1.5}, 'thickness': 0.75, 'value': value}
        }
    ))
    fig.update_layout(height=200, width=400, margin=dict(t=50, b=20, l=0, r=0))
    return fig

# Funktion zur Erstellung der Liniendiagramme für die Breaking Load Predictions-Seite, welche Vorhersagen und Mindestbruchlasten anzeigen, sowie Punkte unter dem Mindest Breaking Load rot kennzeichnet
def create_line_chart_dashboard(df, thickness, title=""):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['predicted_breaking_load'], mode='lines+markers', name='Predicted Breaking Load', line=dict(width=1.5)))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=[MIN_BREAKING_LOADS[thickness]]*len(df), mode='lines', name='Minimum Breaking Load [N]', line=dict(dash='dot', width=1, color=RED_COLOR)))
    below_min_load = df[df['predicted_breaking_load'] < MIN_BREAKING_LOADS[thickness]]
    fig.add_trace(go.Scatter(x=below_min_load['timestamp'], y=below_min_load['predicted_breaking_load'], mode='markers', name='Below Min Load', marker=dict(color=RED_COLOR, size=8, symbol='x')))
    fig.update_layout(title=title, xaxis_title='Timestamp', yaxis_title='Breaking Load (N)', height=250, margin=dict(t=80, b=10, l=0, r=0))
    return fig

# Funktion zur Erstellung eines Liniendiagramms für die Quality Insights-Seite, das den Verlauf des Prozentsatzes von Gipsfaserplatten zeigt, die die Mindestanforderung an die Bruchlast überschreiten.
def create_line_chart_percentage(df, time_filter):
    df['time_period'] = df['timestamp'].dt.to_period('W').dt.to_timestamp() if time_filter == "Last Month" else df['timestamp'].dt.to_period('Y').dt.to_timestamp()
    grouped = df.groupby('time_period').apply(lambda x: (x['predicted_breaking_load'] >= x['min_breaking_load']).mean() * 100).reset_index(name='percentage_exceeding_minimum')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grouped['time_period'], y=grouped['percentage_exceeding_minimum'], mode='lines+markers', name='Percentage Exceeding Minimum Load'))
    fig.update_layout(title="Percentage of Boards Exceeding Minimum Load Over Time", xaxis_title="Time Period", yaxis_title="Percentage (%)", height=300)
    return fig

# Funktion zur Erstellung eines Liniendiagramms für die durchschnittliche Abweichung vom Mindest-Breaking-Load im Zeitverlauf
def create_line_chart_deviation(df, time_filter):
    df_below_min = df[df['predicted_breaking_load'] < df['min_breaking_load']]
    df_below_min['time_period'] = df_below_min['timestamp'].dt.to_period('W').dt.to_timestamp() if time_filter == "Last Month" else df_below_min['timestamp'].dt.to_period('Y').dt.to_timestamp()
    grouped = df_below_min.groupby('time_period')['percentage_deviation'].mean().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grouped['time_period'], y=grouped['percentage_deviation'], mode='lines+markers', name='Average Deviation from Minimum Load'))
    fig.update_layout(title="Average Deviation from Minimum Breaking Load Over Time", xaxis_title="Time Period", yaxis_title="Average Deviation (%)", height=300)
    return fig

# Funktion zur Erstellung von Liniendiagrammen für die Modellbewertungsseite, die die Vorhersagen im Vergleich zu Ist-Werten und Benchmark-Werten von der jeweiligen Plattendicke anzeigen
def create_line_chart_model(df, thickness, title=""):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['actual_breaking_load'], mode='lines+markers', name='Actual Breaking Load', line=dict(width=1.5)))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['predicted_breaking_load'], mode='lines+markers', name='Predicted Breaking Load', line=dict(width=1.5, dash='dash')))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=[MIN_BREAKING_LOADS[thickness]] * len(df), mode='lines', name='Minimum Breaking Load [N]', line=dict(dash='dot', width=1, color=RED_COLOR)))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=[BENCHMARK_VALUES[thickness]] * len(df), mode='lines', name='Benchmark Load [N]', line=dict(dash='dashdot', width=1.5, color='grey')))
    fig.update_layout(title=title, xaxis_title='Timestamp', yaxis_title='Breaking Load (N)', height=250, margin=dict(t=50, b=20, l=0, r=0))
    return fig

# Funktion zur Berechnung der Shapley-Werte und Erstellung des Plots
def generate_waterfall_plot(data_row, predicted_value):
    # Anpassung der Bezeichnungen für Ml- Modellkompatibilität
    data_row = data_row.rename(columns={
    'predicted_breaking_load': 'Breaking_Load', 
    'thickness_board': 'Thickness board', 
    'product_group': 'Product_group',
    'timestamp': 'timeutc'
    })
    # "Breaking_Load" entfernen, falls vorhanden, bevor die Vorverarbeitung beginnt
    if 'Breaking_Load' in data_row.columns:
        data_row = data_row.drop(columns=['Breaking_Load'])
    # Sicherstellen, dass data_row alle Merkmale in der richtigen Reihenfolge hat
    data_row = data_row.reindex(columns=feature_names)

    # Daten vorverarbeiten für den SHAP-Explainer, Zielspalten ausschließen
    data_row_preprocessed, _, additional_data = mlmodel.preprocess_data(data_row, imputer=imputer)
    data_row_preprocessed = scaler.transform(data_row_preprocessed)

    # Vorhersage mit dem Modell berechnen
    predicted_value_model = rf_model.predict(data_row_preprocessed)[0]

    # SHAP-Explainer um Shapley-Werte zu erhalten
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(data_row_preprocessed)
    shap_values_adj = shap.Explanation(
        values=shap_values[0], 
        base_values=predicted_value - shap_values[0].sum(),
        data=data_row.iloc[0])
    shap_values_adj.f = predicted_value

    plt.rcParams.update({'font.size': 8})  # Schriftgröße anpassen

    # SHAP Wasserfall Plot generieren
    fig, ax = plt.subplots(figsize=(5, 3))
    shap.waterfall_plot(shap_values_adj, max_display=10)
    plt.tight_layout()
    plt.subplots_adjust(left=0.35)
    return fig
    
# Funktion zum Anzeigen des SHAP Plots beim Klicken auf den Button
def display_shapley_plot(row):
    predicted_value = row['predicted_breaking_load']
    row_df = pd.DataFrame([row])
    waterfall_plot = generate_waterfall_plot(row_df, predicted_value)
    st.pyplot(waterfall_plot)

# Funktion zum Umschalten der Session-States für die SHAP Buttons (Show & Hide)
def toggle_shap(key):
    st.session_state[key] = not st.session_state[key]

# Funktion zur Berechnung der Metriken: MSE, MAE, R2 für das ML-Modell
def calculate_metrics(data):
    mse = mean_squared_error(data['actual_breaking_load'], data['predicted_breaking_load'])
    r2 = r2_score(data['actual_breaking_load'], data['predicted_breaking_load'])
    mae = mean_absolute_error(data['actual_breaking_load'], data['predicted_breaking_load'])

    return mse, r2, mae

# Funktion zur Berechnung der Metriken: MSE, MAE, R2 für Benchmark
def calculate_benchmark_metrics(data):
    data['benchmark'] = data['thickness_board'].map(BENCHMARK_VALUES)
    mse_benchmark = mean_squared_error(data['actual_breaking_load'], data['benchmark'])
    r2_benchmark = r2_score(data['actual_breaking_load'], data['benchmark'])
    mae_benchmark = mean_absolute_error(data['actual_breaking_load'], data['benchmark'])
    
    return mse_benchmark, r2_benchmark, mae_benchmark

# Funktion zur Berechnung der Konfusionsmatrix: True positive (TP), false positive (FP), true negative (TN) und false negative (FN) werden berechnet
def calculate_confusion_matrix(data, predicted_column):
    tp = ((data['actual_breaking_load'] >= data['min_breaking_load']) & (data[predicted_column] >= data['min_breaking_load'])).sum()
    fp = ((data['actual_breaking_load'] < data['min_breaking_load']) & (data[predicted_column] >= data['min_breaking_load'])).sum()
    tn = ((data['actual_breaking_load'] < data['min_breaking_load']) & (data[predicted_column] < data['min_breaking_load'])).sum()
    fn = ((data['actual_breaking_load'] >= data['min_breaking_load']) & (data[predicted_column] < data['min_breaking_load'])).sum()
    return tp, fp, tn, fn

# Funktion für die Darstellung der Confusion Matrix Zellen mit absolutem Wert und Prozentanteil
def format_cell(cell):
    number = cell['value']
    label = cell['label']
    percent = cell['percent']
    if label in ['TP', 'TN']:
        color = 'green'
    elif label in ['FP', 'FN']:
        color = RED_COLOR
    return f'<div style="text-align:center;"><span style="color:{color}; font-weight:bold; font-size:16px;">{number} ({percent}%)</span><br/><small>{label}</small></div>'

# Dynamischen Titel formatieren für die Quality Insights Page da sonst die langen Titel nicht ganz sichtbar sind
def format_title(title, max_length=40):
    # Titel in Wörter aufteilen
    words = title.split()
    wrapped_title = ""
    current_line = ""
    
    # Zeilen bis zur maximalen Länge aufbauen, ohne Wörter zu trennen
    for word in words:
        if len(current_line) + len(word) + 1 > max_length:
            wrapped_title += current_line + "<br>"
            current_line = word
        else:
            if current_line:
                current_line += " "
            current_line += word
    wrapped_title += current_line
    # Schriftgröße anpassen, wenn der Titel lang ist
    font_size = 18 if len(wrapped_title) <= max_length else 14
    return wrapped_title, font_size
            
def main():

    # Daten laden
    dashboard_data = load_data()

    # Logo in der Seitenleiste
    st.sidebar.image(os.path.join(os.path.dirname(__file__), 'Logo.png'), use_container_width=True)
    
    # Navigation in der Seitenleiste
    page = st.sidebar.radio("Navigation", ["Breaking Load Predictions", "Quality Insights", "Model Evaluation"])

    if page == "Breaking Load Predictions":
        st.markdown("<h1 style='margin-top: -60px;'>Breaking Load Predictions for Gypsum Boards</h1>", unsafe_allow_html=True)
        col1, col2 = st.columns((0.4, 0.6), gap='medium') 

        # Linke Spalte: Liniendiagramme für die verschiedene Plattendicken
        with col1:
            st.markdown('##### Breaking Load Predictions Over Time by Board Thickness', unsafe_allow_html=True)
            for thickness in [9.5, 12.5, 15, 18]:
                fig = create_line_chart_dashboard(dashboard_data[dashboard_data['thickness_board'] == thickness], thickness, title=f"{thickness}mm Boards")
                st.plotly_chart(fig, use_container_width=True)
        
        # Rechte Spalte: Gauge Chart und Alerts
        with col2:
            # Aktuellste Daten und Plattendicke für den Gauge Chart Titel
            latest_data = dashboard_data.iloc[-1]
            latest_thickness = latest_data['thickness_board']

            st.markdown(f"##### Current Breaking Load Status ({latest_thickness}mm)", unsafe_allow_html=True)
            
            # Gauge Chart erstellen und anzeigen
            fig_gauge = create_gauge_chart(latest_data['predicted_breaking_load'], latest_data['min_breaking_load'])
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Alerts anzeigen mit dynamischen Titel für den Alert Zähler
            alerts_data = dashboard_data[dashboard_data['predicted_breaking_load'] < dashboard_data['min_breaking_load']]
            alerts_data = alerts_data.sort_values(by='timestamp', ascending=False).reset_index(drop=True)  # Alerts nach Timestamp sortieren und Index zurücksetzen
            alert_count = len(alerts_data)
            icon = "❗"

            st.markdown(f"""
                <div style='display: flex; align-items: center; margin-top: 20px;'>
                    <span style='font-size: 26px; margin-right: 10px; line-height: 1;'>{icon}</span>
                    <span style='font-size: 21px; font-weight: bold; line-height: 1;'>
                        Alerts ({alert_count}) - 
                        <span style='font-weight: normal;'>Boards with Predicted Breaking Load Below Minimum Requirement</span>
                    </span>
                </div>
                <p style='margin-top: 26px;'>
                </p>
                """, unsafe_allow_html=True)

            # Tabellenüberschriften für die Alert Tabelle (in Spalten aufgeteilt, da der "Show Insights" Button in der st.table-Tabelle von Streamlit nicht verwendet werden konnte)
            headers = ["Timestamp", "Product Group", "Thickness Board [mm]", "Predicted Breaking Load[N]", "Minimum Breaking Load[N]", "Deviation from Minimum (%)", "Prediction Explanation"]
            col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
            col1.markdown(f"**{headers[0]}**", unsafe_allow_html=True)
            col2.markdown(f"**{headers[1]}**", unsafe_allow_html=True)
            col3.markdown(f"**{headers[2]}**", unsafe_allow_html=True)
            col4.markdown(f"**{headers[3]}**", unsafe_allow_html=True)
            col5.markdown(f"**{headers[4]}**", unsafe_allow_html=True)
            col6.markdown(f"**{headers[5]}**", unsafe_allow_html=True)
            col7.markdown(f"**{headers[6]}**", unsafe_allow_html=True)

            # Funktion zur Farbformatierung von Werten
            def format_as_black(text):
                return f"<span style='color: black;'>{text}</span>"
            def format_as_red(text):
                return f"<span style='color: {RED_COLOR};'>{text}</span>"

            # Alerts anzeigen
            for i, row in alerts_data.iterrows():
                col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
                col1.markdown(format_as_black(row['timestamp']), unsafe_allow_html=True)
                col2.markdown(format_as_black(row['product_group']), unsafe_allow_html=True)
                col3.markdown(format_as_black(row['thickness_board']), unsafe_allow_html=True)
                col4.markdown(format_as_black(row['predicted_breaking_load']), unsafe_allow_html=True)
                col5.markdown(format_as_black(row['min_breaking_load']), unsafe_allow_html=True)
                col6.markdown(format_as_red(f"{row['percentage_deviation']:.2f}%"), unsafe_allow_html=True)

                # Session-State-Schlüssel für den SHAP-Plot
                shap_key = f"show_shap_{i}"
                if shap_key not in st.session_state:
                    st.session_state[shap_key] = False

                # Label des Buttons Show/Hide Insights basierend auf dem aktuellen Zustand
                button_label = "Show Insights" if not st.session_state[shap_key] else "Hide Insights"

                # Callback-Funktion mit dem entsprechenden Schlüssel
                callback = functools.partial(toggle_shap, key=shap_key)

                # Button mit on_click Parameter erstellen
                col7.button(button_label, key=f"button_{i}", on_click=callback)

                # SHAP Plot anzeigen oder ausblenden
                if st.session_state[shap_key]:
                    st.markdown("#### SHAP Value Explanation")
                    predicted_value = row['predicted_breaking_load']
                    shap_plot = generate_waterfall_plot(row.to_frame().T, predicted_value)
                    st.pyplot(shap_plot)
                
    if page == "Quality Insights":
        st.markdown("<h1 style='margin-top: -60px;'>Quality Insights</h1>", unsafe_allow_html=True)

        # Filter für die Board Thickness und Time Range
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            board_filter = st.selectbox("Select Board Thickness", options=["All Boards", "9.5mm Boards", "12.5mm Boards", "15mm Boards", "18mm Boards"], index=0)
        with filter_col2:
            time_filter = st.selectbox("Select Time Range", options=["Total", "Last Week", "Last Month", "Last Year"], index=0)
        
        # Daten filtern basierend auf den Auswahlkriterien
        filtered_data = dashboard_data
        if board_filter != "All Boards":
            thickness = float(re.search(r'\d+\.?\d*', board_filter).group())
            filtered_data = filtered_data[filtered_data['thickness_board'] == thickness]

             
        if time_filter != "Total":
            if time_filter == "Last Week":
                start_date = dashboard_data['timestamp'].max() - pd.DateOffset(weeks=1)
            elif time_filter == "Last Month":
                start_date = dashboard_data['timestamp'].max() - pd.DateOffset(months=1)
            elif time_filter == "Last Year":
                start_date = dashboard_data['timestamp'].max() - pd.DateOffset(years=1)

            filtered_data = filtered_data[filtered_data['timestamp'] >= start_date]

        # Überprüfung, ob Daten für die ausgewählte Boarddicke vorhanden sind
        if filtered_data.empty:
            st.error("No data available for the selected board thickness.")
            return  # Überspringt das Rendern des restlichen Seiteninhalts
    

        # Dynamischen Titel basierend auf dem Zeitfilter
        if time_filter == "Last Week":
            detailed_insights_title = "Detailed Insights of Last Week"
        elif time_filter == "Last Month":
            detailed_insights_title = "Detailed Insights of Last Month"
        elif time_filter == "Last Year":
            detailed_insights_title = "Detailed Insights of Last Year"
        else:
            detailed_insights_title = "Detailed Insights (Total)"

        st.markdown(f"#### {detailed_insights_title}")
        
        # Bedingte Erstellung der Spalten basierend auf dem Board-Filter (Bei All Boards werden alle vier Spalten angezeigt, sobald ein Board filter gesetzt ist, sind col2 und col4 redundant zu col1 und col3)
        if board_filter == "All Boards":
            # Erstellen von vier Spalten
            col1, col2, col3, col4 = st.columns(4)
            
            # Col1: Quality Index Pie Chart zeigt den prozentualen Anteil der Platten unter und über der minimalen Bruchlast
            with col1:
                total_boards = filtered_data.shape[0]
                boards_above_min = filtered_data[filtered_data['predicted_breaking_load'] >= filtered_data['min_breaking_load']].shape[0]
                quality_index = (boards_above_min / total_boards) * 100 if total_boards > 0 else 0
                fig_quality_index = go.Figure(data=[go.Pie(
                    labels=['Above Min Breaking Load', 'Below Min Breaking Load'],
                    values=[quality_index, 100 - quality_index],
                    hole=.3,
                    marker=dict(colors=['#1f77b4', RED_COLOR])
                )])
                
                wrapped_title, font_size = format_title("Percentage of Boards Above/Below Minimum Breaking Load")
                fig_quality_index.update_layout(
                    title={'text': wrapped_title, 'x': 0.5, 'xanchor': 'center', 'font': {'size': font_size}},
                    height=350
                )
                st.plotly_chart(fig_quality_index, use_container_width=True)
            
            # Col2: Gestapeltes Balkendiagramm zeigt den prozentualen Anteil der Platten unter und über der minimalen Bruchlast, aufgeteilt nach den vier Plattendicken 
            with col2:
                all_thicknesses = [9.5, 12.5, 15, 18]
                percentage_above_min = filtered_data.groupby('thickness_board').apply(
                    lambda x: (x['predicted_breaking_load'] >= x['min_breaking_load']).mean() * 100
                ).reindex(all_thicknesses)
                percentage_above_min = percentage_above_min.reset_index()
                percentage_above_min.columns = ['Thickness', 'Percentage Above Minimum']

                # Berechnung des Prozentsatzes unterhalb des Minimums nur bei vorhandenen Daten; andernfalls werden 100 % der Platten als unterhalb der Mindestanforderung angezeigt, wenn keine Daten vorliegen
                percentage_above_min['Percentage Below Minimum'] = percentage_above_min['Percentage Above Minimum'].apply(
                    lambda x: 100 - x if pd.notnull(x) else None)  # Setzt None, wenn keine Daten vorhanden sind

                # Erstellen eines gestapelten Balkendiagramms
                fig_bar_above_min = go.Figure()
                fig_bar_above_min.add_trace(go.Bar(
                    x=percentage_above_min['Thickness'],
                    y=percentage_above_min['Percentage Above Minimum'],
                    name='Above Min Breaking Load',
                    marker_color='#1f77b4',
                    width=0.3 
                ))
                fig_bar_above_min.add_trace(go.Bar(
                    x=percentage_above_min['Thickness'],
                    y=percentage_above_min['Percentage Below Minimum'],
                    name='Below Min Breaking Load',
                    marker_color=RED_COLOR,
                    width=0.3  
                ))

                wrapped_title, font_size = format_title("Percentage of Boards Above/Below Minimum Breaking Load by Thickness")
                fig_bar_above_min.update_layout(
                    title={'text': wrapped_title, 'x': 0.5, 'xanchor': 'center', 'font': {'size': font_size}},
                    xaxis_title="Board Thickness (mm)",
                    yaxis_title="Percentage (%)",
                    barmode='stack',  # Setzt das Balkendiagramm auf gestapelt
                    xaxis=dict(tickvals=all_thicknesses),
                    yaxis=dict(range=[0, 100]),
                    height=350,
                    showlegend=False  # Legende ausblenden, da identische Legende bereits in col1
                )
                st.plotly_chart(fig_bar_above_min, use_container_width=True)
            
            # Col3: Durchschnittliche Abweichung für Platten unter dem Minimum Breaking Load
            with col3:
                # Filter für Boards unter dem Mindestbruchlast innerhalb der gefilterten Daten
                below_min_data = filtered_data[filtered_data['predicted_breaking_load'] < filtered_data['min_breaking_load']]
                
                # Berechnung der aktuellen durchschnittlichen Abweichung nur mit below_min_data (gefiltert)
                current_avg_deviation = below_min_data['percentage_deviation'].abs().mean() if not below_min_data.empty else 0
                
                # Anzeige der Metrik
                st.markdown(f"""
                    <div style='text-align: center;'>
                        <h3 style="font-size: 15px; font-weight: bold;"><br><br>Average Deviation from Minimum Breaking Load (Boards below Minimum Breaking Load)</h3>
                        <p style='margin-top: 10px;'></p> <!-- Absatz für Abstand -->
                        <p style='font-size: 36px; font-weight: bold; color: {RED_COLOR};'>{current_avg_deviation:.2f}%</p>
                    </div>
                """, unsafe_allow_html=True)
                
            # Col4: Durchschnittliche Abweichung nach Plattendicke für Platten unter dem Minimum Breaking Load
                with col4:
                    below_min_deviation = below_min_data.groupby('thickness_board')['percentage_deviation'].apply(lambda x: x.abs().mean()).reindex(all_thicknesses, fill_value=0).reset_index()
                    below_min_deviation.columns = ['Thickness', 'Avg Deviation Below Min']
                    fig_bar_below_min = go.Figure(go.Bar(
                        x=below_min_deviation['Thickness'],
                        y=below_min_deviation['Avg Deviation Below Min'],
                        marker_color=RED_COLOR,
                        width=0.3
                    ))

                    wrapped_title, font_size = format_title("Average Deviation from Minimum Breaking Load by Thickness (Boards below Minimum Breaking Load)")
                    fig_bar_below_min.update_layout(
                        title={'text': wrapped_title, 'x': 0.5, 'xanchor': 'center', 'font': {'size': font_size}},
                        xaxis_title="Board Thickness (mm)",
                        yaxis_title="Average Deviation (%)",
                        xaxis=dict(tickvals=all_thicknesses),
                        yaxis=dict(range=[0, None]),
                        height=350
                    )
                    st.plotly_chart(fig_bar_below_min, use_container_width=True)
        else:
            # Erstellen von zwei Spalten, wenn der Board-Filter nicht auf "All Boards" gesetzt ist
            col1, col2 = st.columns(2)
            
            # Col1: Quality Index Pie Chart zeigt den prozentualen Anteil der Platten unter und über der minimalen Bruchlast
            with col1:
                total_boards = filtered_data.shape[0]
                boards_above_min = filtered_data[filtered_data['predicted_breaking_load'] >= filtered_data['min_breaking_load']].shape[0]
                quality_index = (boards_above_min / total_boards) * 100 if total_boards > 0 else 0
                fig_quality_index = go.Figure(data=[go.Pie(
                    labels=['Above Minimum Breaking Load', 'Below Minimum Breaking Load'],
                    values=[quality_index, 100 - quality_index],
                    hole=.3,
                    marker=dict(colors=['#1f77b4', RED_COLOR])
                )])
                
                wrapped_title, font_size = format_title("Percentage of Boards Above/Below Minimum Breaking Load")
                fig_quality_index.update_layout(
                    title={'text': wrapped_title, 'x': 0.5, 'xanchor': 'center', 'font': {'size': font_size}},
                    height=350
                )
                st.plotly_chart(fig_quality_index, use_container_width=True)
            
            # Col2: Durchschnittliche Abweichung für Platten unter dem Minimum Breaking Load
            with col2:
                # Filter für Boards unter dem Mindestbruchlast innerhalb der gefilterten Daten
                below_min_data = filtered_data[filtered_data['predicted_breaking_load'] < filtered_data['min_breaking_load']]
                
                # Berechnung der aktuellen durchschnittlichen Abweichung nur mit below_min_data (gefiltert)
                current_avg_deviation = below_min_data['percentage_deviation'].abs().mean() if not below_min_data.empty else 0
                
                # Anzeige der Metrik
                st.markdown(f"""
                    <div style='text-align: center;'>
                        <h3 style="font-size: 15px; font-weight: bold;"><br><br>Average Deviation from Minimum Breaking Load (Boards below Minimum Breaking Load)</h3>
                        <p style='margin-top: 10px;'></p> <!-- Absatz für Abstand -->
                        <p style='font-size: 36px; font-weight: bold; color: {RED_COLOR};'>{current_avg_deviation:.2f}%</p>
                    </div>
                """, unsafe_allow_html=True)

        # Abschnitt Detaillierte Einblicke über die Zeit (wird nur angezeigt, wenn der Zeitfilter nicht auf "Total" gesetzt ist)        
        st.markdown("#### Detailed Insights Over Time")
    
        # Aggregation pro Tag
        filtered_data['date'] = filtered_data['timestamp'].dt.date
        # Für Boards, die die Mindestbruchlast überschreiten
        daily_exceed = filtered_data.groupby('date').apply(
            lambda x: (x['predicted_breaking_load'] >= x['min_breaking_load']).mean() * 100
        ).reset_index(name='percentage_exceeding_minimum')
        # Für durchschnittliche Abweichung bei Boards unter der Mindestbruchlast
        below_min_data = filtered_data[filtered_data['predicted_breaking_load'] < filtered_data['min_breaking_load']]
        
        # **Korrigierte Aggregation: Absolute Abweichung vor dem Gruppieren berechnen**
        daily_deviation = below_min_data.copy()
        daily_deviation['percentage_deviation'] = daily_deviation['percentage_deviation'].abs()
        daily_deviation = daily_deviation.groupby('date')['percentage_deviation'].mean().reset_index(name='average_deviation_below_min')
        
        # **Entfernung der Reindexing-Logik, um nur Tage mit vorhandenen Daten darzustellen**
        # Das bedeutet, dass nur Tage mit Datenpunkten in daily_exceed und daily_deviation enthalten sind

        detailed_col1, detailed_col2 = st.columns(2)
        
        # Liniendiagramm für Boards Exceeding Minimum Breaking Load über die Zeit in col1 erstellen
        with detailed_col1:
            fig_line_chart_percentage = go.Figure()
            fig_line_chart_percentage.add_trace(go.Scatter(
                x=daily_exceed['date'],
                y=daily_exceed['percentage_exceeding_minimum'],
                mode='lines+markers',
                name='Percentage Exceeding Minimum Load',
                marker=dict(color='#1f77b4', size=6)
            ))
            fig_line_chart_percentage.update_layout(
                title="Boards Exceeding Minimum Breaking Load Over Time",
                xaxis_title="Date",
                yaxis_title="Exceeding Minimum Load (%)",
                height=350,
                yaxis=dict(range=[-5, 105]),
                margin=dict(t=60, b=20, l=0, r=0), 
                title_pad=dict(t=10) 
            )
            st.plotly_chart(fig_line_chart_percentage, use_container_width=True)
        
        # Liniendiagramm für durchschnittliche Abweichung unter dem Minimum Breaking Load über die Zeit in col2 erstellen
        with detailed_col2:
            fig_line_chart_deviation = go.Figure()
            fig_line_chart_deviation.add_trace(go.Scatter(
                x=daily_deviation['date'],
                y=daily_deviation['average_deviation_below_min'],
                mode='lines+markers',
                name='Average Deviation Below Min Breaking Load',
                marker=dict(color=RED_COLOR, size=6)
            ))
            fig_line_chart_deviation.update_layout(
                title="Average Deviation from Minimum Breaking Load Over Time<br>(Boards below Minimum Breaking Load)",
                xaxis_title="Date",
                yaxis_title="Average Deviation (%)",
                height=350,
                margin=dict(t=60, b=20, l=0, r=0),  
                title_pad=dict(t=10)
            )
            st.plotly_chart(fig_line_chart_deviation, use_container_width=True)

    if page == "Model Evaluation":
        st.markdown("<h1 style='margin-top: -60px;'>Model Evaluation</h1>", unsafe_allow_html=True)

        # CSS zur Anpassung der Tabellenüberschriften
        st.markdown("""
        <style>
            table.confusion-matrix thead th,
            table.confusion-matrix tbody th {
                font-weight: normal;
            }
        </style>
        """, unsafe_allow_html=True)

        # Berechnet Metrics des ML-Modells und des Benchmarks
        mse_model, r2_model, mae_model= calculate_metrics(dashboard_data)
        mse_benchmark, r2_benchmark, mae_benchmark = calculate_benchmark_metrics(dashboard_data)

        # Berechnung der Konfusionsmatrix für das ML-Modell und den Benchmark
        tp_model, fp_model, tn_model, fn_model = calculate_confusion_matrix(dashboard_data, 'predicted_breaking_load')
        tp_benchmark, fp_benchmark, tn_benchmark, fn_benchmark = calculate_confusion_matrix(dashboard_data, 'benchmark')

        col1, col2 = st.columns((0.55, 0.45), gap='medium')
        
        # Anzeige der Vorhersagen und tatsächlichen Bruchlastwerte je nach Dicke der Gipsplatte
        with col1:
            st.markdown("##### Actual vs. Predicted Breaking Load by Board Thickness")
            for thickness in [9.5, 12.5, 15, 18]:
                fig = create_line_chart_model(dashboard_data[dashboard_data['thickness_board'] == thickness], thickness, title=f"{thickness}mm Boards")
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Erstellen einer Tabelle für die Metriken des Modells und Benchmarks
            metrics_data = {
                "Metrics": ["Mean Squared Error (MSE)", "Mean Absolute Error (MAE)", "Coefficient of Determination (R²)"],
                "ML-Model": [f"{mse_model:.2f}", f"{mae_model:.2f}", f"{r2_model:.2f}"],
                "Benchmark": [f"{mse_benchmark:.2f}", f"{mae_benchmark:.2f}", f"{r2_benchmark:.2f}"]
            }
            metrics_df = pd.DataFrame(metrics_data)
            metrics_df.set_index("Metrics", inplace=True)

            st.markdown("##### Metrics ML Model vs Benchmark")
            # Spaltenaufteilung für die Metriken
            metric_col1, metric_col2, metric_col3 = st.columns(3)

            # Berechnung der Deltas (Differenz zwischen Modell und Benchmark)
            mse_delta = mse_model - mse_benchmark
            mae_delta = mae_model - mae_benchmark
            r2_delta = r2_model - r2_benchmark

            # Anzeige der Metriken mit den jeweiligen Deltas
            metric_col1.metric(label="MSE", value=f"{mse_model:.2f}", delta=f"{mse_delta:+.2f} vs Benchmark", delta_color="inverse")
            metric_col2.metric(label="MAE", value=f"{mae_model:.2f}", delta=f"{mae_delta:+.2f} vs Benchmark", delta_color="inverse")
            metric_col3.metric(label="R² Score", value=f"{r2_model:.2f}", delta=f"{r2_delta:+.2f} vs Benchmark")
            st.markdown("")
            
            # Confusion Matrix für das Machine-Learning-Modell
            st.markdown("##### Confusion Matrix: ML Model")
            
            # Berechnung des Gesamtwerts zur Bestimmung der Prozentwerte in der Confusion Matrix
            total_decisions_model = tp_model + fp_model + tn_model + fn_model
            if total_decisions_model == 0:
                total_decisions_model = 1  # Vermeidung von Division durch Null
            
            # DataFrame der Confusion Matrix mit Werten, Labels und Prozentsätzen
            cm_model_df = pd.DataFrame(
                [
                    [
                        {'value': tp_model, 'label': 'TP', 'percent': round(tp_model / total_decisions_model * 100)},
                        {'value': fn_model, 'label': 'FN', 'percent': round(fn_model / total_decisions_model * 100)}
                    ],
                    [
                        {'value': fp_model, 'label': 'FP', 'percent': round(fp_model / total_decisions_model * 100)},
                        {'value': tn_model, 'label': 'TN', 'percent': round(tn_model / total_decisions_model * 100)}
                    ]
                ],
                index=["Actual: ≥ Min Load", "Actual: < Min Load"],
                columns=["Predicted: ≥ Min Load", "Predicted: < Min Load"]
            )

            # Formatieren der Confusion Matrix für Anzeige in HTML
            cm_model_formatted = cm_model_df.applymap(format_cell)
            cm_model_html = cm_model_formatted.to_html(escape=False, classes='confusion-matrix')
            st.markdown(cm_model_html, unsafe_allow_html=True)

            # Confusion Matrix für den Benchmark
            st.markdown("##### Confusion Matrix: Benchmark")

            # Berechnung des Gesamtwerts zur Bestimmung der Prozentwerte in der Confusion Matrix
            total_decisions_benchmark = tp_benchmark + fp_benchmark + tn_benchmark + fn_benchmark
            if total_decisions_benchmark == 0:
                total_decisions_benchmark = 1  # Vermeidung von Division durch Null

            # DataFrame der Confusion Matrix für den Benchmark mit Werten und Labels
            cm_benchmark_df = pd.DataFrame(
                [
                    [
                        {'value': tp_benchmark, 'label': 'TP', 'percent': round(tp_benchmark / total_decisions_benchmark * 100)},
                        {'value': fn_benchmark, 'label': 'FN', 'percent': round(fn_benchmark / total_decisions_benchmark * 100)}
                    ],
                    [
                        {'value': fp_benchmark, 'label': 'FP', 'percent': round(fp_benchmark / total_decisions_benchmark * 100)},
                        {'value': tn_benchmark, 'label': 'TN', 'percent': round(tn_benchmark / total_decisions_benchmark * 100)}
                    ]
                ],
                index=["Actual: ≥ Min Load", "Actual: < Min Load"],
                columns=["Predicted: ≥ Min Load", "Predicted: < Min Load"]
            )

            # Formatieren der Confusion Matrix für den Benchmark zur HTML-Anzeige
            cm_benchmark_formatted = cm_benchmark_df.applymap(format_cell)
            cm_benchmark_html = cm_benchmark_formatted.to_html(escape=False, classes='confusion-matrix')
            st.markdown(cm_benchmark_html, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
