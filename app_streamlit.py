import os
import time
import json
import requests

try:
    import streamlit as st  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - only used when the dependency is unavailable in the editor environment
    st = None

try:
    import pandas as pd  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency not available in the editor environment
    pd = None

try:
    import plotly.graph_objects as go  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency not available in the editor environment
    go = None

try:
    import google.generativeai as genai  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency not available in the editor environment
    genai = None

try:
    from google.api_core.exceptions import GoogleAPIError, ResourceExhausted, ServerError  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency not available in the editor environment
    class GoogleAPIError(Exception):
        pass

    class ResourceExhausted(Exception):
        pass

    class ServerError(Exception):
        pass

# ------------------------------------------------------------------------------
# 1. Configurazione Iniziale e API Key
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Palinsesto Analisi Match",
    page_icon="⚽",
    layout="wide"
)

# Recupero API Key da secrets o variabile d'ambiente
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    st.warning("⚠️ API Key di Gemini non configurata. Inseriscila nei Secrets di Streamlit.")

# ------------------------------------------------------------------------------
# 2. Funzione Chiamata Gemini con Exponential Backoff
# ------------------------------------------------------------------------------
def call_gemini_with_retry(prompt: str, model_name: str = "gemini-1.5-flash", max_retries: int = 3, delay: float = 2.0) -> str:
    """
    Gestisce la chiamata all'API con retry automatico per evitare blocchi 
    dovuti a rate-limit o server momentaneamente occupati.
    """
    model = genai.GenerativeModel(model_name)
    
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            return response.text
        except (ServerError, ResourceExhausted) as e:
            if attempt == max_retries - 1:
                raise e
            # Attesa esponenziale: 2s, 4s, 8s...
            time.sleep(delay * (2 ** attempt))
        except GoogleAPIError as e:
            raise e

# ------------------------------------------------------------------------------
# 3. Logica di Elaborazione Batch per i Match
# ------------------------------------------------------------------------------
def process_instant_batch(matches_payload: list) -> dict:
    """
    Invia l'intera lista dei match a Gemini per la rielaborazione tattica.
    """
    prompt = f"""
    Sei un analista tattico avanzato di calcio. 
    Analizza i seguenti match e restituisci un JSON con le percentuali ricalcolate 
    (1X2, Over/Under 2.5, Goal/No Goal):
    
    {json.dumps(matches_payload, indent=2)}
    
    Rispondi ESCLUSIVAMENTE con un oggetto JSON valido.
    """
    
    raw_response = call_gemini_with_retry(prompt)
    
    # Pulizia output JSON nel caso la risposta contenga blocchi di codice markdown
    clean_json = raw_response.replace("```json", "").replace("```", "").strip()
    return json.loads(clean_json)

# ------------------------------------------------------------------------------
# 4. Interfaccia Grafica Streamlit
# ------------------------------------------------------------------------------
st.title("⚽ Palinsesto Dettagliato")

# Dati di prova coerenti con l'interfaccia
matches_payload = [
    {
        "id": 1,
        "league": "Inghilterra - Championship",
        "home_team": "Bristol City",
        "away_team": "Lincoln City",
        "date": "2026-09-15",
        "probs": {
            "1": 50.9, "X": 25.5, "2": 23.6,
            "over25": 50.9, "under25": 49.1,
            "goal": 48.1, "nogoal": 51.9
        },
        "recommendation": "No Goal 51.9%"
    }
]

# Bottone principale per il ricalcolo batch
if st.button("⚡ RICALCOLA TUTTI I MATCH CON GEMINI AI (INSTANT BATCH)", type="primary"):
    with st.spinner("Analisi batch in corso con Gemini..."):
        try:
            batch_result = process_instant_batch(matches_payload)
            st.success("Ricalcolo completato con successo!")
            st.session_state['batch_data'] = batch_result
        except (ServerError, ResourceExhausted):
            st.error("⚠️ Errore durante l'analisi batch: Server Google Gemini temporaneamente occupati. Riprova tra qualche istante.")
        except Exception as e:
            st.error(f"⚠️ Si è verificato un errore durante l'elaborazione: {str(e)}")

st.markdown("---")

# Renderizzazione delle schede dei singoli match
for match in matches_payload:
    header_text = f"⚽ {match['home_team']} vs {match['away_team']} — {match['recommendation']}"
    
    with st.expander(header_text, expanded=True):
        col_info, col_rec = st.columns([3, 1])
        with col_info:
            st.caption(f"🏆 {match['league']} | 📅 {match['date']}")
        with col_rec:
            st.success(f"🎯 {match['recommendation']}")
            
        st.subheader("ESITO FINALE (1X2)")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("1 (CASA)", f"{match['probs']['1']}%")
        c2.metric("X (PAREGGIO)", f"{match['probs']['X']}%")
        c3.metric("2 (OSPITE)", f"{match['probs']['2']}%")
        
        # Grafico a barre con Plotly
        fig = go.Figure(data=[
            go.Bar(
                x=["Casa (1)", "Pareggio (X)", "Ospite (2)"],
                y=[match['probs']['1'], match['probs']['X'], match['probs']['2']],
                marker_color=["#00CC96", "#FECB52", "#636efa"],
                text=[f"{match['probs']['1']}%", f"{match['probs']['X']}%", f"{match['probs']['2']}%"],
                textposition="auto"
            )
        ])
        fig.update_layout(
            height=250,
            margin=dict(l=20, r=20, t=20, b=20),
            yaxis=dict(range=[0, 100])
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("MERCATI GOL")
        g1, g2 = st.columns(2)
        g1.metric("Over 2.5", f"{match['probs']['over25']}%")
        g2.metric("Under 2.5", f"{match['probs']['under25']}%")
        
        m1, m2 = st.columns(2)
        m1.metric("Goal", f"{match['probs']['goal']}%")
        m2.metric("No Goal", f"{match['probs']['nogoal']}%")
        
        st.button(f"🔍 Studio Tattico Gemini & Correzione %", key=f"btn_{match['id']}")
