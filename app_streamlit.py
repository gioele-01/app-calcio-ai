import json
import re
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from scipy.stats import poisson

# ---------------------------------------------------------
# CONFIGURAZIONE PAGINA & CSS STILE EMERALD PITCH
# ---------------------------------------------------------
st.set_page_config(
    page_title="Football AI Match Analyzer Pro",
    page_icon="⚽",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .stApp {
        background: linear-gradient(145deg, #06110d 0%, #0c1a14 50%, #06110d 100%);
        color: #ecfdf5;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 760px;
    }

    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #10b981 0%, #34d399 50%, #a7f3d0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        letter-spacing: -0.5px;
        margin-bottom: 0.2rem;
    }

    .sub-title {
        font-size: 0.88rem;
        color: #6ee7b7;
        text-align: center;
        margin-bottom: 1.8rem;
        font-weight: 500;
        opacity: 0.85;
    }

    div[data-testid="stExpander"] {
        background: rgba(15, 31, 24, 0.75) !important;
        border: 1px solid rgba(16, 185, 129, 0.2) !important;
        border-radius: 14px !important;
        backdrop-filter: blur(12px);
        margin-bottom: 1rem !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        color: #34d399 !important;
    }

    div[data-testid="stMetricLabel"] {
        font-size: 0.75rem !important;
        color: #9ca3af !important;
        font-weight: 600 !important;
        text-transform: uppercase;
    }

    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #059669 0%, #10b981 100%) !important;
        color: #ffffff !important;
        font-size: 14px !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.65em 0.5em !important;
        transition: all 0.25s ease !important;
        box-shadow: 0 4px 14px rgba(16, 185, 129, 0.25) !important;
    }

    .stButton>button:hover {
        background: linear-gradient(90deg, #10b981 0%, #34d399 100%) !important;
        box-shadow: 0 6px 20px rgba(52, 211, 153, 0.4) !important;
        transform: translateY(-1px) !important;
        color: #06110d !important;
    }

    .badge-pick {
        background: linear-gradient(90deg, #059669 0%, #10b981 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
        box-shadow: 0 2px 10px rgba(16, 185, 129, 0.3);
    }

    .badge-league {
        background: rgba(52, 211, 153, 0.12);
        color: #a7f3d0;
        border: 1px solid rgba(52, 211, 153, 0.25);
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
    }

    .badge-time {
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.3);
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
    }

    .scalata-card {
        background: rgba(15, 31, 24, 0.85);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-left: 5px solid #10b981;
        padding: 14px;
        border-radius: 12px;
        margin-bottom: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
</style>
""",
    unsafe_allow_html=True,
)

MARGINE_BOOKMAKER = 1.05


def calcola_quota_reale(prob_percentuale):
  if prob_percentuale <= 0:
    return 1.01
  quota_pura = 100.0 / prob_percentuale
  quota_reale = quota_pura / MARGINE_BOOKMAKER
  return max(1.05, round(quota_reale, 2))


# ---------------------------------------------------------
# HEADER APPLICAZIONE
# ---------------------------------------------------------
st.markdown(
    '<div class="main-title">⚽ FOOTBALL AI PRO</div>', unsafe_allow_html=True
)
st.markdown(
    '<div class="sub-title">Algoritmo Quantitativo Multi-Lega & Studio'
    ' Tattico Gemini AI</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# RECUPERO CHIAVI API DAI SECRETS O INPUT MANUALE
# ---------------------------------------------------------
odds_key_secret = st.secrets.get("ODDS_API_KEY", "")
gemini_key_secret = st.secrets.get("GEMINI_API_KEY", "")

with st.expander(
    "🔑 **Configurazione API Studio**", expanded=not bool(odds_key_secret)
):
  if odds_key_secret:
    st.success("✅ Chiave The Odds API attiva")
    api_key = odds_key_secret
  else:
    api_key = st.text_input("Chiave API (The Odds API Key)", type="password")

  if gemini_key_secret:
    st.success("✅ Chiave Google Gemini attiva")
    gemini_api_key = gemini_key_secret
  else:
    gemini_api_key = st.text_input("Chiave API (Google Gemini)", type="password")

# ---------------------------------------------------------
# MAPPATURA CAMPIONATI COMPLETA
# ---------------------------------------------------------
code_map = {
    "🌐 TUTTI I CAMPIONATI PRINCIPALI": {
        "key": "MULTI",
        "home_avg": 1.45,
        "away_avg": 1.15,
        "btts_base": 0.52,
    },
    "🇮🇹 Italia - Serie A": {
        "key": "soccer_italy_serie_a",
        "home_avg": 1.42,
        "away_avg": 1.12,
        "btts_base": 0.52,
    },
    "🇮🇹 Italia - Serie B": {
        "key": "soccer_italy_serie_b",
        "home_avg": 1.30,
        "away_avg": 1.05,
        "btts_base": 0.48,
    },
    "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Inghilterra - Premier League": {
        "key": "soccer_epl",
        "home_avg": 1.55,
        "away_avg": 1.25,
        "btts_base": 0.56,
    },
    "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Inghilterra - Championship": {
        "key": "soccer_efl_champ",
        "home_avg": 1.35,
        "away_avg": 1.10,
        "btts_base": 0.50,
    },
    "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Inghilterra - League One": {
        "key": "soccer_england_league1",
        "home_avg": 1.38,
        "away_avg": 1.12,
        "btts_base": 0.51,
    },
    "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Inghilterra - League Two": {
        "key": "soccer_england_league2",
        "home_avg": 1.40,
        "away_avg": 1.15,
        "btts_base": 0.52,
    },
    "🇪🇸 Spagna - La Liga": {
        "key": "soccer_spain_la_liga",
        "home_avg": 1.38,
        "away_avg": 1.08,
        "btts_base": 0.49,
    },
    "🇪🇸 Spagna - Segunda Division": {
        "key": "soccer_spain_segunda_division",
        "home_avg": 1.25,
        "away_avg": 0.95,
        "btts_base": 0.45,
    },
    "🇩🇪 Germania - Bundesliga": {
        "key": "soccer_germany_bundesliga",
        "home_avg": 1.65,
        "away_avg": 1.35,
        "btts_base": 0.59,
    },
    "🇩🇪 Germania - 2. Bundesliga": {
        "key": "soccer_germany_bundesliga2",
        "home_avg": 1.58,
        "away_avg": 1.30,
        "btts_base": 0.57,
    },
    "🇩🇪 Germania - 3. Liga": {
        "key": "soccer_germany_liga3",
        "home_avg": 1.45,
        "away_avg": 1.20,
        "btts_base": 0.54,
    },
    "🇫🇷 Francia - Ligue 1": {
        "key": "soccer_france_ligue_one",
        "home_avg": 1.40,
        "away_avg": 1.10,
        "btts_base": 0.51,
    },
    "🇫🇷 Francia - Ligue 2": {
        "key": "soccer_france_ligue_two",
        "home_avg": 1.28,
        "away_avg": 0.98,
        "btts_base": 0.46,
    },
    "🇳🇱 Olanda - Eredivisie": {
        "key": "soccer_netherlands_eredivisie",
        "home_avg": 1.68,
        "away_avg": 1.32,
        "btts_base": 0.61,
    },
    "🇵🇹 Portogallo - Primeira Liga": {
        "key": "soccer_portugal_primeira_liga",
        "home_avg": 1.45,
        "away_avg": 1.18,
        "btts_base": 0.53,
    },
    "🇧🇪 Belgio - First Div": {
        "key": "soccer_belgium_first_div",
        "home_avg": 1.52,
        "away_avg": 1.22,
        "btts_base": 0.55,
    },
    "🏴󠁧󠁢󠁳󠁣󠁴󠁿 Scozia - Premiership": {
        "key": "soccer_spl",
        "home_avg": 1.45,
        "away_avg": 1.15,
        "btts_base": 0.51,
    },
    "🇦🇹 Austria - Bundesliga": {
        "key": "soccer_austria_bundesliga",
        "home_avg": 1.50,
        "away_avg": 1.25,
        "btts_base": 0.54,
    },
    "🇨🇭 Svizzera - Super League": {
        "key": "soccer_switzerland_superleague",
        "home_avg": 1.55,
        "away_avg": 1.28,
        "btts_base": 0.56,
    },
    "🇩🇰 Danimarca - Superliga": {
        "key": "soccer_denmark_superliga",
        "home_avg": 1.45,
        "away_avg": 1.20,
        "btts_base": 0.53,
    },
    "🇸🇪 Svezia - Allsvenskan": {
        "key": "soccer_sweden_allsvenskan",
        "home_avg": 1.48,
        "away_avg": 1.18,
        "btts_base": 0.53,
    },
    "🇸🇪 Svezia - Superettan": {
        "key": "soccer_sweden_superettan",
        "home_avg": 1.42,
        "away_avg": 1.15,
        "btts_base": 0.52,
    },
    "🇳🇴 Norvegia - Eliteserien": {
        "key": "soccer_norway_eliteserien",
        "home_avg": 1.60,
        "away_avg": 1.28,
        "btts_base": 0.58,
    },
    "🇫🇮 Finlandia - Veikkausliiga": {
        "key": "soccer_finland_veikkausliiga",
        "home_avg": 1.38,
        "away_avg": 1.12,
        "btts_base": 0.50,
    },
    "🇵🇱 Polonia - Ekstraklasa": {
        "key": "soccer_poland_ekstraklasa",
        "home_avg": 1.40,
        "away_avg": 1.12,
        "btts_base": 0.51,
    },
    "🇹🇷 Turchia - Super League": {
        "key": "soccer_turkey_super_league",
        "home_avg": 1.52,
        "away_avg": 1.20,
        "btts_base": 0.55,
    },
    "🇬🇷 Grecia - Super League": {
        "key": "soccer_greece_super_league",
        "home_avg": 1.38,
        "away_avg": 1.02,
        "btts_base": 0.47,
    },
    "🇷🇺 Russia - Premier League": {
        "key": "soccer_russia_premier_league",
        "home_avg": 1.40,
        "away_avg": 1.08,
        "btts_base": 0.49,
    },
    "🇧🇷 Brasile - Serie A": {
        "key": "soccer_brazil_campeonato",
        "home_avg": 1.48,
        "away_avg": 1.05,
        "btts_base": 0.48,
    },
    "🇧🇷 Brasile - Serie B": {
        "key": "soccer_brazil_serie_b",
        "home_avg": 1.32,
        "away_avg": 0.88,
        "btts_base": 0.42,
    },
    "🇦🇷 Argentina - Primera Div": {
        "key": "soccer_argentina_primera_division",
        "home_avg": 1.25,
        "away_avg": 0.92,
        "btts_base": 0.43,
    },
    "🇨🇱 Cile - Primera Division": {
        "key": "soccer_chile_campeonato",
        "home_avg": 1.40,
        "away_avg": 1.10,
        "btts_base": 0.50,
    },
    "🇲🇽 Messico - Liga MX": {
        "key": "soccer_mexico_ligamx",
        "home_avg": 1.48,
        "away_avg": 1.15,
        "btts_base": 0.52,
    },
    "🇺🇸 USA - MLS": {
        "key": "soccer_usa_mls",
        "home_avg": 1.62,
        "away_avg": 1.22,
        "btts_base": 0.57,
    },
    "🇯🇵 Giappone - J1 League": {
        "key": "soccer_japan_j_league",
        "home_avg": 1.38,
        "away_avg": 1.15,
        "btts_base": 0.50,
    },
    "🇰🇷 Corea del Sud - K League 1": {
        "key": "soccer_korea_kleague1",
        "home_avg": 1.35,
        "away_avg": 1.10,
        "btts_base": 0.49,
    },
    "🇨🇳 Cina - Super League": {
        "key": "soccer_china_superleague",
        "home_avg": 1.50,
        "away_avg": 1.20,
        "btts_base": 0.54,
    },
    "🇦🇺 Australia - A-League": {
        "key": "soccer_australia_aleague",
        "home_avg": 1.58,
        "away_avg": 1.30,
        "btts_base": 0.58,
    },
    "🇪🇺 UEFA Champions League": {
        "key": "soccer_uefa_champs_league",
        "home_avg": 1.60,
        "away_avg": 1.30,
        "btts_base": 0.57,
    },
    "🇪🇺 UEFA Champions Qual.": {
        "key": "soccer_uefa_champs_league_qualification",
        "home_avg": 1.50,
        "away_avg": 1.20,
        "btts_base": 0.54,
    },
    "🇪🇺 UEFA Europa League": {
        "key": "soccer_uefa_europa_league",
        "home_avg": 1.50,
        "away_avg": 1.20,
        "btts_base": 0.55,
    },
    "🇪🇺 UEFA Conference League": {
        "key": "soccer_uefa_europa_conference_league",
        "home_avg": 1.52,
        "away_avg": 1.22,
        "btts_base": 0.55,
    },
    "🇪🇺 UEFA Nations League": {
        "key": "soccer_uefa_nations_league",
        "home_avg": 1.45,
        "away_avg": 1.15,
        "btts_base": 0.52,
    },
    "🌍 Coppa d'Africa (AFCON)": {
        "key": "soccer_africa_cup_of_nations",
        "home_avg": 1.30,
        "away_avg": 0.95,
        "btts_base": 0.44,
    },
    "🌎 Copa Libertadores": {
        "key": "soccer_conmebol_copa_libertadores",
        "home_avg": 1.45,
        "away_avg": 0.98,
        "btts_base": 0.46,
    },
    "🌎 Copa Sudamericana": {
        "key": "soccer_conmebol_copa_sudamericana",
        "home_avg": 1.42,
        "away_avg": 0.95,
        "btts_base": 0.45,
    },
}

TOP_LEAGUES_KEYS = [
    ("🇮🇹 Serie A", "soccer_italy_serie_a"),
    ("🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League", "soccer_epl"),
    ("🇪🇸 La Liga", "soccer_spain_la_liga"),
    ("🇩🇪 Bundesliga", "soccer_germany_bundesliga"),
    ("🇫🇷 Ligue 1", "soccer_france_ligue_one"),
    ("🇳🇱 Eredivisie", "soccer_netherlands_eredivisie"),
    ("🇵🇹 Primeira Liga", "soccer_portugal_primeira_liga"),
    ("🇪🇺 Champions League", "soccer_uefa_champs_league"),
    ("🇪🇺 Europa League", "soccer_uefa_europa_league"),
    ("🇪🇺 Conference League", "soccer_uefa_europa_conference_league"),
    ("🇪🇺 Nations League", "soccer_uefa_nations_league"),
    ("🌍 Coppa d'Africa", "soccer_africa_cup_of_nations"),
]

with st.expander("🎛️ **Filtri Palinsesto & Parametri**", expanded=True):
  campionato_scelto = st.selectbox(
      "🏆 Campionato / Selezione", list(code_map.keys())
  )
  comp_info = code_map[campionato_scelto]
  sport_key = comp_info["key"]

  col_f1, col_f2 = st.columns(2)

  with col_f1:
    filtro_data = st.selectbox(
        "📅 Data Eventi",
        [
            "Tutte le prossime",
            "Solo Oggi",
            "📅 Intervallo di Date Personalizzato",
        ],
    )

  with col_f2:
    mercato_preferito = st.selectbox(
        "🎯 Mercato",
        [
            "Tutti i mercati",
            "Solo 1X2",
            "Solo Over / Under",
            "Solo Goal / No Goal",
        ],
    )

  data_inizio, data_fine = None, None
  if filtro_data == "📅 Intervallo di Date Personalizzato":
    oggi = datetime.now().date()
    dopodomani = oggi + timedelta(days=2)
    range_date = st.date_input("Scegli Periodo (Da -> A)", [oggi, dopodomani])

    if isinstance(range_date, (list, tuple)) and len(range_date) == 2:
      data_inizio, data_fine = range_date[0], range_date[1]
    elif isinstance(range_date, (list, tuple)) and len(range_date) == 1:
      data_inizio = data_fine = range_date[0]

  col_ord1, col_ord2 = st.columns(2)
  with col_ord1:
    modalita_ordinamento = st.selectbox(
        "📌 Ordina Palinsesto per:", ["Campionato", "Orario di inizio"]
    )

  st.markdown("---")

  min_confidence = st.slider(
      "⚡ Confidenza minima (%)",
      min_value=50,
      max_value=90,
      value=50,
      step=5,
  )


# ---------------------------------------------------------
# FETCHING PARTITE DA THE ODDS API
# ---------------------------------------------------------
@st.cache_data(ttl=1800)
def scarica_partite_the_odds_api(s_key, key):
  if not key:
    return None, "⚠️ Inserisci la tua chiave API di The Odds API."

  url = (
      f"https://api.the-odds-api.com/v4/sports/{s_key}/odds/?apiKey={key.strip()}&regions=eu&markets=h2h,totals&dateFormat=iso"
  )

  try:
    res = requests.get(url, timeout=10)
    if res.status_code == 200:
      data = res.json()
      return data, None
    else:
      return None, f"Errore API ({res.status_code}): {res.text}"
  except Exception as e:
    return None, f"Errore connessione: {str(e)}"


# ---------------------------------------------------------
# GEMINI SINGLE-MATCH TACTICAL CORRECTOR
# ---------------------------------------------------------
def studio_tattico_gemini(match_name, p1_math, px_math, p2_math, key):
  if not key:
    return None, "⚠️ Nessuna chiave GEMINI_API_KEY trovata nei Secrets."

  key_clean = key.strip().replace('"', "").replace("'", "")

  prompt = f"""
    Sei un analista tattico quantitativo di calcio.
    Il nostro algoritmo ha calcolato per '{match_name}' le probabilità statistiche base:
    Casa (1): {p1_math:.1f}%, Pareggio (X): {px_math:.1f}%, Ospite (2): {p2_math:.1f}%.

    Valuta attentamente infortuni, turnover, stanchezza da coppe e motivazioni.
    In base alla tua analisi, stabilisci la variazione percentuale (shift) per le due squadre:
    - `home_shift`: tra -8.0 e +8.0 per la casa.
    - `away_shift`: tra -8.0 e +8.0 per l'ospite.

    Rispondi esclusivamente in formato JSON valido con questa struttura esatta:
    {{
        "home_shift": 0.0,
        "away_shift": 0.0,
        "analisi_sintetica": "Analisi sintetica motivata in 3 frasi..."
    }}
    """

  payload = {
      "contents": [{"parts": [{"text": prompt}]}],
      "generationConfig": {"response_mime_type": "application/json"},
  }

  modelli = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]

  for mod in modelli:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{mod}:generateContent?key={key_clean}"
    for intento in range(3):
      try:
        response = requests.post(url, json=payload, timeout=25)
        if response.status_code == 200:
          data = response.json()
          if "candidates" in data and len(data["candidates"]) > 0:
            text_res = data["candidates"][0]["content"]["parts"][0]["text"]
            json_match = re.search(r"\{.*\}", text_res, re.DOTALL)
            if json_match:
              return json.loads(json_match.group(0)), None
            return json.loads(text_res), None
        elif response.status_code in [429, 503]:
          time.sleep(2.0 * (intento + 1))
          continue
        else:
          break
      except Exception:
        time.sleep(1.5)
        continue

  return (
      None,
      "⚠️ Server Gemini momentaneamente occupati. Riprova tra poco.",
  )


# ---------------------------------------------------------
# GEMINI BATCH CORRECTOR
# ---------------------------------------------------------
def studio_tattico_in_blocco_batch(lista_partite, key):
  if not key:
    return None, "⚠️ Nessuna chiave GEMINI_API_KEY nei Secrets."

  key_clean = key.strip().replace('"', "").replace("'", "")
  modelli = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]

  CHUNK_SIZE = 2
  risultati_totali = []

  for i in range(0, len(lista_partite), CHUNK_SIZE):
    chunk = lista_partite[i : i + CHUNK_SIZE]

    info_txt = ""
    for idx, p in enumerate(chunk, 1):
      info_txt += (
          f"{idx}. {p['match']} -> 1: {p['p1']:.1f}%, X: {p['px']:.1f}%, 2:"
          f" {p['p2']:.1f}%\n"
      )

    prompt = f"""
        Sei un analista tattico quantitativo di calcio.
        Analizza il contesto delle seguenti partite:

        {info_txt}

        Per OGNUNA delle partite, calcola uno shift percentuale per la Casa (home_shift da -8.0 a +8.0) e l'Ospite (away_shift da -8.0 a +8.0).

        Rispondi ESCLUSIVAMENTE con una lista JSON con questa struttura:
        [
          {{
            "match": "NomeCasa vs NomeOspite",
            "home_shift": 0.0,
            "away_shift": 0.0,
            "analisi_sintetica": "Analisi tattica sintetica in 2-3 frasi..."
          }}
        ]
        """

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"response_mime_type": "application/json"},
    }

    chunk_successo = False
    for mod in modelli:
      if chunk_successo:
        break
      url = f"https://generativelanguage.googleapis.com/v1beta/models/{mod}:generateContent?key={key_clean}"

      for intento in range(3):
        try:
          response = requests.post(url, json=payload, timeout=30)
          if response.status_code == 200:
            data = response.json()
            if "candidates" in data and len(data["candidates"]) > 0:
              text_res = data["candidates"][0]["content"]["parts"][0]["text"]
              json_match = re.search(r"\[.*\]", text_res, re.DOTALL)
              if json_match:
                parsed_chunk = json.loads(json_match.group(0))
                risultati_totali.extend(parsed_chunk)
                chunk_successo = True
                break
          elif response.status_code in [429, 503]:
            time.sleep(3.0 * (intento + 1))
            continue
          else:
            break
        except Exception:
          time.sleep(2.0)
          continue

    time.sleep(1.5)

  if risultati_totali:
    return risultati_totali, None
  else:
    return (
        None,
        "⚠️ Quota API temporaneamente satura. Riprova tra poco.",
    )


# ---------------------------------------------------------
# CALCOLO PROBABILITÀ E MERCATI ESTESI
# ---------------------------------------------------------
def elab_match_odds(match, comp_info, home_shift=0.0, away_shift=0.0):
  casa = match["home_team"]
  trasferta = match["away_team"]

  prob_1, prob_X, prob_2 = 40.0, 30.0, 30.0
  prob_over, prob_under = None, None

  if match.get("bookmakers"):
    bm = match["bookmakers"][0]
    for m in bm.get("markets", []):
      if m["key"] == "h2h":
        outcomes = {o["name"]: o["price"] for o in m["outcomes"]}
        q1 = outcomes.get(casa, 2.5)
        qX = outcomes.get("Draw", 3.2)
        q2 = outcomes.get(trasferta, 2.8)

        inv_tot = (1 / q1) + (1 / qX) + (1 / q2)
        prob_1 = (1 / q1 / inv_tot) * 100
        prob_X = (1 / qX / inv_tot) * 100
        prob_2 = (1 / q2 / inv_tot) * 100

      elif m["key"] == "totals":
        outcomes = {}
        for o in m.get("outcomes", []):
          name_clean = o["name"].strip()
          point = o.get("point", 2.5)
          if point == 2.5 or "2.5" in name_clean:
            if "Over" in name_clean:
              outcomes["Over"] = o["price"]
            elif "Under" in name_clean:
              outcomes["Under"] = o["price"]

        if "Over" in outcomes and "Under" in outcomes:
          q_over = outcomes["Over"]
          q_under = outcomes["Under"]
          inv_tot = (1 / q_over) + (1 / q_under)
          prob_over = (1 / q_over / inv_tot) * 100
          prob_under = (1 / q_under / inv_tot) * 100

  p1_mod = max(5.0, min(85.0, prob_1 + home_shift))
  p2_mod = max(5.0, min(85.0, prob_2 + away_shift))
  px_mod = max(5.0, 100.0 - (p1_mod + p2_mod))

  tot_mod = p1_mod + px_mod + p2_mod
  p1_final = (p1_mod / tot_mod) * 100
  px_final = (px_mod / tot_mod) * 100
  p2_final = (p2_mod / tot_mod) * 100

  if prob_over is not None:
    gol_attesi_totali = 1.6 + (prob_over / 100.0) * 1.6
  else:
    gol_attesi_totali = comp_info.get("home_avg", 1.4) + comp_info.get(
        "away_avg", 1.1
    )

  forza_casa = p1_final / (p1_final + p2_final + 1e-5)
  lambda_c = max(0.65, gol_attesi_totali * forza_casa)
  lambda_t = max(0.55, gol_attesi_totali * (1.0 - forza_casa))

  matrice_raw = np.zeros((5, 5))
  for i in range(5):
    for j in range(5):
      matrice_raw[i, j] = poisson.pmf(i, lambda_c) * poisson.pmf(j, lambda_t)

  matrice = (matrice_raw / np.sum(matrice_raw)) * 100

  p_o15 = float(
      sum(matrice[i, j] for i in range(5) for j in range(5) if (i + j) > 1)
  )
  p_o25 = float(
      sum(matrice[i, j] for i in range(5) for j in range(5) if (i + j) > 2)
  )
  p_o35 = float(
      sum(matrice[i, j] for i in range(5) for j in range(5) if (i + j) > 3)
  )
  p_u25 = 100.0 - p_o25

  if prob_over is None:
    prob_over = p_o25
    prob_under = p_u25

  p_casa_segna = 1.0 - np.exp(-lambda_c)
  p_trasferta_segna = 1.0 - np.exp(-lambda_t)

  prob_goal_raw = (p_casa_segna * p_trasferta_segna) * 100
  prob_goal = float(
      min(85.0, max(35.0, prob_goal_raw * 0.7 + prob_over * 0.35))
  )
  prob_no_goal = 100.0 - prob_goal

  tutti_gli_esiti = {
      "1": p1_final,
      "X": px_final,
      "2": p2_final,
      "Over 2.5": prob_over,
      "Under 2.5": prob_under,
      "Goal": prob_goal,
      "No Goal": prob_no_goal,
  }

  if mercato_preferito == "Solo 1X2":
    esiti = {"1": p1_final, "X": px_final, "2": p2_final}
  elif mercato_preferito == "Solo Over / Under":
    esiti = {"Over 2.5": prob_over, "Under 2.5": prob_under}
  elif mercato_preferito == "Solo Goal / No Goal":
    esiti = {"Goal": prob_goal, "No Goal": prob_no_goal}
  else:
    esiti = tutti_gli_esiti

  top_pick = max(esiti, key=esiti.get)
  top_perc = esiti[top_pick]

  metriche_estese = {
      "1X2": max(p1_final, px_final, p2_final),
      "BTTS": prob_goal,
      "O1.5": p_o15,
      "O2.5": prob_over,
      "O3.5": p_o35,
      "U2.5": prob_under,
  }

  return (
      top_pick,
      top_perc,
      p1_final,
      px_final,
      p2_final,
      prob_over,
      prob_under,
      prob_goal,
      prob_no_goal,
      matrice[:4, :4],
      metriche_estese,
  )


# ---------------------------------------------------------
# EXECUTION ENGINE MULTI-LEGA
# ---------------------------------------------------------
if st.button("🚀 SCANSIONA PALINSESTO & AVVIA AI"):
  if not api_key:
    st.error("Inserisci la chiave API di The Odds API per continuare.")
  else:
    partite_analizzate = []
    dettagli_matrici = {}
    raw_matches_dict = {}

    ora_attuale_utc = datetime.now(timezone.utc)
    tz_roma = ZoneInfo("Europe/Rome")
    oggi_date = datetime.now(tz_roma).date()

    if sport_key == "MULTI":
      with st.spinner(
          "Scaricamento palinsesti da tutti i campionati principali..."
      ):
        leghe_target = TOP_LEAGUES_KEYS
    else:
      leghe_target = [(campionato_scelto, sport_key)]

    with st.spinner("Calcolo probabilità quantitative in corso..."):
      for l_nome, l_key in leghe_target:
        all_matches, error_msg = scarica_partite_the_odds_api(l_key, api_key)

        if all_matches:
          for m in all_matches:
            casa = m.get("home_team", "").strip()
            trasferta = m.get("away_team", "").strip()

            if not casa or not trasferta or casa.lower() == trasferta.lower():
              continue

            if not m.get("bookmakers") or len(m["bookmakers"]) == 0:
              continue

            raw_date = m["commence_time"]

            try:
              dt_match_utc = datetime.fromisoformat(
                  raw_date.replace("Z", "+00:00")
              )

              if dt_match_utc <= ora_attuale_utc:
                continue

              dt_match_local = dt_match_utc.astimezone(tz_roma)
              orario_str = dt_match_local.strftime("%H:%M")
              match_date_obj = dt_match_local.date()
              commence_time = dt_match_local.strftime("%Y-%m-%d")
            except Exception:
              orario_str = "15:00"
              match_date_obj = oggi_date
              commence_time = raw_date[:10]

            if filtro_data == "Solo Oggi" and match_date_obj != oggi_date:
              continue
            elif (
                filtro_data == "📅 Intervallo di Date Personalizzato"
                and data_inizio
                and data_fine
            ):
              if not (data_inizio <= match_date_obj <= data_fine):
                continue

            nome_match = f"{casa} vs {trasferta}"

            (
                top_pick,
                perc_top,
                p1,
                px,
                p2,
                p_over,
                p_under,
                p_goal,
                p_ng,
                matrice,
                m_estese,
            ) = elab_match_odds(m, comp_info)

            if perc_top >= min_confidence:
              partite_analizzate.append({
                  "lega": l_nome,
                  "data": commence_time,
                  "orario": orario_str,
                  "datetime_raw": raw_date,
                  "match": nome_match,
                  "top_pick": top_pick,
                  "top_perc": perc_top,
                  "p1": p1,
                  "px": px,
                  "p2": p2,
                  "over": p_over,
                  "under": p_under,
                  "goal": p_goal,
                  "no_goal": p_ng,
                  "m_estese": m_estese,
              })
              dettagli_matrici[nome_match] = (casa, trasferta, matrice)
              raw_matches_dict[nome_match] = m

    if partite_analizzate:
      st.session_state["partite"] = partite_analizzate
      st.session_state["dettagli_matrici"] = dettagli_matrici
      st.session_state["raw_matches"] = raw_matches_dict
      st.session_state["campionato_corrente"] = campionato_scelto
      st.rerun()
    else:
      st.error("❌ Nessuna partita futura trovata con i filtri selezionati.")

# ---------------------------------------------------------
# INTERFACCIA UTENTE CON BOTTONI RETTANGOLARI
# ---------------------------------------------------------
if "partite" in st.session_state and st.session_state["partite"]:
  partite = st.session_state["partite"]
  dettagli = st.session_state["dettagli_matrici"]
  raw_m_dict = st.session_state.get("raw_matches", {})
  camp_nome = st.session_state.get("campionato_corrente", "")

  if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = "Palinsesto"

  st.markdown("### 🎛️ **Seleziona Modalità Studio AI**")

  # SELEZIONE TRAMITE RETTANGOLI INTERATTIVI
  col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
  with col_btn1:
    if st.button("⚽ Palinsesto"):
      st.session_state["active_tab"] = "Palinsesto"
      st.rerun()
  with col_btn2:
    if st.button("🎯 Singola"):
      st.session_state["active_tab"] = "Singola"
      st.rerun()
  with col_btn3:
    if st.button("👥 Doppia"):
      st.session_state["active_tab"] = "Doppia"
      st.rerun()
  with col_btn4:
    if st.button("☘️ Tripla"):
      st.session_state["active_tab"] = "Tripla"
      st.rerun()

  col_btn5, col_btn6, col_btn7, col_btn8 = st.columns(4)
  with col_btn5:
    if st.button("📊 Mista"):
      st.session_state["active_tab"] = "Mista"
      st.rerun()
  with col_btn6:
    if st.button("💣 Bomba"):
      st.session_state["active_tab"] = "Bomba"
      st.rerun()
  with col_btn7:
    if st.button("🚀 Scalata"):
      st.session_state["active_tab"] = "Scalata"
      st.rerun()
  with col_btn8:
    if st.button("🎟️ Multipla"):
      st.session_state["active_tab"] = "Multipla"
      st.rerun()

  st.markdown("---")
  current_tab = st.session_state["active_tab"]

  # ---------------------------------------------------------
  # 1. PALINSESTO & ANALISI
  # ---------------------------------------------------------
  if current_tab == "Palinsesto":
    if modalita_ordinamento == "Orario di inizio":
      partite = sorted(partite, key=lambda x: x.get("datetime_raw", ""))
    else:
      partite = sorted(partite, key=lambda x: x.get("lega", ""))

    st.markdown("### 📊 Performance Heatmap per Campionato & Mercato")

    df_heatmap = []
    for p in partite:
      m_e = p.get("m_estese", {})
      df_heatmap.append({
          "Campionato": p.get("lega", camp_nome),
          "1X2": m_e.get("1X2", p["top_perc"]),
          "BTTS": m_e.get("BTTS", p["goal"]),
          "O1.5": m_e.get("O1.5", 70.0),
          "O2.5": m_e.get("O2.5", p["over"]),
          "O3.5": m_e.get("O3.5", 35.0),
          "U2.5": m_e.get("U2.5", p["under"]),
      })

    df_h = pd.DataFrame(df_heatmap)
    if not df_h.empty:
      df_h_grouped = df_h.groupby("Campionato").mean()

      fig_macro_heat = px.imshow(
          df_h_grouped,
          labels=dict(x="Mercati", y="Campionati", color="Affidabilità %"),
          color_continuous_scale="RdYlGn",
          text_auto=".1f",
          range_color=[35, 75],
      )
      fig_macro_heat.update_layout(
          height=280,
          margin=dict(l=10, r=10, t=10, b=10),
          paper_bgcolor="rgba(0,0,0,0)",
          plot_bgcolor="rgba(0,0,0,0)",
          font=dict(color="#a7f3d0"),
      )
      st.plotly_chart(
          fig_macro_heat,
          use_container_width=True,
          key="macro_performance_heatmap",
      )

    st.markdown("---")
    st.markdown(f"### ⚽ Palinsesto Dettagliato ({len(partite)} Eventi)")

    if st.button("⚡ RICALCOLA TUTTI I MATCH CON GEMINI AI (INSTANT BATCH)"):
      if not gemini_api_key:
        st.error(
            "Inserisci la chiave GEMINI_API_KEY nei Secrets prima di continuare."
        )
      else:
        with st.spinner(
            "Gemini sta analizzando il contesto tattico di tutto il"
            " palinsesto..."
        ):
          batch_res, err = studio_tattico_in_blocco_batch(
              partite, gemini_api_key
          )

          if batch_res and isinstance(batch_res, list):
            res_map = {
                item.get("match"): item
                for item in batch_res
                if isinstance(item, dict)
            }

            for p in partite:
              match_key = f"gemini_report_{p['match']}"
              match_info = res_map.get(p["match"])

              if match_info:
                h_s = match_info.get("home_shift", 0.0)
                a_s = match_info.get("away_shift", 0.0)
                report_txt = (
                    "🧠 **Studio Tattico AI:**\n"
                    f"{match_info.get('analisi_sintetica', '')}\n\n⚡ *Shift"
                    f" Applicato:* Casa ({'+' if h_s>=0 else ''}{h_s:.1f}%),"
                    f" Ospite ({'+' if a_s>=0 else ''}{a_s:.1f}%)"
                )

                st.session_state[match_key] = report_txt

                raw_match = raw_m_dict.get(p["match"])
                if raw_match:
                  (
                      new_pick,
                      new_perc,
                      new_p1,
                      new_px,
                      new_p2,
                      new_over,
                      new_under,
                      new_goal,
                      new_ng,
                      new_matrice,
                      new_m_estese,
                  ) = elab_match_odds(
                      raw_match, comp_info, home_shift=h_s, away_shift=a_s
                  )

                  casa_team, trasf_team, _ = dettagli[p["match"]]
                  st.session_state["dettagli_matrici"][p["match"]] = (
                      casa_team,
                      trasf_team,
                      new_matrice,
                  )

                  p["top_pick"] = new_pick
                  p["top_perc"] = new_perc
                  p["p1"], p["px"], p["p2"] = new_p1, new_px, new_p2
                  p["over"], p["under"] = new_over, new_under
                  p["goal"], p["no_goal"] = new_goal, new_ng
                  p["m_estese"] = new_m_estese

            st.success("✅ Analisi in blocco completata per tutte le partite!")
            st.rerun()
          else:
            st.error(f"Errore durante l'analisi batch: {err}")

    st.markdown("---")

    for idx, p in enumerate(partite):
      match_key = f"gemini_report_{p['match']}"
      lega_label = p.get("lega", camp_nome)
      orario_label = p.get("orario", "15:00")
      data_label = p.get("data", "")

      header_card = f"""
            <div style="display: flex; justify-content: space-between; align-items: center; width: 100%;">
                <div>
                    <span class="badge-league">{lega_label}</span>
                    <span class="badge-time">⏰ {data_label} {orario_label}</span>
                    <span style="font-weight: 700; font-size: 1.05rem; color: #f0fdf4;">{p['match']}</span>
                </div>
                <div>
                    <span class="badge-pick">🎯 {p['top_pick']} {p['top_perc']:.1f}%</span>
                </div>
            </div>
            """

      with st.expander(
          f"⏰ {data_label} {orario_label} | {p['match']} — {p['top_pick']}"
          f" ({p['top_perc']:.1f}%)",
          expanded=True,
      ):
        st.markdown(header_card, unsafe_allow_html=True)
        st.write("")

        st.write("**ESITO FINALE (1X2)**")
        c1, c2, c3 = st.columns(3)
        c1.metric("1 (CASA)", f"{p['p1']:.1f}%")
        c2.metric("X (PAREGGIO)", f"{p['px']:.1f}%")
        c3.metric("2 (OSPITE)", f"{p['p2']:.1f}%")

        fig_bar = go.Figure(
            data=[
                go.Bar(
                    x=["Casa (1)", "Pareggio (X)", "Ospite (2)"],
                    y=[p["p1"], p["px"], p["p2"]],
                    marker_color=["#10b981", "#f59e0b", "#3b82f6"],
                    text=[
                        f"{p['p1']:.1f}%",
                        f"{p['px']:.1f}%",
                        f"{p['p2']:.1f}%",
                    ],
                    textposition="auto",
                )
            ]
        )
        fig_bar.update_layout(
            height=200,
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis=dict(range=[0, 100], showgrid=False),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#a7f3d0"),
        )
        st.plotly_chart(
            fig_bar, use_container_width=True, key=f"bar_{idx}_{p['match']}"
        )

        st.write("**MERCATI GOL**")
        m1, m2 = st.columns(2)
        m1.metric("Over 2.5", f"{p['over']:.1f}%")
        m2.metric("Under 2.5", f"{p['under']:.1f}%")

        m3, m4 = st.columns(2)
        m3.metric("Goal", f"{p['goal']:.1f}%")
        m4.metric("No Goal", f"{p['no_goal']:.1f}%")

        st.markdown("---")

        # GESTIONE SINGOLO BOTTONE GEMINI AI
        if match_key in st.session_state:
          st.info(st.session_state[match_key])
          if st.button(
              "🔄 Ripristina Statistica Base", key=f"reload_{idx}_{p['match']}"
          ):
            del st.session_state[match_key]
            st.rerun()
        else:
          if st.button(
              "🧠 Studio Tattico Gemini & Correzione %",
              key=f"btn_ai_{idx}_{p['match']}",
          ):
            if not gemini_api_key:
              st.error(
                  "Inserisci la chiave GEMINI_API_KEY nei Secrets o nel campo"
                  " in alto."
              )
            else:
              with st.spinner("Gemini sta analizando notizie e formazioni..."):
                ai_res, err = studio_tattico_gemini(
                    p["match"], p["p1"], p["px"], p["p2"], gemini_api_key
                )

                if ai_res:
                  h_s = ai_res.get("home_shift", 0.0)
                  a_s = ai_res.get("away_shift", 0.0)
                  report_txt = (
                      "🧠 **Studio Tattico AI:**\n"
                      f"{ai_res.get('analisi_sintetica', '')}\n\n⚡ *Shift"
                      f" Applicato:* Casa ({'+' if h_s>=0 else ''}{h_s:.1f}%),"
                      f" Ospite ({'+' if a_s>=0 else ''}{a_s:.1f}%)"
                  )

                  st.session_state[match_key] = report_txt

                  raw_match = raw_m_dict.get(p["match"])
                  if raw_match:
                    (
                        new_pick,
                        new_perc,
                        new_p1,
                        new_px,
                        new_p2,
                        new_over,
                        new_under,
                        new_goal,
                        new_ng,
                        new_matrice,
                        new_m_estese,
                    ) = elab_match_odds(
                        raw_match, comp_info, home_shift=h_s, away_shift=a_s
                    )

                    casa_team, trasf_team, _ = dettagli[p["match"]]
                    st.session_state["dettagli_matrici"][p["match"]] = (
                        casa_team,
                        trasf_team,
                        new_matrice,
                    )

                    p["top_pick"] = new_pick
                    p["top_perc"] = new_perc
                    p["p1"], p["px"], p["p2"] = new_p1, new_px, new_p2
                    p["over"], p["under"] = new_over, new_under
                    p["goal"], p["no_goal"] = new_goal, new_ng
                    p["m_estese"] = new_m_estese

                  st.rerun()
                else:
                  st.error(err)

  # ---------------------------------------------------------
  # 2. SINGOLA DEL GIORNO
  # ---------------------------------------------------------
  elif current_tab == "Singola":
    st.subheader("🎯 Singola del Giorno (Quota ~1.80)")
    st.write(
        "L'algoritmo seleziona l'evento con il miglior rapporto"
        " rischio/rendimento e **quota vicina a 1.80**."
    )

    target_quota = st.slider(
        "Quota Target desiderata:",
        min_value=1.50,
        max_value=2.20,
        value=1.80,
        step=0.05,
    )
    target_perc = 100.0 / (target_quota * MARGINE_BOOKMAKER)

    best_match = min(
        st.session_state["partite"],
        key=lambda x: abs(x["top_perc"] - target_perc),
    )

    if best_match:
      quota_calcolata = calcola_quota_reale(best_match["top_perc"])
      match_key_s = f"gemini_report_{best_match['match']}"

      st.markdown(f"""
            <div class="scalata-card" style="border-left: 6px solid #f59e0b; background: rgba(20, 35, 28, 0.9);">
                <span class="badge-time">⏰ {best_match.get('data', '')} {best_match.get('orario', '15:00')}</span>
                <span class="badge-league">{best_match.get('lega', '')}</span><br>
                <h3 style="margin: 10px 0 6px 0; color: #f0fdf4;">{best_match['match']}</h3>
                📌 Pronostico Consigliato: <strong style="font-size: 1.1rem; color: #34d399;">{best_match['top_pick']}</strong><br>
                📈 Quota Stimata: <strong style="font-size: 1.2rem; color: #fbbf24;">@{quota_calcolata}</strong> (Confidenza AI: <strong>{best_match['top_perc']:.1f}%</strong>)
            </div>
            """, unsafe_allow_html=True)

      if match_key_s in st.session_state:
        st.info(st.session_state[match_key_s])

  # ---------------------------------------------------------
  # 3. DOPPIA DEL GIORNO
  # ---------------------------------------------------------
  elif current_tab == "Doppia":
    st.subheader("👥 Doppia del Giorno (Quota ~2.50)")
    st.write(
        "L'algoritmo seleziona la **migliore coppia di partite** il cui"
        " prodotto delle quote sia vicino al raddoppio."
    )

    target_quota_doppia = st.slider(
        "Quota Totale Doppia desiderata:",
        min_value=2.00,
        max_value=3.50,
        value=2.50,
        step=0.10,
    )

    lista_p = st.session_state["partite"]

    if len(lista_p) < 2:
      st.warning(
          "Servono almeno 2 partite nel palinsesto per generare una doppia."
      )
    else:
      miglior_coppia = None
      min_diff = 999.0

      for i in range(len(lista_p)):
        for j in range(i + 1, len(lista_p)):
          p1_item, p2_item = lista_p[i], lista_p[j]
          q1 = calcola_quota_reale(p1_item["top_perc"])
          q2 = calcola_quota_reale(p2_item["top_perc"])
          q_tot = round(q1 * q2, 2)

          diff = abs(q_tot - target_quota_doppia)
          if diff < min_diff:
            min_diff = diff
            miglior_coppia = (p1_item, p2_item, q1, q2, q_tot)

      if miglior_coppia:
        m1, m2, q1, q2, q_tot = miglior_coppia

        st.markdown(f"""
                <div class="scalata-card" style="border-left: 6px solid #3b82f6; background: rgba(18, 28, 38, 0.9);">
                    <div style="margin-bottom: 12px; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 10px;">
                        <span class="badge-time">⏰ {m1.get('data', '')} {m1.get('orario', '15:00')}</span>
                        <span class="badge-league">{m1.get('lega', '')}</span><br>
                        <strong style="font-size: 1.05rem; color: #f0fdf4;">1. {m1['match']}</strong><br>
                        👉 Pronostico: <strong style="color: #34d399;">{m1['top_pick']}</strong> | Quota: <strong style="color: #fbbf24;">@{q1:.2f}</strong>
                    </div>
                    <div>
                        <span class="badge-time">⏰ {m2.get('data', '')} {m2.get('orario', '15:00')}</span>
                        <span class="badge-league">{m2.get('lega', '')}</span><br>
                        <strong style="font-size: 1.05rem; color: #f0fdf4;">2. {m2['match']}</strong><br>
                        👉 Pronostico: <strong style="color: #34d399;">{m2['top_pick']}</strong> | Quota: <strong style="color: #fbbf24;">@{q2:.2f}</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        prob_combinata_d = (m1["top_perc"] / 100.0) * (
            m2["top_perc"] / 100.0
        ) * 100
        st.success(f"""
                🎯 **QUOTA TOTALE DOPPIA:** **@{q_tot:.2f}**
                💡 **Probabilità Stimata Combinata:** **{prob_combinata_d:.1f}%**
                """)

  # ---------------------------------------------------------
  # 4. TRIPLA DEL GIORNO
  # ---------------------------------------------------------
  elif current_tab == "Tripla":
    st.subheader("☘️ Tripla del Giorno (Quota ~5.00)")
    st.write(
        "L'algoritmo seleziona la **migliore combinazione di 3 partite** per"
        " raggiungere la quota obiettivo."
    )

    target_quota_tripla = st.slider(
        "Quota Totale Tripla desiderata:",
        min_value=3.50,
        max_value=8.00,
        value=5.00,
        step=0.25,
    )

    lista_p = st.session_state["partite"]

    if len(lista_p) < 3:
      st.warning(
          "Servono almeno 3 partite nel palinsesto per generare una tripla."
      )
    else:
      miglior_tripla = None
      min_diff_t = 999.0

      for i in range(len(lista_p)):
        for j in range(i + 1, len(lista_p)):
          for k in range(j + 1, len(lista_p)):
            p1_i, p2_i, p3_i = lista_p[i], lista_p[j], lista_p[k]
            q1 = calcola_quota_reale(p1_i["top_perc"])
            q2 = calcola_quota_reale(p2_i["top_perc"])
            q3 = calcola_quota_reale(p3_i["top_perc"])
            q_tot = round(q1 * q2 * q3, 2)

            diff = abs(q_tot - target_quota_tripla)
            if diff < min_diff_t:
              min_diff_t = diff
              miglior_tripla = (p1_i, p2_i, p3_i, q1, q2, q3, q_tot)

      if miglior_tripla:
        m1, m2, m3, q1, q2, q3, q_tot = miglior_tripla

        st.markdown(f"""
                <div class="scalata-card" style="border-left: 6px solid #8b5cf6; background: rgba(28, 22, 38, 0.9);">
                    <div style="margin-bottom: 10px; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 8px;">
                        <span class="badge-time">⏰ {m1.get('data', '')} {m1.get('orario', '15:00')}</span>
                        <span class="badge-league">{m1.get('lega', '')}</span><br>
                        <strong style="font-size: 1.05rem; color: #f0fdf4;">1. {m1['match']}</strong> ➔ <strong style="color: #34d399;">{m1['top_pick']}</strong> @{q1:.2f}
                    </div>
                    <div style="margin-bottom: 10px; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 8px;">
                        <span class="badge-time">⏰ {m2.get('data', '')} {m2.get('orario', '15:00')}</span>
                        <span class="badge-league">{m2.get('lega', '')}</span><br>
                        <strong style="font-size: 1.05rem; color: #f0fdf4;">2. {m2['match']}</strong> ➔ <strong style="color: #34d399;">{m2['top_pick']}</strong> @{q2:.2f}
                    </div>
                    <div>
                        <span class="badge-time">⏰ {m3.get('data', '')} {m3.get('orario', '15:00')}</span>
                        <span class="badge-league">{m3.get('lega', '')}</span><br>
                        <strong style="font-size: 1.05rem; color: #f0fdf4;">3. {m3['match']}</strong> ➔ <strong style="color: #34d399;">{m3['top_pick']}</strong> @{q3:.2f}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        prob_combinata_t = (
            (m1["top_perc"] / 100.0)
            * (m2["top_perc"] / 100.0)
            * (m3["top_perc"] / 100.0)
            * 100
        )

        st.success(f"""
                🎯 **QUOTA TOTALE TRIPLA:** **@{q_tot:.2f}**
                💡 **Probabilità Stimata Combinata:** **{prob_combinata_t:.1f}%**
                """)

  # ---------------------------------------------------------
  # 5. MISTA DEL GIORNO
  # ---------------------------------------------------------
  elif current_tab == "Mista":
    st.subheader("📊 Mista del Giorno (Quota ~15.00 - 20.00)")
    st.write(
        "L'algoritmo crea una **schedina mista ad alta quota** selezionando da"
        " 4 a 7 eventi bilanciati."
    )

    target_mista = st.slider(
        "Quota Totale Mista desiderata:",
        min_value=10.0,
        max_value=30.0,
        value=17.5,
        step=0.5,
    )

    lista_p = sorted(
        st.session_state["partite"], key=lambda x: x["top_perc"], reverse=True
    )

    if len(lista_p) < 4:
      st.warning(
          "Servono almeno 4 partite nel palinsesto per generare una mista ad"
          " alta quota."
      )
    else:
      mista_selezionata = []
      q_accumulata = 1.0

      for p_elem in lista_p:
        q_single = calcola_quota_reale(p_elem["top_perc"])
        if (q_accumulata * q_single) <= (target_mista * 1.25):
          mista_selezionata.append((p_elem, q_single))
          q_accumulata *= q_single

        if len(mista_selezionata) >= 7 or q_accumulata >= target_mista:
          break

      if len(mista_selezionata) >= 3:
        st.markdown(
            '<div class="scalata-card" style="border-left: 6px solid #ef4444;'
            ' background: rgba(38, 18, 22, 0.9);">',
            unsafe_allow_html=True,
        )

        prob_comb_mista = 1.0
        txt_mista = (
            "📊 *MISTA DEL GIORNO FOOTBALL AI PRO* 📊\n📅 Data:"
            f" {datetime.today().strftime('%d/%m/%Y')}\n\n"
        )

        for idx_m, (item_m, q_m) in enumerate(mista_selezionata, 1):
          st.markdown(
              f"""
                    <div style="margin-bottom: 8px; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 6px;">
                        <span class="badge-time">⏰ {item_m.get('data', '')} {item_m.get('orario', '15:00')}</span>
                        <span class="badge-league">{item_m.get('lega', '')}</span><br>
                        <strong style="font-size: 0.95rem; color: #f0fdf4;">{idx_m}. {item_m['match']}</strong> ➔
                        👉 <strong style="color: #34d399;">{item_m['top_pick']}</strong> | Quota: <strong style="color: #fbbf24;">@{q_m:.2f}</strong>
                    </div>
                    """,
              unsafe_allow_html=True,
          )

          txt_mista += (
              f"{idx_m}. *{item_m['match']}* [{item_m.get('lega', '')}]\n👉"
              f" Esito: *{item_m['top_pick']}* @{q_m:.2f}\n\n"
          )
          prob_comb_mista *= item_m["top_perc"] / 100.0

        st.markdown("</div>", unsafe_allow_html=True)

        perc_mista_tot = prob_comb_mista * 100
        st.success(f"""
                🎯 **QUOTA TOTALE MISTA:** **@{q_accumulata:.2f}**
                💡 **Probabilità Stimata Combinata:** **{perc_mista_tot:.2f}%**
                """)

        st.download_button(
            label="📥 Scarica Mista (.txt)",
            data=txt_mista,
            file_name=(
                f"mista_del_giorno_{datetime.today().strftime('%Y%m%d')}.txt"
            ),
            mime="text/plain",
        )

  # ---------------------------------------------------------
  # 6. BOMBA DEL GIORNO
  # ---------------------------------------------------------
  elif current_tab == "Bomba":
    st.subheader("💣 Bomba del Giorno (Quota ~100+)")
    st.write(
        "L'algoritmo compone una **schedina bomba ad altissimo moltiplicatore**"
        " accumulando eventi ad alta quota."
    )

    target_bomba = st.slider(
        "Quota Totale Bomba desiderata:",
        min_value=50.0,
        max_value=250.0,
        value=100.0,
        step=10.0,
    )

    lista_p = st.session_state["partite"]

    if len(lista_p) < 4:
      st.warning(
          "Servono almeno 4 partite nel palinsesto per generare una bomba."
      )
    else:
      bomba_selezionata = []
      q_bomba_accumulata = 1.0

      partite_bomba_sort = sorted(
          lista_p,
          key=lambda x: calcola_quota_reale(x["top_perc"]),
          reverse=True,
      )

      for p_elem in partite_bomba_sort:
        q_single = calcola_quota_reale(p_elem["top_perc"])
        bomba_selezionata.append((p_elem, q_single))
        q_bomba_accumulata *= q_single

        if (
            q_bomba_accumulata >= target_bomba
            or len(bomba_selezionata) >= 10
        ):
          break

      if len(bomba_selezionata) >= 3:
        st.markdown(
            '<div class="scalata-card" style="border-left: 6px solid #dc2626;'
            ' background: rgba(45, 12, 18, 0.95);">',
            unsafe_allow_html=True,
        )

        prob_comb_bomba = 1.0
        txt_bomba = (
            "💣 *BOMBA DEL GIORNO FOOTBALL AI PRO (QUOTA 100+)* 💣\n📅"
            f" Data: {datetime.today().strftime('%d/%m/%Y')}\n\n"
        )

        for idx_b, (item_b, q_b) in enumerate(bomba_selezionata, 1):
          st.markdown(
              f"""
                    <div style="margin-bottom: 8px; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 6px;">
                        <span class="badge-time">⏰ {item_b.get('data', '')} {item_b.get('orario', '15:00')}</span>
                        <span class="badge-league">{item_b.get('lega', '')}</span><br>
                        <strong style="font-size: 0.95rem; color: #f0fdf4;">{idx_b}. {item_b['match']}</strong> ➔
                        👉 <strong style="color: #34d399;">{item_b['top_pick']}</strong> | Quota: <strong style="color: #fbbf24;">@{q_b:.2f}</strong>
                    </div>
                    """,
              unsafe_allow_html=True,
          )

          txt_bomba += (
              f"{idx_b}. *{item_b['match']}* [{item_b.get('lega', '')}]\n👉"
              f" Esito: *{item_b['top_pick']}* @{q_b:.2f}\n\n"
          )
          prob_comb_bomba *= item_b["top_perc"] / 100.0

        st.markdown("</div>", unsafe_allow_html=True)

        perc_bomba_tot = prob_comb_bomba * 100
        st.error(f"""
                🔥 **QUOTA TOTALE BOMBA:** **@{q_bomba_accumulata:.2f}**
                💡 **Probabilità Stimata Combinata:** **{perc_bomba_tot:.4f}%**
                """)

        st.download_button(
            label="📥 Scarica Bomba del Giorno (.txt)",
            data=txt_bomba,
            file_name=(
                f"bomba_del_giorno_{datetime.today().strftime('%Y%m%d')}.txt"
            ),
            mime="text/plain",
        )

  # ---------------------------------------------------------
  # 7. SCALATA AI
  # ---------------------------------------------------------
  elif current_tab == "Scalata":
    st.subheader("🚀 Algoritmo Scalata AI (Progressione Cassa)")
    st.write(
        "La Scalata seleziona i match ad **altissima confidenza** ordinandoli"
        " per orario per consentire la re-investizione progressiva."
    )

    col_sc1, col_sc2 = st.columns(2)
    with col_sc1:
      budget_totale = st.number_input(
          "Budget Totale (€)",
          min_value=10.0,
          max_value=1000.0,
          value=100.0,
          step=10.0,
      )
    with col_sc2:
      num_vite = st.number_input(
          "Numero di Vite", min_value=1, max_value=5, value=3
      )

    num_step = st.slider(
        "Numero di Step della Scalata:", min_value=3, max_value=8, value=5
    )

    if st.button("📈 CALCOLA PIANO DI SCALATA AI", key="btn_scalata"):
      partite_cronologiche = sorted(
          st.session_state["partite"], key=lambda x: x.get("datetime_raw", "")
      )
      partite_scalata = [
          p for p in partite_cronologiche if p["top_perc"] >= 55.0
      ][:num_step]

      if len(partite_scalata) < num_step:
        st.warning(
            f"Trovate solo {len(partite_scalata)} partite ad alta confidenza"
            " (≥55%) per la scalata. Prova a ridurre il numero di step."
        )
      else:
        cassa_singola_vita = budget_totale / num_vite
        st.info(
            f"💰 **Cassa per tentativo (1 Vita):** {cassa_singola_vita:.2f}€"
            f" ({num_vite} vite totali)"
        )

        cassa_corrente = cassa_singola_vita
        st.markdown("### 📋 Marcia di Scalata Consigliata:")

        for step_i, match_s in enumerate(partite_scalata, 1):
          quota_stimata = calcola_quota_reale(match_s["top_perc"])
          vincita_step = cassa_corrente * quota_stimata

          st.markdown(
              f"""
                    <div class="scalata-card">
                        <span class="badge-time">⏰ STEP {step_i} — {match_s.get('data', '')} {match_s.get('orario', '15:00')}</span>
                        <span class="badge-league">{match_s.get('lega', '')}</span><br>
                        <h4 style="margin: 8px 0 4px 0; color: #f0fdf4;">{match_s['match']}</h4>
                        👉 Pronostico: <strong>{match_s['top_pick']}</strong> (Confidenza: {match_s['top_perc']:.1f}%)<br>
                        💵 Puntata: <strong>{cassa_corrente:.2f}€</strong> @{quota_stimata} ➔ Vincita: <strong style="color: #34d399;">{vincita_step:.2f}€</strong>
                    </div>
                    """,
              unsafe_allow_html=True,
          )

          cassa_corrente = vincita_step

        moltiplicatore_totale = cassa_corrente / cassa_singola_vita
        st.success(
            f"🎯 **Obiettivo Finale Scalata:** {cassa_corrente:.2f}€"
            f" (Moltiplicatore: **x{moltiplicatore_totale:.2f}**)"
        )

  # ---------------------------------------------------------
  # 8. MULTIPLA TOP PICK
  # ---------------------------------------------------------
  elif current_tab == "Multipla":
    st.subheader("🎟️ Generatore Schedina Multipla")
    num_eventi = st.slider(
        "Numero di eventi per la multipla:",
        min_value=2,
        max_value=10,
        value=4,
        key="slider_multipla",
    )

    partite_ordinate = sorted(
        st.session_state["partite"], key=lambda x: x["top_perc"], reverse=True
    )
    top_eventi = partite_ordinate[:num_eventi]

    if st.button("🎲 GENERA SCHEDINA TOP PICK GLOBALE", key="btn_multipla"):
      prob_combinata = 1.0
      st.markdown("### 📜 La tua Schedina Consigliata:")

      testo_telegram = (
          "⚽ *SCHEDINA MULTI-CAMPIONATO FOOTBALL AI PRO* ⚽\n📅 Data:"
          f" {datetime.today().strftime('%d/%m/%Y')}\n\n"
      )

      for idx_e, ev in enumerate(top_eventi, 1):
        l_info = f"[{ev.get('lega', camp_nome)}]"
        or_info = f"⏰ {ev.get('data', '')} {ev.get('orario', '15:00')}"
        linea = (
            f"{idx_e}. {ev['match']} {l_info} ({or_info}) ➔ {ev['top_pick']}"
            f" ({ev['top_perc']:.1f}%)"
        )
        st.write(f"**{linea}**")
        testo_telegram += (
            f"📌 *{ev['match']}* {l_info} ({or_info})\n👉 Esito:"
            f" *{ev['top_pick']}* (Confidenza: {ev['top_perc']:.1f}%)\n\n"
        )
        prob_combinata *= ev["top_perc"] / 100

      perc_comb_tot = prob_combinata * 100
      st.info(
          "💡 **Probabilità Stimata Combinata della Multipla:**"
          f" {perc_comb_tot:.1f}%"
      )

      st.download_button(
          label="📥 Scarica Schedina per Telegram / WhatsApp (.txt)",
          data=testo_telegram,
          file_name=(
              f"schedina_globale_{datetime.today().strftime('%Y%m%d')}.txt"
          ),
          mime="text/plain",
      )

  # ---------------------------------------------------------
  # HEATMAP RISULTATI ESATTI
  # ---------------------------------------------------------
  st.markdown("---")
  st.subheader("🔥 Heatmap Risultato Esatto")
  match_scelto = st.selectbox("Seleziona Partita:", list(dettagli.keys()))

  if match_scelto:
    casa, trasferta, matrice = dettagli[match_scelto]

    fig_heat = px.imshow(
        matrice,
        labels=dict(
            x=f"Gol {trasferta}", y=f"Gol {casa}", color="Probabilità %"
        ),
        x=["0", "1", "2", "3"],
        y=["0", "1", "2", "3"],
        color_continuous_scale="Greens",
        text_auto=".1f",
    )
    fig_heat.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#a7f3d0"),
    )
    st.plotly_chart(
        fig_heat, use_container_width=True, key=f"heat_{match_scelto}"
    )

elif "partite" in st.session_state:
  st.warning("Nessuna partita futura trovata con i filtri selezionati.")
