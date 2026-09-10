import streamlit as st  # type: ignore[import-not-found]
import requests
import pandas as pd  # type: ignore[import-not-found]
import numpy as np  # type: ignore[import-not-found]
import plotly.express as px  # type: ignore[import-not-found]
import plotly.graph_objects as go  # type: ignore[import-not-found]
from scipy.stats import poisson  # type: ignore[import-not-found]
import json
import re
import time
from datetime import datetime
# ---------------------------------------------------------
# CONFIGURAZIONE PAGINA & CSS STILE EMERALD PITCH
# ---------------------------------------------------------
st.set_page_config(
    page_title="Football AI Match Analyzer Pro",
    page_icon="⚽",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Sfondo scuro toni verde pino / lavagna */
    .stApp {
        background: linear-gradient(145deg, #06110d 0%, #0c1a14 50%, #06110d 100%);
        color: #ecfdf5;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 760px;
    }

    /* Titolo Stile Emerald Glow */
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

    /* Card Espandibili */
    div[data-testid="stExpander"] {
        background: rgba(15, 31, 24, 0.75) !important;
        border: 1px solid rgba(16, 185, 129, 0.2) !important;
        border-radius: 14px !important;
        backdrop-filter: blur(12px);
        margin-bottom: 1rem !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }

    /* Metric Box */
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

    /* Bottoni Gradiente Verde Emerald */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #059669 0%, #10b981 100%);
        color: #ffffff;
        font-size: 15px;
        font-weight: 700;
        border: none;
        border-radius: 10px;
        padding: 0.65em 1em;
        transition: all 0.25s ease;
        box-shadow: 0 4px 14px rgba(16, 185, 129, 0.25);
    }

    .stButton>button:hover {
        background: linear-gradient(90deg, #10b981 0%, #34d399 100%);
        box-shadow: 0 6px 20px rgba(52, 211, 153, 0.4);
        transform: translateY(-1px);
    }

    /* Badge Pick */
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
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# HEADER APPLICAZIONE
# ---------------------------------------------------------
st.markdown('<div class="main-title">⚽ FOOTBALL AI PRO</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Algoritmo Quantitativo Multi-Lega & Studio Tattico Gemini AI</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# RECUPERO CHIAVI API DAI SECRETS O INPUT MANUALE
# ---------------------------------------------------------
odds_key_secret = st.secrets.get("ODDS_API_KEY", "")
gemini_key_secret = st.secrets.get("GEMINI_API_KEY", "")

with st.expander("🔑 **Configurazione API Studio**", expanded=not bool(odds_key_secret)):
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
    "🌐 TUTTI I CAMPIONATI PRINCIPALI": {"key": "MULTI", "home_avg": 1.45, "away_avg": 1.15, "btts_base": 0.52},
    "🇮🇹 Italia - Serie A": {"key": "soccer_italy_serie_a", "home_avg": 1.42, "away_avg": 1.12, "btts_base": 0.52},
    "🇮🇹 Italia - Serie B": {"key": "soccer_italy_serie_b", "home_avg": 1.30, "away_avg": 1.05, "btts_base": 0.48},
    "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Inghilterra - Premier League": {"key": "soccer_epl", "home_avg": 1.55, "away_avg": 1.25, "btts_base": 0.56},
    "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Inghilterra - Championship": {"key": "soccer_efl_champ", "home_avg": 1.35, "away_avg": 1.10, "btts_base": 0.50},
    "🇪🇸 Spagna - La Liga": {"key": "soccer_spain_la_liga", "home_avg": 1.38, "away_avg": 1.08, "btts_base": 0.49},
    "🇩🇪 Germania - Bundesliga": {"key": "soccer_germany_bundesliga", "home_avg": 1.65, "away_avg": 1.35, "btts_base": 0.59},
    "🇫🇷 Francia - Ligue 1": {"key": "soccer_france_ligue_one", "home_avg": 1.40, "away_avg": 1.10, "btts_base": 0.51},
    "🇳🇱 Olanda - Eredivisie": {"key": "soccer_netherlands_eredivisie", "home_avg": 1.68, "away_avg": 1.32, "btts_base": 0.61},
    "🇵🇹 Portogallo - Primeira Liga": {"key": "soccer_portugal_primeira_liga", "home_avg": 1.45, "away_avg": 1.18, "btts_base": 0.53},
    "🇪🇺 UEFA Champions League": {"key": "soccer_uefa_champs_league", "home_avg": 1.60, "away_avg": 1.30, "btts_base": 0.57},
    "🇪🇺 UEFA Europa League": {"key": "soccer_uefa_europa_league", "home_avg": 1.50, "away_avg": 1.20, "btts_base": 0.55},
    "🇪🇺 UEFA Conference League": {"key": "soccer_uefa_europa_conference_league", "home_avg": 1.52, "away_avg": 1.22, "btts_base": 0.55}
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
    ("🇪🇺 Conference League", "soccer_uefa_europa_conference_league")
]

with st.expander("🎛️ **Filtri Palinsesto & Parametri**", expanded=True):
    campionato_scelto = st.selectbox("🏆 Campionato / Selezione", list(code_map.keys()))
    comp_info = code_map[campionato_scelto]
    sport_key = comp_info["key"]

    col_f1, col_f2 = st.columns(2)
    
    with col_f1:
        filtro_data = st.selectbox(
            "📅 Data Eventi", 
            ["Tutte le prossime", "Solo Oggi", "Seleziona Data Specifica"]
        )
    
    with col_f2:
        mercato_preferito = st.selectbox(
            "🎯 Mercato",
            ["Tutti i mercati", "Solo 1X2", "Solo Over / Under", "Solo Goal / No Goal"]
        )

    data_selezionata = None
    if filtro_data == "Seleziona Data Specifica":
        data_selezionata = st.date_input("Scegli data", datetime.today())

    st.markdown("---")
    
    min_confidence = st.slider(
        "⚡ Confidenza minima (%)", 
        min_value=50, 
        max_value=90, 
        value=55, 
        step=5
    )

# ---------------------------------------------------------
# FETCHING PARTITE DA THE ODDS API
# ---------------------------------------------------------
@st.cache_data(ttl=1800)
def scarica_partite_the_odds_api(s_key, key):
    if not key:
        return None, "⚠️ Inserisci la tua chiave API di The Odds API."

    url = f"https://api.the-odds-api.com/v4/sports/{s_key}/odds/?apiKey={key.strip()}&regions=eu&markets=h2h,totals&dateFormat=iso"
    
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

    key_clean = key.strip().replace('"', '').replace("'", "")
    
    prompt = f"""
    Sei un analista tattico quantitativo di calcio.
    Il nostro algoritmo ha calcolato per '{match_name}' le probabilità statistiche base:
    Casa (1): {p1_math:.1f}%, Pareggio (X): {px_math:.1f}%, Ospite (2): {p2_math:.1f}%.

    Valuta attentamente infortuni, turnover, stanchezza da coppe e motivazioni.
    In base alla tua analisi, stabilisci la variazione percentuale (shift) per le due squadre:
    - `home_shift`: tra -8.0 e +8.0 per la casa.
    - `away_shift`: tra -8.0 e +8.0 per l'ospite.

    Rispondi esclusivamente in formato JSON valido con questa struttura:
    {{
        "home_shift": 0.0,
        "away_shift": 0.0,
        "analisi_sintetica": "Analisi sintetica motivata in 3 frasi..."
    }}
    """
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"response_mime_type": "application/json"}
    }
    
    modelli = ["gemini-2.5-flash", "gemini-2.0-flash"]
    
    for mod in modelli:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{mod}:generateContent?key={key_clean}"
        for intento in range(3):
            try:
                response = requests.post(url, json=payload, timeout=25)
                if response.status_code == 200:
                    data = response.json()
                    if 'candidates' in data and len(data['candidates']) > 0:
                        text_res = data['candidates'][0]['content']['parts'][0]['text']
                        json_match = re.search(r'\{.*\}', text_res, re.DOTALL)
                        if json_match:
                            return json.loads(json_match.group(0)), None
                        return json.loads(text_res), None
                elif response.status_code in [429, 503]:
                    time.sleep(3.0 * (intento + 1))
                    continue
                else:
                    break
            except Exception:
                time.sleep(2.0)
                continue

    return None, "⚠️ Server Gemini momentaneamente occupati. Usa il pulsante Instant Batch."

# ---------------------------------------------------------
# GEMINI BATCH CORRECTOR
# ---------------------------------------------------------
def studio_tattico_in_blocco_batch(lista_partite, key):
    if not key:
        return None, "⚠️ Nessuna chiave GEMINI_API_KEY nei Secrets."

    key_clean = key.strip().replace('"', '').replace("'", "")
    modelli = ["gemini-2.5-flash", "gemini-2.0-flash"]
    
    CHUNK_SIZE = 4
    risultati_totali = []
    
    for i in range(0, len(lista_partite), CHUNK_SIZE):
        chunk = lista_partite[i:i + CHUNK_SIZE]
        
        info_txt = ""
        for idx, p in enumerate(chunk, 1):
            info_txt += f"{idx}. {p['match']} -> 1: {p['p1']:.1f}%, X: {p['px']:.1f}%, 2: {p['p2']:.1f}%\n"

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
            "generationConfig": {"response_mime_type": "application/json"}
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
                        if 'candidates' in data and len(data['candidates']) > 0:
                            text_res = data['candidates'][0]['content']['parts'][0]['text']
                            json_match = re.search(r'\[.*\]', text_res, re.DOTALL)
                            if json_match:
                                parsed_chunk = json.loads(json_match.group(0))
                                risultati_totali.extend(parsed_chunk)
                                chunk_successo = True
                                break
                    elif response.status_code in [429, 503]:
                        time.sleep(4.0 * (intento + 1))
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
        return None, "⚠️ Server Google Gemini temporaneamente occupati. Riprova tra qualche istante."

# ---------------------------------------------------------
# CALCOLO PROBABILITÀ E MERCATI ESTESI (O1.5, O2.5, O3.5, U2.5)
# ---------------------------------------------------------
def elab_match_odds(match, comp_info, home_shift=0.0, away_shift=0.0):
    casa = match['home_team']
    trasferta = match['away_team']
    
    prob_1, prob_X, prob_2 = 40.0, 30.0, 30.0
    prob_over, prob_under = None, None

    if match.get('bookmakers'):
        bm = match['bookmakers'][0]
        for m in bm.get('markets', []):
            if m['key'] == 'h2h':
                outcomes = {o['name']: o['price'] for o in m['outcomes']}
                q1 = outcomes.get(casa, 2.5)
                qX = outcomes.get('Draw', 3.2)
                q2 = outcomes.get(trasferta, 2.8)

                inv_tot = (1/q1) + (1/qX) + (1/q2)
                prob_1 = (1/q1 / inv_tot) * 100
                prob_X = (1/qX / inv_tot) * 100
                prob_2 = (1/q2 / inv_tot) * 100

            elif m['key'] == 'totals':
                outcomes = {}
                for o in m.get('outcomes', []):
                    name_clean = o['name'].strip()
                    point = o.get('point', 2.5)
                    if point == 2.5 or '2.5' in name_clean:
                        if 'Over' in name_clean:
                            outcomes['Over'] = o['price']
                        elif 'Under' in name_clean:
                            outcomes['Under'] = o['price']

                if 'Over' in outcomes and 'Under' in outcomes:
                    q_over = outcomes['Over']
                    q_under = outcomes['Under']
                    inv_tot = (1/q_over) + (1/q_under)
                    prob_over = (1/q_over / inv_tot) * 100
                    prob_under = (1/q_under / inv_tot) * 100

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
        gol_attesi_totali = comp_info.get("home_avg", 1.4) + comp_info.get("away_avg", 1.1)

    forza_casa = p1_final / (p1_final + p2_final + 1e-5)
    lambda_c = max(0.65, gol_attesi_totali * forza_casa)
    lambda_t = max(0.55, gol_attesi_totali * (1.0 - forza_casa))

    matrice_raw = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            matrice_raw[i, j] = poisson.pmf(i, lambda_c) * poisson.pmf(j, lambda_t)
            
    matrice = (matrice_raw / np.sum(matrice_raw)) * 100

    # Calcolo dei vari tagli Over/Under dalla matrice
    p_o15 = float(sum(matrice[i, j] for i in range(5) for j in range(5) if (i + j) > 1))
    p_o25 = float(sum(matrice[i, j] for i in range(5) for j in range(5) if (i + j) > 2))
    p_o35 = float(sum(matrice[i, j] for i in range(5) for j in range(5) if (i + j) > 3))
    p_u25 = 100.0 - p_o25

    if prob_over is None:
        prob_over = p_o25
        prob_under = p_u25

    p_casa_segna = 1.0 - np.exp(-lambda_c)
    p_trasferta_segna = 1.0 - np.exp(-lambda_t)
    
    prob_goal_raw = (p_casa_segna * p_trasferta_segna) * 100
    prob_goal = float(min(85.0, max(35.0, prob_goal_raw * 0.7 + prob_over * 0.35)))
    prob_no_goal = 100.0 - prob_goal

    tutti_gli_esiti = {
        "1": p1_final, "X": px_final, "2": p2_final,
        "Over 2.5": prob_over, "Under 2.5": prob_under,
        "Goal": prob_goal, "No Goal": prob_no_goal
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
        "U2.5": prob_under
    }

    return top_pick, top_perc, p1_final, px_final, p2_final, prob_over, prob_under, prob_goal, prob_no_goal, matrice[:4, :4], metriche_estese

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
        oggi_str = datetime.today().strftime('%Y-%m-%d')

        if sport_key == "MULTI":
            with st.spinner("Scaricamento palinsesti da tutti i campionati principali..."):
                leghe_target = TOP_LEAGUES_KEYS
        else:
            leghe_target = [(campionato_scelto, sport_key)]

        with st.spinner("Calcolo probabilità quantitative in corso..."):
            for l_nome, l_key in leghe_target:
                all_matches, error_msg = scarica_partite_the_odds_api(l_key, api_key)

                if all_matches:
                    for m in all_matches:
                        commence_time = m['commence_time'][:10]

                        if filtro_data == "Solo Oggi" and commence_time != oggi_str:
                            continue
                        elif filtro_data == "Seleziona Data Specifica" and data_selezionata and commence_time != data_selezionata.strftime('%Y-%m-%d'):
                            continue

                        casa = m['home_team']
                        trasferta = m['away_team']
                        nome_match = f"{casa} vs {trasferta}"

                        (
                            top_pick, perc_top, p1, px, p2,
                            p_over, p_under, p_goal, p_ng,
                            matrice, m_estese
                        ) = elab_match_odds(m, comp_info)

                        if perc_top >= min_confidence:
                            partite_analizzate.append({
                                "lega": l_nome,
                                "data": commence_time,
                                "match": nome_match,
                                "top_pick": top_pick,
                                "top_perc": perc_top,
                                "p1": p1, "px": px, "p2": p2,
                                "over": p_over, "under": p_under,
                                "goal": p_goal, "no_goal": p_ng,
                                "m_estese": m_estese
                            })
                            dettagli_matrici[nome_match] = (casa, trasferta, matrice)
                            raw_matches_dict[nome_match] = m

        if partite_analizzate:
            st.session_state['partite'] = partite_analizzate
            st.session_state['dettagli_matrici'] = dettagli_matrici
            st.session_state['raw_matches'] = raw_matches_dict
            st.session_state['campionato_corrente'] = campionato_scelto
            st.rerun()
        else:
            st.error("❌ Nessuna partita trovata con i filtri selezionati.")

# ---------------------------------------------------------
# INTERFACCIA UTENTE & DASHBOARD DUMBLESCORE STYLE
# ---------------------------------------------------------
if 'partite' in st.session_state and st.session_state['partite']:
    partite = st.session_state['partite']
    dettagli = st.session_state['dettagli_matrici']
    raw_m_dict = st.session_state.get('raw_matches', {})
    camp_nome = st.session_state.get('campionato_corrente', '')

    # ---------------------------------------------------------
    # MACRO PERFORMANCE HEATMAP DUMBLESCORE STYLE
    # ---------------------------------------------------------
    st.markdown("### 📊 Performance Heatmap per Campionato & Mercato")
    
    df_heatmap = []
    for p in partite:
        m_e = p.get('m_estese', {})
        df_heatmap.append({
            "Campionato": p.get('lega', camp_nome),
            "1X2": m_e.get("1X2", p['top_perc']),
            "BTTS": m_e.get("BTTS", p['goal']),
            "O1.5": m_e.get("O1.5", 70.0),
            "O2.5": m_e.get("O2.5", p['over']),
            "O3.5": m_e.get("O3.5", 35.0),
            "U2.5": m_e.get("U2.5", p['under'])
        })
    
    df_h = pd.DataFrame(df_heatmap)
    if not df_h.empty:
        df_h_grouped = df_h.groupby("Campionato").mean()
        
        fig_macro_heat = px.imshow(
            df_h_grouped,
            labels=dict(x="Mercati", y="Campionati", color="Affidabilità %"),
            color_continuous_scale="RdYlGn",
            text_auto=".1f",
            range_color=[35, 75]
        )
        fig_macro_heat.update_layout(
            height=280,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#a7f3d0')
        )
        st.plotly_chart(fig_macro_heat, use_container_width=True, key="macro_performance_heatmap")

    st.markdown("---")
    st.markdown(f"### ⚽ Palinsesto Dettagliato ({len(partite)} Eventi)")

    # 🧠 PULSANTE ANALISI TATTICA BATCH
    if st.button("⚡ RICALCOLA TUTTI I MATCH CON GEMINI AI (INSTANT BATCH)"):
        if not gemini_api_key:
            st.error("Inserisci la chiave GEMINI_API_KEY nei Secrets prima di continuare.")
        else:
            with st.spinner("Gemini sta analizzando il contesto tattico di tutto il palinsesto..."):
                batch_res, err = studio_tattico_in_blocco_batch(partite, gemini_api_key)
                
                if batch_res and isinstance(batch_res, list):
                    res_map = {item.get("match"): item for item in batch_res if isinstance(item, dict)}
                    
                    for p in partite:
                        match_key = f"gemini_report_{p['match']}"
                        match_info = res_map.get(p['match'])
                        
                        if match_info:
                            h_s = match_info.get("home_shift", 0.0)
                            a_s = match_info.get("away_shift", 0.0)
                            report_txt = f"🧠 **Studio Tattico AI:**\n{match_info.get('analisi_sintetica', '')}\n\n" \
                                         f"⚡ *Shift Applicato:* Casa ({'+' if h_s>=0 else ''}{h_s:.1f}%), Ospite ({'+' if a_s>=0 else ''}{a_s:.1f}%)"
                            
                            st.session_state[match_key] = report_txt
                            
                            raw_match = raw_m_dict.get(p['match'])
                            if raw_match:
                                (
                                    new_pick, new_perc, new_p1, new_px, new_p2,
                                    new_over, new_under, new_goal, new_ng,
                                    new_matrice, new_m_estese
                                ) = elab_match_odds(raw_match, comp_info, home_shift=h_s, away_shift=a_s)

                                casa_team, trasf_team, _ = dettagli[p['match']]
                                st.session_state['dettagli_matrici'][p['match']] = (casa_team, trasf_team, new_matrice)
                                
                                p['top_pick'] = new_pick
                                p['top_perc'] = new_perc
                                p['p1'], p['px'], p['p2'] = new_p1, new_px, new_p2
                                p['over'], p['under'] = new_over, new_under
                                p['goal'], p['no_goal'] = new_goal, new_ng
                                p['m_estese'] = new_m_estese

                    st.success("✅ Analisi in blocco completata per tutte le partite!")
                    st.rerun()
                else:
                    st.error(f"Errore durante l'analisi batch: {err}")

    st.markdown("---")

    for idx, p in enumerate(partite):
        match_key = f"gemini_report_{p['match']}"
        lega_label = p.get('lega', camp_nome)
        
        header_card = f"""
        <div style="display: flex; justify-content: space-between; align-items: center; width: 100%;">
            <div>
                <span class="badge-league">{lega_label}</span>
                <span style="font-weight: 700; font-size: 1.05rem; color: #f0fdf4;">{p['match']}</span>
                <span style="font-size: 0.8rem; color: #6ee7b7; margin-left: 8px;">({p['data']})</span>
            </div>
            <div>
                <span class="badge-pick">🎯 {p['top_pick']} {p['top_perc']:.1f}%</span>
            </div>
        </div>
        """
        
        with st.expander(f"⚽ {p['match']} — {p['top_pick']} ({p['top_perc']:.1f}%)", expanded=True):
            st.markdown(header_card, unsafe_allow_html=True)
            st.write("")

            st.write("**ESITO FINALE (1X2)**")
            c1, c2, c3 = st.columns(3)
            c1.metric("1 (CASA)", f"{p['p1']:.1f}%")
            c2.metric("X (PAREGGIO)", f"{p['px']:.1f}%")
            c3.metric("2 (OSPITE)", f"{p['p2']:.1f}%")

            # 📊 GRAFICO BARRE EMERALD STYLED
            fig_bar = go.Figure(data=[
                go.Bar(
                    x=['Casa (1)', 'Pareggio (X)', 'Ospite (2)'], 
                    y=[p['p1'], p['px'], p['p2']],
                    marker_color=['#10b981', '#f59e0b', '#3b82f6'],
                    text=[f"{p['p1']:.1f}%", f"{p['px']:.1f}%", f"{p['p2']:.1f}%"],
                    textposition='auto'
                )
            ])
            fig_bar.update_layout(
                height=200, 
                margin=dict(l=10, r=10, t=10, b=10), 
                yaxis=dict(range=[0, 100], showgrid=False),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#a7f3d0')
            )
            st.plotly_chart(fig_bar, use_container_width=True, key=f"bar_{idx}_{p['match']}")

            st.write("**MERCATI GOL**")
            m1, m2 = st.columns(2)
            m1.metric("Over 2.5", f"{p['over']:.1f}%")
            m2.metric("Under 2.5", f"{p['under']:.1f}%")

            m3, m4 = st.columns(2)
            m3.metric("Goal", f"{p['goal']:.1f}%")
            m4.metric("No Goal", f"{p['no_goal']:.1f}%")

            st.markdown("---")
            if match_key in st.session_state:
                st.info(st.session_state[match_key])
                if st.button("🔄 Ripristina Statistica Base", key=f"reload_{idx}_{p['match']}"):
                    del st.session_state[match_key]
                    st.rerun()
            else:
                if st.button("🧠 Studio Tattico Gemini & Correzione %", key=f"btn_{idx}_{p['match']}"):
                    with st.spinner("Gemini sta analizzando notizie e formazioni..."):
                        ai_res, err = studio_tattico_gemini(p['match'], p['p1'], p['px'], p['p2'], gemini_api_key)
                        
                        if ai_res:
                            h_s = ai_res.get("home_shift", 0.0)
                            a_s = ai_res.get("away_shift", 0.0)
                            report_txt = f"🧠 **Studio Tattico AI:**\n{ai_res.get('analisi_sintetica', '')}\n\n" \
                                         f"⚡ *Shift Applicato:* Casa ({'+' if h_s>=0 else ''}{h_s:.1f}%), Ospite ({'+' if a_s>=0 else ''}{a_s:.1f}%)"
                            
                            st.session_state[match_key] = report_txt
                            
                            raw_match = raw_m_dict.get(p['match'])
                            if raw_match:
                                (
                                    new_pick, new_perc, new_p1, new_px, new_p2,
                                    new_over, new_under, new_goal, new_ng,
                                    new_matrice, new_m_estese
                                ) = elab_match_odds(raw_match, comp_info, home_shift=h_s, away_shift=a_s)

                                casa_team, trasf_team, _ = dettagli[p['match']]
                                st.session_state['dettagli_matrici'][p['match']] = (casa_team, trasf_team, new_matrice)
                                
                                p['top_pick'] = new_pick
                                p['top_perc'] = new_perc
                                p['p1'], p['px'], p['p2'] = new_p1, new_px, new_p2
                                p['over'], p['under'] = new_over, new_under
                                p['goal'], p['no_goal'] = new_goal, new_ng
                                p['m_estese'] = new_m_estese

                            st.rerun()
                        else:
                            st.error(err)

    # ---------------------------------------------------------
    # GENERATORE SCHEDINA MULTI-CAMPIONATO GLOBALE
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("🎟️ Generatore Schedina Multipla")
    num_eventi = st.slider("Numero di eventi per la multipla:", min_value=2, max_value=8, value=4)

    partite_ordinate = sorted(st.session_state['partite'], key=lambda x: x['top_perc'], reverse=True)
    top_eventi = partite_ordinate[:num_eventi]

    if st.button("🎲 GENERA SCHEDINA TOP PICK GLOBALE"):
        prob_combinata = 1.0
        st.markdown("### 📜 La tua Schedina Consigliata:")

        testo_telegram = f"⚽ *SCHEDINA MULTI-CAMPIONATO FOOTBALL AI PRO* ⚽\n📅 Data: {datetime.today().strftime('%d/%m/%Y')}\n\n"

        for idx_e, ev in enumerate(top_eventi, 1):
            l_info = f"[{ev.get('lega', camp_nome)}]"
            linea = f"{idx_e}. {ev['match']} {l_info} ➔ {ev['top_pick']} ({ev['top_perc']:.1f}%)"
            st.write(f"**{linea}**")
            testo_telegram += f"📌 *{ev['match']}* {l_info}\n👉 Esito: *{ev['top_pick']}* (Confidenza: {ev['top_perc']:.1f}%)\n\n"
            prob_combinata *= (ev['top_perc'] / 100)

        perc_comb_tot = prob_combinata * 100
        testo_telegram += f"💡 *Probabilità Combinata Modello:* {perc_comb_tot:.1f}%\n🤖 Generato con Football AI Match Analyzer"

        st.info(f"💡 **Probabilità Stimata Combinata della Multipla:** {perc_comb_tot:.1f}%")

        st.download_button(
            label="📥 Scarica Schedina pronta per Telegram / WhatsApp (.txt)",
            data=testo_telegram,
            file_name=f"schedina_globale_{datetime.today().strftime('%Y%m%d')}.txt",
            mime="text/plain"
        )

    # ---------------------------------------------------------
    # VISUALIZZAZIONE HEATMAP RISULTATI ESATTI
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("🔥 Heatmap Risultato Esatto")
    match_scelto = st.selectbox("Seleziona Partita:", list(dettagli.keys()))

    if match_scelto:
        casa, trasferta, matrice = dettagli[match_scelto]
        
        fig_heat = px.imshow(
            matrice,
            labels=dict(x=f"Gol {trasferta}", y=f"Gol {casa}", color="Probabilità %"),
            x=['0', '1', '2', '3'],
            y=['0', '1', '2', '3'],
            color_continuous_scale="Greens",
            text_auto=".1f"
        )
        fig_heat.update_layout(
            height=380, 
            margin=dict(l=10, r=10, t=30, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#a7f3d0')
        )
        st.plotly_chart(fig_heat, use_container_width=True, key=f"heat_{match_scelto}")

elif 'partite' in st.session_state:
    st.warning("Nessuna partita trovata con i filtri selezionati.")
