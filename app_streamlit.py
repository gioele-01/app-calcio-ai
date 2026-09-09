# pyright: reportMissingImports=false

try:
    import streamlit as st
    import requests
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    from scipy.stats import poisson
    import json
    import re
    import time
    from datetime import datetime
except ImportError as exc:
    raise ImportError(
        "Missing required runtime dependencies. Install with: pip install streamlit requests numpy plotly scipy"
    ) from exc

# ---------------------------------------------------------
# CONFIGURAZIONE PAGINA & CSS AVANZATO DARK PREMIUM
# ---------------------------------------------------------
st.set_page_config(
    page_title="Football AI Match Analyzer Pro",
    page_icon="⚽",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    /* Importazione font moderni */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Sfondo globale dark sfumato */
    .stApp {
        background: linear-gradient(135deg, #0b0f19 0%, #111827 50%, #0b0f19 100%);
        color: #f3f4f6;
    }

    /* Modifica container centrale */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 760px;
    }

    /* Header e Titolo Fluo */
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
    }

    .sub-title {
        font-size: 0.9rem;
        color: #9ca3af;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 500;
    }

    /* Card Espandibili Stilizzate */
    div[data-testid="stExpander"] {
        background: rgba(30, 41, 59, 0.7) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        backdrop-filter: blur(10px);
        margin-bottom: 1rem !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }

    /* Metric Box personalizzati */
    div[data-testid="stMetricValue"] {
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        color: #38bdf8 !important;
    }

    div[data-testid="stMetricLabel"] {
        font-size: 0.75rem !important;
        color: #9ca3af !important;
        font-weight: 600 !important;
        text-transform: uppercase;
    }

    /* Pulsanti con gradiente e glow */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #2563eb 0%, #1d4ed8 100%);
        color: #ffffff;
        font-size: 15px;
        font-weight: 700;
        border: none;
        border-radius: 10px;
        padding: 0.6em 1em;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }

    .stButton>button:hover {
        background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
        box-shadow: 0 6px 18px rgba(59, 130, 246, 0.5);
        transform: translateY(-1px);
    }

    /* Badge Esito Consigliato */
    .badge-pick {
        background: linear-gradient(90deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.4);
    }

    .badge-league {
        background: rgba(255, 255, 255, 0.1);
        color: #e5e7eb;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
    }

    /* Sfondi per Alert e Info */
    .stAlert {
        border-radius: 10px;
        background-color: rgba(15, 23, 42, 0.9) !important;
        border: 1px solid rgba(56, 189, 248, 0.3) !important;
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
# MAPPATURA CAMPIONATI (THE ODDS API KEYS)
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
    ("🇪🇺 Champions League", "soccer_uefa_champs_league"),
    ("🇪🇺 Europa League", "soccer_uefa_europa_league")
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
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={key_clean}"
    tempi_attesa = [1.0, 3.0, 5.0]
    
    for attesa in tempi_attesa:
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
            elif response.status_code == 429:
                time.sleep(attesa)
                continue
            else:
                return None, f"Errore Gemini ({response.status_code})"
        except Exception:
            time.sleep(attesa)
            continue

    return None, "⚠️ Quota API temporaneamente satura. Usa il pulsante 'Instant Batch'."

# ---------------------------------------------------------
# GEMINI BATCH CORRECTOR (1 SOLA CHIAMATA)
# ---------------------------------------------------------
def studio_tattico_in_blocco_batch(lista_partite, key):
    if not key:
        return None, "⚠️ Nessuna chiave GEMINI_API_KEY nei Secrets."

    key_clean = key.strip().replace('"', '').replace("'", "")
    
    info_txt = ""
    for idx, p in enumerate(lista_partite, 1):
        info_txt += f"{idx}. {p['match']} -> 1: {p['p1']:.1f}%, X: {p['px']:.1f}%, 2: {p['p2']:.1f}%\n"

    prompt = f"""
    Sei un analista tattico quantitativo di calcio.
    Analizza il contesto di ciascuna partita:

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
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={key_clean}"
    
    for intento in range(3):
        try:
            response = requests.post(url, json=payload, timeout=30)
            if response.status_code == 200:
                data = response.json()
                if 'candidates' in data and len(data['candidates']) > 0:
                    text_res = data['candidates'][0]['content']['parts'][0]['text']
                    json_match = re.search(r'\[.*\]', text_res, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group(0)), None
                    return json.loads(text_res), None
            elif response.status_code == 429:
                time.sleep(5)
                continue
            else:
                return None, f"Errore Gemini ({response.status_code})"
        except Exception as e:
            return None, f"Errore connessione: {str(e)}"

    return None, "⚠️ Quota API satura. Riprova tra poco."

# ---------------------------------------------------------
# CALCOLO PROBABILITÀ CON MODIFICATORE TATTICO AI
# ---------------------------------------------------------
def elab_match_odds(match, comp_info, home_shift=0.0, away_shift=0.0):
    casa = match['home_team']
    trasferta = match['away_team']
    
    prob_1, prob_X, prob_2 = 40.0, 30.0, 30.0
    prob_over, prob_under = 50.0, 50.0

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
                outcomes = {o['name']: o['price'] for o in m['outcomes']}
                q_over = outcomes.get('Over', 1.9)
                q_under = outcomes.get('Under', 1.9)

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

    gol_attesi_totali = 1.6 + (prob_over / 100) * 1.6
    forza_casa = p1_final / (p1_final + p2_final + 1e-5)
    
    lambda_c = max(0.65, gol_attesi_totali * forza_casa)
    lambda_t = max(0.55, gol_attesi_totali * (1.0 - forza_casa))

    matrice_raw = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            matrice_raw[i, j] = poisson.pmf(i, lambda_c) * poisson.pmf(j, lambda_t)
            
    matrice = (matrice_raw / np.sum(matrice_raw)) * 100

    p_casa_segna = 1.0 - np.exp(-lambda_c)
    p_trasferta_segna = 1.0 - np.exp(-lambda_t)
    
    prob_goal_raw = (p_casa_segna * p_trasferta_segna) * 100
    prob_goal = float(min(82.0, max(38.0, prob_goal_raw * 0.7 + prob_over * 0.35)))
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

    return top_pick, top_perc, p1_final, px_final, p2_final, prob_over, prob_under, prob_goal, prob_no_goal, matrice

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
                            matrice
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
                                "goal": p_goal, "no_goal": p_ng
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
# INTERFACCIA UTENTE REDESIGN DARK PREMIUM
# ---------------------------------------------------------
if 'partite' in st.session_state and st.session_state['partite']:
    partite = st.session_state['partite']
    dettagli = st.session_state['dettagli_matrici']
    raw_m_dict = st.session_state.get('raw_matches', {})
    camp_nome = st.session_state.get('campionato_corrente', '')

    st.markdown(f"### 📊 Palinsesto Analizzato ({len(partite)} Eventi)")

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
                                    new_matrice
                                ) = elab_match_odds(raw_match, comp_info, home_shift=h_s, away_shift=a_s)

                                casa_team, trasf_team, _ = dettagli[p['match']]
                                st.session_state['dettagli_matrici'][p['match']] = (casa_team, trasf_team, new_matrice)
                                
                                p['top_pick'] = new_pick
                                p['top_perc'] = new_perc
                                p['p1'], p['px'], p['p2'] = new_p1, new_px, new_p2
                                p['over'], p['under'] = new_over, new_under
                                p['goal'], p['no_goal'] = new_goal, new_ng

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
                <span style="font-weight: 700; font-size: 1.05rem; color: #f9fafb;">{p['match']}</span>
                <span style="font-size: 0.8rem; color: #6b7280; margin-left: 8px;">({p['data']})</span>
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

            # 📊 GRAFICO BARRE DARK STYLED
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
                font=dict(color='#9ca3af')
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
                                    new_matrice
                                ) = elab_match_odds(raw_match, comp_info, home_shift=h_s, away_shift=a_s)

                                casa_team, trasf_team, _ = dettagli[p['match']]
                                st.session_state['dettagli_matrici'][p['match']] = (casa_team, trasf_team, new_matrice)
                                
                                p['top_pick'] = new_pick
                                p['top_perc'] = new_perc
                                p['p1'], p['px'], p['p2'] = new_p1, new_px, new_p2
                                p['over'], p['under'] = new_over, new_under
                                p['goal'], p['no_goal'] = new_goal, new_ng

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

        # 📥 PULSANTE DOWNLOAD TXT / TELEGRAM
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
            color_continuous_scale="Darkmint",
            text_auto=".1f"
        )
        fig_heat.update_layout(
            height=380, 
            margin=dict(l=10, r=10, t=30, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#9ca3af')
        )
        st.plotly_chart(fig_heat, use_container_width=True, key=f"heat_{match_scelto}")

elif 'partite' in st.session_state:
    st.warning("Nessuna partita trovata con i filtri selezionati.")
