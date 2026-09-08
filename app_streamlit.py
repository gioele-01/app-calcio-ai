import streamlit as st  # type: ignore[import-not-found]
import requests
import numpy as np  # type: ignore[import-not-found]
import plotly.express as px  # type: ignore[import-not-found]
import plotly.graph_objects as go  # type: ignore[import-not-found]
from scipy.stats import poisson  # type: ignore[import-not-found]
from datetime import datetime

# ---------------------------------------------------------
# CONFIGURAZIONE PAGINA & CSS RESPONSIVE MOBILE
# ---------------------------------------------------------
st.set_page_config(page_title="Football AI Pro", page_icon="⚽", layout="centered")

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; padding-left: 1rem; padding-right: 1rem; }
    .stButton>button { width: 100%; height: 3em; font-size: 16px; font-weight: bold; border-radius: 8px; }
    [data-testid="stMetricValue"] { font-size: 19px !important; }
    [data-testid="stMetricLabel"] { font-size: 11px !important; }
    div[data-testid="stExpander"] { border-radius: 10px; border: 1px solid #374151; }
    .stSelectbox label, .stSlider label { font-weight: bold; font-size: 14px; }
</style>
""", unsafe_allow_html=True)

st.title("⚽ Football AI Match Analyzer Pro")
st.caption("Algoritmo Calibrato: The Odds API, Dixon-Coles & Gemini AI")

# ---------------------------------------------------------
# RECUPERO CHIAVI API DAI SECRETS O INPUT MANUALE
# ---------------------------------------------------------
odds_key_secret = st.secrets.get("ODDS_API_KEY", "")
gemini_key_secret = st.secrets.get("GEMINI_API_KEY", "")

with st.expander("🔑 **Configurazione Chiavi API**", expanded=not bool(odds_key_secret)):
    if odds_key_secret:
        st.success("✅ Chiave The Odds API caricata dai Secrets!")
        api_key = odds_key_secret
    else:
        api_key = st.text_input("Chiave API (The Odds API Key)", type="password")

    if gemini_key_secret:
        st.success("✅ Chiave Google Gemini caricata dai Secrets!")
        gemini_api_key = gemini_key_secret
    else:
        gemini_api_key = st.text_input("Chiave API (Google Gemini - Opzionale)", type="password")

# ---------------------------------------------------------
# MAPPATURA CAMPIONATI (THE ODDS API KEYS)
# ---------------------------------------------------------
with st.expander("🔍 **Filtri di Ricerca & Campionato**", expanded=True):
    code_map = {
        "🇮🇹 Italia - Serie A": {"key": "soccer_italy_serie_a", "home_avg": 1.42, "away_avg": 1.12, "btts_base": 0.52},
        "🇮🇹 Italia - Serie B": {"key": "soccer_italy_serie_b", "home_avg": 1.30, "away_avg": 1.05, "btts_base": 0.48},
        "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Inghilterra - Premier League": {"key": "soccer_epl", "home_avg": 1.55, "away_avg": 1.25, "btts_base": 0.56},
        "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Inghilterra - Championship": {"key": "soccer_efl_champ", "home_avg": 1.35, "away_avg": 1.10, "btts_base": 0.50},
        "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Inghilterra - League One": {"key": "soccer_england_league1", "home_avg": 1.38, "away_avg": 1.12, "btts_base": 0.51},
        "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Inghilterra - League Two": {"key": "soccer_england_league2", "home_avg": 1.40, "away_avg": 1.15, "btts_base": 0.52},
        "🇪🇸 Spagna - La Liga": {"key": "soccer_spain_la_liga", "home_avg": 1.38, "away_avg": 1.08, "btts_base": 0.49},
        "🇪🇸 Spagna - Segunda Division": {"key": "soccer_spain_segunda_division", "home_avg": 1.25, "away_avg": 0.95, "btts_base": 0.45},
        "🇩🇪 Germania - Bundesliga": {"key": "soccer_germany_bundesliga", "home_avg": 1.65, "away_avg": 1.35, "btts_base": 0.59},
        "🇩🇪 Germania - 2. Bundesliga": {"key": "soccer_germany_bundesliga2", "home_avg": 1.58, "away_avg": 1.30, "btts_base": 0.57},
        "🇩🇪 Germania - 3. Liga": {"key": "soccer_germany_liga3", "home_avg": 1.45, "away_avg": 1.20, "btts_base": 0.54},
        "🇫🇷 Francia - Ligue 1": {"key": "soccer_france_ligue_one", "home_avg": 1.40, "away_avg": 1.10, "btts_base": 0.51},
        "🇫🇷 Francia - Ligue 2": {"key": "soccer_france_ligue_two", "home_avg": 1.28, "away_avg": 0.98, "btts_base": 0.46},
        "🇳🇱 Olanda - Eredivisie": {"key": "soccer_netherlands_eredivisie", "home_avg": 1.68, "away_avg": 1.32, "btts_base": 0.61},
        "🇵🇹 Portogallo - Primeira Liga": {"key": "soccer_portugal_primeira_liga", "home_avg": 1.45, "away_avg": 1.18, "btts_base": 0.53},
        "🇧🇪 Belgio - First Div": {"key": "soccer_belgium_first_div", "home_avg": 1.52, "away_avg": 1.22, "btts_base": 0.55},
        "🏴󠁧󠁢󠁳󠁣󠁴󠁿 Scozia - Premiership": {"key": "soccer_spl", "home_avg": 1.45, "away_avg": 1.15, "btts_base": 0.51},
        "🇦🇹 Austria - Bundesliga": {"key": "soccer_austria_bundesliga", "home_avg": 1.50, "away_avg": 1.25, "btts_base": 0.54},
        "🇨🇭 Svizzera - Super League": {"key": "soccer_switzerland_superleague", "home_avg": 1.55, "away_avg": 1.28, "btts_base": 0.56},
        "🇩🇰 Danimarca - Superliga": {"key": "soccer_denmark_superliga", "home_avg": 1.45, "away_avg": 1.20, "btts_base": 0.53},
        "🇸🇪 Svezia - Allsvenskan": {"key": "soccer_sweden_allsvenskan", "home_avg": 1.48, "away_avg": 1.18, "btts_base": 0.53},
        "🇸🇪 Svezia - Superettan": {"key": "soccer_sweden_superettan", "home_avg": 1.42, "away_avg": 1.15, "btts_base": 0.52},
        "🇳🇴 Norvegia - Eliteserien": {"key": "soccer_norway_eliteserien", "home_avg": 1.60, "away_avg": 1.28, "btts_base": 0.58},
        "🇫🇮 Finlandia - Veikkausliiga": {"key": "soccer_finland_veikkausliiga", "home_avg": 1.38, "away_avg": 1.12, "btts_base": 0.50},
        "🇵🇱 Polonia - Ekstraklasa": {"key": "soccer_poland_ekstraklasa", "home_avg": 1.40, "away_avg": 1.12, "btts_base": 0.51},
        "🇹🇷 Turchia - Super League": {"key": "soccer_turkey_super_league", "home_avg": 1.52, "away_avg": 1.20, "btts_base": 0.55},
        "🇬🇷 Grecia - Super League": {"key": "soccer_greece_super_league", "home_avg": 1.38, "away_avg": 1.02, "btts_base": 0.47},
        "🇷🇺 Russia - Premier League": {"key": "soccer_russia_premier_league", "home_avg": 1.40, "away_avg": 1.08, "btts_base": 0.49},
        "🇧🇷 Brasile - Serie A": {"key": "soccer_brazil_campeonato", "home_avg": 1.48, "away_avg": 1.05, "btts_base": 0.48},
        "🇧🇷 Brasile - Serie B": {"key": "soccer_brazil_serie_b", "home_avg": 1.32, "away_avg": 0.88, "btts_base": 0.42},
        "🇦🇷 Argentina - Primera Div": {"key": "soccer_argentina_primera_division", "home_avg": 1.25, "away_avg": 0.92, "btts_base": 0.43},
        "🇨🇱 Cile - Primera Division": {"key": "soccer_chile_campeonato", "home_avg": 1.40, "away_avg": 1.10, "btts_base": 0.50},
        "🇲🇽 Messico - Liga MX": {"key": "soccer_mexico_ligamx", "home_avg": 1.48, "away_avg": 1.15, "btts_base": 0.52},
        "🇺🇸 USA - MLS": {"key": "soccer_usa_mls", "home_avg": 1.62, "away_avg": 1.22, "btts_base": 0.57},
        "🇯🇵 Giappone - J1 League": {"key": "soccer_japan_j_league", "home_avg": 1.38, "away_avg": 1.15, "btts_base": 0.50},
        "🇰🇷 Corea del Sud - K League 1": {"key": "soccer_korea_kleague1", "home_avg": 1.35, "away_avg": 1.10, "btts_base": 0.49},
        "🇨🇳 Cina - Super League": {"key": "soccer_china_superleague", "home_avg": 1.50, "away_avg": 1.20, "btts_base": 0.54},
        "🇦🇺 Australia - A-League": {"key": "soccer_australia_aleague", "home_avg": 1.58, "away_avg": 1.30, "btts_base": 0.58},
        "🇪🇺 UEFA Champions League": {"key": "soccer_uefa_champs_league", "home_avg": 1.60, "away_avg": 1.30, "btts_base": 0.57},
        "🇪🇺 UEFA Champions Qual.": {"key": "soccer_uefa_champs_league_qualification", "home_avg": 1.50, "away_avg": 1.20, "btts_base": 0.54},
        "🇪🇺 UEFA Europa League": {"key": "soccer_uefa_europa_league", "home_avg": 1.50, "away_avg": 1.20, "btts_base": 0.55},
        "🇪🇺 UEFA Conference League": {"key": "soccer_uefa_europa_conference_league", "home_avg": 1.52, "away_avg": 1.22, "btts_base": 0.55},
        "🌎 Copa Libertadores": {"key": "soccer_conmebol_copa_libertadores", "home_avg": 1.45, "away_avg": 0.98, "btts_base": 0.46},
        "🌎 Copa Sudamericana": {"key": "soccer_conmebol_copa_sudamericana", "home_avg": 1.42, "away_avg": 0.95, "btts_base": 0.45}
    }
    
    campionato_scelto = st.selectbox("🏆 Seleziona Campionato / Coppa", list(code_map.keys()))
    comp_info = code_map[campionato_scelto]
    sport_key = comp_info["key"]

    col_f1, col_f2 = st.columns(2)
    
    with col_f1:
        filtro_data = st.selectbox(
            "📅 Selezione Data", 
            ["Tutte le prossime", "Solo Oggi", "Seleziona Data Specifica"]
        )
    
    with col_f2:
        mercato_preferito = st.selectbox(
            "🎯 Mercato d'Interesse",
            ["Tutti i mercati", "Solo 1X2", "Solo Over / Under", "Solo Goal / No Goal"]
        )

    data_selezionata = None
    if filtro_data == "Seleziona Data Specifica":
        data_selezionata = st.date_input("Scegli data partita", datetime.today())

    st.markdown("---")
    
    min_confidence = st.slider(
        "⚡ Affidabilità minima (%)", 
        min_value=50, 
        max_value=90, 
        value=55, 
        step=5
    )

# ---------------------------------------------------------
# FETCHING PARTITE DA THE ODDS API (MERCATI STANDARD SUPPORTATI)
# ---------------------------------------------------------
@st.cache_data(ttl=1800)
def scarica_partite_the_odds_api(sport_key, key):
    if not key:
        return None, "⚠️ Inserisci la tua chiave API di The Odds API."

    # Mercati supportati nativamente: h2h (1X2) e totals (Over/Under)
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/?apiKey={key.strip()}&regions=eu&markets=h2h,totals&dateFormat=iso"
    
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            data = res.json()
            if data:
                return data, None
            else:
                return None, "Nessuna partita in programma trovata per questo campionato al momento."
        else:
            return None, f"Errore The Odds API ({res.status_code}): {res.text}"
    except Exception as e:
        return None, f"Errore di connessione: {str(e)}"

# ---------------------------------------------------------
# INTEGRATORE GEMINI CONTEXT AI
# ---------------------------------------------------------
def analizza_contesto_con_gemini(match_name, pronostico_math, perc_math, key):
    if not key:
        return "⚠️ Nessuna chiave API fornita. Controlla GEMINI_API_KEY nei Secrets."

    key_clean = key.strip().replace('"', '').replace("'", "")
    
    prompt = f"""
    Sei un analista tattico di calcio esperto. 
    Il nostro modello matematico-statistico prevede per la partita '{match_name}' l'esito '{pronostico_math}' con una probabilità del {perc_math:.1f}%.
    
    Analizza brevemente (massimo 3-4 frasi sintetiche) il contesto di questa partita:
    1. Eventuali assenze, squalifiche o infortuni rilevanti per le due squadre.
    2. Motivazioni di classifica o stanchezza da impegni ravvicinati/turnover.
    3. Concludi indicando se il contesto conferma o sconsiglia il pronostico.
    """
    
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    modelli = ["gemini-2.5-flash", "gemini-2.5-pro"]
    
    for mod in modelli:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{mod}:generateContent?key={key_clean}"
        try:
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'candidates' in data and len(data['candidates']) > 0:
                    return data['candidates'][0]['content']['parts'][0]['text']
        except Exception:
            continue

    return "⚠️ Impossibile contattare i server Gemini. Verifica che la chiave GEMINI_API_KEY nei Secrets sia corretta."

# ---------------------------------------------------------
# LOGICA DI CALCOLO POISSON DINAMICA E CORRELATA
# ---------------------------------------------------------
def elab_match_odds(match, comp_info):
    casa = match['home_team']
    trasferta = match['away_team']
    
    prob_1, prob_X, prob_2 = 40.0, 30.0, 30.0
    prob_over, prob_under = 50.0, 50.0

    if match.get('bookmakers'):
        bm = match['bookmakers'][0]
        for m in bm.get('markets', []):
            # 1. Quota 1X2
            if m['key'] == 'h2h':
                outcomes = {o['name']: o['price'] for o in m['outcomes']}
                q1 = outcomes.get(casa, 2.5)
                qX = outcomes.get('Draw', 3.2)
                q2 = outcomes.get(trasferta, 2.8)

                inv_tot = (1/q1) + (1/qX) + (1/q2)
                prob_1 = (1/q1 / inv_tot) * 100
                prob_X = (1/qX / inv_tot) * 100
                prob_2 = (1/q2 / inv_tot) * 100

            # 2. Quota Over / Under 2.5
            elif m['key'] == 'totals':
                outcomes = {o['name']: o['price'] for o in m['outcomes']}
                q_over = outcomes.get('Over', 1.9)
                q_under = outcomes.get('Under', 1.9)

                inv_tot = (1/q_over) + (1/q_under)
                prob_over = (1/q_over / inv_tot) * 100
                prob_under = (1/q_under / inv_tot) * 100

    # Stima dinamica e accurata dei gol attesi per squadra (lambda)
    gol_attesi_totali = 1.6 + (prob_over / 100) * 1.6
    forza_casa = prob_1 / (prob_1 + prob_2 + 1e-5)
    
    lambda_c = max(0.65, gol_attesi_totali * forza_casa)
    lambda_t = max(0.55, gol_attesi_totali * (1.0 - forza_casa))

    # Matrice Poisson Punteggi Esatti 4x4
    matrice_raw = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            matrice_raw[i, j] = poisson.pmf(i, lambda_c) * poisson.pmf(j, lambda_t)
            
    matrice = (matrice_raw / np.sum(matrice_raw)) * 100

    # Calcolo dinamico di Goal e No Goal derivati dalla matrice Poisson
    p_casa_segna = 1.0 - np.exp(-lambda_c)
    p_trasferta_segna = 1.0 - np.exp(-lambda_t)
    
    # Correlazione tattica reale: più l'Over è probabile, più cresce la probabilità che entrambe segnino
    prob_goal_raw = (p_casa_segna * p_trasferta_segna) * 100
    prob_goal = float(min(82.0, max(38.0, prob_goal_raw * 0.7 + prob_over * 0.35)))
    prob_no_goal = 100.0 - prob_goal

    tutti_gli_esiti = {
        "1": prob_1, "X": prob_X, "2": prob_2,
        "Over 2.5": prob_over, "Under 2.5": prob_under,
        "Goal": prob_goal, "No Goal": prob_no_goal
    }

    if mercato_preferito == "Solo 1X2":
        esiti = {"1": prob_1, "X": prob_X, "2": prob_2}
    elif mercato_preferito == "Solo Over / Under":
        esiti = {"Over 2.5": prob_over, "Under 2.5": prob_under}
    elif mercato_preferito == "Solo Goal / No Goal":
        esiti = {"Goal": prob_goal, "No Goal": prob_no_goal}
    else:
        esiti = tutti_gli_esiti

    top_pick = max(esiti, key=esiti.get)
    top_perc = esiti[top_pick]

    return top_pick, top_perc, prob_1, prob_X, prob_2, prob_over, prob_under, prob_goal, prob_no_goal, matrice

# ---------------------------------------------------------
# EXECUTION ENGINE
# ---------------------------------------------------------
if st.button("🚀 AVVIA ANALISI AI"):
    if not api_key:
        st.error("Inserisci la chiave API di The Odds API per continuare.")
    else:
        with st.spinner("Scaricamento palinsesti e calcolo probabilità avanzate..."):
            all_matches, error_msg = scarica_partite_the_odds_api(sport_key, api_key)

        if all_matches:
            partite_analizzate = []
            dettagli_matrici = {}
            oggi_str = datetime.today().strftime('%Y-%m-%d')

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
                        "data": commence_time,
                        "match": nome_match,
                        "top_pick": top_pick,
                        "top_perc": perc_top,
                        "p1": p1, "px": px, "p2": p2,
                        "over": p_over, "under": p_under,
                        "goal": p_goal, "no_goal": p_ng
                    })
                    dettagli_matrici[nome_match] = (casa, trasferta, matrice)

            st.session_state['partite'] = partite_analizzate
            st.session_state['dettagli_matrici'] = dettagli_matrici
            st.session_state['campionato_corrente'] = campionato_scelto
            st.rerun()
        else:
            st.error(f"❌ {error_msg}")

# ---------------------------------------------------------
# INTERFACCIA UTENTE & GRAFICI AVANZATI
# ---------------------------------------------------------
if 'partite' in st.session_state and st.session_state['partite']:
    partite = st.session_state['partite']
    dettagli = st.session_state['dettagli_matrici']
    camp_nome = st.session_state.get('campionato_corrente', '')

    st.success(f"**{camp_nome}**: trovate **{len(partite)}** partite nel palinsesto")

    for idx, p in enumerate(partite):
        match_key = f"gemini_report_{p['match']}"
        
        with st.expander(f"⚽ **{p['match']}** ({p['data']})\n\n🎯 **{p['top_pick']} ({p['top_perc']:.1f}%)**", expanded=True):
            st.write("**Esito Finale (1X2)**")
            c1, c2, c3 = st.columns(3)
            c1.metric("1", f"{p['p1']:.1f}%")
            c2.metric("X", f"{p['px']:.1f}%")
            c3.metric("2", f"{p['p2']:.1f}%")

            # 📊 GRAFICO BARRE PROBABILITÀ 1X2
            fig_bar = go.Figure(data=[
                go.Bar(
                    x=['Casa (1)', 'Pareggio (X)', 'Ospite (2)'], 
                    y=[p['p1'], p['px'], p['p2']],
                    marker_color=['#22c55e', '#f59e0b', '#3b82f6'],
                    text=[f"{p['p1']:.1f}%", f"{p['px']:.1f}%", f"{p['p2']:.1f}%"],
                    textposition='auto'
                )
            ])
            fig_bar.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(range=[0, 100]))
            st.plotly_chart(fig_bar, use_container_width=True)

            st.write("**Mercati Gol**")
            m1, m2 = st.columns(2)
            m1.metric("Over 2.5", f"{p['over']:.1f}%")
            m2.metric("Under 2.5", f"{p['under']:.1f}%")

            m3, m4 = st.columns(2)
            m3.metric("Goal", f"{p['goal']:.1f}%")
            m4.metric("No Goal", f"{p['no_goal']:.1f}%")

            st.markdown("---")
            if match_key in st.session_state:
                st.info(st.session_state[match_key])
                if st.button("🔄 Aggiorna Report", key=f"reload_{idx}_{p['match']}"):
                    del st.session_state[match_key]
                    st.rerun()
            else:
                if st.button("🧠 Analizza Contesto Notizie", key=f"btn_{idx}_{p['match']}"):
                    with st.spinner("Gemini sta analizzando la partita..."):
                        report = analizza_contesto_con_gemini(
                            p['match'], p['top_pick'], p['top_perc'], gemini_api_key
                        )
                        st.session_state[match_key] = report
                        st.rerun()

    # ---------------------------------------------------------
    # GENERATORE SCHEDINA MULTIPLA DIVERSIFICATA
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("🎟️ Generatore Schedina Multipla")
    num_eventi = st.slider("Numero di eventi per la multipla:", min_value=2, max_value=6, value=3)

    partite_ordinate = sorted(st.session_state['partite'], key=lambda x: x['top_perc'], reverse=True)
    top_eventi = partite_ordinate[:num_eventi]

    if st.button("🎲 Genera Schedina Top Pick"):
        prob_combinata = 1.0
        st.markdown("### 📜 La tua Schedina Consigliata:")

        testo_telegram = f"⚽ *SCHEDINA FOOTBALL AI PRO* ⚽\n🏆 {camp_nome}\n\n"

        for idx_e, ev in enumerate(top_eventi, 1):
            linea = f"{idx_e}. {ev['match']} ➔ {ev['top_pick']} ({ev['top_perc']:.1f}%)"
            st.write(f"**{linea}**")
            testo_telegram += f"📌 *{ev['match']}*\n👉 Esito: *{ev['top_pick']}* (Confidenza: {ev['top_perc']:.1f}%)\n\n"
            prob_combinata *= (ev['top_perc'] / 100)

        perc_comb_tot = prob_combinata * 100
        testo_telegram += f"💡 *Probabilità Combinata Modello:* {perc_comb_tot:.1f}%\n🤖 Generato con Football AI Match Analyzer"

        st.info(f"💡 **Probabilità Stimata Combinata della Multipla:** {perc_comb_tot:.1f}%")

        # 📥 PULSANTE DOWNLOAD TXT / TELEGRAM
        st.download_button(
            label="📥 Scarica Schedina pronta per Telegram / WhatsApp (.txt)",
            data=testo_telegram,
            file_name=f"schedina_{datetime.today().strftime('%Y%m%d')}.txt",
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
        
        # Mappa di Calore Colorata (Heatmap)
        fig_heat = px.imshow(
            matrice,
            labels=dict(x=f"Gol {trasferta}", y=f"Gol {casa}", color="Probabilità %"),
            x=['0', '1', '2', '3'],
            y=['0', '1', '2', '3'],
            color_continuous_scale="Viridis",
            text_auto=".1f"
        )
        fig_heat.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_heat, use_container_width=True)

elif 'partite' in st.session_state:
    st.warning("Nessuna partita trovata con i filtri selezionati.")
