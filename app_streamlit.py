import streamlit as st  # type: ignore[import-not-found]
import requests
import numpy as np  # type: ignore[import-not-found]
from scipy.stats import poisson  # type: ignore[import-not-found]
from math import exp, factorial
from datetime import datetime
from google import genai

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
st.caption("Algoritmo Calibrato: Dixon-Coles, Dynamic Goal Markets & Gemini Context AI")

# ---------------------------------------------------------
# RECUPERO AUTOMATICO CHIAVI API DAI SECRETS (O INPUT MANUALE)
# ---------------------------------------------------------
api_key_secret = st.secrets.get("FOOTBALL_API_KEY", "")
gemini_key_secret = st.secrets.get("GEMINI_API_KEY", "")

with st.expander("🔑 **Configurazione Chiavi API**", expanded=not bool(api_key_secret)):
    if api_key_secret:
        st.success("✅ Chiave Football-Data caricata in automatico dai Secrets!")
        api_key = api_key_secret
    else:
        api_key = st.text_input("Chiave API (Football-Data.org)", type="password")

    if gemini_key_secret:
        st.success("✅ Chiave Google Gemini caricata in automatico dai Secrets!")
        gemini_api_key = gemini_key_secret
    else:
        gemini_api_key = st.text_input("Chiave API (Google Gemini - Opzionale)", type="password")

# ---------------------------------------------------------
# FILTRI DI RICERCA & SELEZIONE CAMPIONATO
# ---------------------------------------------------------
with st.expander("🔍 **Filtri di Ricerca & Campionato**", expanded=True):
    code_map = {
        "🇮🇹 Serie A": {"code": "SA", "home_avg": 1.42, "away_avg": 1.12, "btts_base": 0.52},
        "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League": {"code": "PL", "home_avg": 1.55, "away_avg": 1.25, "btts_base": 0.56},
        "🇪🇸 La Liga": {"code": "PD", "home_avg": 1.38, "away_avg": 1.08, "btts_base": 0.49},
        "🇩🇪 Bundesliga": {"code": "BL1", "home_avg": 1.65, "away_avg": 1.35, "btts_base": 0.59},
        "🇫🇷 Ligue 1": {"code": "FL1", "home_avg": 1.40, "away_avg": 1.10, "btts_base": 0.51},
        "🇳🇱 Eredivisie": {"code": "DED", "home_avg": 1.68, "away_avg": 1.32, "btts_base": 0.61},
        "🇵🇹 Primeira Liga": {"code": "PPL", "home_avg": 1.45, "away_avg": 1.18, "btts_base": 0.53},
        "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Championship": {"code": "ELC", "home_avg": 1.35, "away_avg": 1.10, "btts_base": 0.50},
        "🇪🇺 Champions League": {"code": "CL", "home_avg": 1.60, "away_avg": 1.30, "btts_base": 0.57}
    }
    
    campionato_scelto = st.selectbox("🏆 Seleziona Campionato / Coppa", list(code_map.keys()))
    comp_info = code_map[campionato_scelto]
    comp_code = comp_info["code"]

    col_f1, col_f2 = st.columns(2)
    
    with col_f1:
        filtro_data = st.selectbox(
            "📅 Selezione Data", 
            ["Solo Oggi", "Tutte le prossime", "Seleziona Data Specifica"]
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
        value=60, 
        step=5
    )

# ---------------------------------------------------------
# INTEGRATORE GEMINI CONTEXT AI (REST FAST CALL)
# ---------------------------------------------------------
def analizza_contesto_con_gemini(match_name, pronostico_math, perc_math, key):
    if not key:
        return "⚠️ Inserisci la chiave API di Google Gemini nella sezione Configurazione per abilitare l'analisi."

    prompt = f"""
    Sei un analista tattico di calcio esperto. 
    Il nostro modello matematico-statistico prevede per la partita '{match_name}' l'esito '{pronostico_math}' con una probabilità del {perc_math:.1f}%.
    
    Analizza brevemente (massimo 3-4 frasi sintetiche) il contesto di questa partita:
    1. Eventuali assenze, squalifiche o infortuni rilevanti per le due squadre.
    2. Motivazioni di classifica o stanchezza da impegni ravvicinati/turnover.
    3. Concludi indicando se il contesto conferma o sconsiglia il pronostico.
    """
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    
    # Modelli ufficiali e correnti supportati dalle Google Generative Language API
    modelli = ["gemini-2.5-flash", "gemini-3.5-flash", "gemini-2.5-flash-lite"]
    
    for mod in modelli:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{mod}:generateContent?key={key.strip()}"
        try:
            response = requests.post(url, json=payload, timeout=8)
            if response.status_code == 200:
                data = response.json()
                if 'candidates' in data and len(data['candidates']) > 0:
                    return data['candidates'][0]['content']['parts'][0]['text']
            elif response.status_code in [400, 403]:
                return f"⚠️ Errore API ({response.status_code}): Controlla che la chiave Gemini sia attiva e corretta su Google AI Studio."
        except Exception as e:
            continue

    return "⚠️ Impossibile contattare i server Gemini. Verifica che la chiave sia inserita correttamente e attiva su Google AI Studio."
# ---------------------------------------------------------
# LOGICA DATA SCIENCE CALIBRATA
# ---------------------------------------------------------
def tau_dixon_coles(x, y, lambda_casa, lambda_trasferta, rho=-0.05):
    if x == 0 and y == 0:
        return 1 - (lambda_casa * lambda_trasferta * rho)
    elif x == 0 and y == 1:
        return 1 + (lambda_casa * rho)
    elif x == 1 and y == 0:
        return 1 + (lambda_trasferta * rho)
    elif x == 1 and y == 1:
        return 1 - rho
    else:
        return 1.0


def ottieni_stats_casa_trasferta_pesate(matches_giocati, squadra_id, is_home=True, n_partite=6, half_life=3):
    partite_filtrate = []
    partite_generali = []

    for m in reversed(matches_giocati):
        if m['status'] == 'FINISHED':
            is_home_team = (m['homeTeam']['id'] == squadra_id)
            is_away_team = (m['awayTeam']['id'] == squadra_id)

            if is_home_team or is_away_team:
                gf = m['score']['fullTime']['home'] if is_home_team else m['score']['fullTime']['away']
                gs = m['score']['fullTime']['away'] if is_home_team else m['score']['fullTime']['home']
                
                if len(partite_generali) < n_partite:
                    partite_generali.append({'gf': gf, 'gs': gs})

                if (is_home and is_home_team) or (not is_home and is_away_team):
                    if len(partite_filtrate) < n_partite:
                        partite_filtrate.append({'gf': gf, 'gs': gs})

    dataset_finale = partite_filtrate if len(partite_filtrate) >= 2 else partite_generali

    if not dataset_finale:
        return 1.35, 1.15, 0.5

    pesi = [np.exp(-i / half_life) for i in range(len(dataset_finale))]
    somma_pesi = sum(pesi)

    gf_pesati = sum(p['gf'] * w for p, w in zip(dataset_finale, pesi)) / somma_pesi
    gs_pesati = sum(p['gs'] * w for p, w in zip(dataset_finale, pesi)) / somma_pesi

    gol_totali = [p['gf'] + p['gs'] for p in dataset_finale]
    varianza = float(np.var(gol_totali)) if len(gol_totali) > 1 else 0.5

    return gf_pesati, gs_pesati, varianza


def analizza_partita_precisione_pro(
    gf_casa, gs_casa, var_casa,
    gf_trasferta, gs_trasferta, var_trasferta,
    media_camp_casa=1.45, media_camp_trasferta=1.15, btts_base=0.52
):
    gf_casa, gs_casa = max(gf_casa, 1.0), max(gs_casa, 1.0)
    gf_trasferta, gs_trasferta = max(gf_trasferta, 1.0), max(gs_trasferta, 1.0)

    attacco_casa = gf_casa / media_camp_casa
    difesa_casa = gs_casa / media_camp_trasferta
    attacco_trasferta = gf_trasferta / media_camp_trasferta
    difesa_trasferta = gs_trasferta / media_camp_casa

    lambda_casa = max(1.05, attacco_casa * difesa_trasferta * media_camp_casa)
    lambda_trasferta = max(1.05, attacco_trasferta * difesa_casa * media_camp_trasferta)

    max_gol = 6
    matrice = np.zeros((max_gol, max_gol))
    for i in range(max_gol):
        for j in range(max_gol):
            p_base = poisson.pmf(i, lambda_casa) * poisson.pmf(j, lambda_trasferta)
            correzione = tau_dixon_coles(i, j, lambda_casa, lambda_trasferta)
            matrice[i, j] = p_base * correzione

    matrice = matrice / np.sum(matrice)

    prob_1 = np.sum(np.tril(matrice, -1)) * 100
    prob_X = np.sum(np.diag(matrice)) * 100
    prob_2 = np.sum(np.triu(matrice, 1)) * 100

    prob_under_25 = sum(matrice[i, j] for i in range(max_gol) for j in range(max_gol) if i + j <= 2) * 100
    prob_over_25 = 100 - prob_under_25

    p_casa_segna = 1 - poisson.pmf(0, lambda_casa)
    p_trasferta_segna = 1 - poisson.pmf(0, lambda_trasferta)
    prob_goal_raw = p_casa_segna * p_trasferta_segna
    
    prob_goal_calibrata = (0.65 * prob_goal_raw + 0.35 * btts_base) * 100
    prob_no_goal = 100 - prob_goal_calibrata

    tutti_gli_esiti = {
        "1": prob_1, "X": prob_X, "2": prob_2,
        "Over 2.5": prob_over_25, "Under 2.5": prob_under_25,
        "Goal": prob_goal_calibrata, "No Goal": prob_no_goal
    }

    if mercato_preferito == "Solo 1X2":
        esiti_filtrati = {"1": prob_1, "X": prob_X, "2": prob_2}
    elif mercato_preferito == "Solo Over / Under":
        esiti_filtrati = {"Over 2.5": prob_over_25, "Under 2.5": prob_under_25}
    elif mercato_preferito == "Solo Goal / No Goal":
        esiti_filtrati = {"Goal": prob_goal_calibrata, "No Goal": prob_no_goal}
    else:
        esiti_filtrati = tutti_gli_esiti

    esito_top = max(esiti_filtrati, key=esiti_filtrati.get)

    varianza_media = (var_casa + var_trasferta) / 2
    fattore_stabilita = max(0.90, 1.0 - (varianza_media * 0.02))
    affidabilita_corretta = esiti_filtrati[esito_top] * fattore_stabilita

    return (
        esito_top, affidabilita_corretta,
        prob_1, prob_X, prob_2,
        prob_over_25, prob_under_25, prob_goal_calibrata, prob_no_goal,
        matrice
    )

# ---------------------------------------------------------
# EXECUTION ENGINE
# ---------------------------------------------------------
if st.button("🚀 AVVIA ANALISI AI"):
    if not api_key:
        st.error("Inserisci la chiave API di Football-Data.org per continuare.")
    else:
        headers = {"X-Auth-Token": api_key}
        BASE_URL = "https://api.football-data.org/v4/"

        with st.spinner("Calcolo metriche pesate e matrici probabilità..."):
            res_matches = requests.get(f"{BASE_URL}competitions/{comp_code}/matches", headers=headers)

        if res_matches.status_code == 200:
            all_matches = res_matches.json().get("matches", [])
            partite_analizzate = []
            dettagli_matrici = {}
            oggi_str = datetime.today().strftime('%Y-%m-%d')

            for match in all_matches:
                if match['status'] in ['SCHEDULED', 'TIMED']:
                    data_partita = match['utcDate'][:10]

                    if filtro_data == "Solo Oggi" and data_partita != oggi_str:
                        continue
                    elif filtro_data == "Seleziona Data Specifica" and data_partita != data_selezionata.strftime('%Y-%m-%d'):
                        continue

                    casa = match['homeTeam']['name']
                    trasferta = match['awayTeam']['name']
                    casa_id = match['homeTeam']['id']
                    trasf_id = match['awayTeam']['id']
                    nome_match = f"{casa} vs {trasferta}"

                    gf_c, gs_c, var_c = ottieni_stats_casa_trasferta_pesate(
                        all_matches, casa_id, is_home=True, n_partite=6
                    )
                    gf_t, gs_t, var_t = ottieni_stats_casa_trasferta_pesate(
                        all_matches, trasf_id, is_home=False, n_partite=6
                    )

                    (
                        top_pick, perc_top, p1, px, p2,
                        p_over, p_under, p_goal, p_ng,
                        matrice
                    ) = analizza_partita_precisione_pro(
                        gf_c, gs_c, var_c, gf_t, gs_t, var_t,
                        media_camp_casa=comp_info["home_avg"],
                        media_camp_trasferta=comp_info["away_avg"],
                        btts_base=comp_info["btts_base"]
                    )

                    if perc_top >= min_confidence:
                        partite_analizzate.append({
                            "data": data_partita,
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
        else:
            st.error("Errore di connessione API. Verificare la chiave inserita o i permessi del piano.")

# ---------------------------------------------------------
# INTERFACCIA UTENTE (RISULTATI & GEMINI)
# ---------------------------------------------------------
if 'partite' in st.session_state and st.session_state['partite']:
    partite = st.session_state['partite']
    dettagli = st.session_state['dettagli_matrici']
    camp_nome = st.session_state.get('campionato_corrente', '')

    st.success(f"**{camp_nome}**: trovate **{len(partite)}** partite con modello di precisione")

    for idx, p in enumerate(partite):
        match_key = f"gemini_report_{p['match']}"
        
        with st.expander(f"⚽ **{p['match']}**\n\n🎯 **{p['top_pick']} ({p['top_perc']:.1f}%)**", expanded=True):
            st.write("**Esito Finale (1X2)**")
            c1, c2, c3 = st.columns(3)
            c1.metric("1", f"{p['p1']:.1f}%")
            c2.metric("X", f"{p['px']:.1f}%")
            c3.metric("2", f"{p['p2']:.1f}%")

            st.write("**Mercati Gol**")
            m1, m2 = st.columns(2)
            m1.metric("Over 2.5", f"{p['over']:.1f}%")
            m2.metric("Under 2.5", f"{p['under']:.1f}%")

            m3, m4 = st.columns(2)
            m3.metric("Goal", f"{p['goal']:.1f}%")
            m4.metric("No Goal", f"{p['no_goal']:.1f}%")

            # SEZIONE REPORT GEMINI SEMPRE VISIBILE
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

    st.markdown("---")

    st.subheader("🎟️ Generatore Schedina Multipla")
    num_eventi = st.slider("Numero di eventi per la multipla:", min_value=2, max_value=6, value=3)

    if st.button("🎲 Genera Schedina Top Pick"):
        partite_ordinate = sorted(st.session_state['partite'], key=lambda x: x['top_perc'], reverse=True)
        top_eventi = partite_ordinate[:num_eventi]

        prob_combinata = 1.0
        st.markdown("### 📜 La tua Schedina Consigliata:")

        for idx, ev in enumerate(top_eventi, 1):
            st.write(f"**{idx}. {ev['match']}** ({ev['data']}) ➔ **{ev['top_pick']}** (Confidenza: {ev['top_perc']:.1f}%)")
            prob_combinata *= (ev['top_perc'] / 100)

        st.info(f"💡 **Probabilità Stimata Combinata della Multipla:** {prob_combinata * 100:.1f}%")

    st.markdown("---")

    st.subheader("📊 Matrice Punteggio Esatto")
    match_scelto = st.selectbox("Seleziona Partita:", list(dettagli.keys()))

    if match_scelto:
        casa, trasferta, matrice = dettagli[match_scelto]
        st.caption(f"{casa} vs {trasferta}")

        punteggi = []
        for i in range(4):
            for j in range(4):
                punteggi.append((f"{i} - {j}", matrice[i, j] * 100))

        punteggi_ordinati = sorted(punteggi, key=lambda x: x[1], reverse=True)[:6]

        col_a, col_b = st.columns(2)
        for idx, (punteggio, prob) in enumerate(punteggi_ordinati):
            if idx % 2 == 0:
                col_a.metric(f"Risultato {punteggio}", f"{prob:.1f}%")
            else:
                col_b.metric(f"Risultato {punteggio}", f"{prob:.1f}%")

elif 'partite' in st.session_state:
    st.warning("Nessuna partita trovata con i filtri correnti.")
