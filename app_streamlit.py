import streamlit as st
import requests
import numpy as np
import pandas as pd
from math import exp, factorial
from datetime import datetime

st.set_page_config(page_title="Football AI Analyzer Pro", page_icon="⚽", layout="wide")

st.title("⚽ Football AI Match Analyzer (Versione Multi-Campionato)")
st.markdown("Analisi statistica e predittiva basata su **Poisson Modificato (Dixon-Coles)** e **Forma Recente**.")

# ---------------------------------------------------------
# SIDEBAR - CONFIGURAZIONE E MAPPATURA CAMPIONATI
# ---------------------------------------------------------
st.sidebar.header("⚙️ Impostazioni API & Filtri")
api_key = st.sidebar.text_input("Chiave API (Football-Data.org)", type="password")

# Dizionario esteso dei campionati inclusi nel piano Free
code_map = {
    "🇮🇹 Serie A (Italia)": "SA",
    "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League (Inghilterra)": "PL",
    "🇪🇸 La Liga (Spagna)": "PD",
    "🇩🇪 Bundesliga (Germania)": "BL1",
    "🇫🇷 Ligue 1 (Francia)": "FL1",
    "🇳🇱 Eredivisie (Olanda)": "DED",
    "🇵🇹 Primeira Liga (Portogallo)": "PPL",
    "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Championship (Inghilterra)": "ELC",
    "🇪🇺 Champions League": "CL"
}

campionato_scelto = st.sidebar.selectbox("Seleziona Campionato / Coppa", list(code_map.keys()))
comp_code = code_map[campionato_scelto]

st.sidebar.markdown("---")
st.sidebar.header("🔍 Filtri Analisi")
min_confidence = st.sidebar.slider("Affidabilità minima Pronostico (%)", min_value=50, max_value=95, value=65, step=5)

filtro_data = st.sidebar.radio("Filtro Data Matches", ["Tutte le prossime", "Solo partite di Oggi", "Seleziona Data Specifica"])
data_selezionata = None
if filtro_data == "Seleziona Data Specifica":
    data_selezionata = st.sidebar.date_input("Data partita", datetime.today())

# ---------------------------------------------------------
# LOGICA AVANZATA: DIXON-COLES (CORREZIONE POISSON)
# ---------------------------------------------------------
def poisson_pmf(k, lambd):
    """Calcola la massa di probabilità di Poisson senza SciPy."""
    return exp(-lambd) * (lambd ** k) / factorial(k)

def tau_dixon_coles(x, y, lambda_casa, lambda_trasferta, rho=-0.13):
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

def analizza_partita_avanzata(gf_casa, gs_casa, gf_trasferta, gs_trasferta, media_camp_casa=1.45, media_camp_trasferta=1.15):
    gf_casa, gs_casa = max(gf_casa, 0.1), max(gs_casa, 0.1)
    gf_trasferta, gs_trasferta = max(gf_trasferta, 0.1), max(gs_trasferta, 0.1)

    attacco_casa = gf_casa / media_camp_casa
    difesa_casa = gs_casa / media_camp_trasferta
    attacco_trasferta = gf_trasferta / media_camp_trasferta
    difesa_trasferta = gs_trasferta / media_camp_casa

    lambda_casa = attacco_casa * difesa_trasferta * media_camp_casa
    lambda_trasferta = attacco_trasferta * difesa_casa * media_camp_trasferta

    max_gol = 6
    matrice = np.zeros((max_gol, max_gol))
    for i in range(max_gol):
        for j in range(max_gol):
            p_base = poisson_pmf(i, lambda_casa) * poisson_pmf(j, lambda_trasferta)
            correzione = tau_dixon_coles(i, j, lambda_casa, lambda_trasferta)
            matrice[i, j] = p_base * correzione

    matrice = matrice / np.sum(matrice)

    prob_1 = np.sum(np.tril(matrice, -1)) * 100
    prob_X = np.sum(np.diag(matrice)) * 100
    prob_2 = np.sum(np.triu(matrice, 1)) * 100

    prob_under_25 = sum(matrice[i, j] for i in range(max_gol) for j in range(max_gol) if i + j <= 2) * 100
    prob_over_25 = 100 - prob_under_25
    prob_no_goal = (np.sum(matrice[0, :]) + np.sum(matrice[:, 0]) - matrice[0, 0]) * 100
    prob_goal = 100 - prob_no_goal

    tutti_gli_esiti = {
        "1": prob_1, "X": prob_X, "2": prob_2,
        "Over 2.5": prob_over_25, "Under 2.5": prob_under_25,
        "Goal": prob_goal, "No Goal": prob_no_goal
    }
    esito_top = max(tutti_gli_esiti, key=tutti_gli_esiti.get)
    
    return esito_top, tutti_gli_esiti[esito_top], prob_1, prob_X, prob_2, prob_over_25, prob_under_25, prob_goal, prob_no_goal, matrice

# ---------------------------------------------------------
# CALCOLO FORMA RECENTE DALL'API
# ---------------------------------------------------------
def ottieni_forma_recente(matches_giocati, squadra_id, n_partite=5):
    ultime_partite = []
    for m in reversed(matches_giocati):
        if m['status'] == 'FINISHED':
            if m['homeTeam']['id'] == squadra_id:
                ultime_partite.append({
                    'gf': m['score']['fullTime']['home'],
                    'gs': m['score']['fullTime']['away']
                })
            elif m['awayTeam']['id'] == squadra_id:
                ultime_partite.append({
                    'gf': m['score']['fullTime']['away'],
                    'gs': m['score']['fullTime']['home']
                })
            if len(ultime_partite) == n_partite:
                break

    if not ultime_partite:
        return 1.2, 1.2

    media_gf = sum(p['gf'] for p in ultime_partite) / len(ultime_partite)
    media_gs = sum(p['gs'] for p in ultime_partite) / len(ultime_partite)
    return media_gf, media_gs

# ---------------------------------------------------------
# RUN ANALISI
# ---------------------------------------------------------
if st.sidebar.button("🚀 Avvia Analisi"):
    if not api_key:
        st.error("Inserisci una chiave API valida nella barra laterale.")
    else:
        headers = {"X-Auth-Token": api_key}
        BASE_URL = "https://api.football-data.org/v4/"

        with st.spinner(f"Scaricamento dati e forma recente per {campionato_scelto}..."):
            res_matches = requests.get(f"{BASE_URL}competitions/{comp_code}/matches", headers=headers)

        if res_matches.status_code == 200:
            all_matches = res_matches.json().get("matches", [])
            partite_analizzate = []
            dettagli_matrici = {}
            oggi_str = datetime.today().strftime('%Y-%m-%d')

            for match in all_matches:
                if match['status'] in ['SCHEDULED', 'TIMED']:
                    data_partita = match['utcDate'][:10]

                    if filtro_data == "Solo partite di Oggi" and data_partita != oggi_str:
                        continue
                    elif filtro_data == "Seleziona Data Specifica" and data_partita != data_selezionata.strftime('%Y-%m-%d'):
                        continue

                    casa = match['homeTeam']['name']
                    trasferta = match['awayTeam']['name']
                    casa_id = match['homeTeam']['id']
                    trasf_id = match['awayTeam']['id']
                    nome_match = f"{casa} vs {trasferta}"

                    gf_c, gs_c = ottieni_forma_recente(all_matches, casa_id, n_partite=5)
                    gf_t, gs_t = ottieni_forma_recente(all_matches, trasf_id, n_partite=5)

                    top_pick, perc_top, p1, px, p2, p_over, p_under, p_goal, p_ng, matrice = analizza_partita_avanzata(
                        gf_c, gs_c, gf_t, gs_t
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
            st.error(f"Errore API ({res_matches.status_code}): Verificare la chiave o i limiti di accesso per questa competizione.")

# ---------------------------------------------------------
# INTERFACCIA VISIVA
# ---------------------------------------------------------
if 'partite' in st.session_state and st.session_state['partite']:
    partite = st.session_state['partite']
    dettagli = st.session_state['dettagli_matrici']
    camp_nome = st.session_state.get('campionato_corrente', '')

    st.success(f"Analisi completata per **{camp_nome}**: trovate **{len(partite)}** partite corrispondenti ai filtri!")

    st.subheader("📋 Pronostici Partite")
    for p in partite:
        with st.expander(f"⚽ **{p['match']}** ({p['data']})  ➔  PRONOSTICO TOP: **{p['top_pick']} ({p['top_perc']:.1f}%)**", expanded=True):
            # Prima riga: 1X2
            col1, col2, col3 = st.columns(3)
            col1.metric("1 (Vittoria Casa)", f"{p['p1']:.1f}%")
            col2.metric("X (Pareggio)", f"{p['px']:.1f}%")
            col3.metric("2 (Vittoria Trasferta)", f"{p['p2']:.1f}%")

            st.markdown("---")

            # Seconda riga: Over/Under e Goal/No Goal
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Over 2.5 Gol", f"{p['over']:.1f}%")
            c2.metric("Under 2.5 Gol", f"{p['under']:.1f}%")
            c3.metric("Goal (Entrambe segnano)", f"{p['goal']:.1f}%")
            c4.metric("No Goal", f"{p['no_goal']:.1f}%")

    st.markdown("---")
    st.subheader("📊 Dettaglio Risultati Esatti per Singolo Match")
    match_scelto = st.selectbox("Seleziona una partita per analizzare la matrice di punteggi:", list(dettagli.keys()))

    if match_scelto:
        casa, trasferta, matrice = dettagli[match_scelto]
        st.markdown(f"**Probabilità Punteggi Esatti: {casa} (righe) vs {trasferta} (colonne)**")
        for i in range(5):
            cols = st.columns(5)
            for j in range(5):
                prob = matrice[i, j] * 100
                cols[j].metric(f"Risultato {i}-{j}", f"{prob:.1f}%")

elif 'partite' in st.session_state:
    st.warning("Nessuna partita trovata con i filtri selezionati per questo campionato.")