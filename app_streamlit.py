import streamlit as st  # type: ignore[import-not-found]
import requests  # type: ignore[import-not-found, import-untyped]
import numpy as np  # type: ignore[import-not-found]
from scipy.stats import poisson  # type: ignore[import-not-found]
from datetime import datetime

# Configurazione responsive per Mobile
st.set_page_config(page_title="Football AI Mobile", page_icon="⚽", layout="centered")

# CSS Custom per ottimizzare la resa visiva su Smartphone
st.markdown("""
<style>
    /* Riduce i margini superiori per schermi piccoli */
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; padding-left: 1rem; padding-right: 1rem; }
    /* Aumenta la dimensione dei font dei pulsanti e delle metriche */
    .stButton>button { width: 100%; height: 3em; font-size: 16px; font-weight: bold; border-radius: 8px; }
    [data-testid="stMetricValue"] { font-size: 20px !important; }
    [data-testid="stMetricLabel"] { font-size: 12px !important; }
    div[data-testid="stExpander"] { border-radius: 10px; border: 1px solid #374151; }
</style>
""", unsafe_allow_html=True)

st.title("⚽ Football AI Mobile")
st.caption("Analisi statistica basata su Poisson & Dixon-Coles")

# ---------------------------------------------------------
# MENU CONFIGURAZIONE IN-PAGE (PERFETTO PER SMARTPHONE)
# ---------------------------------------------------------
with st.expander("⚙️ **Imposta API e Seleziona Campionato**", expanded=True):
    api_key = st.text_input("Chiave API (Football-Data.org)", type="password")
    
    code_map = {
        "🇮🇹 Serie A": "SA",
        "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League": "PL",
        "🇪🇸 La Liga": "PD",
        "🇩🇪 Bundesliga": "BL1",
        "🇫🇷 Ligue 1": "FL1",
        "🇳🇱 Eredivisie": "DED",
        "🇵🇹 Primeira Liga": "PPL",
        "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Championship": "ELC",
        "🇪🇺 Champions League": "CL"
    }
    
    campionato_scelto = st.selectbox("Campionato / Coppa", list(code_map.keys()))
    comp_code = code_map[campionato_scelto]

with st.expander("🔍 **Filtri di Ricerca**", expanded=False):
    min_confidence = st.slider("Affidabilità minima (%)", min_value=50, max_value=95, value=65, step=5)
    filtro_data = st.radio("Filtro Data", ["Tutte le prossime", "Solo Oggi", "Seleziona Data"])
    data_selezionata = None
    if filtro_data == "Seleziona Data":
        data_selezionata = st.date_input("Data partita", datetime.today())

# ---------------------------------------------------------
# LOGICA MATEMATICA DI POISSON & DIXON-COLES
# ---------------------------------------------------------
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
            p_base = poisson.pmf(i, lambda_casa) * poisson.pmf(j, lambda_trasferta)
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

def ottieni_forma_recente(matches_giocati, squadra_id, n_partite=5):
    ultime_partite = []
    for m in reversed(matches_giocati):
        if m['status'] == 'FINISHED':
            if m['homeTeam']['id'] == squadra_id:
                ultime_partite.append({'gf': m['score']['fullTime']['home'], 'gs': m['score']['fullTime']['away']})
            elif m['awayTeam']['id'] == squadra_id:
                ultime_partite.append({'gf': m['score']['fullTime']['away'], 'gs': m['score']['fullTime']['home']})
            if len(ultime_partite) == n_partite:
                break

    if not ultime_partite:
        return 1.2, 1.2

    media_gf = sum(p['gf'] for p in ultime_partite) / len(ultime_partite)
    media_gs = sum(p['gs'] for p in ultime_partite) / len(ultime_partite)
    return media_gf, media_gs

# ---------------------------------------------------------
# PULSANTE AVVIO ANALISI
# ---------------------------------------------------------
if st.button("🚀 AVVIA ANALISI PARTITE"):
    if not api_key:
        st.error("Inserisci la chiave API per continuare.")
    else:
        headers = {"X-Auth-Token": api_key}
        BASE_URL = "https://api.football-data.org/v4/"

        with st.spinner("Elaborazione dati in corso..."):
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
                    elif filtro_data == "Seleziona Data" and data_partita != data_selezionata.strftime('%Y-%m-%d'):
                        continue

                    casa = match['homeTeam']['name']
                    trasferta = match['awayTeam']['name']
                    nome_match = f"{casa} vs {trasferta}"

                    gf_c, gs_c = ottieni_forma_recente(all_matches, match['homeTeam']['id'], n_partite=5)
                    gf_t, gs_t = ottieni_forma_recente(all_matches, match['awayTeam']['id'], n_partite=5)

                    top_pick, perc_top, p1, px, p2, p_over, p_under, p_goal, p_ng, matrice = analizza_partita_avanzata(gf_c, gs_c, gf_t, gs_t)

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
            st.error("Errore di connessione API. Verificare la chiave inserita.")

# ---------------------------------------------------------
# VISUALIZZAZIONE RISULTATI MOBILE-FRIENDLY
# ---------------------------------------------------------
if 'partite' in st.session_state and st.session_state['partite']:
    partite = st.session_state['partite']
    dettagli = st.session_state['dettagli_matrici']
    camp_nome = st.session_state.get('campionato_corrente', '')

    st.success(f"**{camp_nome}**: trovate **{len(partite)}** partite")

    for p in partite:
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

    st.markdown("---")
    st.subheader("📊 Matrice Punteggio Esatto")
    match_scelto = st.selectbox("Seleziona Partita:", list(dettagli.keys()))

    if match_scelto:
        casa, trasferta, matrice = dettagli[match_scelto]
        st.caption(f"{casa} vs {trasferta}")
        
        # Mostra i primi 6 punteggi più probabili in formato lista verticale per il telefono
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
