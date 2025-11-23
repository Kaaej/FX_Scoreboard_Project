# app.py
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from streamlit_option_menu import option_menu


def get_data_dir() -> Path:
    env_dir = os.environ.get("FX_DATA_DIR") or os.environ.get("DATA_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    
    here = Path(__file__).resolve().parent
    cand = here / "data"
    if cand.exists():
        return cand

    return here.parent / "data"

DATA_DIR = get_data_dir()

# fichiers attendus
SIG_PANEL_FILE   = DATA_DIR / "signals_panel.csv"
BT_PAIR_IS_FILE  = DATA_DIR / "bt_fx_pair_summary.csv"
BT_PORT_IS_FILE  = DATA_DIR / "bt_portfolio_summary.csv"
BT_PAIR_OOS_FILE = DATA_DIR / "bt_oos_fx_pair_summary.csv"
BT_PORT_OOS_FILE = DATA_DIR / "bt_oos_portfolio_summary.csv"
BT_PORT_TS_FILE  = DATA_DIR / "bt_portfolio_timeseries.csv"  # IS uniquement

# colonnes de scores
FACTOR_COLS = [
    "ValueScore",
    "CarryScore",
    "MomentumScore",
    "MacroScore",
    "HedgeCostScore",
]
ALL_SCORE_COLS = ["FX_Score"] + FACTOR_COLS


# ==========================================================
# Chargement des données
# ==========================================================

@st.cache_data
def load_signals_panel() -> pd.DataFrame:
    if not SIG_PANEL_FILE.exists():
        raise FileNotFoundError(f"{SIG_PANEL_FILE} introuvable.")
    df = pd.read_csv(SIG_PANEL_FILE, parse_dates=["Date"])
    df = df.sort_values(["pair", "Date"])
    return df


@st.cache_data
def load_backtest_summaries():
    out = {}
    if BT_PAIR_IS_FILE.exists():
        out["pair_is"] = pd.read_csv(BT_PAIR_IS_FILE)
    if BT_PORT_IS_FILE.exists():
        out["port_is"] = pd.read_csv(BT_PORT_IS_FILE)
    if BT_PAIR_OOS_FILE.exists():
        out["pair_oos"] = pd.read_csv(BT_PAIR_OOS_FILE)
    if BT_PORT_OOS_FILE.exists():
        out["port_oos"] = pd.read_csv(BT_PORT_OOS_FILE)
    return out


@st.cache_data
def load_portfolio_timeseries():
    if not BT_PORT_TS_FILE.exists():
        return None
    df = pd.read_csv(BT_PORT_TS_FILE, parse_dates=["Date"]).sort_values("Date")

    # Colonnes de rendements mensuels
    for col in ["R_dyn", "R_0", "R_100"]:
        if col not in df.columns:
            raise ValueError(f"Colonne {col} manquante dans {BT_PORT_TS_FILE}")

    df["cum_dyn"]  = (1 + df["R_dyn"]).cumprod()
    df["cum_0"]    = (1 + df["R_0"]).cumprod()
    df["cum_100"]  = (1 + df["R_100"]).cumprod()
    return df


# ==========================================================
# Utilitaires
# ==========================================================

def latest_complete_by_pair(sig: pd.DataFrame) -> pd.DataFrame:
    """
    Pour chaque paire, renvoie la dernière ligne où TOUS les facteurs
    ALL_SCORE_COLS sont non-NaN.
    """
    rows = []
    for pair, g in sig.groupby("pair"):
        g2 = g.dropna(subset=ALL_SCORE_COLS)
        if len(g2) == 0:
            continue
        rows.append(g2.iloc[-1])
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).set_index("pair")
    return out


def normalize_weights(raw_weights: dict) -> dict:
    v = np.array(list(raw_weights.values()), dtype=float)
    s = v.sum()
    if s <= 0:
        v = np.ones_like(v) / len(v)
    else:
        v = v / s
    return {k: float(val) for k, val in zip(raw_weights.keys(), v)}


def build_custom_fx_score(latest: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """
    Calcule FX_Score_custom à partir des poids utilisateurs.
    latest : index=pair, colonnes incluant FACTOR_COLS.
    """
    w_vec = np.array([weights[c] for c in FACTOR_COLS])
    X = latest[FACTOR_COLS].to_numpy(dtype=float)
    fx_custom = X @ w_vec
    res = latest.copy()
    res["FX_Score_custom"] = fx_custom
    return res


# ==========================================================
# Pages
# ==========================================================

def page_dashboard(sig: pd.DataFrame, pair: str):
    st.header("Tableau de bord par devise (+ heatmap en bas)")

    col_left, col_right = st.columns([1, 2])

    df_pair = sig[sig["pair"] == pair].copy()
    df_complete = df_pair.dropna(subset=ALL_SCORE_COLS)

    if df_complete.empty:
        st.error(
            f"Aucune date où tous les scores sont disponibles pour {pair}. "
            "Les séries FX ou macro vont probablement plus loin que les derniers scores."
        )
        return

    last_row  = df_complete.iloc[-1]
    last_date = last_row["Date"]
    
    with col_left:
        st.subheader("Vue synthétique (dernière observation complète)")
        st.write(f"**Paire :** {pair}")
        st.write(f"**Date :** {last_date.date()}")

        synth = pd.DataFrame(
            {
                "Score (z-score)": [
                    last_row["FX_Score"],
                    last_row["ValueScore"],
                    last_row["CarryScore"],
                    last_row["MomentumScore"],
                    last_row["MacroScore"],
                    last_row["HedgeCostScore"],
                ]
            },
            index=[
                "FX_Score",
                "ValueScore",
                "CarryScore",
                "MomentumScore",
                "MacroScore",
                "HedgeCostScore",
            ],
        )
        st.table(synth.style.format("{:.4f}"))

        if "HedgeRatio" in df_pair.columns:
            h_last = last_row.get("HedgeRatio", np.nan)
            if pd.notna(h_last):
                st.markdown(f"**HedgeRatio (modèle)** : `{h_last:.2f}`")
            else:
                st.markdown("**HedgeRatio (modèle)** : n/a")

        # warning si les prix FX vont plus loin que la macro
        max_date_all = df_pair["Date"].max()
        if max_date_all > last_date:
            st.info(
                f"Les données de prix FX pour {pair} vont jusqu’au {max_date_all.date()}, "
                f"mais les scores complets ne sont disponibles que "
                f"jusqu’au {last_date.date()}."
            )

    with col_right:
        st.subheader("Historique des scores & hedge")
        df_pair_plot = df_pair.set_index("Date")

        cols_to_plot = ["FX_Score"]
        if "HedgeRatio" in df_pair_plot.columns:
            cols_to_plot.append("HedgeRatio")

        fig = px.line(
            df_pair_plot,
            y=cols_to_plot,
            labels={"value": "Score / HedgeRatio", "Date": "Date", "variable": ""},
        )
        st.plotly_chart(fig, use_container_width=True)

        # =================== Heatmap + poids custom ===================

    st.subheader("Heatmap des scores (FX_Score_custom en bas)")

    latest = latest_complete_by_pair(sig)
    if latest.empty:
        st.error("Impossible de construire la heatmap : aucun point avec scores complets.")
        return

    with st.expander("Ajuster les poids des facteurs (pour FX_Score_custom)", expanded=True):
        cols = st.columns(len(FACTOR_COLS))
        raw_w = {}
        for c, name in zip(cols, FACTOR_COLS):
            with c:
                raw_w[name] = st.slider(
                    label=name,
                    min_value=0,
                    max_value=100,
                    value=25 if name in ["ValueScore", "CarryScore", "MomentumScore"] else 10,
                    step=5,
                    help="Poids relatif de ce facteur dans FX_Score_custom",
                )

        raw_w_float = {k: v / 100.0 for k, v in raw_w.items()}
        weights = normalize_weights(raw_w_float)

        st.caption(
            "Les poids sont automatiquement renormalisés pour sommer à 100 %. "
            f"Vecteur utilisé : {', '.join([f'{k}={weights[k]:.2f}' for k in FACTOR_COLS])}"
        )

    latest_with_custom = build_custom_fx_score(latest, weights)

    mat = latest_with_custom[["FX_Score_custom"] + FACTOR_COLS].T
    mat.index = [
        "FX_Score_custom",
        "ValueScore",
        "CarryScore",
        "MomentumScore",
        "MacroScore",
        "HedgeCostScore",
    ]
    mat_clipped = mat.clip(-3, 3)

    # >>> ICI on ajoute les valeurs dans les cases avec text_auto <<<
    fig_hm = px.imshow(
        mat_clipped,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-3,
        zmax=3,
        labels=dict(x="Paire", y="Facteur", color="Z-score (cap ±3)"),
        text_auto=".1f",          # <-- affiche les valeurs dans chaque case
    )
    fig_hm.update_xaxes(side="top")
    st.plotly_chart(fig_hm, use_container_width=True)

# ----------------------------------------------------------

def page_backtests(sig: pd.DataFrame):
    st.header("Backtests (IS / OOS)")

    bt = load_backtest_summaries()
    tab_is, tab_oos = st.tabs(["In-sample (IS)", "Out-of-sample (OOS)"])

    with tab_is:
        st.subheader("Par paire (IS)")
        if "pair_is" in bt:
            st.dataframe(bt["pair_is"].style.format(precision=3), use_container_width=True)
        else:
            st.warning("Fichier bt_fx_pair_summary.csv introuvable.")

        st.subheader("Portefeuille agrégé (IS)")
        if "port_is" in bt:
            st.dataframe(bt["port_is"].style.format(precision=3), use_container_width=True)
        else:
            st.warning("Fichier bt_portfolio_summary.csv introuvable.")

    with tab_oos:
        st.subheader("Par paire (OOS)")
        if "pair_oos" in bt:
            st.dataframe(bt["pair_oos"].style.format(precision=3), use_container_width=True)
        else:
            st.warning("Fichier bt_oos_fx_pair_summary.csv introuvable.")

        st.subheader("Portefeuille agrégé (OOS)")
        if "port_oos" in bt:
            st.dataframe(bt["port_oos"].style.format(precision=3), use_container_width=True)
        else:
            st.warning("Fichier bt_oos_portfolio_summary.csv introuvable.")


# ----------------------------------------------------------

def page_performance(sig: pd.DataFrame):
    st.header("Performance de la stratégie")

    df_ts = load_portfolio_timeseries()
    if df_ts is None:
        st.error(
            f"{BT_PORT_TS_FILE.name} introuvable dans {DATA_DIR}. "
            "L’onglet performance ne peut pas afficher les courbes."
        )
        return

    st.subheader("Portefeuille agrégé – performance cumulée (IS)")

    fig = px.line(
        df_ts,
        x="Date",
        y=["cum_dyn", "cum_0", "cum_100"],
        labels={"value": "Index de performance (base 1)", "Date": "Date", "variable": ""},
        title="Performance cumulée : dynamique vs 0% hedge vs 100% hedge",
    )
    fig.update_yaxes(type="log")  # optionnel
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
**Lecture :**
- *Dynamique* : hedge piloté par FX\_Score.
- *0% hedge* : exposition FX totalement ouverte.
- *100% hedge* : couverture permanente (coût de hedge maximal).
"""
    )


# ==========================================================
# Application principale
# ==========================================================

def main():
    st.set_page_config(
        page_title="FX Scoreboard",
        page_icon="💱",
        layout="wide",
    )

    try:
        sig = load_signals_panel()
    except Exception as e:
        st.error(f"Erreur lors du chargement des signaux : {e}")
        return

    pairs = sorted(sig["pair"].unique())

    # -------- sidebar : choix de la paire + navigation "barres" --------
    with st.sidebar:
        st.title("FX Scoreboard")
        pair = st.selectbox("Choix de la paire :", pairs, index=0)

        st.markdown("### Navigation")
        page = option_menu(
            menu_title=None,
            options=[
                "Tableau de bord par devise",
                "Backtests (IS / OOS)",
                "Performance",
            ],
            icons=["activity", "bar-chart-line", "graph-up"],  # optionnel
            default_index=0,
            orientation="vertical",
            styles={
                "container": {"padding": "0!important"},
                "icon": {"color": "#ff4b4b", "font-size": "1rem"},
                "nav-link": {
                    "font-size": "0.95rem",
                    "padding": "0.4rem 0.8rem",
                    "border-radius": "0.3rem",
                    "text-align": "left",
                    "margin": "0.15rem 0",
                },
                "nav-link-selected": {
                    "background-color": "#ff4b4b",
                    "color": "white",
                },
            },
        )

        st.markdown("---")
        st.caption(f"Dossier données : `{DATA_DIR}`")

    # -------- routing selon la page choisie --------
    if page == "Tableau de bord par devise":
        page_dashboard(sig, pair)
    elif page == "Backtests (IS / OOS)":
        page_backtests(sig)
    elif page == "Performance":
        page_performance(sig)



if __name__ == "__main__":
    main()
