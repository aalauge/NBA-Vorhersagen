"""
KOBRA – Kader-Stärke (gemeinsames Modul)
=========================================
Wird von drei Stellen benutzt, damit überall exakt dieselbe Rechnung läuft:
  - update_kader.py   (Mac, täglich): rechnet Historie + aktuelle Spielerwerte
  - kobra.py          (GitHub Actions): heutige Kaderstärke für die Vorhersage
  - backtest_kader.py (Backtest)

Idee: Für jedes Team und Spiel – wie gut sind die Spieler, die auflaufen?
  Spielerwert  = Game Score (Hollinger) pro Minute, aus Vorsaison (Gewicht 0,5) + laufender Saison,
                 mit 300 "Pseudo-Minuten" auf Ersatzniveau (Rookies / wenig Daten → vorsichtig)
  Teamwert     = 240 × minutengewichteter Schnitt der Top-8 nach erwarteten Minuten
Wichtig: Spielerwerte nutzen nur Spiele VOR dem jeweiligen Datum (kein Blick in die Zukunft).
"""
import glob
import os

import numpy as np
import pandas as pd

PRIOR_W = 0.5       # Gewicht der Vorsaison-Minuten
SHRINK_MIN = 300    # Pseudo-Minuten auf Ersatzniveau
TOP_N = 8           # Top-8 Spieler nach erwarteten Minuten
REPL_FAKTOR = 0.6   # Ersatzniveau = 60 % des Liga-Schnitts (Game Score pro Minute)
DEFAULT_MIN = 12.0  # erwartete Minuten ohne jede Datenbasis


# ─── Rohdaten ────────────────────────────────────────────────────────────────

def lade_logs(ordner: str) -> pd.DataFrame:
    """Liest daten/player_games_YYYY-YY.csv (nba_api LeagueGameLog, Spieler-Ebene)."""
    teile = []
    for f in sorted(glob.glob(os.path.join(ordner, "player_games_*.csv"))):
        d = pd.read_csv(f)
        d["season"] = int(os.path.basename(f)[13:17])  # 2024-25 → 2024 (wie balldontlie)
        teile.append(d)
    d = pd.concat(teile, ignore_index=True)
    d["GAME_DATE"] = pd.to_datetime(d["GAME_DATE"])
    d["MIN"] = pd.to_numeric(d["MIN"], errors="coerce").fillna(0)
    d["GMSC"] = (d.PTS + 0.4 * d.FGM - 0.7 * d.FGA - 0.4 * (d.FTA - d.FTM) + 0.7 * d.OREB
                 + 0.3 * d.DREB + d.STL + 0.7 * d.AST + 0.7 * d.BLK - 0.4 * d.PF - d.TOV)
    d["home"] = d["MATCHUP"].str.contains("vs.", regex=False)
    return d


def ersatzniveau(d: pd.DataFrame) -> float:
    rate = d.groupby("season").apply(lambda x: x.GMSC.sum() / x.MIN.sum(), include_groups=False).mean()
    return REPL_FAKTOR * float(rate)


def _vorsaison(d: pd.DataFrame) -> pd.DataFrame:
    tot = d.groupby(["PLAYER_ID", "season"]).agg(
        p_min=("MIN", "sum"), p_gs=("GMSC", "sum"), p_gp=("MIN", "size")).reset_index()
    tot["season"] += 1
    return tot


def _wert(p_min, p_gs, p_gp, c_min, c_gs, c_gp, repl):
    """Spielerwert (Game Score/Min) und erwartete Minuten – identische Formel für Historie und live."""
    w_min = PRIOR_W * np.asarray(p_min, float) + np.asarray(c_min, float)
    rate = (PRIOR_W * np.asarray(p_gs, float) + np.asarray(c_gs, float) + SHRINK_MIN * repl) / (w_min + SHRINK_MIN)
    gp = PRIOR_W * np.asarray(p_gp, float) + np.asarray(c_gp, float)
    exp_min = np.where(gp > 0, w_min / np.where(gp > 0, gp, 1), DEFAULT_MIN)
    return rate, np.clip(exp_min, 0, 40)


def team_wert(rate, exp_min) -> float:
    """240 × minutengewichteter Schnitt der Top-N Spieler nach erwarteten Minuten."""
    x = pd.DataFrame({"rate": np.asarray(rate, float), "exp_min": np.asarray(exp_min, float)})
    x = x.sort_values("exp_min", ascending=False, kind="mergesort").head(TOP_N)
    if len(x) == 0:
        return np.nan
    return 240 * float(np.average(x["rate"], weights=x["exp_min"].clip(lower=1)))


# ─── Historie (pro Spiel, nur Spiele davor) ───────────────────────────────────

def kader_historie(d: pd.DataFrame) -> pd.DataFrame:
    """Kaderstärke je Team und Spieltag. Kader = Spieler, die in dem Spiel aufliefen."""
    repl = ersatzniveau(d)
    d = d.sort_values(["PLAYER_ID", "GAME_DATE", "GAME_ID"]).copy()
    grp = d.groupby(["PLAYER_ID", "season"])
    d["c_min"] = grp["MIN"].cumsum() - d["MIN"]
    d["c_gs"] = grp["GMSC"].cumsum() - d["GMSC"]
    d["c_gp"] = grp.cumcount()
    d = d.merge(_vorsaison(d), on=["PLAYER_ID", "season"], how="left").fillna({"p_min": 0, "p_gs": 0, "p_gp": 0})
    d["rate"], d["exp_min"] = _wert(d.p_min, d.p_gs, d.p_gp, d.c_min, d.c_gs, d.c_gp, repl)
    h = (d.groupby(["GAME_DATE", "TEAM_NAME", "season"])
           .apply(lambda x: team_wert(x["rate"], x["exp_min"]), include_groups=False)
           .rename("kader").reset_index()
           .rename(columns={"GAME_DATE": "date", "TEAM_NAME": "team"}))
    return h


# ─── Aktuelle Spielerwerte (nach dem letzten gespielten Spiel) ────────────────

def aktuelle_spielerwerte(d: pd.DataFrame, saison: int, kader: pd.DataFrame) -> pd.DataFrame:
    """
    Spielerwerte für die NÄCHSTEN Spiele der Saison `saison`.
    kader: aktuelle Team-Zuordnung (PLAYER_ID, PLAYER_NAME, TEAM_NAME), z. B. aus nba_api CommonAllPlayers.
    """
    repl = ersatzniveau(d)
    vor = d[d.season == saison - 1].groupby("PLAYER_ID").agg(p_min=("MIN", "sum"), p_gs=("GMSC", "sum"), p_gp=("MIN", "size"))
    cur = d[d.season == saison].groupby("PLAYER_ID").agg(c_min=("MIN", "sum"), c_gs=("GMSC", "sum"), c_gp=("MIN", "size"))
    x = kader.set_index("PLAYER_ID").join(vor).join(cur).fillna({c: 0 for c in ["p_min", "p_gs", "p_gp", "c_min", "c_gs", "c_gp"]}).reset_index()
    x["rate"], x["exp_min"] = _wert(x.p_min, x.p_gs, x.p_gp, x.c_min, x.c_gs, x.c_gp, repl)
    x["repl"] = repl
    return x[["PLAYER_ID", "PLAYER_NAME", "TEAM_NAME", "rate", "exp_min", "repl"]]


# ─── Spiele + Kader zusammenführen ────────────────────────────────────────────

def spiele_aus_logs(d: pd.DataFrame) -> pd.DataFrame:
    """Spiele (Heim/Auswärts, Ergebnis) aus den Spieler-Logs, Format wie kobra.lade_alle_spiele."""
    t = d.groupby(["season", "GAME_ID", "GAME_DATE", "TEAM_NAME", "home"], as_index=False).PTS.sum()
    h = t[t.home].rename(columns={"TEAM_NAME": "home_team", "PTS": "home_score"}).drop(columns="home")
    a = t[~t.home].rename(columns={"TEAM_NAME": "away_team", "PTS": "away_score"}).drop(columns="home")
    g = h.merge(a, on=["season", "GAME_ID", "GAME_DATE"]).rename(columns={"GAME_DATE": "date", "GAME_ID": "game_id"})
    g["home_win"] = (g.home_score > g.away_score).astype(int)
    g["status"] = "Final"
    return g.sort_values(["date", "game_id"]).reset_index(drop=True)


def merge_kader(spiele: pd.DataFrame, historie: pd.DataFrame) -> pd.DataFrame:
    """Hängt home_kader, away_kader, kader_diff an (Spiele ohne Kaderdaten → Diff 0)."""
    h = historie[["date", "team", "kader"]].copy()
    h["date"] = pd.to_datetime(h["date"])
    out = spiele.merge(h.rename(columns={"team": "home_team", "kader": "home_kader"}), on=["date", "home_team"], how="left")
    out = out.merge(h.rename(columns={"team": "away_team", "kader": "away_kader"}), on=["date", "away_team"], how="left")
    out["kader_diff"] = (out["home_kader"] - out["away_kader"]).fillna(0.0)
    return out
