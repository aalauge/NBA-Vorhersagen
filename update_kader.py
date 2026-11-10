"""
KOBRA – Kader-Stärke täglich aktualisieren (läuft lokal auf dem Mac, nba.com blockiert Cloud-Server)
====================================================================================================
1. Lädt die Spieler-Einsätze der laufenden Saison neu (daten/player_games_YYYY-YY.csv)
2. Lädt die aktuelle Team-Zuordnung aller Spieler (nba_api CommonAllPlayers → Trades sofort drin)
3. Schreibt ins Repo:
     kader_historie.csv  – Kaderstärke je Team und Spieltag (Training von kobra.py)
     kader_spieler.csv   – aktuelle Spielerwerte + Team (heutige Vorhersage in kobra.py)

Wird von update_stats.sh nach update_stats.py aufgerufen.
Usage: /opt/anaconda3/bin/python3 update_kader.py
"""
import os
import sys
import time
from datetime import datetime

import pandas as pd
from nba_api.stats.endpoints import commonallplayers

import kader_features as kf
from lade_kader_daten import lade_saison

BASIS = os.path.dirname(os.path.abspath(__file__))
DATEN = os.path.join(BASIS, "daten")


def saison_heute():
    jetzt = datetime.now()
    start = jetzt.year if jetzt.month >= 10 else jetzt.year - 1
    return start, f"{start}-{str(start + 1)[-2:]}"


def lade_team_zuordnung(saison_str: str) -> pd.DataFrame | None:
    for versuch in range(5):
        try:
            df = commonallplayers.CommonAllPlayers(
                is_only_current_season=1, season=saison_str, timeout=90).get_data_frames()[0]
            df = df[(df["ROSTERSTATUS"] == 1) & (df["TEAM_ID"] != 0)].copy()
            df["TEAM_NAME"] = (df["TEAM_CITY"].str.strip() + " " + df["TEAM_NAME"].str.strip())
            return df.rename(columns={"PERSON_ID": "PLAYER_ID", "DISPLAY_FIRST_LAST": "PLAYER_NAME"})[
                ["PLAYER_ID", "PLAYER_NAME", "TEAM_NAME"]]
        except Exception as e:
            warte = 20 * (versuch + 1)
            print(f"    Kader-Fehler ({e.__class__.__name__}) – neuer Versuch in {warte}s")
            time.sleep(warte)
    return None


def main():
    print(f"=== Kader-Update: {datetime.now():%Y-%m-%d %H:%M} ===")
    start, saison_str = saison_heute()
    os.makedirs(DATEN, exist_ok=True)

    # 1) laufende Saison neu laden (leer am Saisonstart ist ok)
    df = lade_saison(saison_str)
    if df is not None and len(df) > 0:
        df.to_csv(os.path.join(DATEN, f"player_games_{saison_str}.csv"), index=False)
        print(f"  ✅ {saison_str}: {len(df)} Spieler-Einsätze bis {df['GAME_DATE'].max()}")
    else:
        print(f"  ⚠️ {saison_str}: keine neuen Einsätze geladen – nutze vorhandene Datei")

    d = kf.lade_logs(DATEN)
    if (d.season == start - 1).sum() == 0:
        print(f"  ❌ Vorsaison fehlt in daten/ – erst lade_kader_daten.py laufen lassen")
        sys.exit(1)

    # 2) Historie
    hist = kf.kader_historie(d)
    hist = hist[hist.season >= start - 3]
    hist["date"] = pd.to_datetime(hist["date"]).dt.strftime("%Y-%m-%d")
    hist.round(3).to_csv(os.path.join(BASIS, "kader_historie.csv"), index=False)
    print(f"  ✅ kader_historie.csv: {len(hist)} Team-Spiele")

    # 3) aktuelle Team-Zuordnung (Fallback: letztes Team laut Einsätzen)
    kader = lade_team_zuordnung(saison_str)
    if kader is None or len(kader) < 300:
        print("  ⚠️ CommonAllPlayers nicht verfügbar – Team = letztes Team laut Einsätzen")
        basis = d[d.season == start] if (d.season == start).sum() > 2000 else d[d.season == start - 1]
        letzte = basis.sort_values("GAME_DATE").groupby("PLAYER_ID").tail(1)
        kader = letzte[["PLAYER_ID", "PLAYER_NAME", "TEAM_NAME"]]
    sp = kf.aktuelle_spielerwerte(d, start, kader)
    sp.round(4).to_csv(os.path.join(BASIS, "kader_spieler.csv"), index=False)
    print(f"  ✅ kader_spieler.csv: {len(sp)} Spieler, {sp.TEAM_NAME.nunique()} Teams")

    # Kontrolle: aktuelle Teamstärke (ohne Ausfälle)
    st = sp.groupby("TEAM_NAME").apply(lambda x: kf.team_wert(x["rate"], x["exp_min"]), include_groups=False)
    print("  Top 5 Kader:", ", ".join(f"{t} {v:.0f}" for t, v in st.sort_values(ascending=False).head(5).items()))


if __name__ == "__main__":
    main()
