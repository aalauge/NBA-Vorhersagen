"""
KOBRA – Kader-Daten laden (Grundlage für das Kader-Feature)
============================================================
Lädt für jede Saison ALLE Spieler-Einsätze (wer hat wann für welches Team wie viele
Minuten / Punkte / Rebounds / Assists gemacht) über nba_api und speichert sie in daten/.

Daraus berechnen wir später für jedes Spiel:
  "Wie stark ist der Kader, der heute für das Team aufläuft?"
– auf Basis der Spielerwerte aus der Vorsaison, egal für welches Team der Spieler damals spielte.
Genau das fehlt KOBRA bisher (LeBron-Wechsel, Haliburton-Rückkehr usw.).

Läuft nur lokal auf dem Mac (nba.com blockiert Cloud-Server, wie bei update_stats.py).
Ändert nichts an kobra.py.

Usage:
    /opt/anaconda3/bin/python3 lade_kader_daten.py
"""

import os
import time

from nba_api.stats.endpoints import leaguegamelog

SAISONS = ["2022-23", "2023-24", "2024-25", "2025-26", "2026-27"]
LAUFEND = "2026-27"  # wird bei jedem Start neu geladen, die anderen nur einmal
ORDNER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "daten")


def lade_saison(saison: str):
    for versuch in range(5):
        try:
            df = leaguegamelog.LeagueGameLog(
                season=saison,
                player_or_team_abbreviation="P",
                season_type_all_star="Regular Season",
                timeout=90,
            ).get_data_frames()[0]
            return df
        except Exception as e:
            warte = 20 * (versuch + 1)
            print(f"    Fehler ({e.__class__.__name__}) – neuer Versuch in {warte}s ({versuch + 1}/5)")
            time.sleep(warte)
    return None


def main():
    os.makedirs(ORDNER, exist_ok=True)
    for saison in SAISONS:
        ziel = os.path.join(ORDNER, f"player_games_{saison}.csv")
        if os.path.exists(ziel) and saison != LAUFEND:
            print(f"  {saison}: schon vorhanden – übersprungen")
            continue
        print(f"  {saison}: lade…", flush=True)
        df = lade_saison(saison)
        if df is None or len(df) == 0:
            print(f"  ❌ {saison}: keine Daten geladen")
            continue
        df.to_csv(ziel, index=False)
        print(f"  ✅ {saison}: {len(df)} Spieler-Einsätze, {df['GAME_ID'].nunique()} Spiele, "
              f"{df['GAME_DATE'].min()} bis {df['GAME_DATE'].max()}")
        time.sleep(5)
    print("\nFertig. Dateien liegen in daten/")


if __name__ == "__main__":
    main()
