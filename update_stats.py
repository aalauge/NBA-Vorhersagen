"""
Holt Spieler-Stats von nba_api und speichert sie als CSV.
Läuft lokal auf deinem Mac (nba.com blockiert Cloud-Server).
"""

import sys
import time
from datetime import datetime

import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats


def lade_spieler_stats(saison: str, min_games: int = 20) -> pd.DataFrame:
    """Spieler-Statistiken von nba_api laden."""
    print(f"  Lade Stats für Saison {saison}...")
    try:
        stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season=saison,
            per_mode_detailed="PerGame",
            timeout=60,
        )
        time.sleep(3)  # NBA.com Rate Limit respektieren
        df = stats.get_data_frames()[0]
        df = df[df["GP"] >= min_games]
        return df[["PLAYER_NAME", "TEAM_ABBREVIATION", "MIN", "PTS", "REB", "AST", "GP"]]
    except Exception as e:
        print(f"  Fehler bei {saison}: {e}")
        return pd.DataFrame()


def main():
    print(f"=== Spieler-Stats Update: {datetime.now().strftime('%Y-%m-%d %H:%M')} ===\n")

    aktuell = lade_spieler_stats("2025-26")
    vorjahr = lade_spieler_stats("2024-25")

    if len(aktuell) == 0 and len(vorjahr) == 0:
        print("\nKeine Stats geladen. Abbruch.")
        sys.exit(1)

    # Impact Score berechnen
    for df in [aktuell, vorjahr]:
        if len(df) > 0:
            df["Impact"] = df["PTS"] * 1.0 + df["AST"] * 1.5 + df["REB"] * 0.8 + df["MIN"] * 0.3
            max_impact = df["Impact"].max()
            if max_impact > 0:
                df["Impact_Score"] = (df["Impact"] / max_impact * 10).round(2)
            else:
                df["Impact_Score"] = 0.0

    # Kombinieren: 60% aktuell, 40% Vorjahr
    if len(aktuell) > 0 and len(vorjahr) > 0:
        kombiniert = aktuell.merge(
            vorjahr[["PLAYER_NAME", "Impact_Score"]],
            on="PLAYER_NAME",
            suffixes=("_Aktuell", "_Vorjahr"),
            how="left",
        )
        kombiniert["Impact_Final"] = (
            kombiniert["Impact_Score_Aktuell"] * 0.6
            + kombiniert["Impact_Score_Vorjahr"].fillna(kombiniert["Impact_Score_Aktuell"]) * 0.4
        ).round(2)
    elif len(aktuell) > 0:
        kombiniert = aktuell.copy()
        kombiniert.rename(columns={"Impact_Score": "Impact_Final"}, inplace=True)
    else:
        kombiniert = vorjahr.copy()
        kombiniert.rename(columns={"Impact_Score": "Impact_Final"}, inplace=True)

    # Speichern
    kombiniert.to_csv("player_stats.csv", index=False)
    print(f"\n✅ {len(kombiniert)} Spieler gespeichert in player_stats.csv")
    print(f"   Stand: {datetime.now().strftime('%Y-%m-%d %H:%M')}")


if __name__ == "__main__":
    main()
