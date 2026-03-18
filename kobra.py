"""
KOBRA – NBA Game Prediction Model
==================================
Predicts today's NBA games using logistic regression with:
- Rolling win rates & scoring averages
- Elo ratings (with season regression)
- Injury impact correction (ESPN scraping + player_stats.csv)

Usage:
    python kobra.py                     # Predict today's games
    python kobra.py --date 2026-03-20   # Predict for a specific date
    python kobra.py --seasons 2023 2024 2025  # Custom training seasons

Requirements: see requirements.txt
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ─── Konfiguration ────────────────────────────────────────────────────────────

API_KEY = os.environ.get("BALLDONTLIE_API_KEY", "")
HEADERS = {"Authorization": API_KEY}
BASE_URL = "https://api.balldontlie.io/v1"

ELO_START = 1500
ELO_K = 20
ELO_SEASON_REGRESSION = 0.25  # 25 % Regression zum Mittelwert zwischen Saisons
FORM_WINDOW = 10

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("kobra")

# ─── ESPN → balldontlie Team-Name-Mapping ─────────────────────────────────────
# ESPN liefert Kurznamen, balldontlie volle Namen.

TEAM_NAME_MAP = {
    "Hawks": "Atlanta Hawks",
    "Celtics": "Boston Celtics",
    "Nets": "Brooklyn Nets",
    "Hornets": "Charlotte Hornets",
    "Bulls": "Chicago Bulls",
    "Cavaliers": "Cleveland Cavaliers",
    "Mavericks": "Dallas Mavericks",
    "Nuggets": "Denver Nuggets",
    "Pistons": "Detroit Pistons",
    "Warriors": "Golden State Warriors",
    "Rockets": "Houston Rockets",
    "Pacers": "Indiana Pacers",
    "Clippers": "LA Clippers",
    "Lakers": "Los Angeles Lakers",
    "Grizzlies": "Memphis Grizzlies",
    "Heat": "Miami Heat",
    "Bucks": "Milwaukee Bucks",
    "Timberwolves": "Minnesota Timberwolves",
    "Pelicans": "New Orleans Pelicans",
    "Knicks": "New York Knicks",
    "Thunder": "Oklahoma City Thunder",
    "Magic": "Orlando Magic",
    "76ers": "Philadelphia 76ers",
    "Suns": "Phoenix Suns",
    "Trail Blazers": "Portland Trail Blazers",
    "Kings": "Sacramento Kings",
    "Spurs": "San Antonio Spurs",
    "Raptors": "Toronto Raptors",
    "Jazz": "Utah Jazz",
    "Wizards": "Washington Wizards",
}

# Auch Reverse-Mapping für flexibles Matching
FULL_TO_SHORT = {v: k for k, v in TEAM_NAME_MAP.items()}


# ─── SCHRITT 1: Spieldaten laden ─────────────────────────────────────────────

def api_request(endpoint: str, params: dict, max_retries: int = 5) -> dict | None:
    """Robuster API-Request mit Retry-Logik."""
    if not API_KEY:
        log.error("BALLDONTLIE_API_KEY nicht gesetzt! Bitte als Umgebungsvariable setzen.")
        sys.exit(1)

    url = f"{BASE_URL}/{endpoint}"
    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=30)
            if r.status_code == 429:
                wait = 60 * (attempt + 1)
                log.warning(f"Rate limit (429). Warte {wait}s… (Versuch {attempt + 1}/{max_retries})")
                time.sleep(wait)
                continue
            if r.status_code != 200:
                log.warning(f"HTTP {r.status_code}: {r.text[:200]}")
                time.sleep(10)
                continue
            return r.json()
        except requests.RequestException as e:
            log.warning(f"Request-Fehler: {e}. Warte 30s… (Versuch {attempt + 1}/{max_retries})")
            time.sleep(30)
    log.error(f"Alle {max_retries} Versuche für {endpoint} fehlgeschlagen.")
    return None


def lade_spiele(saison: int) -> list[dict]:
    """Alle Spiele einer Saison paginiert laden."""
    spiele = []
    cursor = None
    while True:
        params = {"seasons[]": saison, "per_page": 100}
        if cursor:
            params["cursor"] = cursor
        data = api_request("games", params)
        if data is None:
            break
        spiele.extend(data["data"])
        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor:
            break
        time.sleep(1.5)
    return spiele


def lade_alle_spiele(saisons: list[int]) -> pd.DataFrame:
    """Spieldaten aller Saisons laden und als DataFrame zurückgeben."""
    alle = []
    for saison in saisons:
        log.info(f"Lade Saison {saison}…")
        spiele = lade_spiele(saison)
        alle.extend(spiele)
        log.info(f"  → {len(spiele)} Spiele")

    log.info(f"Gesamt: {len(alle)} Spiele geladen")

    df = pd.DataFrame([{
        "game_id": g["id"],
        "date": g["date"][:10],
        "home_team": g["home_team"]["full_name"],
        "away_team": g["visitor_team"]["full_name"],
        "home_score": g["home_team_score"],
        "away_score": g["visitor_team_score"],
        "season": g["season"],
        "status": g["status"],
    } for g in alle])

    df = df[df["status"] == "Final"].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
    df = df.sort_values("date").reset_index(drop=True)

    log.info(f"Gespielte Spiele (Final): {len(df)}")
    return df


# ─── SCHRITT 2: Features berechnen ───────────────────────────────────────────

def _team_spiele(df: pd.DataFrame, team: str, vor_datum: pd.Timestamp, window: int) -> pd.DataFrame:
    """Letzte `window` Spiele eines Teams vor einem Datum."""
    mask = (df["date"] < vor_datum) & ((df["home_team"] == team) | (df["away_team"] == team))
    return df.loc[mask].tail(window)


def _winrate(spiele: pd.DataFrame, team: str) -> float:
    """Gewinnrate eines Teams aus einer Teilmenge von Spielen."""
    if len(spiele) == 0:
        return 0.5
    wins = sum(
        (r["home_team"] == team and r["home_win"] == 1) or
        (r["away_team"] == team and r["home_win"] == 0)
        for _, r in spiele.iterrows()
    )
    return wins / len(spiele)


def _avg_pts(spiele: pd.DataFrame, team: str) -> tuple[float, float]:
    """Durchschnitt (erzielt, kassiert) eines Teams."""
    if len(spiele) == 0:
        return 110.0, 110.0
    scored, conceded = [], []
    for _, r in spiele.iterrows():
        if r["home_team"] == team:
            scored.append(r["home_score"])
            conceded.append(r["away_score"])
        else:
            scored.append(r["away_score"])
            conceded.append(r["home_score"])
    return np.mean(scored), np.mean(conceded)


def berechne_features(df: pd.DataFrame, window: int = FORM_WINDOW) -> pd.DataFrame:
    """Rolling Winrate + Punkte-Features für Heim- und Auswärtsteam."""
    log.info("Berechne Form- und Punkte-Features…")

    records = []
    for idx, row in df.iterrows():
        datum = row["date"]
        heim = row["home_team"]
        ausw = row["away_team"]

        h_spiele = _team_spiele(df, heim, datum, window)
        a_spiele = _team_spiele(df, ausw, datum, window)

        h_scored, h_conceded = _avg_pts(h_spiele, heim)
        a_scored, a_conceded = _avg_pts(a_spiele, ausw)

        records.append({
            "home_winrate": _winrate(h_spiele, heim),
            "away_winrate": _winrate(a_spiele, ausw),
            "home_pts_scored": h_scored,
            "home_pts_conceded": h_conceded,
            "away_pts_scored": a_scored,
            "away_pts_conceded": a_conceded,
        })

    feat_df = pd.DataFrame(records, index=df.index)
    df = pd.concat([df, feat_df], axis=1)
    log.info("Features fertig!")
    return df


# ─── SCHRITT 3: Elo-Rating ───────────────────────────────────────────────────

def berechne_elo(
    df: pd.DataFrame,
    start_elo: float = ELO_START,
    k: float = ELO_K,
    season_regression: float = ELO_SEASON_REGRESSION,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Elo-Ratings berechnen. Zwischen Saisons wird Elo zum Mittelwert
    regressiert (z.B. 25 %). Gibt den DataFrame MIT Elo-Spalten und
    das finale Elo-Dict zurück.
    """
    teams = pd.concat([df["home_team"], df["away_team"]]).unique()
    elo = {team: start_elo for team in teams}

    home_elo_list, away_elo_list = [], []
    prev_season = None

    for _, row in df.iterrows():
        # Season Regression
        if prev_season is not None and row["season"] != prev_season:
            for team in elo:
                elo[team] = elo[team] + season_regression * (start_elo - elo[team])
            log.info(f"Elo-Regression angewandt (Saison {prev_season} → {row['season']})")
        prev_season = row["season"]

        heim = row["home_team"]
        ausw = row["away_team"]
        h_elo = elo[heim]
        a_elo = elo[ausw]

        home_elo_list.append(h_elo)
        away_elo_list.append(a_elo)

        expected_h = 1 / (1 + 10 ** ((a_elo - h_elo) / 400))
        actual_h = row["home_win"]
        elo[heim] += k * (actual_h - expected_h)
        elo[ausw] += k * ((1 - actual_h) - (1 - expected_h))

    df = df.copy()
    df["home_elo"] = home_elo_list
    df["away_elo"] = away_elo_list

    log.info("Elo-Ratings fertig!")
    return df, elo


# ─── SCHRITT 4: Modell ───────────────────────────────────────────────────────

FEATURES = [
    "home_winrate", "away_winrate",
    "home_pts_scored", "home_pts_conceded",
    "away_pts_scored", "away_pts_conceded",
    "home_elo", "away_elo",
]


def trainiere_modell(df: pd.DataFrame, test_season: int):
    """Logistic Regression trainieren, Accuracy auf Testsaison ausgeben."""
    train = df[df["season"] < test_season]
    test = df[df["season"] == test_season]

    if len(train) == 0 or len(test) == 0:
        log.warning("Zu wenig Daten zum Trainieren/Testen.")
        return None, 0.0

    model = LogisticRegression(max_iter=1000)
    model.fit(train[FEATURES], train["home_win"])

    acc = accuracy_score(test["home_win"], model.predict(test[FEATURES]))
    log.info(f"Modell-Genauigkeit (Saison {test_season}): {acc:.1%}")
    return model, acc


# ─── SCHRITT 5: Heutige Spiele laden & vorhersagen ───────────────────────────

def lade_heutige_spiele(datum: str) -> list[dict]:
    """Spiele eines bestimmten Datums laden."""
    data = api_request("games", {"dates[]": datum, "per_page": 100})
    if data is None:
        return []
    spiele = data["data"]
    log.info(f"{len(spiele)} Spiele für {datum}")
    for s in spiele:
        log.info(f"  {s['home_team']['full_name']} vs {s['visitor_team']['full_name']}")
    return spiele


def erstelle_vorhersagen(
    spiele: list[dict],
    model: LogisticRegression,
    df: pd.DataFrame,
    elo: dict[str, float],
) -> pd.DataFrame:
    """
    Vorhersagen für heutige Spiele erstellen.
    Benutzt die letzten bekannten Stats PRO TEAM (unabhängig von Heim/Auswärts-Rolle)
    und das finale Elo-Dict.
    """
    vorhersagen = []
    now = pd.Timestamp.now()

    for spiel in spiele:
        heim = spiel["home_team"]["full_name"]
        ausw = spiel["visitor_team"]["full_name"]

        # Letzte Spiele pro Team (Rolle egal) → korrekte Stats
        h_spiele = _team_spiele(df, heim, now, FORM_WINDOW)
        a_spiele = _team_spiele(df, ausw, now, FORM_WINDOW)

        if len(h_spiele) == 0 or len(a_spiele) == 0:
            log.warning(f"Keine historischen Daten für {heim} vs {ausw} – übersprungen.")
            continue

        h_winrate = _winrate(h_spiele, heim)
        a_winrate = _winrate(a_spiele, ausw)
        h_scored, h_conceded = _avg_pts(h_spiele, heim)
        a_scored, a_conceded = _avg_pts(a_spiele, ausw)

        # Aktuelles Elo aus finalem Dict (nach letztem Spiel)
        h_elo = elo.get(heim, ELO_START)
        a_elo = elo.get(ausw, ELO_START)

        x = pd.DataFrame([{
            "home_winrate": h_winrate,
            "away_winrate": a_winrate,
            "home_pts_scored": h_scored,
            "home_pts_conceded": h_conceded,
            "away_pts_scored": a_scored,
            "away_pts_conceded": a_conceded,
            "home_elo": h_elo,
            "away_elo": a_elo,
        }])

        prob = model.predict_proba(x)[0][1]
        gewinner = heim if prob > 0.5 else ausw
        konfidenz = prob if prob > 0.5 else 1 - prob

        vorhersagen.append({
            "Heimteam": heim,
            "Auswärtsteam": ausw,
            "Heimsieg %": round(prob * 100, 1),
            "Tipp": gewinner,
            "Konfidenz": round(konfidenz * 100, 1),
        })

    return pd.DataFrame(vorhersagen)


# ─── SCHRITT 6: Verletzungen (ESPN) ──────────────────────────────────────────

def lade_verletzungen() -> pd.DataFrame:
    """Verletzte Spieler von ESPN scrapen. Gibt leeren DataFrame bei Fehler zurück."""
    url = "https://www.espn.com/nba/injuries"
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        r.raise_for_status()
    except requests.RequestException as e:
        log.warning(f"ESPN-Verletzungen konnten nicht geladen werden: {e}")
        return pd.DataFrame(columns=["Team", "Spieler", "Status"])

    soup = BeautifulSoup(r.text, "html.parser")
    verletzungen = []

    try:
        tabellen = soup.find_all("div", class_="ResponsiveTable")
        for tabelle in tabellen:
            team_span = tabelle.find("span", class_="injuries__teamName")
            espn_name = team_span.text.strip() if team_span else "Unbekannt"
            # Mapping: ESPN-Kurzname → voller Name
            full_name = TEAM_NAME_MAP.get(espn_name, espn_name)

            rows = tabelle.find_all("tr")[1:]
            for row in rows:
                cols = row.find_all("td")
                if len(cols) >= 4:
                    verletzungen.append({
                        "Team": full_name,
                        "Spieler": cols[0].text.strip(),
                        "Status": cols[3].text.strip(),
                    })
    except Exception as e:
        log.warning(f"Fehler beim Parsen der ESPN-Daten: {e}")

    df = pd.DataFrame(verletzungen)
    if len(df) > 0:
        df = df[df["Status"].str.contains("Out|Doubtful|Day-To-Day|Day to Day|DTD", case=False, na=False)]
    log.info(f"{len(df)} verletzte/fragliche Spieler geladen")
    return df


# ─── SCHRITT 7: Spieler Impact Scores (aus lokaler CSV) ──────────────────────

def lade_impact_scores() -> pd.DataFrame:
    """
    Liest player_stats.csv – wird täglich von deinem Mac aktualisiert
    und ins Repo gepusht. Funktioniert daher auch in der Cloud.
    """
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "player_stats.csv")

    if not os.path.exists(csv_path):
        log.warning(f"player_stats.csv nicht gefunden ({csv_path}). "
                    "Bitte update_stats.py lokal ausführen!")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)

    if "Impact_Final" not in df.columns:
        log.warning("player_stats.csv hat kein Impact_Final – Format falsch?")
        return pd.DataFrame()

    log.info(f"{len(df)} Spieler mit Impact Score aus player_stats.csv geladen")
    return df


# ─── SCHRITT 8: Verletzungskorrektur ─────────────────────────────────────────

def berechne_impact_verlust(
    team_name: str,
    verletzungen_df: pd.DataFrame,
    impact_df: pd.DataFrame,
) -> tuple[float, list[str]]:
    """Impact-Verlust durch verletzte Spieler eines Teams."""
    if len(verletzungen_df) == 0 or len(impact_df) == 0:
        return 0.0, []

    verletzt = verletzungen_df[verletzungen_df["Team"] == team_name]
    total_impact = 0.0
    details = []

    for _, spieler in verletzt.iterrows():
        name = spieler["Spieler"]
        status = spieler["Status"]

        # Flexibles Namens-Matching (Vor- und Nachname)
        teile = name.split()
        if len(teile) >= 2:
            match = impact_df[
                impact_df["PLAYER_NAME"].str.contains(teile[0], case=False, na=False)
                & impact_df["PLAYER_NAME"].str.contains(teile[-1], case=False, na=False)
            ]
        else:
            match = impact_df[impact_df["PLAYER_NAME"].str.contains(name, case=False, na=False)]

        if len(match) > 0:
            impact = match.iloc[0]["Impact_Final"]
            if "Out" in status:
                gewicht = 1.0
            elif "Doubtful" in status:
                gewicht = 0.5
            else:
                gewicht = 0.0  # Day-to-Day: nur anzeigen, nicht einrechnen
            total_impact += impact * gewicht
            if gewicht > 0:
                details.append(f"{name} ({impact:.1f}, {status})")
            else:
                details.append(f"⚠️ {name} ({impact:.1f}, {status})")

    return total_impact, details


def korrigiere_vorhersagen(
    vorhersagen_df: pd.DataFrame,
    verletzungen_df: pd.DataFrame,
    impact_df: pd.DataFrame,
) -> pd.DataFrame:
    """Vorhersagen um Verletzungs-Impact korrigieren."""
    if len(vorhersagen_df) == 0:
        return vorhersagen_df

    ergebnisse = []
    for _, spiel in vorhersagen_df.iterrows():
        heim = spiel["Heimteam"]
        ausw = spiel["Auswärtsteam"]

        h_impact, h_details = berechne_impact_verlust(heim, verletzungen_df, impact_df)
        a_impact, a_details = berechne_impact_verlust(ausw, verletzungen_df, impact_df)

        if h_details:
            log.info(f"  {heim} ohne: {', '.join(h_details)}")
        if a_details:
            log.info(f"  {ausw} ohne: {', '.join(a_details)}")

        # Impact-Differenz verschiebt Wahrscheinlichkeit
        diff = (a_impact - h_impact) * 0.01
        neue_prob = min(0.95, max(0.05, spiel["Heimsieg %"] / 100 + diff))
        gewinner = heim if neue_prob > 0.5 else ausw
        konfidenz = neue_prob if neue_prob > 0.5 else 1 - neue_prob

        ergebnisse.append({
            "Heimteam": heim,
            "Auswärtsteam": ausw,
            "Heimsieg %": round(neue_prob * 100, 1),
            "Tipp": gewinner,
            "Konfidenz": round(konfidenz * 100, 1),
            "Heim_Verletzungen": ", ".join(h_details) if h_details else "-",
            "Ausw_Verletzungen": ", ".join(a_details) if a_details else "-",
        })

    return pd.DataFrame(ergebnisse)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="KOBRA – NBA Game Predictions")
    parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"),
                        help="Datum für Vorhersagen (YYYY-MM-DD)")
    parser.add_argument("--seasons", nargs="+", type=int, default=[2022, 2023, 2024, 2025],
                        help="Trainings-Saisons")
    parser.add_argument("--output", default=None,
                        help="Ausgabe-CSV (default: predictions_DATUM.csv)")
    args = parser.parse_args()

    datum = args.date
    output = args.output or f"predictions_{datum}.csv"

    # 1) Spieldaten laden
    df = lade_alle_spiele(args.seasons)
    if len(df) == 0:
        log.error("Keine Spieldaten geladen. Abbruch.")
        sys.exit(1)

    # 2) Features berechnen
    df = berechne_features(df)

    # 3) Elo berechnen
    df, elo = berechne_elo(df)

    # 4) Modell trainieren
    test_season = max(args.seasons)
    model, acc = trainiere_modell(df, test_season)
    if model is None:
        log.error("Modell konnte nicht trainiert werden. Abbruch.")
        sys.exit(1)

    # 5) Heutige Spiele
    heute_spiele = lade_heutige_spiele(datum)
    if len(heute_spiele) == 0:
        log.warning(f"Keine Spiele für {datum} gefunden.")
        sys.exit(0)

    # 6) Basis-Vorhersagen
    vorhersagen_df = erstelle_vorhersagen(heute_spiele, model, df, elo)
    if len(vorhersagen_df) == 0:
        log.warning("Keine Vorhersagen möglich.")
        sys.exit(0)

    log.info("\n📊 BASIS-VORHERSAGEN:")
    print(vorhersagen_df.to_string(index=False))

    # 7) Verletzungen + Impact
    verletzungen_df = lade_verletzungen()
    impact_df = lade_impact_scores()

    # 8) Finale Vorhersagen
    if len(verletzungen_df) > 0 and len(impact_df) > 0:
        log.info("Korrigiere Vorhersagen mit Verletzungsdaten…")
        finale_df = korrigiere_vorhersagen(vorhersagen_df, verletzungen_df, impact_df)
    else:
        log.warning("Verletzungskorrektur übersprungen (keine Daten).")
        finale_df = vorhersagen_df.copy()
        finale_df["Heim_Verletzungen"] = "-"
        finale_df["Ausw_Verletzungen"] = "-"

    print("\n🏀 FINALE VORHERSAGEN")
    print("=" * 80)
    print(finale_df.to_string(index=False))

    # 9) Speichern
    finale_df.to_csv(output, index=False)
    log.info(f"Gespeichert: {output}")


if __name__ == "__main__":
    main()
