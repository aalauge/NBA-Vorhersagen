"""
KOBRA – NBA Game Prediction Model
==================================
Predicts today's NBA games using logistic regression with:
- Weighted rolling win rates & scoring averages (recency weighting)
- Home/away-specific win rates
- Scoring variance
- Elo ratings (with season regression)
- Injury impact correction (ESPN scraping + player_stats.csv)

Usage:
    python kobra.py                     # Predict today's games
    python kobra.py --date 2026-03-20   # Predict for a specific date
    python kobra.py --seasons 2023 2024 2025  # Custom training seasons

Requirements: see requirements.txt
"""

import argparse
import io
import logging
import os
import re
import sys
import time
from datetime import datetime

import unicodedata

import numpy as np
import pandas as pd
import pdfplumber
import requests
from bs4 import BeautifulSoup
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Optionaler Import – falls fetch_odds.py nicht existiert, läuft das Modell trotzdem
try:
    from fetch_odds import fetch_odds, calculate_ev
    HAS_ODDS = True
except ImportError:
    HAS_ODDS = False

# ─── Konfiguration ────────────────────────────────────────────────────────────

API_KEY = os.environ.get("BALLDONTLIE_API_KEY", "")
HEADERS = {"Authorization": API_KEY}
BASE_URL = "https://api.balldontlie.io/v1"

ELO_START = 1500
ELO_K = 20
ELO_SEASON_REGRESSION = 0.25
FORM_WINDOW = 10

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("kobra")

# ─── ESPN → balldontlie Team-Name-Mapping ─────────────────────────────────────

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

FULL_TO_SHORT = {v: k for k, v in TEAM_NAME_MAP.items()}

ABBR_TO_FULL = {
    "ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets", "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets", "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
    "LAC": "LA Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat", "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans", "NYK": "New York Knicks", "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic", "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings", "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors", "UTA": "Utah Jazz", "WAS": "Washington Wizards",
}


# ─── SCHRITT 1: Spieldaten laden ─────────────────────────────────────────────

def api_request(endpoint: str, params: dict, max_retries: int = 5) -> dict | None:
    if not API_KEY:
        log.error("BALLDONTLIE_API_KEY nicht gesetzt!")
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
    mask = (df["date"] < vor_datum) & ((df["home_team"] == team) | (df["away_team"] == team))
    return df.loc[mask].tail(window)


def _winrate_weighted(spiele: pd.DataFrame, team: str) -> float:
    """Gewichtete Winrate – neuere Spiele zählen mehr."""
    if len(spiele) == 0:
        return 0.5
    n = len(spiele)
    total_weight = 0
    weighted_wins = 0
    for i, (_, r) in enumerate(spiele.iterrows()):
        weight = (i + 1) / n
        won = (r["home_team"] == team and r["home_win"] == 1) or \
              (r["away_team"] == team and r["home_win"] == 0)
        weighted_wins += weight * int(won)
        total_weight += weight
    return weighted_wins / total_weight


def _home_winrate(df: pd.DataFrame, team: str, vor_datum, window: int) -> float:
    """Winrate nur für Heimspiele."""
    mask = (df["date"] < vor_datum) & (df["home_team"] == team)
    spiele = df.loc[mask].tail(window)
    if len(spiele) == 0:
        return 0.5
    return spiele["home_win"].mean()


def _away_winrate(df: pd.DataFrame, team: str, vor_datum, window: int) -> float:
    """Winrate nur für Auswärtsspiele."""
    mask = (df["date"] < vor_datum) & (df["away_team"] == team)
    spiele = df.loc[mask].tail(window)
    if len(spiele) == 0:
        return 0.5
    return (1 - spiele["home_win"]).mean()


def _avg_pts_weighted(spiele: pd.DataFrame, team: str) -> tuple[float, float]:
    """Gewichtete Punkte – neuere Spiele zählen mehr."""
    if len(spiele) == 0:
        return 110.0, 110.0
    n = len(spiele)
    w_scored, w_conceded, w_total = 0, 0, 0
    for i, (_, r) in enumerate(spiele.iterrows()):
        weight = (i + 1) / n
        if r["home_team"] == team:
            w_scored += weight * r["home_score"]
            w_conceded += weight * r["away_score"]
        else:
            w_scored += weight * r["away_score"]
            w_conceded += weight * r["home_score"]
        w_total += weight
    return w_scored / w_total, w_conceded / w_total


def _scoring_variance(spiele: pd.DataFrame, team: str) -> float:
    """Standardabweichung der erzielten Punkte."""
    if len(spiele) < 3:
        return 0.0
    pts = []
    for _, r in spiele.iterrows():
        if r["home_team"] == team:
            pts.append(r["home_score"])
        else:
            pts.append(r["away_score"])
    return np.std(pts)


def berechne_features(df: pd.DataFrame, window: int = FORM_WINDOW) -> pd.DataFrame:
    log.info("Berechne Form- und Punkte-Features…")
    records = []
    for idx, row in df.iterrows():
        datum = row["date"]
        heim = row["home_team"]
        ausw = row["away_team"]

        h_spiele = _team_spiele(df, heim, datum, window)
        a_spiele = _team_spiele(df, ausw, datum, window)

        h_scored_w, h_conceded_w = _avg_pts_weighted(h_spiele, heim)
        a_scored_w, a_conceded_w = _avg_pts_weighted(a_spiele, ausw)

        records.append({
            "home_winrate": _winrate_weighted(h_spiele, heim),
            "away_winrate": _winrate_weighted(a_spiele, ausw),
            "home_pts_scored": h_scored_w,
            "home_pts_conceded": h_conceded_w,
            "away_pts_scored": a_scored_w,
            "away_pts_conceded": a_conceded_w,
            "home_home_winrate": _home_winrate(df, heim, datum, window),
            "away_away_winrate": _away_winrate(df, ausw, datum, window),
            "home_variance": _scoring_variance(h_spiele, heim),
            "away_variance": _scoring_variance(a_spiele, ausw),
        })

    feat_df = pd.DataFrame(records, index=df.index)
    df = pd.concat([df, feat_df], axis=1)
    log.info("Features fertig!")
    return df


# ─── SCHRITT 3: Elo-Rating ───────────────────────────────────────────────────

def berechne_elo(df, start_elo=ELO_START, k=ELO_K, season_regression=ELO_SEASON_REGRESSION):
    teams = pd.concat([df["home_team"], df["away_team"]]).unique()
    elo = {team: start_elo for team in teams}
    home_elo_list, away_elo_list = [], []
    prev_season = None

    for _, row in df.iterrows():
        if prev_season is not None and row["season"] != prev_season:
            for team in elo:
                elo[team] = elo[team] + season_regression * (start_elo - elo[team])
            log.info(f"Elo-Regression angewandt (Saison {prev_season} → {row['season']})")
        prev_season = row["season"]

        heim, ausw = row["home_team"], row["away_team"]
        h_elo, a_elo = elo[heim], elo[ausw]
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
    "home_home_winrate", "away_away_winrate",
    "home_variance", "away_variance",
]


def trainiere_modell(df, test_season):
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
    data = api_request("games", {"dates[]": datum, "per_page": 100})
    if data is None:
        return []
    spiele = data["data"]
    log.info(f"{len(spiele)} Spiele für {datum}")
    for s in spiele:
        log.info(f"  {s['home_team']['full_name']} vs {s['visitor_team']['full_name']}")
    return spiele


def erstelle_vorhersagen(spiele, model, df, elo):
    vorhersagen = []
    now = pd.Timestamp.now()

    for spiel in spiele:
        heim = spiel["home_team"]["full_name"]
        ausw = spiel["visitor_team"]["full_name"]

        h_spiele = _team_spiele(df, heim, now, FORM_WINDOW)
        a_spiele = _team_spiele(df, ausw, now, FORM_WINDOW)

        if len(h_spiele) == 0 or len(a_spiele) == 0:
            log.warning(f"Keine historischen Daten für {heim} vs {ausw} – übersprungen.")
            continue

        h_scored_w, h_conceded_w = _avg_pts_weighted(h_spiele, heim)
        a_scored_w, a_conceded_w = _avg_pts_weighted(a_spiele, ausw)

        h_elo = elo.get(heim, ELO_START)
        a_elo = elo.get(ausw, ELO_START)

        x = pd.DataFrame([{
            "home_winrate": _winrate_weighted(h_spiele, heim),
            "away_winrate": _winrate_weighted(a_spiele, ausw),
            "home_pts_scored": h_scored_w,
            "home_pts_conceded": h_conceded_w,
            "away_pts_scored": a_scored_w,
            "away_pts_conceded": a_conceded_w,
            "home_elo": h_elo,
            "away_elo": a_elo,
            "home_home_winrate": _home_winrate(df, heim, now, FORM_WINDOW),
            "away_away_winrate": _away_winrate(df, ausw, now, FORM_WINDOW),
            "home_variance": _scoring_variance(h_spiele, heim),
            "away_variance": _scoring_variance(a_spiele, ausw),
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


# ─── SCHRITT 6: Verletzungen (NBA PDF + ESPN Fallback) ───────────────────────

def _normalize_ascii(name: str) -> str:
    """Entfernt Akzente/Diakritika: Dončić → Doncic, Porziņģis → Porzingis."""
    nfkd = unicodedata.normalize("NFKD", name)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _finde_spieler(name: str, impact_df: pd.DataFrame) -> pd.DataFrame:
    """Sucht Spieler in impact_df – erst exakt, dann ASCII-normalisiert."""
    if len(impact_df) == 0:
        return pd.DataFrame()

    teile = name.split()
    if len(teile) >= 2:
        match = impact_df[
            impact_df["PLAYER_NAME"].str.contains(teile[0], case=False, na=False)
            & impact_df["PLAYER_NAME"].str.contains(teile[-1], case=False, na=False)
        ]
    else:
        match = impact_df[impact_df["PLAYER_NAME"].str.contains(name, case=False, na=False)]

    if len(match) > 0:
        return match

    # Fallback: ASCII-normalisiert suchen (Dončić → Doncic)
    norm_name = _normalize_ascii(name)
    norm_teile = norm_name.split()
    norm_names = impact_df["PLAYER_NAME"].apply(_normalize_ascii)

    if len(norm_teile) >= 2:
        match = impact_df[
            norm_names.str.contains(norm_teile[0], case=False, na=False)
            & norm_names.str.contains(norm_teile[-1], case=False, na=False)
        ]
    else:
        match = impact_df[norm_names.str.contains(norm_name, case=False, na=False)]

    return match

def _lade_nba_pdf(datum: str) -> pd.DataFrame:
    """
    Lädt den offiziellen NBA Injury Report als PDF und parst ihn.
    URL-Muster: https://ak-static.cms.nba.com/referee/injury/Injury-Report_YYYY-MM-DD_HH00PM.pdf
    Probiert mehrere Zeitstempel durch.
    Nutzt Text-Extraktion (pdfplumber entfernt Spaces innerhalb von Feldern).
    """
    # NBA Injury Reports werden zu mehreren Zeiten am Spieltag veröffentlicht.
    # Wir probieren vom spätesten (neuesten) zum frühesten.
    zeitstempel = [
        "08_30PM", "07_30PM", "07_00PM", "06_30PM", "06_00PM",
        "05_30PM", "05_00PM", "04_30PM", "04_00PM", "03_30PM",
        "03_00PM", "02_30PM", "02_00PM", "01_30PM", "01_00PM",
        "12_30PM", "12_00PM", "11_30AM", "11_00AM",
        "12_00AM",  # Mitternacht des Spieltags – meist erster Report mit vielen NOT YET SUBMITTED
    ]
    base_url = "https://ak-static.cms.nba.com/referee/injury/Injury-Report"

    pdf_bytes = None

    for ts in zeitstempel:
        url = f"{base_url}_{datum}_{ts}.pdf"
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200 and len(r.content) > 500:
                pdf_bytes = r.content
                log.info(f"[NBA PDF] Geladen: {url} ({len(r.content)} bytes)")
                break
            else:
                log.debug(f"[NBA PDF] {ts} nicht verfügbar (HTTP {r.status_code})")
        except requests.RequestException as e:
            log.debug(f"[NBA PDF] {ts} Fehler: {e}")

    if pdf_bytes is None:
        log.warning(f"[NBA PDF] Kein Injury Report für {datum} gefunden.")
        return pd.DataFrame(columns=["Team", "Spieler", "Status"])

    # pdfplumber entfernt Spaces innerhalb von Feldern (z.B. "ChicagoBulls")
    # Mapping: no-space Version → korrekter Vollname
    nba_teams = set(TEAM_NAME_MAP.values())
    nospace_to_full = {t.replace(" ", ""): t for t in nba_teams}

    # Regex: "Nachname, Vorname Status"
    # Unterstützt Multi-Word-Nachnamen (z.B. "Jones Garcia, David")
    # und Multi-Word-Vornamen (z.B. "Niederhauser, Yanic Konan")
    # Behandelt Jr., III, T.J., D'Angelo etc.
    player_re = re.compile(
        r"([A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+)?(?:\s?(?:Jr\.|Sr\.|III|II|IV))?),\s*"
        r"([A-Z][a-zA-Z'\-\.]+(?:\s+[A-Z][a-zA-Z'\-\.]+)?(?:\s?[A-Z]\.)?)\s+"
        r"(Out|Doubtful|Questionable|Probable|Available)\b"
    )

    # Text aus allen Seiten extrahieren
    verletzungen = []
    current_team = None

    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                for line in text.split("\n"):
                    line = line.strip()
                    if not line or "NOTYETSUBMITTED" in line.replace(" ", ""):
                        continue
                    if "InjuryReport:" in line.replace(" ", "") or "GameDate" in line.replace(" ", ""):
                        continue
                    if re.match(r"Page\s*\d+\s*of\s*\d+", line.replace(" ", "")):
                        continue

                    # Team erkennen (no-space Version, längste zuerst)
                    for nospace, full in sorted(nospace_to_full.items(), key=lambda x: len(x[0]), reverse=True):
                        if nospace in line.replace(" ", ""):
                            current_team = full
                            break

                    # Spieler + Status erkennen
                    m = player_re.search(line)
                    if m and current_team:
                        last_name = m.group(1)
                        first_name = m.group(2)
                        status = m.group(3)

                        # Merged Suffixes bereinigen: "ButlerIII" → "Butler"
                        last_name = re.sub(r"(III|II|IV|Jr\.|Sr\.)$", "", last_name).strip()

                        verletzungen.append({
                            "Team": current_team,
                            "Spieler": f"{first_name} {last_name}",
                            "Status": status,
                        })

    except Exception as e:
        log.warning(f"[NBA PDF] Fehler beim Parsen: {e}")
        return pd.DataFrame(columns=["Team", "Spieler", "Status"])

    df = pd.DataFrame(verletzungen)
    if len(df) == 0:
        log.warning("[NBA PDF] Keine Verletzungen im PDF gefunden.")
        return df

    # Duplikate entfernen (Spieler kann in mehreren Matchups stehen)
    df = df.drop_duplicates(subset=["Team", "Spieler"], keep="last")

    log.info(f"[NBA PDF] {len(df)} Spieler geparst "
             f"(Out: {(df['Status'] == 'Out').sum()}, "
             f"Doubtful: {(df['Status'] == 'Doubtful').sum()}, "
             f"Questionable: {(df['Status'] == 'Questionable').sum()}, "
             f"Probable: {(df['Status'] == 'Probable').sum()})")
    return df


def _lade_espn_verletzungen() -> pd.DataFrame:
    """Fallback: ESPN Injuries scrapen (bisherige Logik)."""
    url = "https://www.espn.com/nba/injuries"
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        r.raise_for_status()
    except requests.RequestException as e:
        log.warning(f"[ESPN] Verletzungen konnten nicht geladen werden: {e}")
        return pd.DataFrame(columns=["Team", "Spieler", "Status"])

    soup = BeautifulSoup(r.text, "html.parser")
    verletzungen = []

    try:
        tabellen = soup.find_all("div", class_="ResponsiveTable")
        for tabelle in tabellen:
            team_span = tabelle.find("span", class_="injuries__teamName")
            espn_name = team_span.text.strip() if team_span else "Unbekannt"
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
        log.warning(f"[ESPN] Fehler beim Parsen: {e}")

    df = pd.DataFrame(verletzungen)
    if len(df) == 0:
        return df

    df = df[df["Status"].str.contains("Out|Doubtful|Day-To-Day", case=False, na=False)].copy()
    log.info(f"[ESPN] {len(df)} verletzte/fragliche Spieler geladen")
    return df


def lade_verletzungen(impact_df=None, datum=None) -> pd.DataFrame:
    """
    Lädt Verletzungsdaten: NBA Official Injury Report (PDF) mit ESPN-Fallback.
    Validiert Team-Zuordnungen gegen player_stats.csv.
    """
    if datum is None:
        datum = datetime.now().strftime("%Y-%m-%d")

    # 1) Versuche NBA Official PDF
    df = _lade_nba_pdf(datum)

    # 2) Fallback auf ESPN wenn PDF leer oder fehlgeschlagen
    if len(df) == 0:
        log.info("[Injuries] NBA PDF leer/nicht verfügbar – Fallback auf ESPN…")
        df = _lade_espn_verletzungen()

    if len(df) == 0:
        log.warning("[Injuries] Keine Verletzungsdaten aus beiden Quellen.")
        return pd.DataFrame(columns=["Team", "Spieler", "Status"])

    # Probable und Available rausfiltern (die spielen)
    df = df[~df["Status"].str.contains("Available|Probable", case=False, na=False)].copy()

    # Questionable → Day-To-Day (nur Anzeige, kein Penalty)
    df["Status"] = df["Status"].replace({"Questionable": "Day-To-Day"})

    log.info(f"[Injuries] {len(df)} relevante Spieler nach Filter")

    # Nur Logging: Abweichungen zwischen Quelle und player_stats.csv melden (NICHT überschreiben!)
    # NBA PDF / ESPN ist die Echtzeit-Quelle – player_stats.csv kann bei Trades veraltet sein.
    if impact_df is not None and len(impact_df) > 0 and "TEAM_ABBREVIATION" in impact_df.columns:
        abweichungen = 0
        for idx, row in df.iterrows():
            name = row["Spieler"]
            match = _finde_spieler(name, impact_df)

            if len(match) > 0:
                csv_team_abbr = match.iloc[0]["TEAM_ABBREVIATION"]
                csv_team_full = ABBR_TO_FULL.get(csv_team_abbr.upper())
                if csv_team_full and csv_team_full != row["Team"]:
                    log.info(
                        f"  Team-Abweichung (nicht korrigiert): {name} Source={row['Team']} vs Stats={csv_team_full}"
                    )
                    abweichungen += 1

        if abweichungen > 0:
            log.info(f"  {abweichungen} Abweichungen Quelle vs Stats (Quelle wird vertraut)")

    return df



def lade_impact_scores() -> pd.DataFrame:
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "player_stats.csv")
    if not os.path.exists(csv_path):
        log.warning(f"player_stats.csv nicht gefunden ({csv_path}).")
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    if "Impact_Final" not in df.columns:
        log.warning("player_stats.csv hat kein Impact_Final – Format falsch?")
        return pd.DataFrame()
    log.info(f"{len(df)} Spieler mit Impact Score geladen")
    return df


# ─── SCHRITT 8: Verletzungskorrektur ─────────────────────────────────────────

def berechne_impact_verlust(team_name, verletzungen_df, impact_df):
    """Out=100%, Doubtful=50%, Day-to-Day=0% (nur Anzeige mit ⚠️).
    Spieler ohne Impact-Score in player_stats.csv werden mit Default-Wert angezeigt."""
    if len(verletzungen_df) == 0:
        return 0.0, []

    verletzt = verletzungen_df[verletzungen_df["Team"] == team_name]
    total_impact = 0.0
    details = []
    DEFAULT_IMPACT = 3.0  # Konservativer Default für unbekannte Spieler

    for _, spieler in verletzt.iterrows():
        name = spieler["Spieler"]
        status = spieler["Status"]

        # Impact Score suchen (mit Unicode-Normalisierung)
        impact = None
        match = _finde_spieler(name, impact_df)
        if len(match) > 0:
            impact = match.iloc[0]["Impact_Final"]

        # Fallback: Default-Impact für unbekannte Spieler
        if impact is None:
            impact = DEFAULT_IMPACT
            log.warning(f"  Kein Impact-Score für {name} ({team_name}) – verwende Default {DEFAULT_IMPACT}")

        if "Out" in status:
            gewicht = 1.0
        elif "Doubtful" in status:
            gewicht = 0.5
        else:
            gewicht = 0.0  # Day-to-Day: kein Impact, nur Anzeige

        total_impact += impact * gewicht

        if "Day-To-Day" in status:
            details.append(f"⚠️ {name} ({impact:.1f}, {status})")
        else:
            details.append(f"{name} ({impact:.1f}, {status})")

    return total_impact, details


def korrigiere_vorhersagen(vorhersagen_df, verletzungen_df, impact_df):
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


# ─── SCHRITT 9: Quoten-Integration (optional) ───────────────────────────────

def enrich_with_odds(predictions_df, datum):
    if not HAS_ODDS:
        log.info("[Odds] fetch_odds nicht verfügbar – übersprungen.")
        return predictions_df

    odds_path = f"odds/odds_{datum}.csv"
    if not os.path.exists(odds_path):
        log.warning(f"[Odds] Keine Quoten-Datei gefunden: {odds_path}")
        predictions_df["Heim Quote"] = None
        predictions_df["Ausw Quote"] = None
        predictions_df["EV"] = None
        predictions_df["Edge"] = None
        predictions_df["Value Bet"] = False
        return predictions_df

    odds_df = pd.read_csv(odds_path)
    odds_df = odds_df.rename(columns={
        "home_team": "Heimteam", "away_team": "Auswärtsteam",
        "home_odds": "Heim Quote", "away_odds": "Ausw Quote",
    })
    merged = predictions_df.merge(
        odds_df[["Heimteam", "Auswärtsteam", "Heim Quote", "Ausw Quote"]],
        on=["Heimteam", "Auswärtsteam"], how="left",
    )

    ev_list, edge_list, value_bet_list = [], [], []
    for _, row in merged.iterrows():
        if pd.isna(row.get("Heim Quote")) or pd.isna(row.get("Ausw Quote")):
            ev_list.append(None)
            edge_list.append(None)
            value_bet_list.append(False)
            continue
        _, ev, _, edge = calculate_ev(row["Heimsieg %"] / 100, row["Heim Quote"], row["Ausw Quote"])
        ev_list.append(ev)
        edge_list.append(edge)
        value_bet_list.append(ev > 0)

    merged["EV"] = ev_list
    merged["Edge"] = edge_list
    merged["Value Bet"] = value_bet_list
    log.info(f"[Odds] {merged['Heim Quote'].notna().sum()}/{len(merged)} Spiele mit Quoten")
    return merged


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="KOBRA – NBA Game Predictions")
    parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"),
                        help="Datum für Vorhersagen (YYYY-MM-DD)")
    parser.add_argument("--seasons", nargs="+", type=int, default=[2023, 2024, 2025],
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
    impact_df = lade_impact_scores()
    verletzungen_df = lade_verletzungen(impact_df, datum=datum)

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

    # 9) Quoten anreichern + Speichern
    # finale_df = enrich_with_odds(finale_df, datum)  # deaktiviert
    finale_df.to_csv(output, index=False)
    log.info(f"Gespeichert: {output}")


if __name__ == "__main__":
    main()
