import os
import requests
import pandas as pd
from datetime import datetime, timezone

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY   = os.environ.get("ODDS_API_KEY", "5c5e8f1d3d6210d7bba3684a45cf49ed")
SPORT     = "basketball_nba"
REGIONS   = "eu"          # eu = dezimale Quoten europäischer Buchmacher
MARKETS   = "h2h"
ODDS_FMT  = "decimal"
BASE_URL  = "https://api.the-odds-api.com/v4/sports"

# The Odds API Teamnamen → Balldontlie Teamnamen
TEAM_MAP = {
    "Atlanta Hawks": "Atlanta Hawks",
    "Boston Celtics": "Boston Celtics",
    "Brooklyn Nets": "Brooklyn Nets",
    "Charlotte Hornets": "Charlotte Hornets",
    "Chicago Bulls": "Chicago Bulls",
    "Cleveland Cavaliers": "Cleveland Cavaliers",
    "Dallas Mavericks": "Dallas Mavericks",
    "Denver Nuggets": "Denver Nuggets",
    "Detroit Pistons": "Detroit Pistons",
    "Golden State Warriors": "Golden State Warriors",
    "Houston Rockets": "Houston Rockets",
    "Indiana Pacers": "Indiana Pacers",
    "LA Clippers": "LA Clippers",
    "Los Angeles Lakers": "Los Angeles Lakers",
    "Memphis Grizzlies": "Memphis Grizzlies",
    "Miami Heat": "Miami Heat",
    "Milwaukee Bucks": "Milwaukee Bucks",
    "Minnesota Timberwolves": "Minnesota Timberwolves",
    "New Orleans Pelicans": "New Orleans Pelicans",
    "New York Knicks": "New York Knicks",
    "Oklahoma City Thunder": "Oklahoma City Thunder",
    "Orlando Magic": "Orlando Magic",
    "Philadelphia 76ers": "Philadelphia 76ers",
    "Phoenix Suns": "Phoenix Suns",
    "Portland Trail Blazers": "Portland Trail Blazers",
    "Sacramento Kings": "Sacramento Kings",
    "San Antonio Spurs": "San Antonio Spurs",
    "Toronto Raptors": "Toronto Raptors",
    "Utah Jazz": "Utah Jazz",
    "Washington Wizards": "Washington Wizards",
}

def fetch_odds():
    url = f"{BASE_URL}/{SPORT}/odds/"
    params = {
        "apiKey":     API_KEY,
        "regions":    REGIONS,
        "markets":    MARKETS,
        "oddsFormat": ODDS_FMT,
    }

    resp = requests.get(url, params=params, timeout=15)

    # Verbleibende Credits loggen
    remaining = resp.headers.get("x-requests-remaining", "?")
    used      = resp.headers.get("x-requests-used", "?")
    print(f"[Odds API] Credits used: {used} | remaining: {remaining}")

    if resp.status_code != 200:
        print(f"[Odds API] Fehler {resp.status_code}: {resp.text}")
        return pd.DataFrame()

    data = resp.json()
    rows = []

    for game in data:
        home_raw = game.get("home_team", "")
        away_raw = game.get("away_team", "")
        home_team = TEAM_MAP.get(home_raw, home_raw)
        away_team = TEAM_MAP.get(away_raw, away_raw)

        # Commence time → Datum
        commence = game.get("commence_time", "")
        game_date = commence[:10] if commence else datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Beste verfügbare Quote pro Seite (Maximum über alle Bookmaker)
        best_home = 1.0
        best_away = 1.0
        bookmaker_name = ""

        for bookmaker in game.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                for outcome in market.get("outcomes", []):
                    name  = outcome.get("name", "")
                    price = outcome.get("price", 1.0)
                    if name == home_raw and price > best_home:
                        best_home = price
                        bookmaker_name = bookmaker.get("title", "")
                    elif name == away_raw and price > best_away:
                        best_away = price

        if best_home == 1.0 and best_away == 1.0:
            continue  # Keine Quoten gefunden

        rows.append({
            "date":      game_date,
            "home_team": home_team,
            "away_team": away_team,
            "home_odds": round(best_home, 3),
            "away_odds": round(best_away, 3),
            "bookmaker": bookmaker_name,
        })

    df = pd.DataFrame(rows)
    return df


def calculate_ev(model_home_prob, home_odds, away_odds):
    """
    EV aus Modell-Wahrscheinlichkeit und Bookie-Quote.
    Gibt zurück: (tipp, ev, bookie_implied_prob, edge)
    """
    # Bookie-Wahrscheinlichkeit bereinigt um Overround
    raw_home = 1 / home_odds
    raw_away = 1 / away_odds
    overround = raw_home + raw_away
    bookie_home_prob = raw_home / overround
    bookie_away_prob = raw_away / overround

    model_away_prob = 1 - model_home_prob

    ev_home = (model_home_prob * home_odds - 1) * 100
    ev_away = (model_away_prob * away_odds - 1) * 100

    if ev_home >= ev_away:
        return "home", round(ev_home, 1), round(bookie_home_prob, 4), round(model_home_prob - bookie_home_prob, 4)
    else:
        return "away", round(ev_away, 1), round(bookie_away_prob, 4), round(model_away_prob - bookie_away_prob, 4)


if __name__ == "__main__":
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    os.makedirs("odds", exist_ok=True)

    df = fetch_odds()

    if df.empty:
        print("[Odds API] Keine Quoten gefunden oder Fehler.")
    else:
        out_path = f"odds/odds_{today}.csv"
        df.to_csv(out_path, index=False)
        print(f"[Odds API] {len(df)} Spiele gespeichert → {out_path}")
        print(df.to_string(index=False))
