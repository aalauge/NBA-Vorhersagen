"""
KOBRA – Discord Ergebnis-Notifications
=======================================
Holt Ergebnisse, vergleicht mit Predictions, sendet an Discord.

Usage:
    python notify_ergebnisse.py --datum 2026-04-13
    python notify_ergebnisse.py                        # gestern

Benötigte Umgebungsvariablen:
    BALLDONTLIE_API_KEY     – API Key für Ergebnisse
    DISCORD_WEBHOOK_ERGEBNISSE    – Webhook für #ergebnisse
"""

import argparse
import os
import sys
from datetime import datetime, timedelta

import pandas as pd
import requests


# ─── Team-Namen ───────────────────────────────────────────────────────────────

FULL_TO_SHORT = {
    "Atlanta Hawks": "Hawks", "Boston Celtics": "Celtics", "Brooklyn Nets": "Nets",
    "Charlotte Hornets": "Hornets", "Chicago Bulls": "Bulls", "Cleveland Cavaliers": "Cavaliers",
    "Dallas Mavericks": "Mavericks", "Denver Nuggets": "Nuggets", "Detroit Pistons": "Pistons",
    "Golden State Warriors": "Warriors", "Houston Rockets": "Rockets", "Indiana Pacers": "Pacers",
    "LA Clippers": "Clippers", "Los Angeles Lakers": "Lakers", "Memphis Grizzlies": "Grizzlies",
    "Miami Heat": "Heat", "Milwaukee Bucks": "Bucks", "Minnesota Timberwolves": "Timberwolves",
    "New Orleans Pelicans": "Pelicans", "New York Knicks": "Knicks",
    "Oklahoma City Thunder": "Thunder", "Orlando Magic": "Magic",
    "Philadelphia 76ers": "76ers", "Phoenix Suns": "Suns",
    "Portland Trail Blazers": "Trail Blazers", "Sacramento Kings": "Kings",
    "San Antonio Spurs": "Spurs", "Toronto Raptors": "Raptors",
    "Utah Jazz": "Jazz", "Washington Wizards": "Wizards",
}


def short(name: str) -> str:
    return FULL_TO_SHORT.get(name, name)


# ─── Ergebnisse holen ─────────────────────────────────────────────────────────

def hole_ergebnisse(datum: str, api_key: str) -> list[dict]:
    url = "https://api.balldontlie.io/v1/games"
    params = {"dates[]": datum, "per_page": 100}
    headers = {"Authorization": api_key}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=30)
        if r.status_code != 200:
            print(f"API Fehler: {r.status_code}")
            return []
        spiele = r.json()["data"]
        return [s for s in spiele if s["status"] == "Final"]
    except Exception as e:
        print(f"API Fehler: {e}")
        return []


# ─── Discord senden ───────────────────────────────────────────────────────────

def sende_discord(webhook_url: str, embeds: list[dict]) -> bool:
    try:
        r = requests.post(webhook_url, json={"embeds": embeds}, timeout=15)
        return r.status_code in (200, 204)
    except Exception as e:
        print(f"Discord Fehler: {e}")
        return False


# ─── Tier-Konstanten ─────────────────────────────────────────────────────────

MIN_QUOTE = 1.25
STRONG_EV_THRESHOLD = 10.0


def klassifiziere_tier(ev, quote) -> str:
    if pd.isna(ev) or pd.isna(quote) or quote == 0:
        return "skip"
    if ev > 0 and quote >= MIN_QUOTE and ev >= STRONG_EV_THRESHOLD:
        return "strong"
    if ev > 0 and quote >= MIN_QUOTE:
        return "value"
    return "skip"


# ─── Ergebnisse formatieren und senden ────────────────────────────────────────

def sende_ergebnisse(predictions_df: pd.DataFrame, ergebnisse: list[dict],
                     odds_df: pd.DataFrame, datum_display: str, webhook: str):

    # Ergebnisse als Dict: Heimteam -> (home_score, away_score, winner)
    results = {}
    for spiel in ergebnisse:
        heim = spiel["home_team"]["full_name"]
        h_score = spiel["home_team_score"]
        a_score = spiel["visitor_team_score"]
        winner = heim if h_score > a_score else spiel["visitor_team"]["full_name"]
        results[heim] = {"home_score": h_score, "away_score": a_score, "winner": winner}

    # Merge predictions mit odds + Tier berechnen
    if odds_df is not None and len(odds_df) > 0:
        merged = predictions_df.merge(odds_df, on="Heimteam", how="left")
        merged["EV"] = (merged["Konfidenz"] / 100 * merged["Quote"] - 1) * 100
        merged["Tier"] = merged.apply(lambda r: klassifiziere_tier(r["EV"], r["Quote"]), axis=1)
    else:
        merged = predictions_df.copy()
        merged["Quote"] = None
        merged["EV"] = None
        merged["Tier"] = "skip"

    strong_lines = []
    vb_lines = []
    rest_lines = []
    strong_correct, strong_total = 0, 0
    vb_correct, vb_total = 0, 0
    all_correct, all_total = 0, 0

    for _, row in merged.iterrows():
        heim_full = row["Heimteam"]
        ausw_full = row["Auswärtsteam"]
        heim = short(heim_full)
        ausw = short(ausw_full)
        tipp = row["Tipp"]

        if heim_full not in results:
            continue

        res = results[heim_full]
        h_score = res["home_score"]
        a_score = res["away_score"]
        winner = res["winner"]
        richtig = tipp == winner

        all_total += 1
        if richtig:
            all_correct += 1

        score_str = f"{h_score} – {a_score}"
        tier = row["Tier"]

        if tier in ("strong", "value"):
            target_lines = strong_lines if tier == "strong" else vb_lines
            counter_total = "strong_total" if tier == "strong" else "vb_total"
            counter_correct = "strong_correct" if tier == "strong" else "vb_correct"

            if tier == "strong":
                strong_total += 1
            else:
                vb_total += 1

            tier_label = "🔥" if tier == "strong" else "✅"

            if richtig:
                if tier == "strong":
                    strong_correct += 1
                else:
                    vb_correct += 1
                profit = (row["Quote"] - 1) * 10
                target_lines.append(f"✅  **{heim}** {score_str} **{ausw}**")
                target_lines.append(f"      {tier_label} Pick: {short(tipp)}  ·  +{profit:.2f} EUR")
            else:
                target_lines.append(f"❌  **{heim}** {score_str} **{ausw}**")
                target_lines.append(f"      {tier_label} Pick: {short(tipp)}  ·  −10.00 EUR")
            target_lines.append("")
        else:
            if richtig:
                rest_lines.append(f"✅  **{heim}** {score_str} **{ausw}**")
            else:
                rest_lines.append(f"❌  **{heim}** {score_str} **{ausw}**")
            rest_lines.append(f"      Pick: {short(tipp)}")
            rest_lines.append("")

    embeds = []

    # 🔥 Strong Value Ergebnisse
    if len(strong_lines) > 0:
        embeds.append({
            "color": 0xFF9500,
            "author": {"name": "🐍 KOBRA Ergebnisse"},
            "title": f"🔥  Strong Value  ·  {datum_display}",
            "description": "\n".join(strong_lines).strip(),
            "footer": {"text": f"Strong Value: {strong_correct}/{strong_total} richtig"},
        })

    # ✅ Value Bet Ergebnisse
    if len(vb_lines) > 0:
        embeds.append({
            "color": 0x57F287,
            "author": {"name": "🐍 KOBRA Ergebnisse"},
            "title": f"💰  Value Bets  ·  {datum_display}",
            "description": "\n".join(vb_lines).strip(),
            "footer": {"text": f"Value Bets: {vb_correct}/{vb_total} richtig"},
        })

    # Graues Embed: Restliche Spiele
    if len(rest_lines) > 0:
        picks_total = strong_total + vb_total
        picks_correct = strong_correct + vb_correct
        embeds.append({
            "color": 0x95A5A6,
            "title": f"📊  Weitere Spiele  ·  {datum_display}",
            "description": "\n".join(rest_lines).strip(),
            "footer": {"text": f"Gesamt: {all_correct}/{all_total} richtig ({all_correct/all_total*100:.0f}%)"},
        })

    if len(embeds) > 0:
        if sende_discord(webhook, embeds):
            print(f"✅ Ergebnisse gesendet ({all_correct}/{all_total} richtig)")
        else:
            print("❌ Ergebnisse senden fehlgeschlagen")
    else:
        print("⚠️ Keine Ergebnisse gefunden")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="KOBRA – Ergebnis-Notifications")
    parser.add_argument("--datum", default=None,
                        help="Datum (YYYY-MM-DD), default = gestern")
    parser.add_argument("--predictions", default=None,
                        help="Pfad zur Predictions-CSV")
    parser.add_argument("--odds", default=None,
                        help="Pfad zur Odds-CSV (optional)")
    args = parser.parse_args()

    if args.datum is None:
        gestern = datetime.now() - timedelta(days=1)
        datum = gestern.strftime("%Y-%m-%d")
    else:
        datum = args.datum

    datum_display = datetime.strptime(datum, "%Y-%m-%d").strftime("%d.%m.%Y")

    api_key = os.environ.get("BALLDONTLIE_API_KEY", "")
    webhook = os.environ.get("DISCORD_WEBHOOK_ERGEBNISSE", "")

    if not api_key:
        print("❌ BALLDONTLIE_API_KEY nicht gesetzt!")
        sys.exit(1)
    if not webhook:
        print("❌ DISCORD_WEBHOOK_ERGEBNISSE nicht gesetzt!")
        sys.exit(1)

    # Predictions CSV finden
    pred_path = args.predictions
    if pred_path is None:
        pred_path = f"results/predictions_{datum}.csv"
        if not os.path.exists(pred_path):
            pred_path = "predictions.csv"

    if not os.path.exists(pred_path):
        print(f"❌ Predictions nicht gefunden: {pred_path}")
        sys.exit(1)

    df = pd.read_csv(pred_path)
    print(f"🏀 {len(df)} Predictions geladen")

    # Odds CSV laden (optional)
    odds_df = None
    if args.odds and os.path.exists(args.odds):
        odds_df = pd.read_csv(args.odds)
        print(f"📊 {len(odds_df)} Quoten geladen")

    # Ergebnisse holen
    ergebnisse = hole_ergebnisse(datum, api_key)
    print(f"📡 {len(ergebnisse)} Ergebnisse geholt für {datum}")

    if len(ergebnisse) == 0:
        print("⚠️ Keine Ergebnisse gefunden")
        sys.exit(0)

    sende_ergebnisse(df, ergebnisse, odds_df, datum_display, webhook)


if __name__ == "__main__":
    main()
