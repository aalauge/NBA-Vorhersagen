"""
KOBRA – Discord Webhook Notifications
======================================
Sendet Vorhersagen und Value Bets automatisch an Discord.

Zwei Modi:
    --morgens   Frühprognose an #vorhersagen (Free)
    --abends    Finale Version an #vorhersagen + Value Bets an #daily-picks (Premium)

Benötigte Umgebungsvariablen:
    DISCORD_WEBHOOK_FREE    – Webhook-URL für #vorhersagen
    DISCORD_WEBHOOK_PREMIUM – Webhook-URL für #daily-picks

Usage:
    python notify_discord.py --morgens                          # Frühprognose
    python notify_discord.py --abends --odds odds_today.csv     # Finale + Value Bets
"""

import argparse
import os
import sys
from datetime import datetime

import pandas as pd
import requests


# ─── Team-Namen kürzen ────────────────────────────────────────────────────────

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


# ─── Discord Embed senden ─────────────────────────────────────────────────────

def sende_discord(webhook_url: str, embeds: list[dict]) -> bool:
    payload = {"embeds": embeds}
    try:
        r = requests.post(webhook_url, json=payload, timeout=15)
        if r.status_code in (200, 204):
            return True
        else:
            print(f"Discord Fehler: {r.status_code} – {r.text}")
            return False
    except Exception as e:
        print(f"Discord Fehler: {e}")
        return False


# ─── Morgens: Frühprognose (Free) ─────────────────────────────────────────────

def sende_fruehprognose(df: pd.DataFrame, datum: str, webhook_free: str):
    lines = []
    for _, row in df.iterrows():
        heim = short(row["Heimteam"])
        ausw = short(row["Auswärtsteam"])
        tipp = short(row["Tipp"])
        konf = row["Konfidenz"]

        lines.append(f"🏀  **{heim}** vs **{ausw}**")
        lines.append(f"      ➜ {tipp} ({konf}%)")
        lines.append("")

    embed = {
        "color": 0x00D4AA,
        "author": {"name": "🐍 KOBRA Vorhersagen"},
        "title": f"{datum}  ·  Frühprognose",
        "description": "\n".join(lines).strip(),
        "footer": {"text": f"📊 {len(df)} Spiele  ·  Finale Version folgt heute Abend"},
    }

    if sende_discord(webhook_free, [embed]):
        print(f"✅ Frühprognose gesendet ({len(df)} Spiele)")
    else:
        print("❌ Frühprognose fehlgeschlagen")


# ─── Tier-Konstanten ─────────────────────────────────────────────────────────

MIN_QUOTE = 1.25       # Mindestquote für Value Bets
STRONG_EV_THRESHOLD = 10.0  # EV >= 10% = Strong Value


# ─── Abends: Finale Vorhersage (Free) + Value Bets (Premium) ─────────────────

def klassifiziere_tier(ev: float, quote: float) -> str:
    """Bestimmt den Tier eines Spiels basierend auf EV und Quote."""
    if pd.isna(ev) or pd.isna(quote) or quote == 0:
        return "skip"
    if ev > 0 and quote >= MIN_QUOTE and ev >= STRONG_EV_THRESHOLD:
        return "strong"
    if ev > 0 and quote >= MIN_QUOTE:
        return "value"
    return "skip"


def skip_grund(ev: float, quote: float) -> str:
    """Gibt den Skip-Grund als kurzen Text zurück."""
    if pd.isna(quote) or quote == 0:
        return "Keine Quote verfügbar"
    if quote < MIN_QUOTE:
        return f"Quote {quote:.2f}  ·  Quote zu niedrig (<{MIN_QUOTE})"
    return f"Quote {quote:.2f}  ·  EV {ev:.1f}%  →  Negativer EV"


def sende_abend(df: pd.DataFrame, odds_df: pd.DataFrame, datum: str,
                webhook_free: str, webhook_premium: str):

    merged = df.merge(odds_df, on="Heimteam", how="left")
    merged["EV"] = (merged["Konfidenz"] / 100 * merged["Quote"] - 1) * 100
    merged["EV"] = merged["EV"].round(1)
    merged["Tier"] = merged.apply(lambda r: klassifiziere_tier(r["EV"], r["Quote"]), axis=1)

    # ─── Free: Finale Vorhersage ───
    free_lines = []
    for _, row in merged.iterrows():
        heim = short(row["Heimteam"])
        ausw = short(row["Auswärtsteam"])
        tipp = short(row["Tipp"])
        konf = row["Konfidenz"]

        free_lines.append(f"🏀  **{heim}** vs **{ausw}**")
        free_lines.append(f"      ➜ {tipp} ({konf}%)")
        free_lines.append("")

    free_embed = {
        "color": 0x00D4AA,
        "author": {"name": "🐍 KOBRA Vorhersagen"},
        "title": f"{datum}  ·  Final",
        "description": "\n".join(free_lines).strip(),
        "footer": {"text": f"📊 {len(merged)} Spiele  ·  Inkl. Verletzungskorrekturen"},
    }

    if sende_discord(webhook_free, [free_embed]):
        print(f"✅ Finale Vorhersage gesendet ({len(merged)} Spiele)")
    else:
        print("❌ Finale Vorhersage fehlgeschlagen")

    # ─── Premium: Strong Value + Value Bets + Skips ───
    strong_bets = merged[merged["Tier"] == "strong"]
    value_bets = merged[merged["Tier"] == "value"]
    skips = merged[merged["Tier"] == "skip"]

    premium_embeds = []
    pick_nr = 1

    # 🔥 Strong Value Embed
    if len(strong_bets) > 0:
        strong_lines = []
        for _, row in strong_bets.iterrows():
            tipp = short(row["Tipp"])
            gegner = short(row["Auswärtsteam"] if row["Tipp"] == row["Heimteam"] else row["Heimteam"])

            strong_lines.append(f"**PICK #{pick_nr}  ·  🔥 STRONG VALUE**")
            strong_lines.append(f"**{tipp}**")
            strong_lines.append(f"vs {gegner}  ·  Quote {row['Quote']:.2f}  ·  EV +{row['EV']:.1f}%")
            strong_lines.append("")
            pick_nr += 1

        premium_embeds.append({
            "color": 0xFF9500,
            "author": {"name": "🐍 KOBRA Picks"},
            "title": f"🔥  Strong Value  ·  {datum}",
            "description": "\n".join(strong_lines).strip(),
            "footer": {"text": f"{len(strong_bets)} Strong Value Picks  ·  EV ≥ {STRONG_EV_THRESHOLD:.0f}%"},
        })

    # ✅ Value Bet Embed
    if len(value_bets) > 0:
        vb_lines = []
        for _, row in value_bets.iterrows():
            tipp = short(row["Tipp"])
            gegner = short(row["Auswärtsteam"] if row["Tipp"] == row["Heimteam"] else row["Heimteam"])

            vb_lines.append(f"**PICK #{pick_nr}  ·  ✅ VALUE BET**")
            vb_lines.append(f"**{tipp}**")
            vb_lines.append(f"vs {gegner}  ·  Quote {row['Quote']:.2f}  ·  EV +{row['EV']:.1f}%")
            vb_lines.append("")
            pick_nr += 1

        premium_embeds.append({
            "color": 0x57F287,
            "author": {"name": "🐍 KOBRA Picks"},
            "title": f"✅  Value Bets  ·  {datum}",
            "description": "\n".join(vb_lines).strip(),
            "footer": {"text": f"{len(value_bets)} Value Bets  ·  EV 0–{STRONG_EV_THRESHOLD:.0f}%"},
        })

    # ⛔ Skip Embed
    if len(skips) > 0:
        skip_lines = []
        for _, row in skips.iterrows():
            tipp = short(row["Tipp"])
            gegner = short(row["Auswärtsteam"] if row["Tipp"] == row["Heimteam"] else row["Heimteam"])

            skip_lines.append(f"~~{tipp} vs {gegner}~~")
            skip_lines.append(f"→ {skip_grund(row['EV'], row['Quote'])}")
            skip_lines.append("")

        premium_embeds.append({
            "color": 0xED4245,
            "title": f"⛔  Skips  ·  {datum}",
            "description": "\n".join(skip_lines).strip(),
        })

    # Senden
    if len(premium_embeds) > 0:
        total_picks = len(strong_bets) + len(value_bets)
        total = len(merged)
        if sende_discord(webhook_premium, premium_embeds):
            print(f"✅ Premium Picks gesendet ({len(strong_bets)} Strong, {len(value_bets)} Value, {len(skips)} Skips)")
        else:
            print("❌ Premium Picks fehlgeschlagen")
    else:
        print("⚠️ Keine Spiele mit Quoten – Premium nicht gesendet")


# ─── Odds CSV laden ───────────────────────────────────────────────────────────

def lade_odds(pfad: str) -> pd.DataFrame:
    """
    Erwartet CSV mit: Heimteam, Quote
    Beispiel:
        Heimteam,Quote
        Oklahoma City Thunder,1.35
        Los Angeles Lakers,1.55
        Boston Celtics,1.12
    """
    if not os.path.exists(pfad):
        print(f"❌ Odds-Datei nicht gefunden: {pfad}")
        sys.exit(1)
    df = pd.read_csv(pfad)
    if "Heimteam" not in df.columns or "Quote" not in df.columns:
        print("❌ Odds-CSV muss Spalten 'Heimteam' und 'Quote' haben")
        sys.exit(1)
    print(f"📊 {len(df)} Quoten geladen aus {pfad}")
    return df


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="KOBRA – Discord Notifications")
    parser.add_argument("--morgens", action="store_true", help="Frühprognose senden")
    parser.add_argument("--abends", action="store_true", help="Finale + Value Bets senden")
    parser.add_argument("--predictions", default="predictions.csv",
                        help="Pfad zur Predictions-CSV")
    parser.add_argument("--odds", default=None,
                        help="Pfad zur Odds-CSV (nur für --abends)")
    parser.add_argument("--datum", default=datetime.now().strftime("%d.%m.%Y"),
                        help="Datum für die Überschrift (DD.MM.YYYY)")
    args = parser.parse_args()

    if not args.morgens and not args.abends:
        print("Bitte --morgens oder --abends angeben")
        sys.exit(1)

    webhook_free = os.environ.get("DISCORD_WEBHOOK_FREE", "")
    webhook_premium = os.environ.get("DISCORD_WEBHOOK_PREMIUM", "")

    if not webhook_free:
        print("❌ DISCORD_WEBHOOK_FREE nicht gesetzt!")
        sys.exit(1)

    if not os.path.exists(args.predictions):
        print(f"❌ Predictions nicht gefunden: {args.predictions}")
        sys.exit(1)
    df = pd.read_csv(args.predictions)
    print(f"🏀 {len(df)} Spiele geladen")

    if args.morgens:
        sende_fruehprognose(df, args.datum, webhook_free)

    if args.abends:
        if not webhook_premium:
            print("❌ DISCORD_WEBHOOK_PREMIUM nicht gesetzt!")
            sys.exit(1)
        if not args.odds:
            print("❌ --odds Pfad benötigt für Abend-Modus")
            sys.exit(1)
        odds_df = lade_odds(args.odds)
        sende_abend(df, odds_df, args.datum, webhook_free, webhook_premium)


if __name__ == "__main__":
    main()
