"""
Sendet die KOBRA-Vorhersagen automatisch an Telegram.
Kann an einzelne Chats oder Gruppen senden.

Benötigte Umgebungsvariablen:
    TELEGRAM_BOT_TOKEN  - Bot-Token von BotFather
    TELEGRAM_CHAT_IDS   - Komma-getrennte Chat-IDs (z.B. "123456,789012,-100987654")
                          Gruppen-IDs beginnen mit einem Minus (z.B. -100987654321)
"""

import os
import sys
import pandas as pd
from datetime import datetime
import requests


def sende_telegram(bot_token: str, chat_id: str, nachricht: str) -> bool:
    """Sendet eine Nachricht an einen Telegram-Chat."""
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": nachricht,
        "parse_mode": "Markdown",
    }
    try:
        r = requests.post(url, data=payload, timeout=15)
        if r.status_code == 200:
            return True
        else:
            print(f"Fehler bei Chat {chat_id}: {r.text}")
            return False
    except Exception as e:
        print(f"Fehler bei Chat {chat_id}: {e}")
        return False


def erstelle_nachricht(df: pd.DataFrame, datum: str) -> str:
    """Baut die Telegram-Nachricht aus den Vorhersagen."""
    nachricht = f"🏀 *KOBRA – NBA Vorhersagen {datum}*\n"
    nachricht += f"📊 {len(df)} Spiele heute\n"
    nachricht += "─" * 28 + "\n\n"

    for _, row in df.iterrows():
        heim = row["Heimteam"]
        ausw = row["Auswärtsteam"]
        tipp = row["Tipp"]
        konfidenz = row["Konfidenz"]
        heim_pct = row["Heimsieg %"]

        # Konfidenz-Emoji
        if konfidenz >= 85:
            emoji = "🔥"
        elif konfidenz >= 70:
            emoji = "✅"
        elif konfidenz >= 60:
            emoji = "👀"
        else:
            emoji = "⚠️"

        nachricht += f"{emoji} *{heim}* vs *{ausw}*\n"
        nachricht += f"   ➡️ Tipp: *{tipp}* ({konfidenz}%)\n"

        # Verletzungen (gekürzt)
        heim_verl = row.get("Heim_Verletzungen", "-")
        ausw_verl = row.get("Ausw_Verletzungen", "-")
        if heim_verl and heim_verl != "-":
            # Nur Spielernamen, gekürzt
            namen = [v.split("(")[0].strip() for v in str(heim_verl).split(",")]
            if len(namen) > 3:
                nachricht += f"   🏥 {heim}: {', '.join(namen[:3])} +{len(namen)-3}\n"
            else:
                nachricht += f"   🏥 {heim}: {', '.join(namen)}\n"
        if ausw_verl and ausw_verl != "-":
            namen = [v.split("(")[0].strip() for v in str(ausw_verl).split(",")]
            if len(namen) > 3:
                nachricht += f"   🏥 {ausw}: {', '.join(namen[:3])} +{len(namen)-3}\n"
            else:
                nachricht += f"   🏥 {ausw}: {', '.join(namen)}\n"

        nachricht += "\n"

    nachricht += "─" * 28 + "\n"
    nachricht += "🤖 _KOBRA Prediction Model_"

    return nachricht


def main():
    # Token und Chat-IDs aus Umgebungsvariablen oder Argumenten
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_ids_str = os.environ.get("TELEGRAM_CHAT_IDS", "")

    if not bot_token:
        print("TELEGRAM_BOT_TOKEN nicht gesetzt!")
        sys.exit(1)

    if not chat_ids_str:
        print("TELEGRAM_CHAT_IDS nicht gesetzt!")
        sys.exit(1)

    chat_ids = [cid.strip() for cid in chat_ids_str.split(",")]

    # Vorhersage-Datei finden
    datum = datetime.now().strftime("%Y-%m-%d")
    dateiname = "predictions.csv"

    if not os.path.exists(dateiname):
        # Fallback: Mit Datum
        dateiname = f"predictions_{datum}.csv"

    if not os.path.exists(dateiname):
        print(f"Keine Vorhersage-Datei gefunden!")
        nachricht = f"⚠️ *KOBRA – Keine Vorhersagen für {datum}*\n\nKeine Spiele gefunden oder Fehler im Workflow."
        for cid in chat_ids:
            sende_telegram(bot_token, cid, nachricht)
        sys.exit(0)

    # CSV laden und Nachricht erstellen
    df = pd.read_csv(dateiname)
    nachricht = erstelle_nachricht(df, datum)

    # An alle Chat-IDs senden
    erfolg = 0
    for cid in chat_ids:
        if sende_telegram(bot_token, cid, nachricht):
            erfolg += 1
            print(f"✅ Gesendet an {cid}")

    print(f"\n{erfolg}/{len(chat_ids)} Nachrichten gesendet")


if __name__ == "__main__":
    main()
