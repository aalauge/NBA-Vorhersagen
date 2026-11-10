#!/bin/bash
# ─── KOBRA Stats Update ──────────────────────────────────
# Läuft täglich im Hintergrund auf deinem Mac.
# Holt Spieler-Stats und pusht sie ins GitHub-Repo.
#
# WICHTIG: Passe den Pfad unten an, falls dein Repo woanders liegt!
# ──────────────────────────────────────────────────────────

REPO_DIR="$HOME/kobra-nba"
LOG_FILE="$REPO_DIR/update_log.txt"

echo "$(date): Stats-Update gestartet" >> "$LOG_FILE"

cd "$REPO_DIR" || { echo "$(date): Ordner nicht gefunden!" >> "$LOG_FILE"; exit 1; }

# Python-Skript ausführen
/opt/anaconda3/bin/python3 update_stats.py >> "$LOG_FILE" 2>&1
/opt/anaconda3/bin/python3 update_kader.py >> "$LOG_FILE" 2>&1

# Nur pushen wenn sich was geändert hat
git add player_stats.csv kader_spieler.csv kader_historie.csv 2>/dev/null
if git diff --cached --quiet; then
    echo "$(date): Keine Änderungen, nichts zu pushen." >> "$LOG_FILE"
else
    git commit -m "📊 Stats + Kader Update $(date +%Y-%m-%d)"
    if git pull --rebase && git push; then
        echo "$(date): Stats gepusht!" >> "$LOG_FILE"
    else
        echo "$(date): ❌ PUSH FEHLGESCHLAGEN (Login/Token prüfen!)" >> "$LOG_FILE"
    fi
fi

echo "$(date): Fertig." >> "$LOG_FILE"
echo "---" >> "$LOG_FILE"
