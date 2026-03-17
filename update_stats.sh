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

# Nur pushen wenn sich was geändert hat
if git diff --quiet player_stats.csv 2>/dev/null; then
    echo "$(date): Keine Änderungen, nichts zu pushen." >> "$LOG_FILE"
else
    git add player_stats.csv
    git commit -m "📊 Player Stats Update $(date +%Y-%m-%d)"
    git pull --rebase && git push
    echo "$(date): Stats gepusht!" >> "$LOG_FILE"
fi

echo "$(date): Fertig." >> "$LOG_FILE"
echo "---" >> "$LOG_FILE"
