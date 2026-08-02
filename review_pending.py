#!/usr/bin/env python3
"""
review_pending.py — record a human decision on a pending UFFIS reading.

This is the piece that closes the loop: run_uffis_pipeline.py queues ORANGE/RED
readings into uffis_data.db, but nothing was actually able to confirm or reject
them and have it stick. This script does that.

Intended to be run via the "UFFIS review" GitHub Actions workflow (workflow_dispatch
with input fields — no command line needed, just fill in a form on github.com), but
also works run directly:

    python3 review_pending.py --review-id 5 --decision CONFIRMED \\
        --reviewer "Traffic Police - Zone 3" --notes "Confirmed waterlogging on site"

Effects of a CONFIRMED decision:
  - alert_review_queue.status -> CONFIRMED, cleared_for_public_alert -> 1
  - alert_dissemination_log gets a row per Tier-2 channel for that alert band
    (RED -> SMS/social/loudspeaker, ORANGE -> app/WhatsApp advisory)

Effects of a REJECTED decision:
  - alert_review_queue.status -> REJECTED, cleared_for_public_alert stays 0
  - nothing is added to alert_dissemination_log — the gate works.

Either way, PENDING_REVIEWS.md is regenerated so the reviewed row drops off the list.
"""

import argparse
import sqlite3
import sys
from datetime import datetime, timezone

import pandas as pd

DB_PATH = "uffis_data.db"
PENDING_MD_PATH = "PENDING_REVIEWS.md"

TIER2_CHANNELS = {
    "RED":    ["SMS cell-broadcast (public)", "Official social media post", "Ward loudspeaker announcement"],
    "ORANGE": ["In-app push notification (advisory)", "Local ward WhatsApp group (advisory)"],
}


def log(msg):
    print(f"[{datetime.now(timezone.utc).isoformat()}] {msg}", flush=True)


def ensure_dissemination_table(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS alert_dissemination_log (
        dissemination_id INTEGER PRIMARY KEY AUTOINCREMENT,
        review_id INTEGER NOT NULL,
        tier TEXT NOT NULL,
        channel TEXT NOT NULL,
        message_summary TEXT,
        sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (review_id) REFERENCES alert_review_queue(review_id)
    );""")


def record_decision(review_id, decision, reviewer, notes, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    ensure_dissemination_table(conn)

    row = cur.execute(
        "SELECT locality, alert_band, fused_score, status FROM alert_review_queue WHERE review_id = ?",
        (review_id,)
    ).fetchone()

    if row is None:
        conn.close()
        sys.exit(f"No review with review_id={review_id}. Check PENDING_REVIEWS.md for valid IDs.")

    locality, band, score, current_status = row
    if current_status != "PENDING_REVIEW":
        conn.close()
        sys.exit(f"review_id={review_id} ({locality}) is already {current_status}, not PENDING_REVIEW. "
                  f"Refusing to overwrite an existing decision.")

    cleared = 1 if decision == "CONFIRMED" else 0
    cur.execute(
        "UPDATE alert_review_queue SET status = ?, cleared_for_public_alert = ?, "
        "reviewed_by = ?, reviewed_at = CURRENT_TIMESTAMP, reviewer_notes = ? WHERE review_id = ?",
        (decision, cleared, reviewer, notes, review_id)
    )

    if decision == "CONFIRMED":
        for channel in TIER2_CHANNELS.get(band, []):
            cur.execute(
                "INSERT INTO alert_dissemination_log (review_id, tier, channel, message_summary) "
                "VALUES (?, 'Tier 2 - public', ?, ?)",
                (review_id, channel, f"{band} alert confirmed for {locality} (score {score:.1f})")
            )
        log(f"CONFIRMED review_id={review_id} ({locality}, {band}) — "
            f"{len(TIER2_CHANNELS.get(band, []))} dissemination channel(s) fired.")
    else:
        log(f"REJECTED review_id={review_id} ({locality}, {band}) — no dissemination. "
            f"Reason on file: {notes or '(none given)'}")

    conn.commit()
    conn.close()


def regenerate_pending_md(db_path=DB_PATH, md_path=PENDING_MD_PATH):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(
        "SELECT review_id, locality, alert_band, fused_score, nlp_score, rain_score, queued_at "
        "FROM alert_review_queue WHERE status = 'PENDING_REVIEW' ORDER BY fused_score DESC", conn
    )
    conn.close()

    lines = [
        "# Pending human reviews",
        "",
        f"Last updated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "Nothing here is public yet. To confirm or reject a row, go to the **Actions** tab -> "
        "**UFFIS review** -> **Run workflow**, and fill in the `review_id` from the table below.",
        "",
    ]
    if df.empty:
        lines.append("_Nothing pending right now._")
    else:
        lines.append("| review_id | locality | band | score | nlp | rain | queued_at |")
        lines.append("|---|---|---|---|---|---|---|")
        for _, r in df.iterrows():
            lines.append(f"| {r.review_id} | {r.locality} | {r.alert_band} | {r.fused_score:.1f} "
                         f"| {r.nlp_score:.1f} | {r.rain_score:.1f} | {r.queued_at} |")

    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    log(f"Wrote {md_path} — {len(df)} row(s) still awaiting review.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--review-id", type=int, required=True)
    parser.add_argument("--decision", choices=["CONFIRMED", "REJECTED"], required=True)
    parser.add_argument("--reviewer", required=True, help='e.g. "Traffic Police - Zone 3"')
    parser.add_argument("--notes", default="", help="Optional context, e.g. what a field check found")
    args = parser.parse_args()

    record_decision(args.review_id, args.decision, args.reviewer, args.notes)
    regenerate_pending_md()
