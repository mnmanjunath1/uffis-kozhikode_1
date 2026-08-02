#!/usr/bin/env python3
"""
run_uffis_pipeline.py — UFFIS Kozhikode scheduled pipeline.

Reuses the exact logic already validated in the notebook (classifier training in
Section 2, locality extraction in Section 9a, forecast fetch in Section 6a, fusion
engine in Section 7, review queue in Section 10a) — this script just runs them
unattended, on a timer, instead of inside a manually-executed notebook cell.

Each run:
  1. Retrains the SVM classifier fresh from UFFIS_training_data.csv (sub-second on
     64 rows — simpler and safer than persisting a model file that could go stale).
  2. Pulls live headlines for Kozhikode via Google News RSS and classifies each one.
  3. Pulls a live rainfall forecast per locality via Open-Meteo (falls back to a
     placeholder if the API is unreachable — same graceful-degradation pattern as
     the notebook's Section 6a).
  4. Computes the fusion score + alert band per locality.
  5. Queues ORANGE/RED readings for human review in uffis_data.db — this does NOT
     send any public alert by itself. A human still has to confirm each one
     (Section 10a's rule holds here too: the model never gets the final word).
  6. Writes location_summary.json — a timestamped snapshot the dashboard can fetch.

Run manually:
    python3 run_uffis_pipeline.py

Run on a schedule:
    see uffis_pipeline.yml alongside this file (GitHub Actions, free, no server needed)

Requires: UFFIS_training_data.csv in the same directory.
"""

import json
import os
import re
import sqlite3
import sys
import urllib.request
from datetime import datetime, timezone

try:
    import feedparser
except ImportError:
    sys.exit("Missing dependency: pip install feedparser --break-system-packages")

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

DB_PATH = "uffis_data.db"
TRAINING_CSV = "UFFIS_training_data.csv"
SNAPSHOT_PATH = "location_summary.json"
PENDING_MD_PATH = "PENDING_REVIEWS.md"
HEADLINE_LOG_PATH = "live_headlines_log.csv"

# Same 19 localities as notebook Section 9, with the 3 coastline-offset corrections
# from earlier validation (West Hill, Kozhikode Beach, Vellayil).
KOZHIKODE_LOCATIONS = {
    "Palayam": (11.2536, 75.7794), "Mavoor Road": (11.2612, 75.7967), "Kallai": (11.2465, 75.7739),
    "Nadakkavu": (11.2632, 75.7850), "Puthiyara": (11.2523, 75.7860), "Eranhipalam": (11.2745, 75.7870),
    "Chevayur": (11.2802, 75.7778), "West Hill": (11.2890, 75.7620),
    "Beypore": (11.1774, 75.8069), "Medical College": (11.2977, 75.7803), "Mini Bypass": (11.2661, 75.7909),
    "SM Street": (11.2508, 75.7773), "Kozhikode Beach": (11.2545, 75.7710),
    "Railway Station": (11.2483, 75.7796), "KSRTC": (11.2500, 75.7850), "Arayidathupalam": (11.2650, 75.7825),
    "Kottooli": (11.2400, 75.7750), "Vellayil": (11.2650, 75.7740), "Ashokapuram": (11.2580, 75.7870),
}
LOCALITY_ALIASES = {"Mavoor Road": ["Mavoor Road", "Mavoor"]}
DISTRICT_ONLY_LOCALITIES = {"Chathamangalam": (11.2960, 75.9150), "Kodiyathur": (11.2875, 75.9875)}
NEIGHBORING_DISTRICTS = ["Wayanad", "Malappuram", "Kannur", "Idukki"]

# Fusion weights (Section 3.1 of the Technical Report: inherited assumptions, not
# fitted against outcome data — see that section before changing these).
W_NLP, W_RAIN = 0.45, 0.30
BONUS = 8 + 10 + 10 + 10  # corroboration + credibility + gdacs + location


def log(msg):
    print(f"[{datetime.now(timezone.utc).isoformat()}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 1. Classifier (Section 2)
# ---------------------------------------------------------------------------
def train_classifier():
    df = pd.read_csv(TRAINING_CSV)
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    X = vectorizer.fit_transform(df["Title"])
    y = df["Label"]
    model = SVC(kernel="linear", probability=True, class_weight="balanced", random_state=42)
    model.fit(X, y)
    log(f"Classifier trained on {len(df)} rows from {TRAINING_CSV}.")
    return vectorizer, model


# ---------------------------------------------------------------------------
# 2. Locality extraction (Section 9a)
# ---------------------------------------------------------------------------
def _find(term, text):
    return re.search(r"\b" + re.escape(term) + r"\b", text, re.IGNORECASE) is not None


def extract_localities(title):
    corp = [name for name in KOZHIKODE_LOCATIONS
            if any(_find(alias, title) for alias in LOCALITY_ALIASES.get(name, [name]))]
    district_only = [name for name in DISTRICT_ONLY_LOCALITIES if _find(name, title)]
    neighboring = [d for d in NEIGHBORING_DISTRICTS if _find(d, title)]
    city_level = (not corp) and (not district_only) and _find("Kozhikode", title)
    return corp, district_only, neighboring, city_level


# ---------------------------------------------------------------------------
# 3. Live RSS ingestion (Section 24, promoted into the scheduled pipeline)
# ---------------------------------------------------------------------------
def fetch_live_headlines(query="Kozhikode flooding"):
    encoded_query = query.replace(" ", "%20")
    rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"
    try:
        parsed = feedparser.parse(rss_url)
    except Exception as e:
        log(f"RSS fetch failed ({type(e).__name__}: {e}) — continuing with 0 live headlines.")
        return []
    entries = getattr(parsed, "entries", [])
    log(f"Fetched {len(entries)} live headlines for query '{query}'.")
    return [{"title": e.title, "published": getattr(e, "published", None)} for e in entries]


# ---------------------------------------------------------------------------
# 4. Live rainfall forecast (Section 6a)
# ---------------------------------------------------------------------------
def fetch_rainfall_forecast(lat, lon, forecast_hours=6, timeout=10):
    url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
           f"&hourly=precipitation&forecast_hours={forecast_hours}&timezone=Asia%2FKolkata")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = json.loads(resp.read())
        precip_mm = data["hourly"]["precipitation"][:forecast_hours]
        peak_mm = max(precip_mm) if precip_mm else 0.0
        return round(min(100.0, (peak_mm / 16.0) * 100.0), 1)
    except Exception as e:
        log(f"  forecast unavailable for ({lat:.4f},{lon:.4f}): {type(e).__name__}: {e} — using placeholder.")
        return None


def get_rain_score(locality, placeholder=30.0):
    lat, lon = KOZHIKODE_LOCATIONS[locality]
    live = fetch_rainfall_forecast(lat, lon)
    return live if live is not None else placeholder


# ---------------------------------------------------------------------------
# 5. Fusion engine (Section 7)
# ---------------------------------------------------------------------------
def fused_score(nlp_score, rain_score, w_nlp=W_NLP, w_rain=W_RAIN):
    return float(np.clip(w_nlp * nlp_score + w_rain * rain_score + BONUS, 0, 100))


def alert_band(score):
    if score < 25: return "GREEN"
    if score < 50: return "YELLOW"
    if score < 75: return "ORANGE"
    return "RED"


# ---------------------------------------------------------------------------
# 6. Human review queue (Section 10a) — automated queuing only. A human still
#    has to confirm/reject before cleared_for_public_alert is ever set to 1.
# ---------------------------------------------------------------------------
def init_db(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.execute("""CREATE TABLE IF NOT EXISTS alert_review_queue (
        review_id INTEGER PRIMARY KEY AUTOINCREMENT,
        locality TEXT NOT NULL, fused_score REAL NOT NULL, alert_band TEXT NOT NULL,
        nlp_score REAL, rain_score REAL, status TEXT NOT NULL DEFAULT 'PENDING_REVIEW',
        cleared_for_public_alert INTEGER NOT NULL DEFAULT 0,
        queued_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        reviewed_by TEXT, reviewed_at TIMESTAMP, reviewer_notes TEXT
    );""")
    conn.execute("""CREATE TABLE IF NOT EXISTS alert_dissemination_log (
        dissemination_id INTEGER PRIMARY KEY AUTOINCREMENT,
        review_id INTEGER NOT NULL,
        tier TEXT NOT NULL,
        channel TEXT NOT NULL,
        message_summary TEXT,
        sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (review_id) REFERENCES alert_review_queue(review_id)
    );""")
    conn.commit()
    conn.close()


def queue_for_review(locality, score, band, nlp_score, rain_score, db_path=DB_PATH):
    status = "PENDING_REVIEW" if band in ("ORANGE", "RED") else "AUTO_INFORMATIONAL"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO alert_review_queue (locality, fused_score, alert_band, nlp_score, rain_score, status) "
        "VALUES (?, ?, ?, ?, ?, ?)", (locality, score, band, nlp_score, rain_score, status)
    )
    conn.commit()
    conn.close()
    return status


def append_to_headline_log(classified, log_path=HEADLINE_LOG_PATH):
    """Appends every genuinely live-fetched headline (real text, real fetch timestamp)
    to a growing CSV — this is what actually extends real-data coverage forward through
    next month and beyond, automatically, every 15 minutes, without anyone needing to
    hand-search for news. Deduplicates on title so re-fetching the same still-trending
    headline across runs doesn't pad the log with repeats.

    Note: this log is NOT the same file as UFFIS_training_data.csv, and rows here don't
    have a Label yet - they're unlabelled real headlines, not automatically-trusted
    ground truth. Folding reviewed/labelled rows from this log into the official
    training set periodically is a human step (see Way Forward: "Larger, real-world
    dataset"), kept deliberately separate from automatic classifier retraining so the
    training data's quality doesn't silently drift.
    """
    if not classified:
        return 0

    new_rows = pd.DataFrame([{
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "title": c["title"],
        "published": c.get("published"),
        "nlp_confidence": c["nlp_confidence"],
        "corp_localities": "; ".join(c["corp_localities"]) or "",
        "district_only": "; ".join(c["district_only"]) or "",
        "neighboring_districts": "; ".join(c["neighboring_districts"]) or "",
        "city_level": c["city_level"],
    } for c in classified])

    if os.path.exists(log_path):
        existing = pd.read_csv(log_path)
        combined = pd.concat([existing, new_rows], ignore_index=True)
        before = len(combined)
        combined = combined.drop_duplicates(subset=["title"], keep="first")
        added = len(combined) - len(existing)
    else:
        combined = new_rows.drop_duplicates(subset=["title"], keep="first")
        added = len(combined)

    combined.to_csv(log_path, index=False)
    log(f"Appended to {log_path} — {added} new unique headline(s), {len(combined)} total logged since tracking began.")
    return added


def write_pending_reviews_md(db_path=DB_PATH, md_path=PENDING_MD_PATH):
    """Human-readable list of everything still awaiting a human decision - .db files
    aren't readable in GitHub's file viewer, this is. See HOW_TO_REVIEW.md for how
    to act on a row (or run the "UFFIS review" workflow with the review_id below)."""
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
    log(f"Wrote {md_path} — {len(df)} row(s) awaiting review.")


# ---------------------------------------------------------------------------
# Main pipeline run
# ---------------------------------------------------------------------------
def run():
    log("=== UFFIS pipeline run starting ===")
    init_db()
    vectorizer, model = train_classifier()

    headlines = fetch_live_headlines()
    classified = []
    for h in headlines:
        X = vectorizer.transform([h["title"]])
        proba = model.predict_proba(X)[0][1]  # P(flood-risk)
        corp, district_only, neighboring, city_level = extract_localities(h["title"])
        classified.append({**h, "nlp_confidence": round(proba * 100, 1),
                            "corp_localities": corp, "district_only": district_only,
                            "neighboring_districts": neighboring, "city_level": city_level})

    # Per-locality NLP signal = the strongest live headline confidence mentioning it today.
    locality_nlp = {name: 0.0 for name in KOZHIKODE_LOCATIONS}
    for c in classified:
        for loc in c["corp_localities"]:
            locality_nlp[loc] = max(locality_nlp[loc], c["nlp_confidence"])

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "live_headlines_fetched": len(headlines),
        "localities": {},
    }

    pending_count = 0
    for locality in KOZHIKODE_LOCATIONS:
        nlp_score = locality_nlp[locality]
        rain_score = get_rain_score(locality)
        score = fused_score(nlp_score, rain_score)
        band = alert_band(score)
        status = queue_for_review(locality, score, band, nlp_score, rain_score)
        if status == "PENDING_REVIEW":
            pending_count += 1
        summary["localities"][locality] = {
            "fused_score": round(score, 1),
            "alert_band": band,
            "nlp_score": nlp_score,
            "rain_score": rain_score,
            "review_status": status,
        }

    with open(SNAPSHOT_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    append_to_headline_log(classified)
    write_pending_reviews_md()

    log(f"Wrote {SNAPSHOT_PATH} — {pending_count} locality/localities queued for human review "
        f"(nothing is public until a reviewer confirms it in the Alerts tab).")
    log("=== UFFIS pipeline run complete ===")
    return summary


if __name__ == "__main__":
    run()
