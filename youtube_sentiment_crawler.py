"""
YouTube Comment Sentiment Analyzer
====================================
Target  : Iran-Israel-USA conflict YouTube comments
Actors  : Ali Khamenei · Vladimir Putin · Xi Jinping
Method  : VADER + TextBlob ensemble NLP

Requirements
------------
    pip install google-api-python-client vaderSentiment textblob langdetect pandas tqdm

    # TextBlob needs its corpus on first run:
    python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"

Usage
-----
    # With real YouTube API key:
    export YOUTUBE_API_KEY="AIza..."
    python youtube_sentiment_crawler.py

    # Demo mode (no API key needed):
    python youtube_sentiment_crawler.py
"""

import os
import json
import time
import pandas as pd
from datetime import datetime
from collections import defaultdict

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

API_KEY = os.getenv("YOUTUBE_API_KEY", "")

SEARCH_QUERIES = [
    "Iran Israel war attack 2024",
    "Iran USA conflict strike 2024",
    "Iran Israel missile attack response",
    "Israel Iran war retaliation",
    "Iran nuclear deal USA 2024",
    "Iran attack Israel October 2024",
    "Khamenei Iran war speech",
    "Iran proxy war Hezbollah Hamas",
]

# Keywords per actor for mention detection (lowercase)
ACTORS = {
    "Khamenei": [
        "khamenei", "ali khamenei", "supreme leader iran",
        "pemimpin iran", "ayatollah", "rahbar", "iran leader",
    ],
    "Putin": [
        "putin", "vladimir putin", "russia president", "kremlin",
        "presiden rusia", "russia iran", "putin iran",
    ],
    "Xi Jinping": [
        "xi jinping", "xi jinping", "xinping", "china president",
        "presiden china", "beijing", "china iran", "xi iran",
    ],
}

MAX_COMMENTS_PER_VIDEO = 200
MAX_VIDEOS_PER_QUERY   = 5

RAW_OUTPUT_FILE      = "iran_conflict_comments_raw.csv"
ANALYZED_OUTPUT_FILE = "analyzed_comments.csv"
RESULTS_JSON_FILE    = "sentiment_results.json"


# ─── YOUTUBE CRAWLER ──────────────────────────────────────────────────────────

def get_youtube_client():
    try:
        from googleapiclient.discovery import build
        return build("youtube", "v3", developerKey=API_KEY)
    except ImportError:
        print("❌  Missing library. Run: pip install google-api-python-client")
        return None
    except Exception as e:
        print(f"❌  Failed to build YouTube client: {e}")
        return None


def search_videos(youtube, query: str, max_results: int = 5) -> list:
    """Return a list of video metadata dicts for the given query."""
    try:
        response = youtube.search().list(
            q=query,
            part="id,snippet",
            maxResults=max_results,
            type="video",
            order="relevance",
            relevanceLanguage="en",
        ).execute()
    except Exception as e:
        print(f"  ⚠️  Search error for '{query}': {e}")
        return []

    videos = []
    for item in response.get("items", []):
        if item["id"].get("videoId"):
            videos.append({
                "video_id":    item["id"]["videoId"],
                "title":       item["snippet"]["title"],
                "channel":     item["snippet"]["channelTitle"],
                "published_at": item["snippet"]["publishedAt"],
                "query":       query,
            })
    return videos


def get_comments(youtube, video_id: str, max_comments: int = 200) -> list:
    """Fetch up to max_comments top-level comments from a video."""
    comments = []
    next_page_token = None

    try:
        while len(comments) < max_comments:
            response = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=min(100, max_comments - len(comments)),
                pageToken=next_page_token,
                textFormat="plainText",
                order="relevance",
            ).execute()

            for item in response.get("items", []):
                snippet = item["snippet"]["topLevelComment"]["snippet"]
                comments.append({
                    "video_id":    video_id,
                    "comment_id":  item["id"],
                    "text":        snippet["textDisplay"],
                    "author":      snippet["authorDisplayName"],
                    "like_count":  snippet["likeCount"],
                    "published_at": snippet["publishedAt"],
                    "reply_count": item["snippet"]["totalReplyCount"],
                })

            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break

    except Exception:
        # Comments disabled or quota exceeded — silently skip
        pass

    return comments


def crawl_youtube(queries: list = None) -> list:
    """Crawl YouTube for all queries; return deduplicated list of comment dicts."""
    if queries is None:
        queries = SEARCH_QUERIES

    youtube = get_youtube_client()
    if not youtube:
        return []

    all_comments    = []
    seen_video_ids  = set()
    seen_comment_ids = set()

    print(f"\n🔍  Starting YouTube crawl ({len(queries)} queries)…\n")

    for query in queries:
        print(f"  📌  {query}")
        videos = search_videos(youtube, query, MAX_VIDEOS_PER_QUERY)
        print(f"       {len(videos)} videos found")

        for video in videos:
            vid_id = video["video_id"]
            if vid_id in seen_video_ids:
                continue
            seen_video_ids.add(vid_id)

            print(f"       🎬  {video['title'][:70]}…")
            comments = get_comments(youtube, vid_id, MAX_COMMENTS_PER_VIDEO)

            for c in comments:
                if c["comment_id"] not in seen_comment_ids:
                    seen_comment_ids.add(c["comment_id"])
                    c["video_title"]  = video["title"]
                    c["channel"]      = video["channel"]
                    c["search_query"] = query
                    all_comments.append(c)

            print(f"            → {len(comments)} comments (total so far: {len(all_comments)})")
            time.sleep(0.5)

        time.sleep(1)

    print(f"\n✅  Total unique comments collected: {len(all_comments)}\n")
    return all_comments


# ─── NLP HELPERS ──────────────────────────────────────────────────────────────

# Instantiate VADER once (not inside a loop)
_vader_analyzer = None

def _get_vader():
    global _vader_analyzer
    if _vader_analyzer is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _vader_analyzer = SentimentIntensityAnalyzer()
    return _vader_analyzer


def detect_actor_mentions(text: str) -> dict:
    """Return {actor_name: bool} for each actor in ACTORS."""
    text_lower = text.lower()
    return {
        actor: any(kw in text_lower for kw in keywords)
        for actor, keywords in ACTORS.items()
    }


def analyze_sentiment_vader(text: str) -> dict:
    """VADER sentiment — optimised for social media text."""
    try:
        analyzer = _get_vader()
        scores   = analyzer.polarity_scores(text)
        compound = scores["compound"]
        label    = "positive" if compound >= 0.05 else ("negative" if compound <= -0.05 else "neutral")
        return {
            "vader_compound": compound,
            "vader_label":    label,
            "vader_pos":      scores["pos"],
            "vader_neg":      scores["neg"],
            "vader_neu":      scores["neu"],
        }
    except Exception:
        return {"vader_compound": 0.0, "vader_label": "neutral",
                "vader_pos": 0.0, "vader_neg": 0.0, "vader_neu": 1.0}


def analyze_sentiment_textblob(text: str) -> dict:
    """TextBlob sentiment — rule-based polarity & subjectivity."""
    try:
        from textblob import TextBlob
        blob        = TextBlob(text)
        polarity    = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        label       = "positive" if polarity > 0.05 else ("negative" if polarity < -0.05 else "neutral")
        return {
            "tb_polarity":     polarity,
            "tb_subjectivity": subjectivity,
            "tb_label":        label,
        }
    except Exception:
        return {"tb_polarity": 0.0, "tb_subjectivity": 0.0, "tb_label": "neutral"}


def detect_language(text: str) -> str:
    """Detect the language of the comment text."""
    try:
        from langdetect import detect
        return detect(text)
    except Exception:
        return "unknown"


def classify_tone(text: str) -> list:
    """Tag comment with one or more tone labels based on keyword matching."""
    t = text.lower()
    tones = []
    # Deduplicated word lists
    if any(w in t for w in ["war", "bomb", "kill", "destroy", "attack", "fight", "death", "dead"]):
        tones.append("aggressive")
    if any(w in t for w in ["support", "agree", "brave", "strong", "hero", "great", "good"]):
        tones.append("supportive")
    if any(w in t for w in ["wrong", "bad", "evil", "hate", "stupid", "liar", "criminal", "dictator", "tyrant"]):
        tones.append("opposing")
    if any(w in t for w in ["peace", "diplomacy", "negotiate", "ceasefire", "dialogue", "stop war"]):
        tones.append("peace-seeking")
    return tones if tones else ["neutral"]


# ─── ANALYSIS PIPELINE ────────────────────────────────────────────────────────

def analyze_comments(comments: list) -> tuple:
    """
    Run the full NLP pipeline.
    Returns (results_list, actor_sentiments_dict).
    """
    print(f"🧠  Running NLP on {len(comments)} comments…\n")

    results          = []
    actor_sentiments = defaultdict(
        lambda: {"positive": 0, "negative": 0, "neutral": 0, "total": 0, "scores": []}
    )

    for i, comment in enumerate(comments):
        if i % 200 == 0:
            print(f"    {i}/{len(comments)} processed…")

        text = comment.get("text", "").strip()
        if len(text) < 5:
            continue

        vader    = analyze_sentiment_vader(text)
        tb       = analyze_sentiment_textblob(text)
        mentions = detect_actor_mentions(text)
        tones    = classify_tone(text)
        lang     = detect_language(text)

        # Ensemble: average VADER compound + TextBlob polarity
        ensemble_score = round(
            (vader["vader_compound"] + tb["tb_polarity"]) / 2, 4
        )
        ensemble_label = (
            "positive" if ensemble_score >= 0.05
            else "negative" if ensemble_score <= -0.05
            else "neutral"
        )

        row = {
            **comment,
            **vader,
            **tb,
            "language":          lang,
            "tones":             ", ".join(tones),
            "ensemble_score":    ensemble_score,
            "ensemble_label":    ensemble_label,
            "mentions_khamenei": mentions.get("Khamenei", False),
            "mentions_putin":    mentions.get("Putin", False),
            "mentions_xi":       mentions.get("Xi Jinping", False),
        }
        results.append(row)

        for actor, mentioned in mentions.items():
            if mentioned:
                actor_sentiments[actor][ensemble_label] += 1
                actor_sentiments[actor]["total"]        += 1
                actor_sentiments[actor]["scores"].append(ensemble_score)

    print(f"  ✅  Done — {len(results)} comments analysed.\n")
    return results, dict(actor_sentiments)


def generate_report(results: list, actor_sentiments: dict) -> tuple:
    """Build summary report dict and analysed DataFrame."""
    df = pd.DataFrame(results)

    overall = {
        "positive": int((df["ensemble_label"] == "positive").sum()),
        "negative": int((df["ensemble_label"] == "negative").sum()),
        "neutral":  int((df["ensemble_label"] == "neutral").sum()),
    }

    report = {
        "generated_at":       datetime.now().isoformat(),
        "total_comments":     len(results),
        "overall_sentiment":  overall,
        "actors":             {},
        "top_comments":       {},
        "language_distribution": df["language"].value_counts().head(10).to_dict(),
    }

    # Per-actor stats
    for actor, stats in actor_sentiments.items():
        scores = stats.pop("scores", [])
        avg    = round(sum(scores) / len(scores), 4) if scores else 0.0
        total  = max(stats["total"], 1)
        report["actors"][actor] = {
            **stats,
            "avg_sentiment_score": avg,
            "sentiment_ratio": {
                "positive_pct": round(stats["positive"] / total * 100, 1),
                "negative_pct": round(stats["negative"] / total * 100, 1),
                "neutral_pct":  round(stats["neutral"]  / total * 100, 1),
            },
        }

    # Top / bottom 3 comments per actor
    col_map = {
        "Khamenei":  "mentions_khamenei",
        "Putin":     "mentions_putin",
        "Xi Jinping": "mentions_xi",
    }
    for actor, col in col_map.items():
        if col not in df.columns:
            continue
        adf = df[df[col] == True]
        if adf.empty:
            continue
        report["top_comments"][actor] = {
            "most_positive": adf.nlargest(3, "ensemble_score")[
                ["text", "ensemble_score", "like_count"]].to_dict("records"),
            "most_negative": adf.nsmallest(3, "ensemble_score")[
                ["text", "ensemble_score", "like_count"]].to_dict("records"),
        }

    return report, df


# ─── DEMO DATA (no API key required) ──────────────────────────────────────────

def generate_demo_data(n: int = 500) -> list:
    """Generate synthetic comments for dashboard testing."""
    import random

    templates = [
        # (text, base_score, primary_actor)
        ("Khamenei is leading Iran with strength against Western imperialism!", 0.70, "khamenei"),
        ("Khamenei's rhetoric only brings suffering to ordinary Iranians.", -0.62, "khamenei"),
        ("The Supreme Leader Khamenei is playing a very dangerous game.", -0.33, "khamenei"),
        ("Khamenei proved Iran won't bow to USA pressure. Respect!", 0.82, "khamenei"),
        ("Khamenei's policies are destabilizing the entire Middle East.", -0.71, "khamenei"),
        ("Ayatollah Khamenei has maintained Iran's sovereignty for decades.", 0.30, "khamenei"),
        ("Iran under Khamenei is becoming more isolated every day.", -0.40, "khamenei"),
        ("Khamenei is a religious leader defending Muslim causes globally.", 0.50, "khamenei"),
        ("Putin and Iran are allies against NATO aggression.", 0.40, "putin"),
        ("Putin is supplying drones to Iran, making him complicit in attacks.", -0.83, "putin"),
        ("Russia under Putin will support Iran no matter what.", 0.20, "putin"),
        ("Putin is using Iran–Israel conflict to distract from Ukraine.", -0.55, "putin"),
        ("Putin's alliance with Iran is purely strategic, not ideological.", -0.18, "putin"),
        ("Russia and Putin stand with Iran against Western hegemony.", 0.61, "putin"),
        ("Putin should stay out of Middle East conflicts entirely.", -0.32, "putin"),
        ("Putin is playing geopolitical chess while others play checkers.", 0.45, "putin"),
        ("Xi Jinping is trying to mediate peace in the Middle East.", 0.54, "xi"),
        ("China under Xi is buying Iranian oil, indirectly funding this war.", -0.63, "xi"),
        ("Xi's Belt and Road helps Iran bypass US sanctions.", 0.22, "xi"),
        ("Xi Jinping met Iranian leaders to strengthen bilateral ties.", 0.10, "xi"),
        ("China and Xi are enabling Iran's aggression by buying their oil.", -0.70, "xi"),
        ("Xi is smart to stay neutral while others bleed.", 0.38, "xi"),
        ("Xi Jinping's China blocked UN action on Iran — disgraceful.", -0.48, "xi"),
        ("Under Xi, China–Iran relations have deepened significantly.", 0.05, "xi"),
        ("This war will only bring more suffering to civilians.", -0.81, "none"),
        ("Iran has every right to defend itself against Israeli strikes.", 0.63, "none"),
        ("The USA must stop arming Israel and push for a ceasefire.", -0.30, "none"),
        ("We need diplomacy, not bombs! Stop the escalation now!", 0.22, "none"),
        ("The axis of resistance grows stronger every day.", 0.48, "khamenei"),
        ("All three — Khamenei, Putin, Xi — are authoritarians blocking peace.", -0.77, "multiple"),
    ]

    results = []
    for i in range(n):
        text, base_score, actor_tag = random.choice(templates)
        noise = random.uniform(-0.15, 0.15)
        score = round(max(-1.0, min(1.0, base_score + noise)), 4)

        results.append({
            "video_id":    f"demo_video_{random.randint(1, 20)}",
            "comment_id":  f"demo_{i}",
            "text":        text,
            "author":      f"User_{random.randint(1000, 9999)}",
            "like_count":  random.randint(0, 500),
            "published_at": f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}T12:00:00Z",
            "reply_count": random.randint(0, 50),
            "video_title": f"Iran Israel conflict analysis vol.{random.randint(1, 10)}",
            "channel":     random.choice(["Al Jazeera", "BBC News", "CNN", "DW News", "Independent"]),
            "search_query": random.choice(SEARCH_QUERIES[:4]),
        })

    return results


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  🌍  Iran–Israel Conflict · YouTube Sentiment Analyzer")
    print("  Actors: Khamenei | Putin | Xi Jinping")
    print("=" * 60)

    # Step 1: Collect comments
    if not API_KEY:
        print("\n⚠️   No YOUTUBE_API_KEY found → running in DEMO MODE.\n"
              "    Set the env var to enable real crawling.\n")
        raw_comments = generate_demo_data(500)
    else:
        raw_comments = crawl_youtube()
        if not raw_comments:
            print("⚠️   Crawling returned no comments → falling back to demo data.\n")
            raw_comments = generate_demo_data(500)

    # Step 2: Save raw
    pd.DataFrame(raw_comments).to_csv(RAW_OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"💾  Raw comments → {RAW_OUTPUT_FILE}")

    # Step 3: NLP
    results, actor_sentiments = analyze_comments(raw_comments)

    # Step 4: Report
    report, df_analyzed = generate_report(results, actor_sentiments)
    df_analyzed.to_csv(ANALYZED_OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"💾  Analysed CSV → {ANALYZED_OUTPUT_FILE}")

    with open(RESULTS_JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"💾  JSON report  → {RESULTS_JSON_FILE}")

    # Step 5: Print summary
    print("\n" + "=" * 60)
    print("  📊  RESULTS SUMMARY")
    print("=" * 60)
    print(f"\n  Total comments: {report['total_comments']}")
    print("  Overall sentiment:")
    total = report["total_comments"]
    for label, count in report["overall_sentiment"].items():
        print(f"    {label:10s}: {count:5d}  ({count/max(total,1)*100:.1f}%)")

    print("\n  Actor breakdown:")
    for actor, data in report["actors"].items():
        r = data["sentiment_ratio"]
        print(f"\n  🎯  {actor}")
        print(f"      Mentions : {data['total']}")
        print(f"      Avg score: {data['avg_sentiment_score']:+.4f}")
        print(f"      Positive : {r['positive_pct']}%")
        print(f"      Negative : {r['negative_pct']}%")
        print(f"      Neutral  : {r['neutral_pct']}%")

    print(f"\n✅  Done!  Load {RESULTS_JSON_FILE} into the dashboard.\n")
    return report


if __name__ == "__main__":
    main()
