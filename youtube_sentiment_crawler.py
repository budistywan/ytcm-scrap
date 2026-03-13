"""
YouTube Comment Sentiment Analyzer — v2
=========================================
Studi kasus: Serangan Israel-USA ke Iran (Feb-Mar 2026)
Target audiens: Komentar Indonesia di YouTube

DUA MODUL ANALISIS:
  A) SERANGAN — Sentimen publik Indonesia terhadap pelaku & aktor
     Fokus: USA, Israel, Iran, Khamenei, Syiah, China, Rusia
     Tujuan: Apakah komentar Indonesia bisa 'jernih' melihat (hubungan diplomatik,
             sesama Islam), atau masih terbawa dikotomi Khamenei/Syiah?
             (konteks: Indonesia mayoritas Muslim Sunni, non-Syiah)

  B) EKONOMI — Penutupan Selat Hormuz & dampak minyak (sekitar 10 Mar 2026)
     Fokus: USA, Israel, Iran, China, Rusia, harga minyak
     Tujuan: Apakah diskusi ekonomi lebih substantif atau sekadar penonton konflik?

Requirements
------------
    pip install google-api-python-client vaderSentiment textblob langdetect pandas tqdm

Usage
-----
    export YOUTUBE_API_KEY="AIza..."
    python youtube_sentiment_crawler_v2.py
"""

import os, json, time, re
import pandas as pd
from datetime import datetime
from collections import defaultdict

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

API_KEY = os.getenv("YOUTUBE_API_KEY", "")

# ─── KANAL INDONESIA TARGET (Opsi B) ─────────────────────────────────────────
# Channel ID didapat dari URL kanal YouTube masing-masing.
# Format: youtube.com/channel/CHANNEL_ID atau youtube.com/@handle
# Untuk verifikasi: buka kanal → About → Share → Copy channel ID
#
# TIER 1 — Kanal berita besar, audience luas, banyak komentar konflik luar negeri
# 8 kanal dipilih berdasarkan: audience luas, aktif, banyak komentar berita luar negeri
# Quota estimasi: 8 kanal × 3 query × 2 modul × 100 unit = 4.800 unit search
#                 + ~270 video × 2 halaman × 1 unit = ~540 unit comments
#                 Total: ~5.340 unit (aman di bawah 10.000/hari)
# Channel ID diverifikasi dari youtube.com/@handle → About → Share → Copy channel ID
INDONESIAN_CHANNELS = {
    "Kompas TV":     "UCTat5Lfqv8CvvzXMTQTcMpQ",  # @KompasTV
    "CNN Indonesia": "UCuATgZgGbHkRRCOlP2HFsEg",  # @CNNIndonesia
    "tvOne":         "UCkRTiF_tEz6RqDiHBGBdKzQ",  # @tvOneNews
    "Metro TV":      "UCls31GKZS6VWZMH2BqsMTiw",  # @Metro_TV
    "detikcom":      "UCHFBZxMhfMVBvjFV8R4Zomg",  # @detikcom
    "Narasi":        "UCp9XuBFb_7bEFlCeVA0DKUQ",  # @narasi
    "Tempo.co":      "UCQI4bu7-kDHtBhXlMJJleVA",  # @tempodotco
    "Liputan6":      "UCVNaosgIaBWd5rAv5cNYl0w",  # @liputan6
}

# 3 query per kanal — cukup untuk menangkap video relevan tanpa buang quota
QUERIES_PER_CHANNEL_SERANGAN = [
    "iran israel serangan",
    "khamenei",
    "perang iran israel 2026",
]

QUERIES_PER_CHANNEL_EKONOMI = [
    "selat hormuz minyak",
    "harga minyak naik iran",
    "dampak ekonomi iran indonesia",
]

MAX_VIDEOS_PER_CHANNEL_QUERY = 3   # 3 video per query per kanal
MAX_COMMENTS_PER_VIDEO       = 200  # komentar per video

# ── MODUL A: SERANGAN (Feb 28, 2026 — Khamenei tewas) ─────────────────────────
# Semua query BAHASA INDONESIA — target: komentar publik Indonesia
# relevanceLanguage="id" dipaksa untuk semua query
QUERIES_SERANGAN = [
    "serangan israel iran 2026",
    "israel serang iran khamenei mati 2026",
    "iran diserang israel amerika 2026",
    "khamenei meninggal serangan israel",
    "reaksi indonesia serangan iran israel",
    "pendapat orang indonesia soal perang iran israel",
    "iran israel perang maret 2026",
    "khamenei tewas israel serang iran",
    "indonesia tanggapi serangan iran israel",
    "perang timur tengah iran israel 2026",
]

# ── MODUL B: EKONOMI (Mar 10, 2026 — Penutupan Selat Hormuz) ──────────────────
# Semua query BAHASA INDONESIA — target: komentar publik Indonesia
# relevanceLanguage="id" dipaksa untuk semua query
QUERIES_EKONOMI = [
    "iran tutup selat hormuz harga minyak",
    "selat hormuz ditutup iran dampak indonesia",
    "harga minyak naik iran israel 2026",
    "dampak perang iran minyak indonesia",
    "krisis minyak selat hormuz 2026",
    "bbm naik gara gara iran hormuz",
    "pertamina minyak iran selat hormuz",
    "dampak ekonomi indonesia perang iran",
    "harga bensin naik perang timur tengah",
    "indonesia impor minyak iran hormuz ditutup",
]

# ─── AKTOR & ENTITAS yang dideteksi ───────────────────────────────────────────

# Modul A: Aktor geopolitik + dimensi agama/sektarian
ENTITIES_SERANGAN = {
    # Pelaku serangan
    "Israel": [
        "israel", "idf", "zionis", "zionist", "tel aviv", "netanyahu",
        "israelis", "pasukan israel",
    ],
    "USA": [
        "usa", "america", "amerika", "united states", "washington",
        "pentagon", "biden", "trump", "us military", "militer as",
    ],
    # Korban / pihak diserang
    "Iran": [
        "iran", "iranian", "persia", "teheran", "tehran", "irgc",
        "republik islam", "islamic republic",
    ],
    "Khamenei": [
        "khamenei", "ali khamenei", "supreme leader", "pemimpin tertinggi",
        "ayatollah", "rahbar", "mursyid", "wali faqih",
    ],
    # Dimensi sektarian — INI KUNCI ANALISIS
    "Syiah": [
        "syiah", "shia", "syi'ah", "syiah iran", "shia islam",
        "sunni syiah", "sunni shia", "sektarian", "hezbollah", "hizbullah",
        "houthi", "houthi", "milisi syiah", "shia militia",
    ],
    # Pendukung Iran
    "China": [
        "china", "cina", "tiongkok", "beijing", "xi jinping",
        "chinese", "prc", "rrc",
    ],
    "Rusia": [
        "russia", "rusia", "russion", "moscow", "moskow", "putin",
        "kremlin", "vladimir",
    ],
}

# Modul B: Entitas ekonomi + geopolitik
ENTITIES_EKONOMI = {
    "USA": [
        "usa", "america", "amerika", "united states", "washington",
        "us sanctions", "sanksi as", "trump", "biden",
    ],
    "Israel": [
        "israel", "idf", "zionis", "zionist", "netanyahu", "tel aviv",
    ],
    "Iran": [
        "iran", "iranian", "persia", "teheran", "irgc", "republik islam",
        "hormuz", "selat hormuz", "strait of hormuz",
    ],
    "China": [
        "china", "cina", "tiongkok", "beijing", "xi jinping",
        "chinese oil", "minyak china", "pembeli minyak iran",
    ],
    "Rusia": [
        "russia", "rusia", "moscow", "putin", "kremlin",
        "russian oil", "minyak rusia", "opec",
    ],
    # Dimensi ekonomi — KUNCI ANALISIS MODUL B
    "Minyak_Energi": [
        "minyak", "oil", "bbm", "bensin", "solar", "pertamina",
        "harga minyak", "oil price", "crude oil", "brent",
        "energi", "energy", "bahan bakar", "fuel",
        "opec", "cadangan minyak", "oil reserve",
    ],
    "Dampak_Indonesia": [
        "indonesia", "rupiah", "ihsg", "inflasi", "inflation",
        "subsidi", "subsidy", "anggaran", "budget", "apbn",
        "daya beli", "purchasing power", "ekonomi indonesia",
        "impor minyak", "import minyak",
    ],
}

# ─── KAMUS TONE khusus Indonesia ──────────────────────────────────────────────
#
# KERANGKA ANALISIS MODUL A — 3 KUTUB JERNIH vs DIKOTOMI:
#
#   JERNIH (1): Perspektif diplomatik — melihat Iran sebagai negara berdaulat,
#               terlepas dari sekte. Bicara hubungan bilateral, hukum internasional.
#
#   JERNIH (2): Perspektif sesama Islam — solidaritas kemanusiaan yang TIDAK
#               membedakan Syiah/Sunni, bicara penderitaan warga sipil.
#
#   DIKOTOMI: Respons terbawa frame Syiah vs Sunni — Iran = Syiah = bukan saudara
#             sepenuhnya, atau sebaliknya membela Iran KARENA sesama Muslim tanpa
#             melihat konteks diplomatik/hukum.
#
TONE_PATTERNS = {

    # ── TANDA JERNIH ─────────────────────────────────────────────────────────

    # Jernih-1: Perspektif diplomatik / hubungan antarnegara
    "jernih_diplomatik": [
        "hubungan diplomatik", "kedaulatan", "sovereignty", "hukum internasional",
        "international law", "pembunuhan pemimpin negara", "state assassination",
        "pelanggaran hukum", "violation", "pbb", "un", "dewan keamanan",
        "kepentingan nasional", "national interest", "hubungan bilateral",
        "kebijakan luar negeri", "foreign policy", "non blok", "non-aligned",
        "geopolitik", "geopolitics", "konsekuensi diplomatik",
        "pengakuan kedaulatan", "invasi", "agresi",
    ],

    # Jernih-2: Perspektif sesama Muslim tanpa frame sektarian
    # (solidaritas kemanusiaan, tidak mempermasalahkan Syiah-nya)
    "jernih_kemanusiaan": [
        "warga sipil", "civilian", "korban sipil", "korban tak berdosa",
        "anak-anak", "perempuan", "kemanusiaan", "humanitarian",
        "hak asasi", "ham", "human rights", "penderitaan",
        "sesama manusia", "sesama muslim", "saudara seiman",
        "tanpa memandang", "tak peduli syiah atau sunni",
        "bukan soal syiah", "bukan urusan sekte",
        "muslim manapun", "semua muslim",
    ],

    # ── TANDA DIKOTOMI SEKTARIAN ─────────────────────────────────────────────

    # Dikotomi aktif: mempermasalahkan ke-Syiah-an Iran secara eksplisit
    "dikotomi_syiah": [
        "syiah sesat", "syiah kafir", "syiah bukan islam", "syiah bukan muslim",
        "mereka syiah", "iran syiah", "khamenei syiah", "bela syiah",
        "ngapain bela syiah", "syiah dapat karma", "syiah memang begitu",
        "tidak perlu bela syiah", "syiah vs sunni", "sunni jangan bela syiah",
        "syiah musuh sunni", "mereka bukan saudara kita",
        "bukan saudara seiman", "beda akidah",
    ],

    # Dikotomi pasif: membela Iran KARENA label Islam/Muslim, tanpa analisis
    # (solidaritas sektarian terselubung — Iran = Muslim = harus dibela)
    "solidaritas_buta": [
        "bela iran karena muslim", "iran saudara muslim", "sesama muslim harus bela",
        "karena iran islam", "iran negara islam", "republik islam harus kita dukung",
        "allahu akbar iran", "takbir untuk iran", "doa untuk iran menang",
        "iran pasti menang karena allah", "iran dilindungi allah",
        "iran pejuang islam", "iran benteng islam",
        "hidup iran", "viva iran", "iran kuat", "semangat iran",
    ],

    # ── TANDA LAIN ───────────────────────────────────────────────────────────

    # Analitis — menggunakan data/fakta/sejarah (lintas Syiah-Sunni)
    "analitis": [
        "analisis", "sebenarnya", "faktanya", "data menunjukkan",
        "secara historis", "sejarahnya", "track record", "menurut saya",
        "perlu dipahami", "konteksnya adalah", "latar belakang",
        "strategi", "skenario", "implikasi", "konsekuensi",
        "jika dilihat dari", "dari sudut pandang",
    ],

    # Reaktif emosional — kemarahan tanpa frame sektarian maupun analisis
    "reaktif_emosional": [
        "bangsat", "anjing", "laknat", "kutuk", "keparat", "biadab",
        "pembunuh", "killer", "murderer", "genocide", "genosida",
        "monster", "evil", "jahat", "brutal", "barbar",
    ],

    # Nuansa ekonomi lokal (relevan juga di Modul A, lebih kuat di B)
    "nuansa_ekonomi_lokal": [
        "bbm naik", "bensin naik", "harga naik", "mahal", "susah",
        "rakyat kecil", "masyarakat bawah", "kita yang rugi",
        "indonesia rugi", "dampaknya ke kita", "dampak ke indonesia",
    ],
}

MAX_COMMENTS_PER_VIDEO = 300   # Lebih banyak per video
MAX_VIDEOS_PER_QUERY   = 5

# Filter bahasa — PRIORITAS INDONESIA
TARGET_LANGUAGES = ["id", "en"]  # id = Indonesia, en = English

RAW_OUTPUT_SERANGAN   = "raw_serangan.csv"
RAW_OUTPUT_EKONOMI    = "raw_ekonomi.csv"
ANALYZED_SERANGAN     = "analyzed_serangan.csv"
ANALYZED_EKONOMI      = "analyzed_ekonomi.csv"
RESULTS_SERANGAN      = "results_serangan.json"
RESULTS_EKONOMI       = "results_ekonomi.json"
COMBINED_RESULTS      = "sentiment_results.json"   # untuk dashboard


# ─── YOUTUBE CRAWLER ──────────────────────────────────────────────────────────

def get_youtube_client():
    try:
        from googleapiclient.discovery import build
        return build("youtube", "v3", developerKey=API_KEY)
    except ImportError:
        print("❌  pip install google-api-python-client")
        return None
    except Exception as e:
        print(f"❌  YouTube client error: {e}")
        return None


def search_videos_in_channel(youtube, channel_id: str, channel_name: str,
                              query: str, max_results: int = 3) -> list:
    """Cari video di dalam satu kanal Indonesia spesifik."""
    try:
        resp = youtube.search().list(
            q=query,
            part="id,snippet",
            channelId=channel_id,
            maxResults=max_results,
            type="video",
            order="date",          # terbaru dulu — relevan untuk kejadian 2026
        ).execute()
    except Exception as e:
        print(f"    ⚠️  Search error [{channel_name}] '{query}': {e}")
        return []

    return [
        {
            "video_id":    item["id"]["videoId"],
            "title":       item["snippet"]["title"],
            "channel":     item["snippet"]["channelTitle"],
            "channel_id":  channel_id,
            "published_at": item["snippet"]["publishedAt"],
            "query":       query,
        }
        for item in resp.get("items", [])
        if item["id"].get("videoId")
    ]


def search_videos(youtube, query: str, max_results: int = 5,
                  relevance_language: str = "id") -> list:
    """Fallback: search global (tidak dipakai di Opsi B)."""
    try:
        resp = youtube.search().list(
            q=query, part="id,snippet", maxResults=max_results,
            type="video", order="relevance", relevanceLanguage=relevance_language,
        ).execute()
    except Exception as e:
        print(f"  ⚠️  Search error '{query}': {e}")
        return []
    return [
        {"video_id": item["id"]["videoId"], "title": item["snippet"]["title"],
         "channel": item["snippet"]["channelTitle"],
         "published_at": item["snippet"]["publishedAt"], "query": query}
        for item in resp.get("items", []) if item["id"].get("videoId")
    ]


def get_comments(youtube, video_id: str, max_comments: int = 300) -> list:
    """Ambil komentar top-level. Prioritaskan relevance (banyak likes)."""
    comments, next_token = [], None
    try:
        while len(comments) < max_comments:
            resp = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=min(100, max_comments - len(comments)),
                pageToken=next_token,
                textFormat="plainText",
                order="relevance",
            ).execute()

            for item in resp.get("items", []):
                sn = item["snippet"]["topLevelComment"]["snippet"]
                comments.append({
                    "video_id":    video_id,
                    "comment_id":  item["id"],
                    "text":        sn["textDisplay"],
                    "author":      sn["authorDisplayName"],
                    "like_count":  sn["likeCount"],
                    "published_at": sn["publishedAt"],
                    "reply_count": item["snippet"]["totalReplyCount"],
                })

            next_token = resp.get("nextPageToken")
            if not next_token:
                break
    except Exception:
        pass
    return comments


def crawl(channel_queries: list, label: str) -> list:
    """
    Opsi B: Crawl dari kanal Indonesia spesifik.
    channel_queries = QUERIES_PER_CHANNEL_SERANGAN atau QUERIES_PER_CHANNEL_EKONOMI
    Setiap query dijalankan di setiap kanal dalam INDONESIAN_CHANNELS.
    """
    youtube = get_youtube_client()
    if not youtube:
        return []

    all_comments, seen_vids, seen_ids = [], set(), set()
    print(f"\n🔍  [{label}] Crawl {len(INDONESIAN_CHANNELS)} kanal × {len(channel_queries)} query…\n")

    for ch_name, ch_id in INDONESIAN_CHANNELS.items():
        print(f"  📺  {ch_name}")
        ch_videos = []

        for query in channel_queries:
            videos = search_videos_in_channel(
                youtube, ch_id, ch_name, query, MAX_VIDEOS_PER_CHANNEL_QUERY
            )
            for v in videos:
                if v["video_id"] not in seen_vids:
                    seen_vids.add(v["video_id"])
                    ch_videos.append(v)
            time.sleep(0.3)

        print(f"       {len(ch_videos)} video unik ditemukan")

        for v in ch_videos:
            print(f"       🎬  {v['title'][:65]}")
            comments = get_comments(youtube, v["video_id"], MAX_COMMENTS_PER_VIDEO)
            for c in comments:
                if c["comment_id"] not in seen_ids:
                    seen_ids.add(c["comment_id"])
                    c["video_title"]  = v["title"]
                    c["channel"]      = ch_name
                    c["channel_id"]   = ch_id
                    c["search_query"] = v["query"]
                    c["query_lang"]   = "id"
                    all_comments.append(c)
            print(f"            → {len(comments)} komentar (total: {len(all_comments)})")
            time.sleep(0.5)

        time.sleep(1)

    print(f"\n✅  [{label}] {len(all_comments)} komentar unik terkumpul\n")
    return all_comments


# ─── NLP HELPERS ──────────────────────────────────────────────────────────────

_vader = None
def get_vader():
    global _vader
    if _vader is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _vader = SentimentIntensityAnalyzer()
    return _vader


def detect_language(text: str) -> str:
    try:
        from langdetect import detect
        return detect(text)
    except Exception:
        return "unknown"


def vader_sentiment(text: str) -> dict:
    try:
        sc = get_vader().polarity_scores(text)
        c  = sc["compound"]
        return {"vader_compound": c, "vader_pos": sc["pos"],
                "vader_neg": sc["neg"], "vader_neu": sc["neu"],
                "vader_label": "positive" if c>=0.05 else "negative" if c<=-0.05 else "neutral"}
    except Exception:
        return {"vader_compound":0,"vader_pos":0,"vader_neg":0,"vader_neu":1,"vader_label":"neutral"}


def textblob_sentiment(text: str) -> dict:
    try:
        from textblob import TextBlob
        b = TextBlob(text)
        p, s = b.sentiment.polarity, b.sentiment.subjectivity
        return {"tb_polarity": p, "tb_subjectivity": s,
                "tb_label": "positive" if p>0.05 else "negative" if p<-0.05 else "neutral"}
    except Exception:
        return {"tb_polarity":0,"tb_subjectivity":0,"tb_label":"neutral"}


def detect_entities(text: str, entity_dict: dict) -> dict:
    """Deteksi mention entitas. Return {entity: True/False}."""
    t = text.lower()
    return {ent: any(kw in t for kw in kws) for ent, kws in entity_dict.items()}


def detect_tones(text: str) -> dict:
    """Deteksi tone komentar Indonesia. Return {tone: True/False}."""
    t = text.lower()
    return {tone: any(kw in t for kw in kws) for tone, kws in TONE_PATTERNS.items()}


def classify_comment_type(tones: dict, lang: str) -> str:
    """
    Klasifikasikan komentar ke dalam salah satu tipe.

    HIERARKI KLASIFIKASI MODUL A:
    ─────────────────────────────────────────────────────────────
    JERNIH:
      jernih_diplomatik  → melihat Iran sebagai negara berdaulat,
                           bicara hukum internasional / geopolitik
      jernih_kemanusiaan → solidaritas tanpa frame sektarian,
                           tidak mempermasalahkan Syiah/Sunni

    DIKOTOMI:
      dikotomi_syiah     → eksplisit mempermasalahkan ke-Syiah-an Iran
      solidaritas_buta   → membela Iran KARENA label Islam/Muslim,
                           tanpa analisis substansial

    LAIN:
      analitis           → berbasis data/fakta tapi tidak termasuk
                           kerangka jernih di atas
      reaktif            → kemarahan emosional murni
      ekonomi_lokal      → menyentuh dampak ke Indonesia
      netral             → tidak terklasifikasi
    ─────────────────────────────────────────────────────────────
    """
    # Jernih: prioritas tertinggi — meski ada elemen lain
    if tones.get("jernih_diplomatik"):
        return "jernih_diplomatik"
    if tones.get("jernih_kemanusiaan") and not tones.get("dikotomi_syiah"):
        return "jernih_kemanusiaan"

    # Dikotomi sektarian
    if tones.get("dikotomi_syiah"):
        return "dikotomi_syiah"
    if tones.get("solidaritas_buta") and not tones.get("jernih_kemanusiaan"):
        return "solidaritas_buta"

    # Analitis umum
    if tones.get("analitis"):
        return "analitis"

    # Lain-lain
    if tones.get("reaktif_emosional"):
        return "reaktif"
    if tones.get("nuansa_ekonomi_lokal"):
        return "ekonomi_lokal"
    return "netral"


# ─── PIPELINE UTAMA ───────────────────────────────────────────────────────────

def analyze(comments: list, entity_dict: dict, label: str) -> tuple:
    print(f"🧠  [{label}] Analisis NLP {len(comments)} komentar…")

    results = []
    entity_stats = defaultdict(lambda: {
        "positive":0,"negative":0,"neutral":0,"total":0,
        "scores":[],"id_comments":0
    })

    comment_types = defaultdict(int)
    id_comment_types = defaultdict(int)

    for i, c in enumerate(comments):
        if i % 300 == 0:
            print(f"    {i}/{len(comments)}…")

        text = c.get("text","").strip()
        if len(text) < 5:
            continue

        vader  = vader_sentiment(text)
        tb     = textblob_sentiment(text)
        lang   = detect_language(text)
        ents   = detect_entities(text, entity_dict)
        tones  = detect_tones(text)
        ctype  = classify_comment_type(tones, lang)

        ens_score = round((vader["vader_compound"] + tb["tb_polarity"]) / 2, 4)
        ens_label = "positive" if ens_score>=0.05 else "negative" if ens_score<=-0.05 else "neutral"

        is_id = lang == "id"

        row = {
            **c,
            **vader, **tb,
            "language": lang,
            "is_indonesian": is_id,
            "ensemble_score": ens_score,
            "ensemble_label": ens_label,
            "comment_type": ctype,
            **{f"tone_{k}": v for k,v in tones.items()},
            **{f"ent_{k}": v for k,v in ents.items()},
        }
        results.append(row)

        comment_types[ctype] += 1
        if is_id:
            id_comment_types[ctype] += 1

        for ent, mentioned in ents.items():
            if mentioned:
                entity_stats[ent][ens_label] += 1
                entity_stats[ent]["total"]   += 1
                entity_stats[ent]["scores"].append(ens_score)
                if is_id:
                    entity_stats[ent]["id_comments"] += 1

    print(f"  ✅  Selesai — {len(results)} komentar dianalisis\n")
    return results, dict(entity_stats), dict(comment_types), dict(id_comment_types)


def build_report(results, entity_stats, comment_types, id_comment_types,
                 module_label, module_desc, queries_used):
    if not results:
        return {
            "module": module_label, "description": module_desc,
            "generated_at": datetime.now().isoformat(),
            "queries_used": queries_used,
            "total_comments": 0, "indonesian_comments": 0,
            "english_comments": 0,
            "overall_sentiment": {"positive":0,"negative":0,"neutral":0},
            "indonesian_sentiment": {"positive":0,"negative":0,"neutral":0},
            "comment_type_distribution": {},
            "indonesian_comment_type_distribution": {},
            "entities": {}, "language_distribution": {},
            "top_comments": {}, "top_indonesian_comments": {},
            "error": "No comments collected — check channel IDs and queries.",
        }

    df = pd.DataFrame(results)

    id_df = df[df["is_indonesian"] == True] if "is_indonesian" in df.columns else pd.DataFrame()
    en_df = df[df["language"] == "en"] if "language" in df.columns else pd.DataFrame()

    def calc_actor(stats):
        out = {}
        for ent, s in stats.items():
            scores = s.pop("scores", [])
            total  = max(s["total"], 1)
            avg    = round(sum(scores)/len(scores), 4) if scores else 0
            out[ent] = {
                **s,
                "avg_sentiment_score": avg,
                "sentiment_ratio": {
                    "positive_pct": round(s["positive"]/total*100, 1),
                    "negative_pct": round(s["negative"]/total*100, 1),
                    "neutral_pct":  round(s["neutral"]/total*100, 1),
                },
            }
        return out

    report = {
        "module":        module_label,
        "description":   module_desc,
        "generated_at":  datetime.now().isoformat(),
        "queries_used":  queries_used,
        "total_comments": len(results),
        "indonesian_comments": int(df["is_indonesian"].sum()),
        "english_comments": int((df["language"]=="en").sum()),
        "overall_sentiment": {
            "positive": int((df["ensemble_label"]=="positive").sum()),
            "negative": int((df["ensemble_label"]=="negative").sum()),
            "neutral":  int((df["ensemble_label"]=="neutral").sum()),
        },
        "indonesian_sentiment": {
            "positive": int((id_df["ensemble_label"]=="positive").sum()) if len(id_df)>0 else 0,
            "negative": int((id_df["ensemble_label"]=="negative").sum()) if len(id_df)>0 else 0,
            "neutral":  int((id_df["ensemble_label"]=="neutral").sum())  if len(id_df)>0 else 0,
        },
        "comment_type_distribution": comment_types,
        "indonesian_comment_type_distribution": id_comment_types,
        "entities": calc_actor(entity_stats),
        "language_distribution": df["language"].value_counts().head(15).to_dict(),
        "top_comments": _top_comments(df),
        "top_indonesian_comments": _top_comments(id_df) if len(id_df)>0 else {},
    }
    return report


def _top_comments(df):
    """Top 3 positif & negatif per entitas."""
    if df is None or len(df) == 0:
        return {}
    out = {}
    ent_cols = [c for c in df.columns if c.startswith("ent_")]
    for col in ent_cols:
        ent_name = col.replace("ent_","")
        adf = df[df[col]==True]
        if len(adf) == 0:
            continue
        cols = [c for c in ["text","ensemble_score","like_count","language","comment_type"] if c in adf.columns]
        out[ent_name] = {
            "most_positive": adf.nlargest(3,"ensemble_score")[cols].to_dict("records"),
            "most_negative": adf.nsmallest(3,"ensemble_score")[cols].to_dict("records"),
        }
    return out


# ─── DEMO DATA ────────────────────────────────────────────────────────────────

def make_demo_data_serangan(n=400):
    import random
    templates = [
        # Indonesian — identitas agama
        ("Innalillahi, saudara kita di Iran syahid oleh kekejaman Israel dan Amerika!", -0.55, "id"),
        ("Khamenei memang syiah tapi tetap saudara seiman, doa kami menyertai Iran", -0.2, "id"),
        ("Yang meninggal bukan cuma syiah, ada warga sipil tak berdosa juga", -0.6, "id"),
        ("Sebagai Muslim Sunni saya menolak syiah tapi juga menolak pembunuhan ini", -0.4, "id"),
        ("Syiah atau Sunni bukan urusannya, ini tentang kedaulatan negara", 0.1, "id"),
        ("Amerika dan Israel biadab, membunuh pemimpin negara berdaulat!", -0.7, "id"),
        ("Iran harus kena karma karena mendukung Hizbullah teror", -0.3, "id"),
        ("Analisis saya: ini bagian dari strategi AS untuk destabilisasi Timur Tengah sebelum pemilu", 0.0, "id"),
        ("Kita tidak perlu membela Iran hanya karena sesama Muslim, harus objektif", 0.1, "id"),
        ("Khamenei bukan perwakilan Islam, dia pemimpin negara Iran saja", 0.0, "id"),
        # Indonesian — analitis
        ("Secara geopolitik, kematian Khamenei akan menciptakan kekosongan kekuasaan yang berbahaya", -0.2, "id"),
        ("Data menunjukkan Iran sudah lemah secara ekonomi sebelum serangan ini", 0.1, "id"),
        ("Pertanyaan pentingnya: siapa yang akan gantikan Khamenei dan apa dampaknya ke OPEC?", 0.0, "id"),
        # English — internasional
        ("The killing of Khamenei is illegal under international law", -0.5, "en"),
        ("Good, one less terrorist leader in the world", 0.5, "en"),
        ("This will destabilize the whole region", -0.6, "en"),
        ("Iran asked for this by supporting Hamas and Hezbollah", 0.2, "en"),
        ("Russia and China will not let Iran fall, watch the next move", 0.0, "en"),
        ("China is watching carefully, Taiwan might be next on their mind", -0.1, "en"),
        ("Putin will use this to distract from Ukraine", -0.3, "en"),
    ]
    results = []
    for i in range(n):
        text, score, lang = random.choice(templates)
        noise = random.uniform(-0.1, 0.1)
        results.append({
            "video_id": f"demo_{random.randint(1,15)}",
            "comment_id": f"demo_s_{i}",
            "text": text,
            "author": f"User_{random.randint(1000,9999)}",
            "like_count": random.randint(0, 300),
            "published_at": f"2026-02-{random.randint(28,28)}T{random.randint(0,23):02d}:00:00Z",
            "reply_count": random.randint(0, 30),
            "video_title": f"Israel serang Iran Khamenei tewas vol{random.randint(1,8)}",
            "channel": random.choice(["Kompas TV","CNN Indonesia","tvOne","Narasi","Metro TV"]),
            "search_query": random.choice(QUERIES_SERANGAN[:4]),
            "query_lang": lang,
        })
    return results


def make_demo_data_ekonomi(n=400):
    import random
    templates = [
        # Indonesian — nuansa ekonomi lokal
        ("BBM pasti naik lagi gara-gara selat hormuz ditutup, rakyat kecil yang kena", -0.6, "id"),
        ("Pertamina harus antisipasi ini, cadangan minyak kita terbatas!", -0.4, "id"),
        ("Harga bensin Indonesia pasti naik, pemerintah harus subsidi!", -0.5, "id"),
        ("Dampaknya ke IHSG sudah keliatan, saham energi naik tapi manufaktur turun", -0.1, "id"),
        ("Indonesia sebagai net importer minyak paling terdampak di ASEAN", -0.3, "id"),
        ("Kita perlu percepat energi terbarukan, ini bukti ketergantungan minyak bahaya", -0.1, "id"),
        ("Rupiah pasti melemah kalau harga minyak dunia terus naik", -0.4, "id"),
        # Indonesian — penonton konflik (tanpa nuansa ekonomi lokal)
        ("Iran tutup selat hormuz, Amerika pasti panik!", 0.2, "id"),
        ("Bagus! Iran kasih pelajaran ke Amerika dengan tutup hormuz!", 0.5, "id"),
        ("Iran kuat! Tutup hormuz bikin barat kelimpungan hahaha", 0.4, "id"),
        ("Doa kita untuk Iran, semoga kuat menghadapi blokade ekonomi Barat", -0.1, "id"),
        # English
        ("Oil at $200 is going to crash the global economy", -0.7, "en"),
        ("China has been stockpiling oil for this exact scenario", 0.1, "en"),
        ("Russia benefits enormously from higher oil prices", 0.3, "en"),
        ("The Hormuz closure is Iran's last economic weapon", 0.0, "en"),
        ("US strategic petroleum reserve will be deployed immediately", 0.1, "en"),
        ("This is why energy independence matters", 0.0, "en"),
        ("Saudi Arabia will increase production to compensate", 0.2, "en"),
    ]
    results = []
    for i in range(n):
        text, score, lang = random.choice(templates)
        noise = random.uniform(-0.1, 0.1)
        results.append({
            "video_id": f"demo_e{random.randint(1,15)}",
            "comment_id": f"demo_e_{i}",
            "text": text,
            "author": f"User_{random.randint(1000,9999)}",
            "like_count": random.randint(0, 200),
            "published_at": f"2026-03-{random.randint(10,13)}T{random.randint(0,23):02d}:00:00Z",
            "reply_count": random.randint(0, 20),
            "video_title": f"Iran tutup selat hormuz harga minyak naik vol{random.randint(1,6)}",
            "channel": random.choice(["Kompas TV","CNN Indonesia","CNBC Indonesia","Bloomberg","Reuters"]),
            "search_query": random.choice(QUERIES_EKONOMI[:4]),
            "query_lang": lang,
        })
    return results


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  🇮🇩  YouTube Sentiment Analyzer — Perspektif Indonesia")
    print("  Modul A: Serangan Israel-USA ke Iran (Feb 28, 2026)")
    print("  Modul B: Penutupan Selat Hormuz (Mar 10, 2026)")
    print("=" * 65)

    demo_mode = not bool(API_KEY)
    if demo_mode:
        print("\n⚠️   Demo mode (tidak ada YOUTUBE_API_KEY).\n")

    # ── MODUL A ──
    print("\n" + "─"*65)
    print("  MODUL A: SERANGAN")
    print("─"*65)
    raw_a = crawl(QUERIES_PER_CHANNEL_SERANGAN, "SERANGAN") if not demo_mode else make_demo_data_serangan(400)

    pd.DataFrame(raw_a).to_csv(RAW_OUTPUT_SERANGAN, index=False, encoding="utf-8")
    print(f"💾  Raw → {RAW_OUTPUT_SERANGAN}")

    res_a, estats_a, ctypes_a, id_ctypes_a = analyze(raw_a, ENTITIES_SERANGAN, "SERANGAN")
    pd.DataFrame(res_a).to_csv(ANALYZED_SERANGAN, index=False, encoding="utf-8")

    report_a = build_report(res_a, estats_a, ctypes_a, id_ctypes_a,
                            "A_SERANGAN",
                            "Sentimen publik Indonesia atas serangan Israel-USA ke Iran. Fokus: apakah komentar Indonesia bisa jernih melihat (hubungan diplomatik, sesama Islam), atau terbawa dikotomi Khamenei/Syiah. Konteks: Indonesia mayoritas Muslim Sunni, non-Syiah.",
                            QUERIES_SERANGAN)
    with open(RESULTS_SERANGAN, "w", encoding="utf-8") as f:
        json.dump(report_a, f, indent=2, ensure_ascii=False)
    print(f"💾  Hasil → {RESULTS_SERANGAN}")

    # ── MODUL B ──
    print("\n" + "─"*65)
    print("  MODUL B: EKONOMI / SELAT HORMUZ")
    print("─"*65)
    raw_b = crawl(QUERIES_PER_CHANNEL_EKONOMI, "EKONOMI") if not demo_mode else make_demo_data_ekonomi(400)

    pd.DataFrame(raw_b).to_csv(RAW_OUTPUT_EKONOMI, index=False, encoding="utf-8")
    print(f"💾  Raw → {RAW_OUTPUT_EKONOMI}")

    res_b, estats_b, ctypes_b, id_ctypes_b = analyze(raw_b, ENTITIES_EKONOMI, "EKONOMI")
    pd.DataFrame(res_b).to_csv(ANALYZED_EKONOMI, index=False, encoding="utf-8")

    report_b = build_report(res_b, estats_b, ctypes_b, id_ctypes_b,
                            "B_EKONOMI",
                            "Sentimen publik Indonesia atas penutupan Selat Hormuz. Fokus: apakah diskusi ekonomi substantif atau sekadar penonton konflik.",
                            QUERIES_EKONOMI)
    with open(RESULTS_EKONOMI, "w", encoding="utf-8") as f:
        json.dump(report_b, f, indent=2, ensure_ascii=False)
    print(f"💾  Hasil → {RESULTS_EKONOMI}")

    # ── COMBINED untuk dashboard ──
    combined = {"module_A": report_a, "module_B": report_b}
    with open(COMBINED_RESULTS, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    print(f"💾  Combined → {COMBINED_RESULTS}")

    # ── SUMMARY ──
    print("\n" + "="*65)
    print("  📊  RINGKASAN")
    print("="*65)
    for label, report in [("SERANGAN", report_a), ("EKONOMI", report_b)]:
        total = report["total_comments"]
        id_n  = report["indonesian_comments"]
        print(f"\n  [{label}]")
        print(f"  Total komentar : {total}")
        if total == 0:
            print(f"  ⚠️  Tidak ada komentar — periksa channel ID di INDONESIAN_CHANNELS")
            continue
        print(f"  Komentar ID    : {id_n} ({round(id_n/max(total,1)*100,1)}%)")
        print(f"  Tipe komentar Indonesia:")
        for k,v in report.get("indonesian_comment_type_distribution",{}).items():
            print(f"    {k:25s}: {v}")

    print(f"\n✅  Selesai! Load {COMBINED_RESULTS} ke dashboard.\n")
    return combined


if __name__ == "__main__":
    main()
