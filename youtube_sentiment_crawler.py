"""
youtube_discourse_analyzer.py
==============================
Frame-Aware Indonesian Discourse Analyzer
Versi 3.0 — rewrite substantif per brief revisi

Tujuan riset:
  Membedakan frame diskursus komentar publik Indonesia terhadap dua peristiwa:
  - Modul A: Serangan Israel–AS ke Iran & kematian Khamenei (window: 2026-02-28 s/d 2026-03-05)
  - Modul B: Penutupan Selat Hormuz & dampak ekonomi (window: 2026-03-10 s/d 2026-03-16)

Output utama: discourse frame classification (multi-label), bukan sentiment.
Sentiment (VADER/TextBlob) hanya metadata tambahan, bukan hasil inti.

PERUBAHAN DARI VERSI SEBELUMNYA:
  - Single-label hierarchical → multi-label frame classification
  - Entity detection dipisah dari frame detection
  - Substring matching → regex word-boundary + phrase matching
  - Normalisasi teks sebelum matching (lowercase, URL strip, slang ringan)
  - Query dipecah per frame (bukan hanya topik umum)
  - Event window filtering via publishedAfter/publishedBefore
  - VADER/TextBlob hanya sebagai metadata opsional
  - Lexicon modular dan editable di CONFIG
  - Modul audit/evaluasi manual
  - Output lengkap: CSV per komentar + JSON agregat + ringkasan per label

KETERBATASAN YANG TERSISA (lihat juga WARNING di output):
  - Rule-based = indikasi awal, bukan ground truth
  - Label sektarian WAJIB diaudit manual sebelum klaim akademik
  - VADER/TextBlob tidak dioptimalkan untuk Bahasa Indonesia informal
  - Komentar YouTube tidak mewakili seluruh opini publik Indonesia
"""

import os
import re
import csv
import json
import random
import logging
from datetime import datetime, timezone
from collections import defaultdict

import googleapiclient.discovery
from langdetect import detect, LangDetectException

# Opsional — hanya metadata tambahan
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    from textblob import TextBlob
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════════
# KONFIGURASI UTAMA — semua parameter riset di sini, mudah diubah
# ══════════════════════════════════════════════════════════════════════════════

CONFIG = {

    # ── API ──────────────────────────────────────────────────────────────────
    "api_key": os.environ.get("YOUTUBE_API_KEY", ""),
    "max_results_per_query": 5,      # video per query
    "max_comments_per_video": 100,   # komentar per video

    # ── KANAL INDONESIA TERVERIFIKASI ─────────────────────────────────────────
    # Format: {"nama_tampilan": "channel_id"}
    # Channel ID diverifikasi manual dari halaman YouTube masing-masing kanal.
    "channels": {
        "Metro TV":      "UCzl0OrB3-ehunyotIQvK77A",   # @metrotvnews ✓
        "iNews":         "UC_7n5QZdUIpPc0-C_s9hshg",   # @officialinews ✓
        "Kompas TV":     "UCneA4BuveCEgJql1m7lwFag",   # @kompastv ✓
        "CNN Indonesia": "UCIohHXwCEKxWCLvAguJ-GjA",   # @cnnidofficial — perlu verifikasi ulang
        # Tambahkan kanal baru di sini:
        # "tvOne":       "CHANNEL_ID_TVOONE",
        # "Narasi":      "CHANNEL_ID_NARASI",
        # "Liputan6":    "CHANNEL_ID_LIPUTAN6",
    },

    # ── EVENT WINDOWS ────────────────────────────────────────────────────────
    # Format: ISO 8601 (UTC). YouTube API gunakan publishedAfter/publishedBefore.
    "modules": {
        "A": {
            "name": "Serangan Israel–AS ke Iran & Kematian Khamenei",
            "window_start": "2026-02-28T00:00:00Z",
            "window_end":   "2026-03-05T23:59:59Z",
        },
        "B": {
            "name": "Penutupan Selat Hormuz & Dampak Ekonomi",
            "window_start": "2026-03-10T00:00:00Z",
            "window_end":   "2026-03-16T23:59:59Z",
        },
    },

    # ── QUERY PER FRAME (dipecah berdasarkan frame, bukan hanya topik umum) ──
    # Setiap modul punya query set per frame. Mudah ditambah tanpa ubah logika.
    "queries": {
        "A": {
            "geopolitik": [
                "serangan AS Israel ke Iran",
                "Iran diserang Amerika Israel",
                "agresi Israel ke Iran",
                "pembunuhan Khamenei",
                "reaksi dunia serangan Iran",
                "konflik Iran Israel 2026",
            ],
            "sektarian": [
                "Khamenei Syiah",
                "Iran Syiah",
                "Syiah vs Sunni Iran",
                "sesama muslim bela Iran",
                "jangan bela Syiah",
                "Iran saudara muslim",
                "Syiah bukan Islam",
                "Sunni dukung Iran",
            ],
        },
        "B": {
            "ekonomi": [
                "Selat Hormuz Indonesia",
                "BBM naik Iran",
                "perang Iran harga minyak Indonesia",
                "Pertamina Hormuz",
                "impor minyak Indonesia Timur Tengah",
                "harga Pertalite Iran",
                "sembako naik karena minyak",
                "cadangan minyak Indonesia",
            ],
            "geopolitik": [
                "Hormuz ditutup Iran",
                "Iran tutup selat Hormuz",
                "dampak Hormuz global",
            ],
        },
    },

    # ── AUDIT ────────────────────────────────────────────────────────────────
    "audit_sample_size": 150,   # komentar per modul untuk audit manual

    # ── OUTPUT ───────────────────────────────────────────────────────────────
    "output_dir": "output",
}

# ══════════════════════════════════════════════════════════════════════════════
# LEXICON — modular dan editable. Pisahkan per frame.
# Setiap entry bisa berupa: string (phrase) atau dict {pattern, note}
#
# CATATAN DESAIN:
#   - Semua matching pakai regex word-boundary (\b) atau phrase, BUKAN substring sederhana.
#   - Token pendek berisiko (<= 3 karakter seperti "un") tidak dipakai sebagai keyword tunggal.
#   - Setiap frame mempunyai lexicon sendiri untuk kemudahan audit dan modifikasi.
# ══════════════════════════════════════════════════════════════════════════════

LEXICON = {

    # ── ENTITY LEXICON ────────────────────────────────────────────────────────
    # Dipisah dari frame. Mendeteksi penyebutan entitas, bukan frame-nya.
    "entities": {
        "USA":       [r"\bamerika\b", r"\bamerican?\b", r"\bas\b(?!\w)", r"\busa\b", r"\bwashington\b", r"\bbiden\b", r"\btrump\b"],
        "Israel":    [r"\bisrael\b", r"\bzionis\b", r"\bnetanyahu\b", r"\btel.?aviv\b", r"\bidf\b"],
        "Iran":      [r"\biran\b", r"\bteheran\b", r"\btehran\b", r"\bpersia\b"],
        "Khamenei":  [r"\bkhamenei\b", r"\bkhomeini\b", r"\bkhamene[i']?\b", r"\bali khamenei\b", r"\bmojtaba\b"],
        "Syiah":     [r"\bsyi[i']?ah\b", r"\bshia\b", r"\bsyi[ae]\b"],
        "Sunni":     [r"\bsunni\b", r"\bahlus sunnah\b", r"\bahlussunnah\b"],
        "China":     [r"\bcina\b", r"\bchina\b", r"\btiongkok\b", r"\bbeijing\b", r"\bxi jinping\b"],
        "Rusia":     [r"\brusia\b", r"\brussia\b", r"\bmoskow?\b", r"\bputin\b"],
        "Hormuz":    [r"\bhormuz\b", r"\bselat hormuz\b", r"\bhormuz strait\b"],
        "Pertamina": [r"\bpertamina\b"],
        "BBM":       [r"\bbbm\b", r"\bpertalite\b", r"\bpertamax\b", r"\bsolar\b(?!\s+system)", r"\bbensol\b"],
        "Minyak":    [r"\bminyak\b", r"\boil\b", r"\bcrude\b", r"\bpetroleum\b", r"\bkilang\b"],
    },

    # ── FRAME LEXICONS ────────────────────────────────────────────────────────

    "geo_diplomatik": {
        # Framing geopolitik, kedaulatan, hukum internasional
        "phrases": [
            r"\bkedaulatan\b", r"\bhukum internasional\b", r"\bpbb\b", r"\bsanksi\b",
            r"\bembargo\b", r"\bhubungan bilateral\b", r"\bdiplomasi\b", r"\bkonflik geopolitik\b",
            r"\bketegangan kawasan\b", r"\bkeseimbangan kekuatan\b", r"\bproxy war\b",
            r"\bperang proxy\b", r"\bgeopolitik\b", r"\bstabilitas kawasan\b",
            r"\bkepentingan nasional\b", r"\bkawasan timur tengah\b", r"\bnato\b",
            r"\bperjanjian\b", r"\bgencatan senjata\b", r"\bnorma internasional\b",
            r"\bpelanggaran kedaulatan\b", r"\bagresi militer\b", r"\bserang duluan\b",
            r"\bhak mempertahankan diri\b", r"\bself.?defense\b",
        ],
        "note": "Frame geopolitik/diplomatik — analisis berbasis kedaulatan dan hukum internasional",
    },

    "kemanusiaan": {
        # Framing kemanusiaan universal, korban sipil
        "phrases": [
            r"\bkorban sipil\b", r"\brakyat\b.*\b(menderita|sengsara)\b",
            r"\b(menderita|sengsara)\b.*\brakyat\b",
            r"\banak.?anak\b.*\b(tewas|meninggal|korban)\b",
            r"\b(tewas|meninggal|korban)\b.*\banak.?anak\b",
            r"\bwarga sipil\b", r"\brefugee\b", r"\bpengungsi\b",
            r"\btragedi kemanusiaan\b", r"\bgenosida\b", r"\bhak asasi\b",
            r"\bham\b(?!\w)", r"\bperang tidak manusiawi\b",
            r"\bpenderitaan\b", r"\bselamatkan nyawa\b", r"\bjangan (ada )?korban\b",
        ],
        "note": "Framing kemanusiaan universal — tidak berbasis identitas agama/etnis",
    },

    "solidaritas_emosional": {
        # Dukungan/penolakan berbasis emosi atau identitas keagamaan tanpa analisis
        "phrases": [
            r"\bdukung iran\b", r"\bbela iran\b", r"\ballahu akbar\b.*\biran\b",
            r"\biran\b.*\ballahu akbar\b",
            r"\bsemoga iran menang\b", r"\bhajar (terus|mereka)\b",
            r"\bkafir (menyerang|vs)\b", r"\bmaju terus iran\b",
            r"\bsemangat iran\b", r"\biran kuat\b", r"\biran tidak takut\b",
            r"\bkita (semua )?dukung\b.*\biran\b",
            r"\biran\b.*\b(semangat|terus|maju)\b",
        ],
        "note": "Dukungan emosional terhadap Iran — belum tentu sektarian, bisa patriotik atau simpati umum",
    },

    "sektarian_syiah_sunni": {
        # Frame sektarian — label sensitif, WAJIB diaudit manual
        "phrases": [
            r"\bsyi[i']?ah\b.*\b(bukan islam|sesat|kafir)\b",
            r"\b(bukan islam|sesat|kafir)\b.*\bsyi[i']?ah\b",
            r"\bjangan bela syi[i']?ah\b",
            r"\biran syi[i']?ah\b.*\bbukan (saudara|islam)\b",
            r"\bsyi[i']?ah bukan (saudara|islam)\b",
            r"\bsunni (vs|lawan) syi[i']?ah\b",
            r"\bsyi[i']?ah (vs|lawan) sunni\b",
            r"\brafidhah\b", r"\bmazhab (sesat|salah)\b",
            r"\bahlul bait\b.*\b(sesat|salah)\b",
            r"\bsyi[i']?ah\b.*\b(musuh|lawan)\b.*\bsunni\b",
            r"\btidak perlu bela iran.*syi[i']?ah\b",
            r"\bbela iran karena (islam|muslim)\b",
        ],
        "note": "⚠️ LABEL SENSITIF — wajib audit manual. Hanya positif jika frame sektarian eksplisit terbaca.",
    },

    "anti_barat": {
        # Frame anti-Barat, anti-imperialisme, anti-Amerika
        "phrases": [
            r"\bimperialis\b", r"\bpenjajah\b", r"\bhegemoni (barat|as|amerika)\b",
            r"\bwakil (iblis|setan)\b.*\b(as|israel|barat)\b",
            r"\bkaki tangan barat\b", r"\bpion barat\b", r"\bbarat munafik\b",
            r"\bamerika biadab\b", r"\binternasional bungkam\b",
            r"\bdunia (diam|bungkam|tutup mata)\b",
            r"\bpbb tidak berguna\b", r"\bpbb gagal\b",
            r"\bdua standard\b", r"\bdouble standard\b",
            r"\bnew world order\b", r"\bnwo\b(?!\w)",
            r"\bbarat\b.*\b(jahat|laknat|biadab)\b",
        ],
        "note": "Frame anti-Barat/anti-imperialisme — bisa tumpang tindih dengan geo_diplomatik",
    },

    "ekonomi_global": {
        # Dampak ekonomi global, harga minyak dunia
        "phrases": [
            r"\bharga minyak (dunia|global|internasional)\b",
            r"\bpasar (minyak|energi|komoditas)\b",
            r"\bopec\b", r"\bsupply chain\b", r"\brantai pasok\b",
            r"\bgangguan (pasokan|distribusi|supply)\b",
            r"\bselat hormuz\b.*\bjalur\b",
            r"\bpengiriman minyak\b", r"\btanker\b", r"\bblokade\b",
            r"\bkrisis energi (global|dunia)\b",
        ],
        "note": "Dampak ekonomi global — harga minyak dunia, rantai pasok",
    },

    "ekonomi_lokal_indonesia": {
        # Dampak ke Indonesia secara spesifik
        "phrases": [
            r"\bbbm naik\b", r"\bpertalite (mahal|naik)\b", r"\bpertamax naik\b",
            r"\bsolar naik\b", r"\bharga (bbm|bensin|solar|pertalite) naik\b",
            r"\bharga barang naik\b", r"\bsembako naik\b", r"\brakyat susah\b",
            r"\bpertamina (siap|bisa|mampu|bagaimana)\b",
            r"\bimpor minyak\b.*\bindonesia\b", r"\bindonesia\b.*\bimpor minyak\b",
            r"\bdolar naik\b", r"\bnilai tukar\b.*\bnegeri\b",
            r"\befek ke indonesia\b", r"\bdampak ke (indonesia|kita|rakyat)\b",
            r"\bindonesia (terdampak|kena dampak)\b",
            r"\bujung.?ujung(nya)?\b.*\brakyat\b",
            r"\bcadangan minyak\b.*\bindonesia\b",
            r"\bindonesia\b.*\bcadangan minyak\b",
            r"\benergy security\b.*\bindonesia\b",
            r"\bketahanan energi\b",
        ],
        "note": "Dampak ekonomi lokal ke Indonesia — BBM, Pertamina, sembako, impor minyak",
    },

    "netral_informasional": {
        # Komentar yang hanya menginformasikan ulang berita, tanpa stance
        "phrases": [
            r"^\s*(breaking|berita)\s*:\s*",
            r"\bmenurut (laporan|berita|media)\b",
            r"\bdilaporkan bahwa\b",
            r"\bupdate\s*:\s*",
            r"\binfo\s*:\s*",
            r"\bfyi\s*:?\b",
        ],
        "note": "Komentar informatif/update berita tanpa frame diskursif yang jelas",
    },

    "noise_spam": {
        # Komentar spam, promosi, tidak relevan
        "phrases": [
            r"\bsubscribe\b", r"\bfollow\b.*\bchannel\b",
            r"\bjual\b.*\bmurah\b", r"\bharga\b.*\bpromo\b",
            r"(wa|whatsapp|telegram|ig)\s*:?\s*\+?\d{8,}",
            r"\bslot\b.*\bonline\b", r"\bjudi (online|slot)\b",
            r"\bagen\b.*\bresmi\b",
        ],
        "note": "Spam, promosi, konten tidak relevan — dikecualikan dari analisis",
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

# Kamus normalisasi slang ringan — tambahkan sesuai kebutuhan
SLANG_DICT = {
    r"\bgak\b":     "tidak",
    r"\bnggak\b":   "tidak",
    r"\bnga\b":     "tidak",
    r"\bgakk\b":    "tidak",
    r"\bga\b(?!\w)":"tidak",
    r"\byg\b":      "yang",
    r"\bkrn\b":     "karena",
    r"\bkrna\b":    "karena",
    r"\bgara2\b":   "gara-gara",
    r"\bkita2\b":   "kita",
    r"\bbaik2\b":   "baik-baik",
    r"\bbnyk\b":    "banyak",
    r"\bdgn\b":     "dengan",
    r"\bdri\b":     "dari",
    r"\bprtama\b":  "pertama",
    r"\bsdh\b":     "sudah",
    r"\bblm\b":     "belum",
    r"\bjg\b":      "juga",
    r"\bspt\b":     "seperti",
}

def preprocess(text: str) -> str:
    """
    Normalisasi teks sebelum matching.
    Urutan: lowercase → strip URL → strip mention → strip simbol berlebih
            → normalisasi spasi → normalisasi slang ringan
    """
    if not text:
        return ""
    t = text.lower()
    # Hapus URL
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    # Hapus mention @user
    t = re.sub(r"@\w+", " ", t)
    # Hapus tanda baca yang tidak bermakna (pertahankan tanda hubung dan apostrof dalam kata)
    t = re.sub(r"[^\w\s\-']", " ", t)
    # Normalisasi spasi
    t = re.sub(r"\s+", " ", t).strip()
    # Normalisasi slang
    for pattern, replacement in SLANG_DICT.items():
        t = re.sub(pattern, replacement, t)
    return t

# ══════════════════════════════════════════════════════════════════════════════
# ENTITY DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_entities(cleaned_text: str) -> list[str]:
    """
    Deteksi entitas dengan regex word-boundary.
    Mengembalikan list entitas yang terdeteksi (bisa lebih dari satu).
    """
    detected = []
    for entity_name, patterns in LEXICON["entities"].items():
        for pattern in patterns:
            if re.search(pattern, cleaned_text, re.IGNORECASE):
                detected.append(entity_name)
                break  # cukup satu pattern match per entitas
    return detected

# ══════════════════════════════════════════════════════════════════════════════
# FRAME DETECTION (multi-label)
# ══════════════════════════════════════════════════════════════════════════════

FRAME_LABELS = [
    "geo_diplomatik",
    "kemanusiaan",
    "solidaritas_emosional",
    "sektarian_syiah_sunni",
    "anti_barat",
    "ekonomi_global",
    "ekonomi_lokal_indonesia",
    "netral_informasional",
    "noise_spam",
]

def detect_frames(cleaned_text: str, original_text: str) -> dict:
    """
    Multi-label frame detection.
    Setiap komentar bisa memiliki lebih dari satu label.
    Mengembalikan dict:
      {
        "frames": ["geo_diplomatik", "ekonomi_lokal_indonesia"],
        "match_details": {"geo_diplomatik": ["pola yang match"], ...}
      }

    DESAIN:
    - Matching pakai regex dari LEXICON, bukan substring sederhana.
    - Tidak ada hierarki — semua frame yang match dilaporkan.
    - noise_spam diperiksa pertama; jika positif, frame lain diabaikan.
    """
    frames = []
    match_details = {}

    # 1. Cek noise_spam terlebih dahulu
    noise_matches = _match_phrases(cleaned_text, LEXICON["noise_spam"]["phrases"])
    if noise_matches:
        return {"frames": ["noise_spam"], "match_details": {"noise_spam": noise_matches}}

    # 2. Deteksi semua frame lain
    for frame in FRAME_LABELS:
        if frame in ("noise_spam", "netral_informasional"):
            continue
        if frame not in LEXICON:
            continue
        matches = _match_phrases(cleaned_text, LEXICON[frame]["phrases"])
        if matches:
            frames.append(frame)
            match_details[frame] = matches

    # 3. Cek netral_informasional terakhir (hanya jika tidak ada frame lain)
    if not frames:
        ni_matches = _match_phrases(original_text, LEXICON["netral_informasional"]["phrases"])
        if ni_matches:
            frames = ["netral_informasional"]
            match_details["netral_informasional"] = ni_matches

    # 4. Jika benar-benar tidak ada frame — label sebagai "tidak_terklasifikasi"
    if not frames:
        frames = ["tidak_terklasifikasi"]

    return {"frames": frames, "match_details": match_details}


def _match_phrases(text: str, patterns: list) -> list:
    """Helper: jalankan semua regex pattern, kembalikan list pattern yang match."""
    matched = []
    for pattern in patterns:
        try:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                matched.append(m.group(0))
        except re.error as e:
            logging.warning(f"Regex error pada pattern '{pattern}': {e}")
    return matched

# ══════════════════════════════════════════════════════════════════════════════
# LANGUAGE DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def is_indonesian(text: str) -> bool:
    """
    Deteksi bahasa Indonesia.
    Mengembalikan True jika terdeteksi sebagai 'id'.
    Catatan: langdetect bisa false positive untuk teks sangat pendek.
    Teks < 15 karakter dikecualikan untuk mengurangi noise.
    """
    if len(text.strip()) < 15:
        return False
    try:
        return detect(text) == "id"
    except LangDetectException:
        return False

# ══════════════════════════════════════════════════════════════════════════════
# SENTIMENT (metadata opsional — bukan output utama)
# ══════════════════════════════════════════════════════════════════════════════

def get_sentiment_metadata(text: str) -> dict:
    """
    VADER + TextBlob sebagai metadata tambahan, bukan output inti.
    Catatan metodologis: keduanya dioptimalkan untuk Bahasa Inggris.
    Hasil untuk Bahasa Indonesia informal (campur kode, singkatan, emoji)
    tidak akurat — gunakan hanya sebagai referensi, bukan kesimpulan.
    Mengembalikan dict kosong jika library tidak tersedia.
    """
    if not SENTIMENT_AVAILABLE:
        return {}
    vader = SentimentIntensityAnalyzer()
    vs = vader.polarity_scores(text)
    tb = TextBlob(text).sentiment.polarity
    ensemble = (vs["compound"] + tb) / 2
    label = "positif" if ensemble > 0.05 else "negatif" if ensemble < -0.05 else "netral"
    return {
        "vader_compound": round(vs["compound"], 4),
        "textblob_polarity": round(tb, 4),
        "ensemble_score": round(ensemble, 4),
        "sentiment_label": label,
        "catatan": "METADATA SAJA — tidak akurat untuk Bahasa Indonesia informal",
    }

# ══════════════════════════════════════════════════════════════════════════════
# YOUTUBE API CRAWLER
# ══════════════════════════════════════════════════════════════════════════════

def build_youtube_client():
    api_key = CONFIG["api_key"]
    if not api_key:
        raise ValueError("YOUTUBE_API_KEY belum diset. Export env var atau isi CONFIG['api_key'].")
    return googleapiclient.discovery.build("youtube", "v3", developerKey=api_key)


def fetch_videos(youtube, channel_id: str, query: str, module_id: str) -> list[dict]:
    """
    Ambil video dari channel tertentu dalam event window modul.
    Menggunakan publishedAfter + publishedBefore untuk filtering waktu.
    """
    mod = CONFIG["modules"][module_id]
    try:
        resp = youtube.search().list(
            part="snippet",
            channelId=channel_id,
            q=query,
            type="video",
            order="relevance",
            maxResults=CONFIG["max_results_per_query"],
            publishedAfter=mod["window_start"],
            publishedBefore=mod["window_end"],
        ).execute()
    except Exception as e:
        logging.error(f"Error fetching videos untuk '{query}' di channel {channel_id}: {e}")
        return []

    videos = []
    for item in resp.get("items", []):
        videos.append({
            "video_id":    item["id"]["videoId"],
            "video_title": item["snippet"]["title"],
            "published_at": item["snippet"]["publishedAt"],
            "channel_name": item["snippet"]["channelTitle"],
        })
    return videos


def fetch_comments(youtube, video_id: str) -> list[dict]:
    """Ambil komentar top-level dari video."""
    comments = []
    try:
        resp = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=CONFIG["max_comments_per_video"],
            textFormat="plainText",
            order="relevance",
        ).execute()
        for item in resp.get("items", []):
            s = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "comment_id":   item["id"],
                "original_text": s["textDisplay"],
                "published_at": s["publishedAt"],
            })
    except Exception as e:
        logging.warning(f"Komentar tidak bisa diambil untuk video {video_id}: {e}")
    return comments

# ══════════════════════════════════════════════════════════════════════════════
# PROSES SATU KOMENTAR
# ══════════════════════════════════════════════════════════════════════════════

def process_comment(comment: dict, video_meta: dict, module_id: str) -> dict | None:
    """
    Proses satu komentar:
    1. Deteksi bahasa — skip jika bukan Indonesia
    2. Preprocessing
    3. Entity detection
    4. Frame detection (multi-label)
    5. Sentiment metadata (opsional)

    Mengembalikan None jika komentar bukan bahasa Indonesia.
    """
    original = comment["original_text"]

    if not is_indonesian(original):
        return None

    cleaned = preprocess(original)
    entities = detect_entities(cleaned)
    frame_result = detect_frames(cleaned, original)
    sentiment = get_sentiment_metadata(original) if SENTIMENT_AVAILABLE else {}

    return {
        # Identifikasi sumber
        "module_name":    module_id,
        "video_id":       video_meta["video_id"],
        "video_title":    video_meta["video_title"],
        "channel_name":   video_meta["channel_name"],
        "published_at":   comment.get("published_at", ""),
        "comment_id":     comment["comment_id"],
        # Teks
        "original_text":  original,
        "cleaned_text":   cleaned,
        # Analisis
        "detected_entities":      json.dumps(entities, ensure_ascii=False),
        "detected_frames":        json.dumps(frame_result["frames"], ensure_ascii=False),
        "frame_match_details":    json.dumps(frame_result["match_details"], ensure_ascii=False),
        # Metadata sentimen (opsional)
        "sentiment_vader":        sentiment.get("vader_compound", ""),
        "sentiment_textblob":     sentiment.get("textblob_polarity", ""),
        "sentiment_ensemble":     sentiment.get("ensemble_score", ""),
        "sentiment_label":        sentiment.get("sentiment_label", ""),
        # Bahasa
        "detected_as_indonesian": True,
    }

# ══════════════════════════════════════════════════════════════════════════════
# AGREGASI & RINGKASAN
# ══════════════════════════════════════════════════════════════════════════════

def build_summary(results: list[dict]) -> dict:
    """
    Bangun ringkasan agregat dari semua hasil:
    - Jumlah per frame
    - Jumlah per entitas
    - Cross-tab entity × frame
    - Contoh komentar representatif per frame
    - Perbandingan Modul A vs Modul B
    """
    frame_count = defaultdict(int)
    entity_count = defaultdict(int)
    entity_frame_crosstab = defaultdict(lambda: defaultdict(int))
    frame_examples = defaultdict(list)
    module_counts = defaultdict(lambda: defaultdict(int))

    for r in results:
        mod = r["module_name"]
        frames = json.loads(r["detected_frames"])
        entities = json.loads(r["detected_entities"])

        for f in frames:
            frame_count[f] += 1
            module_counts[mod][f] += 1
            if len(frame_examples[f]) < 5:
                frame_examples[f].append({
                    "text":    r["original_text"][:200],
                    "module":  mod,
                    "video":   r["video_title"][:80],
                    "channel": r["channel_name"],
                })

        for e in entities:
            entity_count[e] += 1
            for f in frames:
                entity_frame_crosstab[e][f] += 1

    return {
        "total_indonesian_comments": len(results),
        "frame_distribution": dict(frame_count),
        "entity_distribution": dict(entity_count),
        "entity_frame_crosstab": {k: dict(v) for k, v in entity_frame_crosstab.items()},
        "frame_examples": {k: v for k, v in frame_examples.items()},
        "comparison_by_module": {k: dict(v) for k, v in module_counts.items()},
    }

# ══════════════════════════════════════════════════════════════════════════════
# AUDIT / EVALUASI MANUAL
# ══════════════════════════════════════════════════════════════════════════════

def export_audit_sample(results: list[dict], module_id: str, out_dir: str):
    """
    Ekspor sample untuk audit manual.
    Prioritaskan komentar dengan frame sektarian_syiah_sunni (perlu audit wajib).
    Lengkapi dengan random sample.
    """
    sektarian = [r for r in results if r["module_name"] == module_id
                 and "sektarian_syiah_sunni" in r["detected_frames"]]
    others    = [r for r in results if r["module_name"] == module_id
                 and "sektarian_syiah_sunni" not in r["detected_frames"]]

    random.shuffle(others)
    n = CONFIG["audit_sample_size"]
    # Selalu sertakan semua sektarian, sisanya dari others
    sample = sektarian + others[:max(0, n - len(sektarian))]
    sample = sample[:n]

    filepath = os.path.join(out_dir, f"audit_modul_{module_id}.csv")
    cols = ["comment_id","original_text","detected_frames","frame_match_details",
            "detected_entities","video_title","channel_name","published_at",
            # Kolom audit manual — diisi oleh peneliti
            "audit_frame_manual","audit_benar_salah","audit_catatan"]

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sample:
            w.writerow({**r, "audit_frame_manual": "", "audit_benar_salah": "", "audit_catatan": ""})

    logging.info(f"Audit sample Modul {module_id}: {len(sample)} komentar → {filepath}")
    print(f"\n⚠️  AUDIT WAJIB: File audit Modul {module_id} tersimpan di {filepath}")
    print(f"   {len(sektarian)} komentar dengan flag sektarian_syiah_sunni termasuk dalam sample.")
    print(f"   Kolom 'audit_frame_manual', 'audit_benar_salah', 'audit_catatan' diisi manual peneliti.")


def compute_audit_metrics(audit_filepath: str) -> dict:
    """
    Baca file audit yang sudah diisi manual.
    Hitung precision indikatif per frame.
    Jalankan fungsi ini setelah audit manual selesai.
    """
    if not os.path.exists(audit_filepath):
        print(f"File audit tidak ditemukan: {audit_filepath}")
        return {}

    correct = defaultdict(int)
    total   = defaultdict(int)
    fp_examples = defaultdict(list)
    fn_examples = defaultdict(list)

    with open(audit_filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("audit_benar_salah"):
                continue
            frames = json.loads(row["detected_frames"])
            is_correct = row["audit_benar_salah"].strip().lower() in ("y","ya","benar","1","true")
            for fr in frames:
                total[fr] += 1
                if is_correct:
                    correct[fr] += 1
                else:
                    fp_examples[fr].append(row["original_text"][:100])

    precision = {fr: round(correct[fr] / total[fr], 3) if total[fr] else None
                 for fr in total}
    return {
        "precision_per_frame": precision,
        "false_positive_examples": {k: v[:3] for k, v in fp_examples.items()},
        "catatan": "Precision indikatif — bukan evaluasi formal. Sample size kecil, hasil tidak representatif secara statistik.",
    }

# ══════════════════════════════════════════════════════════════════════════════
# EXPORT
# ══════════════════════════════════════════════════════════════════════════════

CSV_COLUMNS = [
    "module_name", "video_id", "video_title", "channel_name", "published_at",
    "comment_id", "original_text", "cleaned_text",
    "detected_entities", "detected_frames", "frame_match_details",
    "sentiment_vader", "sentiment_textblob", "sentiment_ensemble", "sentiment_label",
    "detected_as_indonesian",
]

def export_results(results: list[dict], summary: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # 1. CSV komentar level
    csv_path = os.path.join(out_dir, "discourse_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    logging.info(f"CSV: {csv_path} ({len(results)} baris)")

    # 2. JSON agregat
    json_path = os.path.join(out_dir, "summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logging.info(f"JSON summary: {json_path}")

    # 3. File ringkasan readable (teks)
    txt_path = os.path.join(out_dir, "ringkasan.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("RINGKASAN ANALISIS DISCOURSE PUBLIK INDONESIA\n")
        f.write("Konflik Iran–Israel–USA | Frame-Aware Discourse Analyzer\n")
        f.write("=" * 70 + "\n\n")

        f.write("⚠️  WARNING METODOLOGIS:\n")
        f.write("  1. Komentar YouTube TIDAK mewakili seluruh opini publik Indonesia.\n")
        f.write("  2. Distribusi komentar dipengaruhi algoritma platform & pemilihan video.\n")
        f.write("  3. Rule-based frame detection adalah INDIKASI AWAL, bukan ground truth.\n")
        f.write("  4. Label sektarian_syiah_sunni WAJIB dicek manual sebelum klaim akademik.\n")
        f.write("  5. Sentiment VADER/TextBlob adalah metadata tambahan — tidak akurat untuk\n")
        f.write("     Bahasa Indonesia informal.\n\n")

        f.write(f"Total komentar Indonesia: {summary['total_indonesian_comments']}\n\n")

        f.write("DISTRIBUSI FRAME:\n")
        for frame, count in sorted(summary["frame_distribution"].items(), key=lambda x: -x[1]):
            note = LEXICON.get(frame, {}).get("note", "")
            f.write(f"  {frame:35s} {count:4d}   {note}\n")
        f.write("\n")

        f.write("DISTRIBUSI ENTITAS:\n")
        for ent, count in sorted(summary["entity_distribution"].items(), key=lambda x: -x[1]):
            f.write(f"  {ent:20s} {count:4d}\n")
        f.write("\n")

        f.write("PERBANDINGAN MODUL A vs B:\n")
        for mod, frames in summary["comparison_by_module"].items():
            mod_name = CONFIG["modules"].get(mod, {}).get("name", mod)
            f.write(f"\n  Modul {mod} — {mod_name}:\n")
            for fr, cnt in sorted(frames.items(), key=lambda x: -x[1]):
                f.write(f"    {fr:35s} {cnt:4d}\n")
        f.write("\n")

        f.write("CONTOH KOMENTAR REPRESENTATIF PER FRAME:\n\n")
        for frame, examples in summary["frame_examples"].items():
            f.write(f"  [{frame}]\n")
            for ex in examples[:3]:
                f.write(f"    [{ex['channel']}] \"{ex['text'][:120]}\"\n")
            f.write("\n")

    logging.info(f"Ringkasan: {txt_path}")
    return csv_path, json_path, txt_path

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
    out_dir = CONFIG["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    youtube = build_youtube_client()
    all_results = []
    seen_comments = set()  # deduplifikasi antar query

    # ── Iterasi modul × kanal × frame × query ────────────────────────────────
    for module_id, queries_by_frame in CONFIG["queries"].items():
        mod_name = CONFIG["modules"][module_id]["name"]
        logging.info(f"\n{'='*60}")
        logging.info(f"MODUL {module_id}: {mod_name}")
        logging.info(f"Window: {CONFIG['modules'][module_id]['window_start']} → {CONFIG['modules'][module_id]['window_end']}")

        for channel_name, channel_id in CONFIG["channels"].items():
            logging.info(f"\n  Kanal: {channel_name}")

            for frame_group, queries in queries_by_frame.items():
                for query in queries:
                    logging.info(f"    Query [{frame_group}]: {query}")
                    videos = fetch_videos(youtube, channel_id, query, module_id)
                    if not videos:
                        continue

                    for vid in videos:
                        comments = fetch_comments(youtube, vid["video_id"])
                        for raw_comment in comments:
                            cid = raw_comment["comment_id"]
                            if cid in seen_comments:
                                continue
                            seen_comments.add(cid)

                            processed = process_comment(raw_comment, vid, module_id)
                            if processed:
                                all_results.append(processed)

    logging.info(f"\nTotal komentar Indonesia: {len(all_results)}")

    if not all_results:
        logging.warning("Tidak ada komentar Indonesia yang berhasil diproses.")
        return

    # ── Summary & export ──────────────────────────────────────────────────────
    summary = build_summary(all_results)
    csv_p, json_p, txt_p = export_results(all_results, summary, out_dir)

    # ── Audit sample per modul ────────────────────────────────────────────────
    for module_id in CONFIG["modules"]:
        export_audit_sample(all_results, module_id, out_dir)

    print(f"\n{'='*60}")
    print("SELESAI.")
    print(f"  CSV komentar : {csv_p}")
    print(f"  JSON summary : {json_p}")
    print(f"  Ringkasan    : {txt_p}")
    print(f"\n  File audit tersimpan di: {out_dir}/audit_modul_*.csv")
    print("  Isi kolom audit_frame_manual + audit_benar_salah secara manual,")
    print("  lalu jalankan: compute_audit_metrics('output/audit_modul_A.csv')")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run()
