# 🌍 Iran–Israel Conflict · YouTube Sentiment Analyzer

Sistem crawling komentar YouTube + analisis sentimen NLP terhadap tiga aktor utama blok Iran:  
**Ali Khamenei · Vladimir Putin · Xi Jinping**

---

## 📁 Struktur File

```
├── youtube_sentiment_crawler.py   # Script crawling + NLP pipeline
├── dashboard.html                 # Dashboard visualisasi interaktif
├── requirements.txt               # Python dependencies
└── README.md
```

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt

# Download TextBlob corpus (sekali saja):
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"
```

### 2. Dapatkan YouTube Data API v3 Key
- Buka [Google Cloud Console](https://console.cloud.google.com)
- Buat project baru → Enable **YouTube Data API v3**
- Buat API Key di **Credentials**

### 3. Set API Key & Jalankan
```bash
export YOUTUBE_API_KEY="AIza..."
python youtube_sentiment_crawler.py
```

> **Demo mode:** Jalankan tanpa API key untuk testing dengan 500 data sintetis.

### 4. Lihat Hasil di Dashboard
Buka `dashboard.html` di browser → klik **▲ LOAD JSON** → pilih `sentiment_results.json`

---

## 📊 Output Files

| File | Keterangan |
|---|---|
| `iran_conflict_comments_raw.csv` | Komentar mentah dari YouTube |
| `analyzed_comments.csv` | Komentar + skor sentimen lengkap |
| `sentiment_results.json` | Ringkasan per aktor (untuk dashboard) |

---

## 🧠 NLP Pipeline

```
Komentar YouTube
      │
      ├─→ VADER Sentiment       (compound score -1 to +1)
      ├─→ TextBlob Sentiment    (polarity + subjectivity)
      ├─→ Ensemble Average      (VADER + TextBlob) / 2
      ├─→ Actor Mention Detection (keyword matching)
      ├─→ Language Detection    (langdetect)
      └─→ Tone Classification   (aggressive / supportive / opposing / peace-seeking)
```

**Threshold klasifikasi:**
- Positive : ensemble score ≥ +0.05
- Negative : ensemble score ≤ −0.05
- Neutral  : −0.05 < score < +0.05

---

## 🔑 Konfigurasi

Edit di bagian `CONFIGURATION` dalam script:

```python
SEARCH_QUERIES          # Query pencarian video
ACTORS                  # Keyword per aktor
MAX_COMMENTS_PER_VIDEO  # Default: 200
MAX_VIDEOS_PER_QUERY    # Default: 5
```

---

## ⚠️ Catatan API Quota

YouTube Data API v3 memberi **10.000 unit/hari** gratis.  
Estimasi penggunaan script default: ~2.000–4.000 unit per run.

---

## 📜 License

MIT
