# 📈 stock-news-transcriber

![Python](https://img.shields.io/badge/python-3.10%2B-blue?style=for-the-badge&logo=python)
![AI Powered](https://img.shields.io/badge/AI-Powered-orange?style=for-the-badge)
![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)

**An automated pipeline for transcribing, verifying, and summarizing Thai stock market analysis videos. Powered by Typhoon ASR and Gemini Flash with an agentic workflow.**

---

## 📖 About The Project

This project was developed to solve the challenge of processing long-form Thai financial news. It automates the workflow of converting audio to text, correcting specific financial jargon, verifying stock entities against market data, and generating infographic-ready summaries.

It utilizes **Typhoon ASR** for high-accuracy Thai speech recognition and **Gemini Flash** for context-aware text processing, wrapped in a robust Python script that implements agentic logic (Self-Correction & Verification).

## 🚀 Key Features

* **🧠 Agentic Correction Workflow:** Implements a loop that doesn't just transcribe, but actively verifies and corrects financial terms using a local Knowledge Base.
* **⚡ Adaptive Chunking Strategy:** Automatically adjusts processing logic based on video duration:
    * *Short (< 30 min):* High-speed concurrent processing.
    * *Long (> 1 hr):* Stability-focused sequential processing to prevent API rate limits.
* **🛡️ Smart Entity Verification:**
    * **Fuzzy Matching:** Corrects ASR errors (e.g., "HIKOM" -> "THCOM") using `TheFuzz`.
    * **Contextual Safety:** Filters out false positives (e.g., confusing "Citi" bank with "CI" stock).
* **🔗 Context-Aware Processing:** Handles long videos by passing the "previous context" to the LLM in chunks, ensuring sentences aren't cut off and the narrative remains fluid.
* **📊 Infographic-Ready Summary:** Outputs a structured Markdown summary highlighting Market Overview, Top Gainers/Losers, and Strategy.

## 🛠️ Tech Stack

* **ASR Engine:** [Typhoon Audio API](https://opentyphoon.ai/) (Realtime Thai ASR)
* **LLM Engine:** [Google Gemini 2.0 Flash](https://ai.google.dev/) (via LangChain)
* **Audio Processing:** `yt-dlp`, `pydub`, `ffmpeg`
* **Logic & Verification:** `TheFuzz`, `LangChain Text Splitters`

## 📦 Installation

1.  **Install dependencies**
    It is recommended to use a virtual environment (venv).
    ```bash
    pip install -r requirements.txt
    ```
    *> **Note:** Ensure `ffmpeg` is installed and added to your system path.*

2.  **Configuration**
    Open the script file (e.g., `main.py`) and set your API keys:
    ```python
    TYPHOON_API_KEY = "your_typhoon_key"
    GOOGLE_API_KEY = "your_gemini_key"
    ```

## ⚙️ Setup Knowledge Base

To make the entity verification work effectively, ensure you have the following JSON files in the root directory (or let the script generate/read them):

* `knowledge_base.json`: Contains mapping of stock tickers to their aliases (e.g., `"ADVANC.BK": ["advanc", "แอดวานซ์"]`).
* `finance_terms.json`: Custom dictionary for specific jargon correction.

## 🚀 Usage

Run the script via command line by providing a YouTube URL:

```bash
python main.py --url "[https://www.youtube.com/watch?v=YOUR_VIDEO_ID](https://www.youtube.com/watch?v=YOUR_VIDEO_ID)"
