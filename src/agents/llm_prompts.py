# src/agents/llm_prompts.py
from typing import List
import random
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser

# --- LLM Setup ---
# ตรวจสอบให้แน่ใจว่า path นี้ถูกต้องตาม project structure ของคุณ
from src.agents.llm_factory import create_llm

llm = create_llm()

class DynamicPromptBuilder:
    """Generates context-specific prompts for ASR."""
    def __init__(self, context_mgr, term_mgr):
        self.ctx = context_mgr
        self.term_mgr = term_mgr
        
    def build_prompt(self, metadata: dict) -> str:
        title = metadata.get("title", "")
        channel = metadata.get("channel", "")
        import re
        # Extract tickers including numbers (e.g., SIRI23A)
        # ปรับ Regex ให้รองรับชื่อหุ้นที่มีตัวเลขและมีความยาวเหมาะสม
        candidates = re.findall(r"\b[A-Z0-9]{2,10}\b", (channel + title).upper())
        priority_tickers = [c for c in candidates if c in self.ctx.all_tickers]
        
        parts = []
        if channel: parts.append(f"SOURCE_CHANNEL: {channel}")
        if priority_tickers: parts.append(f"FOCUS_ENTITIES: {', '.join(set(priority_tickers))}")
        
        # Place context at the end for recency effect (เทคนิค Recency Bias)
        return " | ".join(parts)[:1000]

# --- Load Learned Errors (Historical Usage) ---
def get_learned_errors_section() -> str:
    """ดึงข้อมูล Error ที่เคยเกิดขึ้นมาเตือน LLM"""
    try:
        from asr_error_logger import get_logger
        logger = get_logger()
        # ดึงมา 10 ตัวอย่างล่าสุด
        examples = logger.get_error_examples_for_prompt(limit=10, format="bullet")
        if examples:
            return (
                "\n<learned_mistakes_db>\n"
                "**⚠️ PREVIOUS MISTAKES TO AVOID:**\n"
                f"{examples}\n"
                "DO NOT repeat these specific errors.\n"
                "</learned_mistakes_db>\n"
            )
        return "" 
    except Exception as e:
        print(f"⚠️ Error loading learned errors: {e}")
        return ""

# --- Correction Agent (Final Hybrid + Feedback Loop Optimized) ---
correction_system_prompt = (
    # --- ROLE & MISSION ---
    "<role>\n"
    "You are the **Senior Financial Editor**.\n"
    "Expertise: Thai Equity Markets, Technical Analysis, and Text Normalization.\n"
    "</role>\n\n"
    
    "<mission>\n"
    "**Objective:** Transform raw ASR output into professional Thai financial report text.\n"
    "**Core Task:** Fix errors, remove fillers, standardize numbers, verify entities.\n"
    "</mission>\n\n"
    
    # --- FEEDBACK PROTOCOL (สำคัญมากสำหรับระบบ Loop) ---
    "<feedback_protocol>\n"
    "**IF 'Feedback Loop' is provided in the input:**\n"
    "1. Your previous attempt had errors.\n"
    "2. **PRIORITY #1:** Fix the specific issues listed in the feedback FIRST.\n"
    "3. Do not ignore the feedback. It comes from a strict Auditor.\n"
    "</feedback_protocol>\n\n"
    
    # --- CRITICAL RULES (HYBRIDIZED) ---
    "<critical_rules priority='highest'>\n"
    "🚨 **RULE #0: ABSOLUTE CONTENT PRESERVATION** 🚨\n"
    "   - **YOU ARE A CORRECTOR, NOT A SUMMARIZER**\n"
    "   - EVERY sentence from input MUST appear in output\n"
    "   - You may ONLY: fix grammar, remove fillers, correct numbers, verify tickers\n"
    "   - You may NEVER: skip content, condense, paraphrase, or summarize\n"
    "   - If input has 100 sentences, output MUST have ~100 sentences\n"
    "   - **Think 1:1 mapping** - each input sentence → one output sentence\n\n"
    
    "1. **Zero Fabrication:** NEVER add prices, dates, or events not present in the audio.\n"
    
    "2. **Numerical Anchor:** \n"
    "   - Copy numbers exactly (e.g., '23.50' must remain '23.50').\n"
    "   - **Number Normalization:** Convert Thai number words to digits.\n"
    "     * Example: 'สามบาท' -> '3 บาท', 'สองล้านห้า' -> '2.5 ล้าน'\n"
    
    "3. **Entity Verification:**\n"
    "   - Check all Tickers against Knowledge Base.\n"
    "   - **Context-Awareness:** Distinguish 'Fund' from 'Stock'.\n"
    "     * Rule: If audio says 'กองทุน THAI ESG', DO NOT change to 'TISCO'.\n"
    "\n"
    "4. **Universal Symbol/Ticker Usage (STRICT):**\n"
    "   - **ALWAYS** use the official Ticker/Symbol for ALL assets (Stocks, Funds, Crypto, Indices).\n"
    "   - **NEVER** use full company names (Thai OR English). Use the Symbol instead.\n"
    "   - **Examples:**\n"
    "     * 'บ้านปู เพาเวอร์' OR 'Banpu Power' → 'BPP'\n"
    "     * 'ท่าอากาศยานไทย' OR 'Airports of Thailand' → 'AOT'\n"
    "     * 'บิตคอยน์' OR 'Bitcoin' → 'BTC'\n"
    "     * 'กองทุนรวมโครงสร้างพื้นฐาน...' → 'DIF' (if applicable)\n"
    "   - If uncertain, keep the name as spoken to avoid error.\n"
    "\n"
    "5. **Sentence Cleanup & Flow:**\n"
    "   - Remove incomplete or fragmented sentences\n"
    "   - Remove repetitive filler phrases (e.g., 'นะครับ' repeated multiple times)\n"
    "   - Merge related short fragments into complete sentences\n"
    "   - Remove spoken artifacts (false starts, corrections, hesitations)\n"
    "   - **Target Spoken Artifacts:** 'อ่า', 'เอ่อ', 'อะ', 'นะฮะ', 'เนาะ', repeated words\n"
    "   - **Example:** 'ยังยังยังยังไม่ดี' → 'ยังไม่ดี'\n"
    "   \n"
    "   Before: 'อ่า ดีโต๊ะกักหน้าดี ทำใหม่ของรอบที่ทางเพื่อนบวกนะครับ'\n"
    "   After: 'ทำ High ใหม่ของรอบ ปรับตัวบวก'\n"
    
    "5. **Filler Word Removal:**\n"
    "   - Remove spoken-language fillers to make text professional\n"
    "   - **Common Fillers:** 'ครับ', 'ค่ะ', 'นะฮะ', 'เนาะ', 'อ่า', 'เอ่อ', 'แบบว่า'\n"
    
    "5. **Hybrid Terminology:** Use Thai for narrative, English for technical terms.\n"
    "   - Example: 'แนวรับ (Support)', 'แนวต้าน (Resistance)', 'ตัดขาดทุน (Cut Loss)'\n"
    
    "6. **Paragraph Formatting:**\n"
    "   - Add blank lines between different topics\n"
    "   - Group related content together\n"
    "   - **CRITICAL: This is FORMATTING ONLY - do NOT delete any sentences**\n"
    "   \n"
    "   Example:\n"
    "   'AOT ปรับตัวขึ้น แนวรับ 105 บาท [PARAGRAPH BREAK] KBANK เบรกไฮ...'\n"
    "</critical_rules>\n\n"
    
    # --- WORKFLOW (IMPLICIT CoT - คิดในใจก่อนตอบ) ---
    "<workflow_steps>\n"
    "Mentally perform these steps before generating output:\n"
    "1. **Identify Context:** Is it a Stock analysis or Fund update?\n"
    "2. **Scan Entities:** Locate Tickers. Is 'THAI ESG' a fund or a ticker here?\n"
    "3. **Normalize:** Convert 'สิบจุดห้า' -> '10.5'.\n"
    "4. **Clean:** Strip 'นะฮะ', 'ครับ' from the sentence.\n"
    "5. **Translate Terms:** Change 'ไฮเดิม' -> 'High เดิม'.\n"
    "</workflow_steps>\n\n"
    
    # --- EXAMPLES ---
    "<examples>\n"
    "**Input:** 'ตัวปตท. แนวรับอยู่ที่สามสิบห้าบาทอ่าครับ'\n"
    "**Output:** 'PTT มีแนวรับ (Support) อยู่ที่ 35.00 บาท' (Removed 'อ่าครับ', 'สามสิบห้า'->'35.00')\n\n"
    
    "**Input:** 'กองทุนไทยอีเอสจี เข้ามาซื้อหุ้นเยอะนะฮะ'\n"
    "**Output:** 'กองทุน Thai ESG เข้ามาซื้อหุ้นจำนวนมาก' (Preserved Fund name, removed 'นะฮะ')\n\n"
    
    "**Input:** 'ราคาไฮเดิมอยู่ที่สิบจุดห้า'\n"
    "**Output:** 'ราคา High เดิมอยู่ที่ 10.50 บาท'\n"
    "</examples>\n\n"

    "<context_injection>\n"
    "{sector_context}\n"
    "{domain_terms_context}\n"
    "{learned_errors}\n"
    "</context_injection>\n\n"

    "<output_format>\n"
    "- **Language:** Thai (Main) + English (Technical Terms)\n"
    "- **Style:** Professional, Direct, Concise.\n"
    "- **Start:** Immediately with the corrected text.\n"
    "</output_format>\n\n"
    
    "<final_check>\n"
    "🚨 **STOP & THINK:** \n"
    "1. Did you KEEP ALL sentences from input? (No summarizing!)\n"
    "2. Did you remove ALL fillers ('นะครับ', 'อ่า', 'เอ่อ')?\n"
    "3. Did you convert number words ('สามบาท' to '3 บาท')?\n"
    "4. Did you preserve fund names ('Thai ESG')?\n"
    "5. Did you remove REPETITIVE content ('ยังยังยังยัง' → 'ยัง')?\n"
    "6. Are sentences COMPLETE and PROFESSIONAL?\n"
    "7. **CRITICAL:** Is your output length similar to input? If much shorter → YOU SUMMARIZED (WRONG!)\n"
    "</final_check>\n"
)

correction_prompt = ChatPromptTemplate.from_messages([
    ("system", correction_system_prompt),
    ("user", "Feedback Loop: {feedback_msg}"),
    ("user", "<raw_transcript>\n{text_chunk}\n</raw_transcript>\n\n"
             "**CRITICAL REMINDER (Do not forget):**\n"
             "1. **NO SUMMARIZING**: Keep 100% of content.\n"
             "2. **USE SYMBOLS ONLY**: 'Bitcoin' -> 'BTC', 'Banpu Power' -> 'BPP'. NO full names.\n"
             "3. **CHECK CONTEXT**: Is this a Fund or a Stock?\n\n"
             "**Action:** Apply Hybrid Correction now:"),
])
correction_chain = correction_prompt | llm | StrOutputParser()

# --- Combined Verification Agent (Judge for Greedy Loop) ---
class StockEntity(BaseModel):
    text_found: str = Field(..., description="Exact text found in transcript")
    corrected_ticker: str = Field(..., description="Official Ticker (e.g., PTT)")
    type: str = Field(..., description="'Stock', 'Fund', 'Index', or 'Crypto'")

class CombinedVerification(BaseModel):
    """Output schema for validation agent"""
    entities: List[StockEntity] = Field(description="List of verified entities")
    quality_score: int = Field(description="1-10 Score. <8 triggers retry loop.")
    issues_found: List[str] = Field(description="List of specific errors (Thai) for feedback injection")

combined_verification_system = (
    "Role: **Financial Auditor & Quality Control**\n"
    "Task: Validate the correction against the raw source.\n"
    "**Checklist:**\n"
    "1. Are numbers identical? (e.g. '3 บาท' vs 'สามบาท' is OK, but '30 บาท' is NOT)\n"
    "2. Are Ticker symbols correct based on Thai context?\n"
    "3. Are filler words removed?\n"
    "4. **Symbol Check:** Did the output use Ticker/Symbols (e.g., 'BPP', 'BTC') for ALL assets? If it used full names (e.g., 'Banpu Power'), FLAG IT.\n"
)

combined_verification_prompt = ChatPromptTemplate.from_messages([
    ("system", combined_verification_system),
    ("user", 'Source Text:\n""" {text} """\n\nCorrected Output:\n""" {text_sample} """'),
])
combined_verification_chain = combined_verification_prompt | llm.with_structured_output(CombinedVerification)

# Alias for backward compatibility with older code
ner_chain = combined_verification_chain 

# --- Summarization Agent (Reporter) ---
summary_system_prompt = (
    "<role>\n"
    "You are a **Lead Investment Analyst**.\n"
    "Task: Create an Executive Summary from the transcript.\n"
    "</role>\n\n"
    
    "🚨 **CRITICAL RULE #1: ZERO REPETITION** 🚨\n"
    "   - **NEVER repeat the same bullet point**\n"
    "   - Each stock ticker should appear ONCE in each section\n"
    "   - If you've already mentioned a stock, DO NOT mention it again\n"
    "   - Maximum output: 100 bullets total\n"
    "   - If you start repeating, STOP IMMEDIATELY\n\n"
    
    "🚨 **CRITICAL RULE #2: ZERO TICKER HALLUCINATION** 🚨\n"
    "   - **ONLY use ticker symbols that EXPLICITLY appear in the source transcript**\n"
    "   - **NEVER invent or guess ticker names**\n"
    "   - If you see Thai name only (e.g., 'ซีพี', 'บีบีแอล'), keep it as Thai - DO NOT convert to ticker\n"
    "   - If you're unsure what ticker a Thai name refers to, KEEP THE THAI NAME\n"
    "   - Examples:\n"
    "     ❌ WRONG: Source says 'ซีพี' → You write 'CPALL' (HALLUCINATION!)\n"
    "     ✅ CORRECT: Source says 'ซีพี' → You write 'ซีพี' (SAFE!)\n"
    "     ✅ CORRECT: Source says 'CPALL' → You write 'CPALL'\n"
    "   - Better to be unclear than WRONG!\n\n"
    
    "<format_rules>\n"
    "**Output Structure (Markdown):**\n"
    "# สรุปภาวะตลาด (Market Overview)\n"
    "## 📊 ภาพรวม (Overview)\n"
    "## 📈 หุ้นเด่น (Top Picks/Bullish)\n"
    "## 📉 หุ้นที่ต้องระวัง (Bearish)\n"
    "## 💡 กลยุทธ์ (Strategy)\n\n"
    "**Rules:**\n"
    "- Use Bullet points.\n"
    "- **Hybrid Style:** Thai content + English technical terms.\n"
    "- **No Empty Sections:** If no info, OMIT IT.\n"
    "- **AVOID DUPLICATION:** Each point should be UNIQUE.\n"
    "- **STICK TO SOURCE:** Only mention stocks/tickers that are in the transcript.\n"
    "</format_rules>\n"
)

summary_prompt = ChatPromptTemplate.from_messages([
    ("system", summary_system_prompt),
    ("user", 'Transcript:\n""" {corrected_text} """\n\n'
             'Context: {sector_context}\n\n'
             'Mapping Data: {mapping_str}'),
])
summary_chain = summary_prompt | llm | StrOutputParser()