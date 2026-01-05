import time
import random
from config import MAX_RETRIES, LLM_REQUEST_DELAY_SECONDS, TEXT_CHUNK_SIZE
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser

def process_large_text_robust(text, chain, chunk_size=TEXT_CHUNK_SIZE, rate_limiter=None, extra_inputs=None):
    """
    Split text into chunks and process each with retry logic + Rate Limiter.
    
    OPTIMIZATION: Apply smart cache before LLM processing
    """
    if extra_inputs is None: extra_inputs = {}
    
    # OPTIMIZATION: Pre-process with smart cache
    from smart_cache import get_smart_cache
    cache = get_smart_cache()
    text, cache_stats = cache.process(text)
    
    if cache_stats["total_replacements"] > 0:
        print(f"   🎯 Smart Cache: {cache_stats['total_replacements']} instant corrections applied")
    
    print(f"   ... Splitting text (Size: {len(text)}, Chunk: {chunk_size})...")
    
    # 1. Split Text
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=200)
    chunks = splitter.create_documents([text])
    
    chunks = [d.page_content for d in chunks]
    total_chunks = len(chunks)
    processed_parts = []

    print(f"   ... Processing {total_chunks} chunks sequentially (Delay: {LLM_REQUEST_DELAY_SECONDS}s, Smart Mode: {rate_limiter is not None})...")

    # 2. Process each chunk
    for chunk_index, input_text in enumerate(chunks):
        
        # [CHECK] Budget
        if rate_limiter and not rate_limiter.can_make_request():
            # IMPORTANT: Fallback to original text to preserve context for Summary
            print(f"      ⛔ Budget Exhausted! Skipping LLM for chunk {chunk_index+1}/{total_chunks}. Keeping original text.")
            processed_parts.append(input_text)
            continue

        chunk_success = False
        
        for attempt in range(MAX_RETRIES):
            try:
                # [RECORD] Request start
                if rate_limiter:
                    rate_limiter.record_request()
                else:
                    # Legacy fixed delay for backward compatibility
                    if chunk_index > 0 or attempt > 0:
                        time.sleep(LLM_REQUEST_DELAY_SECONDS)

                # Merge chunk with extra context variables
                input_args = {"text_chunk": input_text}
                input_args.update(extra_inputs)
                
                res = chain.invoke(input_args)
                
                # Handle output type
                if hasattr(res, 'content'):
                    out_text = res.content
                else: 
                    out_text = str(res)
                
                processed_parts.append(out_text)
                chunk_success = True
                break
            
            except Exception as e:
                error_str = str(e)
                is_quota_error = ("429" in error_str or 
                                "RESOURCE_EXHAUSTED" in error_str or
                                "quota" in error_str.lower())
                
                # [BACKOFF] Smart wait
                if rate_limiter:
                    wait_time = rate_limiter.get_backoff_delay(attempt, is_quota_error)
                else:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)

                if is_quota_error:
                    # Try key rotation
                    try:
                        from key_rotation import rotate_key
                        from src.agents.llm_factory import create_llm
                        from src.agents import llm_prompts
                        
                        if rotate_key():
                            print(f"      🔄 Quota exceeded! Rotated to next API key...")
                            # Recreate LLM and chains
                            llm_prompts.llm = create_llm()
                            llm_prompts.correction_chain = llm_prompts.correction_prompt | llm_prompts.llm | llm_prompts.StrOutputParser()
                            continue  # Retry immediately with new key
                        else:
                            print(f"      ❌ All API keys exhausted!")
                    except ImportError:
                        pass # Key rotation not available

                print(f"      ⚠️ Chunk {chunk_index+1} Error (Attempt {attempt+1}/{MAX_RETRIES}): {e}. Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
        
        if not chunk_success:
            print(f"      ❌ Chunk {chunk_index+1} Failed after retries. Using original text.")
            processed_parts.append(input_text)

        # [DELAY] Smart inter-request delay
        if chunk_index < total_chunks - 1:
            if rate_limiter:
                delay = rate_limiter.get_smart_delay()
                if delay > 0:
                    print(f"      ⏳ Smart Delay: {delay:.2f}s")
                    time.sleep(delay)
            else:
                time.sleep(LLM_REQUEST_DELAY_SECONDS)

    return "\n\n".join(processed_parts)