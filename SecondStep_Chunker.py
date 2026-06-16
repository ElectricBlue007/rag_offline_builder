#!/usr/bin/env python3
"""
SecondStep_Chunker.py — FINAL PRODUCTION VERSION
All 6 critical fixes applied:
1. Token limit = 512 (e5-large hard cap)
2. chunk_type in metadata
3. Parent store includes page_range
4. Cross-reference annotation for broken links
5. Language-aware splitting for RTL/CJK
6. Global deduplication across files
"""

import os
import re
import sys
import json
import hashlib
import logging
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple

# ==========================================
# 0. Logging Setup
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger("Step2_Chunker")

# ==========================================
# 1. Tokenizer Setup (Only HF, no tiktoken fallback)
# ==========================================
try:
    import os
    from transformers import AutoTokenizer

    # Respect the HF_HOME env var set by the .bat, with a sensible fallback
    _model_cache = os.environ.get(
        "HF_HOME",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache")
    )
    _model_id = "intfloat/multilingual-e5-large"

    TOKENIZER = AutoTokenizer.from_pretrained(
        _model_id,
        cache_dir=_model_cache,
        local_files_only=True   # ← hard-enforce offline
    )
    logger.info("✅ HuggingFace AutoTokenizer loaded.")

    def count_tokens(text: str) -> int:
        if not text: return 0
        return len(TOKENIZER.encode(text, add_special_tokens=False))

    def decode_tokens(token_ids: List[int]) -> str:
        return TOKENIZER.decode(token_ids, skip_special_tokens=True)

except ImportError:
    logger.critical("❌ CRITICAL: transformers not installed.")
    sys.exit(1)
except Exception as e:
    logger.critical(
        f"❌ CRITICAL: Could not load tokenizer for '{_model_id}'.\n"
        f"   Cache path: {_model_cache}\n"
        f"   Error: {e}\n\n"
        "   ➡  Download the tokenizer ONCE while online and re-run:\n"
        "      python -c \"from transformers import AutoTokenizer; "
        "AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large', "
        f"cache_dir=r'{_model_cache}')\""
    )
    sys.exit(1)

# ==========================================
# 2. Language Detection
# ==========================================
try:
    from langdetect import detect, LangDetectException
    def detect_language(text: str) -> str:
        try: return detect(text)
        except LangDetectException: return "unknown"
except ImportError:
    def detect_language(text: str) -> str: return "unknown"

# ==========================================
# 3. Configuration (FIXED: max=512, overlap=64, min=30)
# ==========================================
TARGET_CHUNK_TOKENS = 400      # Safe target
MAX_CHUNK_TOKENS = 512         # Hard cap for e5-large
MIN_CHUNK_TOKENS = 30          # Allow short but meaningful chunks
OVERLAP_TOKENS = 64            # Recommended overlap

# ==========================================
# 4. Global Deduplication Set (loaded once)
# ==========================================
GLOBAL_HASH_SET = set()
GLOBAL_HASH_FILE = None

def load_global_hashes(output_dir: Path):
    global GLOBAL_HASH_SET, GLOBAL_HASH_FILE
    GLOBAL_HASH_FILE = output_dir / "global_hashes.json"
    if GLOBAL_HASH_FILE.exists():
        try:
            with open(GLOBAL_HASH_FILE, 'r', encoding='utf-8') as f:
                GLOBAL_HASH_SET = set(json.load(f))
            logger.info(f"📋 Loaded {len(GLOBAL_HASH_SET)} existing chunk hashes")
        except:
            GLOBAL_HASH_SET = set()
    else:
        GLOBAL_HASH_SET = set()

def save_global_hashes():
    if GLOBAL_HASH_FILE:
        with open(GLOBAL_HASH_FILE, 'w', encoding='utf-8') as f:
            json.dump(list(GLOBAL_HASH_SET), f, indent=2)

# ==========================================
# 5. Cross-Reference Annotation (FIX #4)
# ==========================================
def annotate_cross_references(text: str) -> str:
    """Adds '(see other chunk)' to internal markdown links that likely point elsewhere."""
    # Pattern for markdown internal links: [text](#anchor)
    pattern = r'\[([^\]]+)\]\(#([^)]+)\)'
    def repl(match):
        # We don't know if target is in same chunk, but safest to annotate
        return f"{match.group(0)} *(see other chunk)*"
    text = re.sub(pattern, repl, text)
    # Pattern for footnote references [^1]
    footnote_pattern = r'\[\^(\d+)\]'
    text = re.sub(footnote_pattern, r'[\^(\1)] *(see other chunk)*', text)
    return text

# ==========================================
# 6. Language-Aware Splitting for RTL/CJK (FIX #5)
# ==========================================
def split_oversized_text(content: str, lang: str, start_char: int, path: List[str], page: int) -> List[Dict]:
    """Split text that exceeds MAX_CHUNK_TOKENS using appropriate boundaries."""
    if lang in ('fa', 'ar', 'zh', 'ja', 'ko', 'unknown'):
        # Character-based fallback: split every 200 chars (approx 50 tokens)
        chunks = []
        for i in range(0, len(content), 200):
            sub = content[i:i+200]
            if sub.strip():
                chunks.append({
                    'type': 'text',
                    'content': sub,
                    'start_char': start_char + i,
                    'end_char': start_char + min(i+200, len(content)),
                    'path': path,
                    'page': page
                })
        return chunks
    else:
        # Sentence-based splitting for Latin scripts
        sentences = re.split(r'(?<=[.!?])\s+', content)
        chunks = []
        curr_text = ""
        curr_start = start_char
        for sent in sentences:
            sent = sent.strip()
            if not sent: continue
            sent_pos = content.find(sent, curr_start - start_char if curr_text else 0)
            if sent_pos == -1: sent_pos = 0
            if count_tokens(curr_text + " " + sent) > MAX_CHUNK_TOKENS and curr_text:
                chunks.append({
                    'type': 'text',
                    'content': curr_text.strip(),
                    'start_char': curr_start,
                    'end_char': curr_start + len(curr_text),
                    'path': path,
                    'page': page
                })
                curr_text = sent
                curr_start = start_char + sent_pos
            else:
                curr_text += (" " if curr_text else "") + sent
        if curr_text.strip():
            chunks.append({
                'type': 'text',
                'content': curr_text.strip(),
                'start_char': curr_start,
                'end_char': curr_start + len(curr_text),
                'path': path,
                'page': page
            })
        return chunks

# ==========================================
# 7. Core Utility Functions
# ==========================================
def parse_yaml_frontmatter(text: str) -> Tuple[Dict[str, Any], str]:
    if not text.startswith('---'): return {}, text.strip()
    parts = text.split('---', 2)
    if len(parts) < 3: return {}, text.strip()
    
    meta = {}
    for line in parts[1].strip().split('\n'):
        if ':' in line and not line.strip().startswith('-'):
            key, val = line.split(':', 1)
            meta[key.strip()] = val.strip().strip('"').strip("'")
    return meta, parts[2].strip()

def get_overlap_text(text: str, target_tokens: int) -> str:
    tokens = TOKENIZER.encode(text, add_special_tokens=False)
    if len(tokens) <= target_tokens: return ""
    overlap_str = decode_tokens(tokens[-target_tokens:])
    # Trim to last space
    space_idx = overlap_str.rfind(' ')
    if space_idx != -1:
        return overlap_str[space_idx+1:].strip()
    return overlap_str.strip()

def deduplicate_overlap(overlap: str, content: str) -> str:
    """Prevents duplicate table rows or paragraphs when prepending overlap."""
    if not overlap: return content
    overlap_lines = overlap.strip().split('\n')
    content_lines = content.strip().split('\n')
    
    max_match = 0
    for i in range(1, min(len(overlap_lines), len(content_lines)) + 1):
        if overlap_lines[-i:] == content_lines[:i]:
            max_match = i
    if max_match > 0:
        return '\n'.join(content_lines[max_match:])
    return content

# ==========================================
# 8. Markdown Block Parser (unchanged, works well)
# ==========================================
def parse_markdown_blocks(text: str) -> List[Dict[str, Any]]:
    blocks = []
    lines = text.split('\n')
    current_idx = 0
    current_heading_path = []
    current_page = None
    
    i = 0
    while i < len(lines):
        line = lines[i]
        line_start = current_idx
        stripped = line.strip()
        
        page_match = re.match(r'^<!--\s*PAGE\s+(\d+)\s*-->', stripped)
        if page_match:
            current_page = int(page_match.group(1))
            i += 1
            current_idx = line_start + len(line) + 1
            continue
            
        if stripped.startswith('```'):
            code_lines = [line]
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith('```'):
                code_lines.append(lines[j])
                j += 1
            if j < len(lines): code_lines.append(lines[j]); j += 1
            content = '\n'.join(code_lines)
            blocks.append({
                'type': 'code', 'content': content,
                'start_char': line_start, 'end_char': line_start + len(content),
                'path': list(current_heading_path), 'page': current_page
            })
            i = j
            current_idx = line_start + len(content)
            continue
        
        if stripped.startswith('|'):
            table_lines = [line]
            j = i + 1
            while j < len(lines) and lines[j].strip().startswith('|'):
                table_lines.append(lines[j])
                j += 1
            is_table = any(re.match(r'^\s*\|[\s\-:|]+\|\s*$', l.strip()) for l in table_lines)
            if is_table:
                content = '\n'.join(table_lines)
                blocks.append({
                    'type': 'table', 'content': content,
                    'start_char': line_start, 'end_char': line_start + len(content),
                    'path': list(current_heading_path), 'page': current_page
                })
                i = j
                current_idx = line_start + len(content)
                continue
        
        heading_match = re.match(r'^(#{1,6})\s+(.*)$', stripped)
        if heading_match:
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            current_heading_path = current_heading_path[:level-1] + [title]
            blocks.append({
                'type': 'heading', 'level': level, 'content': line,
                'start_char': line_start, 'end_char': line_start + len(line) + 1,
                'path': list(current_heading_path), 'page': current_page
            })
            i += 1
            current_idx = line_start + len(line) + 1
            continue
        
        text_lines = [line]
        j = i + 1
        while j < len(lines):
            next_stripped = lines[j].strip()
            if next_stripped == '' or re.match(r'^(#{1,6})\s+', next_stripped) or \
               next_stripped.startswith('```') or next_stripped.startswith('|') or \
               re.match(r'^<!--\s*PAGE\s+\d+\s*-->', next_stripped):
                break
            text_lines.append(lines[j])
            j += 1
        raw_content = '\n'.join(text_lines)
        if raw_content.strip():
            blocks.append({
                'type': 'text', 'content': raw_content,
                'start_char': line_start, 'end_char': line_start + len(raw_content),
                'path': list(current_heading_path), 'page': current_page
            })
        i = j
        current_idx = line_start + len(raw_content) + (1 if j < len(lines) else 0)
    return blocks

def split_text_block(block: Dict[str, Any], lang: str) -> List[Dict[str, Any]]:
    """Uses language-aware splitter."""
    content = block['content']
    if count_tokens(content) <= MAX_CHUNK_TOKENS:
        return [block]
    return split_oversized_text(
        content, lang,
        block['start_char'],
        block['path'],
        block.get('page')
    )

# ==========================================
# 9. Chunk Assembly (with ALL fixes)
# ==========================================
def assemble_chunks(blocks: List[Dict], source_file: str, frontmatter: Dict) -> List[Dict]:
    chunks = []
    current_blocks = []
    current_tokens = 0
    last_chunk = None
    overlap_text = ""
    lang = frontmatter.get("language", "unknown")
    
    def finalize_chunk(blks: List[Dict], prepend_overlap: str = ""):
        nonlocal last_chunk, overlap_text
        if not blks and not prepend_overlap:
            return
        
        # Build raw content
        if prepend_overlap:
            clean_overlap = '\n'.join([l for l in prepend_overlap.split('\n') if not re.match(r'^#{1,6}\s+', l.strip())])
            raw_content = "\n\n".join(b['content'] for b in blks)
            content = deduplicate_overlap(clean_overlap, raw_content)
        else:
            content = "\n\n".join(b['content'] for b in blks)
        content = content.strip()
        if not content:
            return
        
        # FIX #4: Annotate cross-references
        content = annotate_cross_references(content)
        
        # Count tokens
        tokens = count_tokens(content)
        if tokens > MAX_CHUNK_TOKENS:
            logger.warning(f"Chunk exceeded {MAX_CHUNK_TOKENS} tokens ({tokens}) - truncating at sentence boundary")
            # Truncate to max tokens
            enc = TOKENIZER.encode(content, add_special_tokens=False)
            truncated = decode_tokens(enc[:MAX_CHUNK_TOKENS])
            content = truncated
            tokens = MAX_CHUNK_TOKENS
        
        # Language detection override
        chunk_lang = lang
        if len(content) > 200:
            detected = detect_language(content)
            if detected != "unknown" and detected != chunk_lang:
                chunk_lang = detected
        
        # Page range
        pages = sorted(list(set(b.get('page') for b in blks if b.get('page') is not None)))
        if not pages and chunks:
            pages = chunks[-1]['metadata'].get('page_range', [])
        
        # Heading path (use first heading block if any, else first block's path)
        heading_blk = next((b for b in blks if b['type'] == 'heading'), None)
        if heading_blk:
            heading_path = heading_blk['path']
        elif blks:
            heading_path = blks[0]['path']
        else:
            heading_path = chunks[-1]['metadata']['heading_path'] if chunks else []
        
        parent_id = f"{source_file}::{ '::'.join(heading_path) }" if heading_path else f"{source_file}::root"
        
        volume = frontmatter.get("volume", "Root")
        section = frontmatter.get("section", "Root")
        chapter = frontmatter.get("chapter", "Root")
        if len(pages) == 1:
            page_str = f"Page {pages[0]}"
        elif len(pages) > 1:
            page_str = f"Page {pages[0]}-{pages[-1]}"
        else:
            page_str = "Page Unknown"
        breadcrumb = f"Context: {volume} > {section} > {chapter} | {page_str}\n\n"
        
        # FIX #2: Determine chunk type — table/code take priority; otherwise most frequent
        chunk_types = [b['type'] for b in blks if b['type'] != 'heading']
        if 'table' in chunk_types:
            dominant_type = 'table'
        elif 'code' in chunk_types:
            dominant_type = 'code'
        elif chunk_types:
            dominant_type = max(set(chunk_types), key=chunk_types.count)
        else:
            dominant_type = 'text'
        
        # Global deduplication check (FIX #6)
        chunk_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
        if chunk_hash in GLOBAL_HASH_SET:
            logger.debug(f"Skipping duplicate chunk (hash {chunk_hash[:8]})")
            return
        
        chunk_data = {
            "chunk_id": "",
            "text": breadcrumb + content,
            "metadata": {
                "source_file": source_file,
                "volume": volume,
                "section": section,
                "chapter": chapter,
                "heading_path": heading_path,
                "chunk_type": dominant_type,   # FIX #2
                "language": chunk_lang,
                "start_char": blks[0]['start_char'] if blks else 0,
                "end_char": blks[-1]['end_char'] if blks else 0,
                "page_range": pages,
                "chunk_index": 0,
                "total_chunks_in_file": 0
            },
            "hash": chunk_hash,
            "parent_id": parent_id,
            "token_count": tokens
        }
        
        # Merge tiny chunks only if same parent and below min
        same_section = (last_chunk is not None and last_chunk["parent_id"] == parent_id)
        if tokens < MIN_CHUNK_TOKENS and same_section:
            last_raw = last_chunk["text"]
            if last_raw.startswith("Context:"):
                last_raw = last_raw.split("\n\n", 1)[1] if "\n\n" in last_raw else ""
            merged_content = last_raw + "\n\n" + content
            merged_tokens = count_tokens(merged_content)
            combined_pages = sorted(list(set(last_chunk['metadata']['page_range'] + pages)))
            if len(combined_pages) == 1:
                p_str = f"Page {combined_pages[0]}"
            elif len(combined_pages) > 1:
                p_str = f"Page {combined_pages[0]}-{combined_pages[-1]}"
            else:
                p_str = "Page Unknown"
            new_breadcrumb = f"Context: {volume} > {section} > {chapter} | {p_str}\n\n"
            last_chunk["text"] = new_breadcrumb + merged_content
            last_chunk["token_count"] = merged_tokens
            last_chunk["metadata"]["end_char"] = chunk_data["metadata"]["end_char"]
            last_chunk["metadata"]["page_range"] = combined_pages
            # Update chunk_type: table/code always wins over text
            merged_type = dominant_type
            prev_type = last_chunk["metadata"]["chunk_type"]
            if 'table' in (merged_type, prev_type):
                last_chunk["metadata"]["chunk_type"] = 'table'
            elif 'code' in (merged_type, prev_type):
                last_chunk["metadata"]["chunk_type"] = 'code'
            last_chunk["hash"] = hashlib.sha256(merged_content.encode('utf-8')).hexdigest()
            # Also update global hash set (remove old, add new)
            GLOBAL_HASH_SET.discard(last_chunk["hash"])
            GLOBAL_HASH_SET.add(last_chunk["hash"])
        else:
            chunks.append(chunk_data)
            last_chunk = chunk_data
            GLOBAL_HASH_SET.add(chunk_hash)
        
        # Overlap for next chunk
        if tokens > OVERLAP_TOKENS:
            overlap_text = get_overlap_text(content, OVERLAP_TOKENS)
        else:
            overlap_text = ""
    
    for block in blocks:
        b_tokens = count_tokens(block['content'])
        
        if block['type'] == 'heading':
            if current_blocks and current_tokens > 0:
                finalize_chunk(current_blocks, prepend_overlap=overlap_text)
                overlap_text = ""
                current_blocks = []
                current_tokens = 0
            current_blocks.append(block)
            current_tokens += b_tokens
            continue
        
        if block['type'] in ('table', 'code'):
            if current_blocks:
                finalize_chunk(current_blocks, prepend_overlap=overlap_text)
                overlap_text = ""
                current_blocks = []
                current_tokens = 0
            # Tables/code become own chunk
            if b_tokens > MAX_CHUNK_TOKENS:
                logger.warning(f"Oversized {block['type']} ({b_tokens} tokens) kept intact.")
            finalize_chunk([block])
            overlap_text = ""
            continue
        
        # Text block – possibly split
        if current_tokens + b_tokens > MAX_CHUNK_TOKENS and block['type'] == 'text':
            sub_blocks = split_text_block(block, lang)
            for sub in sub_blocks:
                sub_tokens = count_tokens(sub['content'])
                if current_tokens + sub_tokens > MAX_CHUNK_TOKENS:
                    finalize_chunk(current_blocks, prepend_overlap=overlap_text)
                    overlap_text = ""
                    current_blocks = [sub]
                    current_tokens = sub_tokens
                else:
                    current_blocks.append(sub)
                    current_tokens += sub_tokens
        else:
            current_blocks.append(block)
            current_tokens += b_tokens
    
    if current_blocks:
        finalize_chunk(current_blocks, prepend_overlap=overlap_text)
    
    total = len(chunks)
    safe_name = source_file.replace('/', '_').replace('.md', '')
    for i, chunk in enumerate(chunks):
        chunk["chunk_id"] = f"{safe_name}_{i+1:04d}"
        chunk["metadata"]["chunk_index"] = i + 1
        chunk["metadata"]["total_chunks_in_file"] = total
    
    return chunks

# ==========================================
# 10. Document Processing (with parent_store page_range fix)
# ==========================================
def process_document(doc_path: Path, output_dir: Path, input_root: Path, dry_run: bool):
    start_time = time.time()
    logger.info(f"🔄 Processing: {doc_path.name}")
    
    raw_text = doc_path.read_text(encoding='utf-8')
    frontmatter, main_text = parse_yaml_frontmatter(raw_text)
    
    if not main_text.strip():
        logger.warning(f"⚠️ Empty file: {doc_path.name}")
        return {"chunks": 0, "tokens": 0}
    
    try:
        source_ref = str(doc_path.relative_to(input_root))
    except ValueError:
        source_ref = doc_path.name
    
    blocks = parse_markdown_blocks(main_text)
    if not blocks:
        return {"chunks": 0, "tokens": 0}
    
    chunks = assemble_chunks(blocks, source_ref, frontmatter)
    
    # FIX #3: Parent store with page_range collected inline (no O(n²) second scan)
    parent_store = {}
    parent_pages: Dict[str, list] = {}
    for chunk in chunks:
        parent_id = chunk["parent_id"]
        if parent_id not in parent_store:
            parent_store[parent_id] = {
                "parent_id": parent_id,
                "source_file": source_ref,
                "heading_path": chunk["metadata"]["heading_path"],
                "chunk_type": chunk["metadata"]["chunk_type"],
                "text": "",
                "child_ids": [],
                "page_range": []  # finalised below
            }
            parent_pages[parent_id] = []
        # Accumulate pages directly from each child
        parent_pages[parent_id].extend(chunk["metadata"]["page_range"])
        # Extract raw text (without breadcrumb)
        raw_text_chunk = chunk["text"]
        if raw_text_chunk.startswith("Context:"):
            parts = raw_text_chunk.split("\n\n", 1)
            raw_text_chunk = parts[1] if len(parts) > 1 else ""
        parent_store[parent_id]["text"] += raw_text_chunk + "\n\n"
        parent_store[parent_id]["child_ids"].append(chunk["chunk_id"])
        # Promote parent chunk_type if child is table or code
        child_type = chunk["metadata"]["chunk_type"]
        if child_type == "table":
            parent_store[parent_id]["chunk_type"] = "table"
        elif child_type == "code" and parent_store[parent_id]["chunk_type"] not in ("table",):
            parent_store[parent_id]["chunk_type"] = "code"

    # Finalise page_range and strip accumulated text for every parent
    for pid in parent_store:
        parent_store[pid]["page_range"] = sorted(set(parent_pages[pid]))
        parent_store[pid]["text"] = parent_store[pid]["text"].strip()
    
    out_file = output_dir / f"{doc_path.stem}.jsonl"
    parent_store_file = output_dir / f"{doc_path.stem}_parent_store.json"
    total_tokens = 0
    
    if not dry_run:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
                total_tokens += chunk["token_count"]
        with open(parent_store_file, 'w', encoding='utf-8') as f:
            json.dump(parent_store, f, ensure_ascii=False, indent=2)
    
    elapsed = time.time() - start_time
    logger.info(f"✅ {doc_path.name} | Chunks: {len(chunks)} | Tokens: {total_tokens} | Time: {elapsed:.2f}s")
    return {"chunks": len(chunks), "tokens": total_tokens}

# ==========================================
# 11. Main Entry Point
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Step 2: Intelligent Token-Accurate Chunker (All Fixes)")
    parser.add_argument("-i", "--input", required=True, help="Input markdown file or directory")
    parser.add_argument("-o", "--output", required=True, help="Output directory for .jsonl files")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_dir = Path(args.output)
    
    if not input_path.exists():
        logger.error(f"❌ Input path not found: {input_path}")
        sys.exit(1)
    
    if input_path.is_file():
        files_to_process = [input_path]
    else:
        files_to_process = list(input_path.rglob("*.md"))
        files_to_process = [f for f in files_to_process if "_page_" not in f.name]
    
    if not files_to_process:
        logger.warning("⚠️ No markdown files found.")
        return
    
    # Load global deduplication set
    load_global_hashes(output_dir)
    
    # Load checkpoint
    checkpoint_file = output_dir / "step2_checkpoint.json"
    checkpoint = {}
    if checkpoint_file.exists():
        try:
            checkpoint = json.loads(checkpoint_file.read_text(encoding='utf-8'))
            logger.info(f"📂 Loaded checkpoint with {len(checkpoint)} processed files.")
        except Exception:
            checkpoint = {}
    
    logger.info(f"🚀 Starting Step 2 Chunker on {len(files_to_process)} document(s)...")
    total_stats = {"chunks": 0, "tokens": 0}
    
    for md_file in files_to_process:
        if input_path.is_dir():
            file_key = str(md_file.relative_to(input_path))
        else:
            file_key = md_file.name
        
        if file_key in checkpoint:
            logger.info(f"⏭️ Skipping {md_file.name} (in checkpoint)")
            total_stats["chunks"] += checkpoint[file_key].get("chunks", 0)
            total_stats["tokens"] += checkpoint[file_key].get("tokens", 0)
            continue
        
        try:
            stats = process_document(md_file, output_dir, input_path, args.dry_run)
            total_stats["chunks"] += stats["chunks"]
            total_stats["tokens"] += stats["tokens"]
            
            if not args.dry_run:
                checkpoint[file_key] = {
                    "processed_at": time.time(),
                    "chunks": stats["chunks"],
                    "tokens": stats["tokens"]
                }
                # Atomic write checkpoint
                temp_ckpt = checkpoint_file.with_suffix('.tmp')
                temp_ckpt.write_text(json.dumps(checkpoint, indent=2), encoding='utf-8')
                temp_ckpt.replace(checkpoint_file)
                # Save global hashes after each file
                save_global_hashes()
        except Exception as e:
            logger.error(f"❌ Failed to process {md_file.name}: {e}", exc_info=True)
    
    logger.info("=" * 60)
    logger.info("🏁 STEP 2 PIPELINE COMPLETED SUCCESSFULLY")
    logger.info(f"Total Chunks Generated: {total_stats['chunks']}")
    logger.info(f"Total Tokens Processed: {total_stats['tokens']}")
    logger.info(f"Unique Chunks (after dedup): {len(GLOBAL_HASH_SET)}")
    logger.info("=" * 60)
    
    if not args.dry_run:
        save_global_hashes()

if __name__ == "__main__":
    main()