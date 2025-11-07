"""Utilities for computing text embeddings."""

import numpy as np
from typing import List, Optional, Dict
import concurrent.futures
from tqdm.auto import tqdm
import tiktoken
import os
import time
from pathlib import Path
import glob
import torch
import openai
from .utils import filter_invalid_texts
from .batch import BatchRequest, OpenAIBatchExecutor, choose_backend, make_custom_id
from .llm_api import resolve_model_name
from sentence_transformers import SentenceTransformer
import torch
import gc


# Use environment variable for cache dir if set, otherwise use default
CACHE_DIR = os.getenv('EMB_CACHE_DIR') or os.path.join(Path(__file__).parent.parent, 'emb_cache')
EMBED_AUTO_BATCH_THRESHOLD = int(os.getenv('HYPOTHESAES_EMBED_BATCH_THRESHOLD', '2000'))

def _embed_batch_openai(
        batch: List[str], 
        model: str, 
        client,
        max_tokens: int = 8192, 
        max_retries: int = 3, 
        backoff_factor: float = 3.0,
        timeout: float = 10.0
) -> List[List[float]]:
    """Helper function for batch embedding using OpenAI API."""
    # Truncate texts to max tokens
    enc = tiktoken.get_encoding("cl100k_base")  # encoding for OpenAI text-embedding models
    truncated_batch = []
    for text in batch:
        tokens = enc.encode(text.strip())
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
            text = enc.decode(tokens)
        truncated_batch.append(text)
    
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(
                input=truncated_batch,
                model=model,
                timeout=timeout
            )
            return [data.embedding for data in response.data]
            
        except (openai.RateLimitError, openai.APITimeoutError) as e:
            if attempt == max_retries - 1:  # Last attempt
                raise e
            
            wait_time = timeout * (backoff_factor ** attempt)
            if attempt > 0:
                print(f"API error: {e}; retrying in {wait_time:.1f}s... ({attempt + 1}/{max_retries})")
            time.sleep(wait_time)


def load_embedding_cache(cache_name: str) -> dict:
    """Load cached embeddings from chunked files."""
    if not cache_name:
        return {}
    
    cache_dir = f"{CACHE_DIR}/{cache_name}"
    if not os.path.exists(cache_dir):
        return {}
    
    text2embedding = {}
    chunk_files = sorted(glob.glob(f"{cache_dir}/chunk_*.npy"))
    
    start_time = time.time()
    for chunk_file in tqdm(chunk_files, desc="Loading embedding chunks"):
        # Each chunk file contains a list of (text, embedding) tuples
        chunk_data = np.load(chunk_file, allow_pickle=True)
        for text, emb in chunk_data:
            text2embedding[text] = emb
            
    load_time = time.time() - start_time
    print(f"Loaded {len(text2embedding)} embeddings in {load_time:.1f}s")
            
    return text2embedding


def update_embedding_cache(cache_name: str, text2embedding: dict, chunk_size: int = 50000) -> None:
    """Update cache files in chunks."""
    if not cache_name:
        return
        
    cache_dir = f"{CACHE_DIR}/{cache_name}"
    os.makedirs(cache_dir, exist_ok=True)
    
    # Convert to list of (text, embedding) tuples for storage
    items = list(text2embedding.items())
    
    for i in range(0, len(items), chunk_size):
        chunk_items = items[i:i + chunk_size]
        chunk_num = i // chunk_size
        chunk_path = f"{cache_dir}/chunk_{chunk_num:03d}.npy"
        np.save(chunk_path, np.array(chunk_items, dtype=object))
        

def _get_next_chunk_index(cache_name: str) -> int:
    """Determine the next available chunk index for a cache."""
    if not cache_name:
        return 0
        
    cache_dir = f"{CACHE_DIR}/{cache_name}"
    if not os.path.exists(cache_dir):
        return 0
        
    chunk_files = glob.glob(f"{cache_dir}/chunk_*.npy")
    if not chunk_files:
        return 0
        
    indices = [int(os.path.basename(f).split("_")[1].split(".")[0]) for f in chunk_files]
    return max(indices) + 1
    
def _save_embedding_chunk(cache_name: str, chunk_embeddings: dict, chunk_idx: int) -> int:
    """Save a chunk of embeddings to disk."""
    if not cache_name or not chunk_embeddings:
        return chunk_idx
        
    cache_dir = f"{CACHE_DIR}/{cache_name}"
    os.makedirs(cache_dir, exist_ok=True)
    
    chunk_path = f"{cache_dir}/chunk_{chunk_idx:03d}.npy"
    chunk_items = list(chunk_embeddings.items())
    np.save(chunk_path, np.array(chunk_items, dtype=object))
    print(f"Saved {len(chunk_items)} embeddings to {chunk_path}")
    
    return chunk_idx + 1

def get_openai_embeddings(
    texts: List[str],
    model: str = "text-embedding-3-small",
    batch_size: int = 256,
    n_workers: int = 5,
    cache_name: Optional[str] = None,
    show_progress: bool = True,
    chunk_size: int = 50000,
    timeout: float = 10.0,
    backend: str = "live",
    auto_batch_threshold: int = EMBED_AUTO_BATCH_THRESHOLD,
) -> Dict[str, np.ndarray]:
    """Get embeddings using OpenAI API with parallel processing and chunked caching."""
    # Filter out None values and empty strings
    texts = filter_invalid_texts(texts)
    
    # Setup cache
    text2embedding = load_embedding_cache(cache_name)
    texts_to_embed = [text for text in texts if text not in text2embedding]
    
    if not texts_to_embed:
        return text2embedding
    
    backend_mode = choose_backend(backend, len(texts_to_embed), auto_batch_threshold)

    if backend_mode == "batch":
        batch_embeddings = _batch_embed_openai(
            texts=texts_to_embed,
            model=model,
            show_progress=show_progress,
        )
        text2embedding.update(batch_embeddings)
        if cache_name:
            next_chunk_idx = _get_next_chunk_index(cache_name)
            items = list(batch_embeddings.items())
            for i in range(0, len(items), chunk_size):
                chunk_dict = dict(items[i:i + chunk_size])
                next_chunk_idx = _save_embedding_chunk(cache_name, chunk_dict, next_chunk_idx)
        return text2embedding

    from .llm_api import get_client
    client = get_client()

    # Process in chunks
    next_chunk_idx = _get_next_chunk_index(cache_name)

    # Create chunk ranges
    chunk_ranges = [(i, min(i+chunk_size, len(texts_to_embed))) 
                   for i in range(0, len(texts_to_embed), chunk_size)]

    # Outer progress bar for chunks
    chunk_iterator = chunk_ranges
    if show_progress:
        chunk_iterator = tqdm(chunk_iterator, desc="Processing chunks", total=len(chunk_ranges))

    for chunk_start, chunk_end in chunk_iterator:
        chunk_texts = texts_to_embed[chunk_start:chunk_end]
        chunk_embeddings = {}
        
        # Process chunk in batches with parallel workers
        batches = [chunk_texts[i:i+batch_size] for i in range(0, len(chunk_texts), batch_size)]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            for batch in batches:
                futures.append(executor.submit(_embed_batch_openai, batch, model, client, timeout=timeout))
            
            # Process results as they complete
            iterator = concurrent.futures.as_completed(futures)
            if show_progress:
                iterator = tqdm(iterator, total=len(batches), desc=f"Chunk {next_chunk_idx}")
                
            for future in iterator:
                batch_result = future.result()
                batch_idx = futures.index(future)
                batch = batches[batch_idx]
                
                for text, embedding in zip(batch, batch_result):
                    chunk_embeddings[text] = embedding
                    text2embedding[text] = embedding
        
        # Save completed chunk
        next_chunk_idx = _save_embedding_chunk(cache_name, chunk_embeddings, next_chunk_idx)
    
    return text2embedding

def get_local_embeddings(
    texts: List[str],
    model: str = "nomic-ai/modernbert-embed-base",
    batch_size: int = 128,
    show_progress: bool = True,
    cache_name: Optional[str] = None,
    chunk_size: int = 50000,
    device: Optional[torch.device] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
) -> Dict[str, np.ndarray]:
    """Get embeddings using local SentenceTransformer model with chunked caching."""
    # Filter out None values and empty strings
    texts = filter_invalid_texts(texts)
    
    # Setup cache
    text2embedding = load_embedding_cache(cache_name)
    texts_to_embed = [text for text in texts if text not in text2embedding]
    
    if not texts_to_embed:
        return text2embedding
    
    # Load model
    transformer_model = SentenceTransformer(model, device=device)
    print(f"Loaded model {model} to {device}")
    
    # Process in chunks
    next_chunk_idx = _get_next_chunk_index(cache_name)
    
    # Create chunk ranges
    chunk_ranges = [(i, min(i+chunk_size, len(texts_to_embed))) 
                   for i in range(0, len(texts_to_embed), chunk_size)]
    
    # Outer progress bar for chunks
    chunk_iterator = chunk_ranges
    if show_progress:
        chunk_iterator = tqdm(chunk_iterator, desc="Processing chunks", total=len(chunk_ranges))
    
    for chunk_start, chunk_end in chunk_iterator:
        chunk_texts = texts_to_embed[chunk_start:chunk_end]
        chunk_embeddings = {}
        
        # Process chunk in batches
        batch_iterator = range(0, len(chunk_texts), batch_size)
        if show_progress:
            batch_iterator = tqdm(batch_iterator, desc=f"Chunk {next_chunk_idx}")
            
        for i in batch_iterator:
            batch = chunk_texts[i:i+batch_size]
            if "nomic-ai" in model:
                prefixed_batch = ["clustering: " + text for text in batch]
            elif "instructor" in model:
                prefixed_batch = [["Represent the text for classification: ", text] for text in batch]
            else:
                prefixed_batch = batch
            batch_embs = transformer_model.encode(prefixed_batch, batch_size=batch_size)
            
            for text, embedding in zip(batch, batch_embs):
                chunk_embeddings[text] = embedding
                text2embedding[text] = embedding
        
        # Save completed chunk
        next_chunk_idx = _save_embedding_chunk(cache_name, chunk_embeddings, next_chunk_idx)

    del transformer_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return text2embedding


def _batch_embed_openai(
    texts: List[str],
    model: str,
    show_progress: bool = False,
) -> Dict[str, np.ndarray]:
    executor = OpenAIBatchExecutor(
        endpoint="/v1/embeddings",
        task_name="embeddings",
    )
    model_id = resolve_model_name(model)
    requests: List[BatchRequest] = []
    mapping: Dict[str, str] = {}

    for text in texts:
        truncated_text = _truncate_to_tokens(text)
        custom_id = make_custom_id("embedding")
        body = {
            "model": model_id,
            "input": truncated_text,
        }
        requests.append(BatchRequest(custom_id=custom_id, url="/v1/embeddings", body=body))
        mapping[custom_id] = text

    result = executor.execute(
        requests,
        metadata={"type": "embeddings"},
        show_progress=show_progress,
        progress_desc=f"Embedding {len(texts)} texts (batch)",
    )
    embeddings: Dict[str, np.ndarray] = {}

    for custom_id, response_body in result.responses.items():
        text = mapping.get(custom_id)
        if text is None:
            continue
        data = response_body.get("data", [])
        if not data:
            continue
        vector = np.array(data[0].get("embedding", []), dtype=np.float32)
        embeddings[text] = vector

    if result.errors:
        for custom_id, error in result.errors.items():
            text = mapping.get(custom_id, "?")
            print(f"Batch embedding error for text '{text[:40]}...': {error}")

    return embeddings


def _truncate_to_tokens(text: str, max_tokens: int = 8192) -> str:
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text.strip())
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens])
