#!/usr/bin/env python3
"""
Send extracted English texts from Thief Simulator 2 to Inference API for translation into Indonesian.

Uses the https://us.inference.heroku.com/v1/chat/completions endpoint with claude-4-sonnet model.
Processes texts in batches to handle large files (e.g., 500 KB input).
Saves translated texts in the same JSON structure for reintegration.

Features:
- Batch processing with configurable batch size
- Enhanced error handling with response logging
- Standard POST request handling
- Progress tracking with tqdm
"""

import json
import requests
import argparse
import logging
import sys
import time
import os
import re
from typing import List, Dict, Any
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Configure logging with rotation to prevent large log files
from logging.handlers import RotatingFileHandler

logging.basicConfig(
    level=logging.DEBUG,  # Enable debug logging
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler('translation_api_errors.log', maxBytes=1024*1024, backupCount=2),  # 1MB max, 2 backups
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# API configuration
INFERENCE_URL = "https://us.inference.heroku.com/v1/chat/completions"
INFERENCE_KEY = "YOU TOKEN KEY HEROKU ADDONS AI"
INFERENCE_MODEL_ID = "claude-4-sonnet"

# Translation prompt template
TRANSLATION_PROMPT = """
Translate English game text from "Thief Simulator 2" to natural Indonesian. Keep gaming context and tone.

Rules:
1. Translate only "word" field to Indonesian
2. Preserve placeholders like {player_name}, %d
3. Keep original structure (entry_idx, trans_idx, id unchanged)
4. Use natural gaming terminology
5. Maintain sentence case

Input: {input_json}

Output JSON in json format:
[
{"entry_idx": 0, "trans_idx": 0, "id": "example", "word": "Indonesian translation"},
...
]
"""

class InferenceAPITranslator:
    """Handles translation requests to the Inference API with optimized batch processing"""
    
    def __init__(self, url: str, token: str, model_id: str, max_workers: int = 3):
        self.url = url
        self.token = token
        self.model_id = model_id
        self.max_workers = max_workers
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        # Create session with connection pooling for efficient multi-requests
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        # Configure connection pooling
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=max_workers,
            pool_maxsize=max_workers * 2,
            max_retries=3
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        
        # Thread lock for logging
        self._log_lock = threading.Lock()
    
    def send_translation_request(self, input_json: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Send translation request to the Inference API and parse response"""
        # Prepare the prompt with the input JSON - use replace to avoid format conflicts
        json_str = json.dumps(input_json, ensure_ascii=False, indent=2)
        prompt = TRANSLATION_PROMPT.replace("{input_json}", json_str)
        
        # API payload - standard POST request
        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}]
        }
        # Print formatted JSON payload for API testing
        # print("Postman/Insomnia Request Body:")
        # print(json.dumps(payload, indent=2, ensure_ascii=False))
        
        logger.warning(f"Sending translation request for {len(input_json)} entries...")
        response = None
        try:
            # Send standard POST request
            response = self.session.post(self.url, json=payload, timeout=300)
            response.raise_for_status()
            
            # Handle standard JSON response
            response_json = response.json()
            
            # Extract content from response
            if "choices" not in response_json or len(response_json["choices"]) == 0:
                logger.error("Invalid response structure: missing choices")
                raise ValueError("Invalid response structure from API")
            
            content = response_json["choices"][0].get("message", {}).get("content", "")
            
            # Log response for debugging
            logger.debug(f"Response received, content length: {len(content)} characters")
            
            if not content.strip():
                logger.error("Empty response content received")
                raise ValueError("Empty response from API")
            
            # Extract JSON from response content
            translated_json = None
            
            # Method 1: Look for JSON wrapped in triple backticks
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', content, re.IGNORECASE)
            if json_match:
                json_content = json_match.group(1).strip()
                try:
                    translated_json = json.loads(json_content)
                    logger.debug("Successfully extracted JSON using backticks method")
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON from backticks: {e}")
            
            # Method 2: Look for JSON array pattern
            if not translated_json:
                json_match = re.search(r'(\[\s*{[\s\S]*?\])', content)
                if json_match:
                    json_content = json_match.group(1).strip()
                    try:
                        translated_json = json.loads(json_content)
                        logger.debug("Successfully extracted JSON using array pattern method")
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse JSON from array pattern: {e}")
            
            # Method 3: Try to parse the entire content as JSON
            if not translated_json:
                try:
                    translated_json = json.loads(content.strip())
                    logger.debug("Successfully parsed entire content as JSON")
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse entire content as JSON: {e}")
            
            if not translated_json:
                logger.error(f"Could not extract valid JSON from response. Content preview: {content[:500]}...")
                raise ValueError("Could not extract valid JSON from API response")
            
            # Validate the response structure
            if not isinstance(translated_json, list):
                logger.error(f"Expected list but got {type(translated_json)}")
                raise ValueError("API response is not a list")
            
            # Validate each entry
            for i, entry in enumerate(translated_json):
                if not isinstance(entry, dict):
                    logger.error(f"Entry {i} is not a dictionary: {entry}")
                    raise ValueError(f"Invalid entry structure at index {i}")
                
                required_keys = ["entry_idx", "trans_idx", "id", "word"]
                missing_keys = [key for key in required_keys if key not in entry]
                if missing_keys:
                    logger.error(f"Entry {i} missing keys: {missing_keys}")
                    raise ValueError(f"Entry {i} missing required keys: {missing_keys}")
            
            logger.info(f"Received {len(translated_json)} translated entries")
            return translated_json
            
        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP request failed: {e}")
            if response:
                logger.error(f"Response status: {response.status_code}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            raise
        except Exception as e:
            logger.error(f"API request failed: {e}")
            raise
    
    def _safe_log(self, level, message):
        """Thread-safe logging"""
        with self._log_lock:
            logger.log(level, message)
    
    def _process_single_batch(self, batch_data):
        """Process a single batch with error handling and retry logic"""
        batch, batch_num, total_batches = batch_data
        
        try:
            batch_translated = self.send_translation_request(batch)
            if batch_translated and len(batch_translated) == len(batch):
                self._safe_log(logging.INFO, f"Successfully translated batch {batch_num}/{total_batches} ({len(batch)} items)")
                return {'success': True, 'data': batch_translated, 'batch_num': batch_num}
            else:
                self._safe_log(logging.WARNING, f"Batch {batch_num} returned incomplete results")
                raise ValueError("Incomplete translation results")
                
        except Exception as e:
            self._safe_log(logging.ERROR, f"Failed to translate batch {batch_num}/{total_batches}: {e}")
            
            # Retry with smaller batch sizes if batch has multiple items
            if len(batch) > 1:
                self._safe_log(logging.INFO, f"Retrying batch {batch_num} with individual items...")
                successful_items = []
                
                for i, item in enumerate(batch):
                    try:
                        item_translated = self.send_translation_request([item])
                        if item_translated and len(item_translated) == 1:
                            successful_items.extend(item_translated)
                        else:
                            self._safe_log(logging.WARNING, f"Item {i+1} in batch {batch_num} failed, using original")
                            successful_items.append(item)
                    except Exception as e2:
                        self._safe_log(logging.ERROR, f"Item {i+1} in batch {batch_num} failed: {e2}, using original")
                        successful_items.append(item)
                    
                    # Small delay between individual items
                    time.sleep(0.2)
                
                return {'success': True, 'data': successful_items, 'batch_num': batch_num, 'fallback': True}
            else:
                # Single item batch failed, use original
                self._safe_log(logging.WARNING, f"Using original text for failed single-item batch {batch_num}")
                return {'success': True, 'data': batch, 'batch_num': batch_num, 'fallback': True}
    
    def _process_batches_concurrent(self, batches_data, max_concurrent=None):
        """Process multiple batches concurrently with controlled parallelism"""
        if max_concurrent is None:
            max_concurrent = min(self.max_workers, len(batches_data))
        
        results = [None] * len(batches_data)
        
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            # Submit all batch jobs
            future_to_index = {
                executor.submit(self._process_single_batch, batch_data): i 
                for i, batch_data in enumerate(batches_data)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    results[index] = result
                except Exception as e:
                    batch_num = batches_data[index][1]
                    self._safe_log(logging.ERROR, f"Batch {batch_num} processing failed completely: {e}")
                    # Use original data as fallback
                    results[index] = {
                        'success': True, 
                        'data': batches_data[index][0], 
                        'batch_num': batch_num, 
                        'fallback': True
                    }
        
        return results
    
    def close(self):
        """Clean up resources"""
        if hasattr(self, 'session'):
            self.session.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def translate_file(self, input_file: str, output_file: str = None, batch_size: int = 10) -> List[Dict[str, Any]]:
        """Read extracted English texts, send to API in batches, and save translations"""
        if output_file is None:
            base_name = os.path.splitext(input_file)[0]
            output_file = f"{base_name}_translated.json"
        
        logger.info(f"Processing {input_file} for translation...")
        
        # Validate input file exists
        if not os.path.exists(input_file):
            logger.error(f"Input file does not exist: {input_file}")
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                input_data = json.load(f)
            
            if not input_data:
                logger.warning("No English texts found in input file")
                # Create empty output file
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump([], f, ensure_ascii=False, indent=2)
                return []
            
            logger.info(f"Loaded {len(input_data)} entries from input file")
            
            # Validate input structure
            invalid_entries = []
            for i, entry in enumerate(input_data):
                if not isinstance(entry, dict):
                    invalid_entries.append(f"Entry {i}: not a dictionary")
                    continue
                    
                required_keys = ["entry_idx", "trans_idx", "id", "word"]
                missing_keys = [key for key in required_keys if key not in entry]
                if missing_keys:
                    invalid_entries.append(f"Entry {i}: missing keys {missing_keys}")
                    continue
                    
                # Check if word field is empty or None
                if not entry.get("word") or not str(entry["word"]).strip():
                    logger.warning(f"Entry {i} has empty 'word' field, skipping translation")
            
            if invalid_entries:
                logger.error(f"Found {len(invalid_entries)} invalid entries:")
                for error in invalid_entries[:10]:  # Show first 10 errors
                    logger.error(f"  {error}")
                if len(invalid_entries) > 10:
                    logger.error(f"  ... and {len(invalid_entries) - 10} more errors")
                raise ValueError(f"Input JSON contains {len(invalid_entries)} invalid entries")
            
            # Validate batch size
            if batch_size < 1:
                logger.warning(f"Invalid batch size {batch_size}, using default of 10")
                batch_size = 10
            elif batch_size > len(input_data):
                logger.info(f"Batch size {batch_size} is larger than input size {len(input_data)}, using {len(input_data)}")
                batch_size = len(input_data)
            
            # Prepare batches for concurrent processing
            total_batches = (len(input_data) + batch_size - 1) // batch_size
            batches_data = []
            
            for i in range(0, len(input_data), batch_size):
                batch = input_data[i:i + batch_size]
                batch_num = i // batch_size + 1
                batches_data.append((batch, batch_num, total_batches))
            
            logger.info(f"Processing {total_batches} batches with up to {self.max_workers} concurrent workers")
            logger.info(f"Batch size: {batch_size}, Total items: {len(input_data)}")
            
            # Process batches with concurrent execution and progress tracking
            translated_data = []
            successful_batches = 0
            fallback_batches = 0
            start_time = time.time()
            
            # Process in chunks to avoid overwhelming the API
            chunk_size = max(1, min(self.max_workers * 2, total_batches))
            logger.info(f"Processing in chunks of {chunk_size} batches")
            
            with tqdm(total=total_batches, desc="Translating batches") as pbar:
                for chunk_start in range(0, len(batches_data), chunk_size):
                    chunk_end = min(chunk_start + chunk_size, len(batches_data))
                    chunk_batches = batches_data[chunk_start:chunk_end]
                    
                    # Process this chunk concurrently
                    chunk_results = self._process_batches_concurrent(chunk_batches)
                    
                    # Collect results in order
                    for result in chunk_results:
                        if result and result['success']:
                            translated_data.extend(result['data'])
                            if result.get('fallback'):
                                fallback_batches += 1
                            else:
                                successful_batches += 1
                            pbar.update(1)
                        else:
                            logger.error(f"Batch {result.get('batch_num', 'unknown')} failed completely")
                            fallback_batches += 1
                            pbar.update(1)
                    
                    # Delay between chunks to respect rate limits
                    if chunk_end < len(batches_data):
                        time.sleep(1)
            
            # Log performance statistics
            end_time = time.time()
            processing_time = end_time - start_time
            items_per_second = len(input_data) / processing_time if processing_time > 0 else 0
            
            logger.info(f"Batch processing completed in {processing_time:.2f} seconds")
            logger.info(f"Processing rate: {items_per_second:.2f} items/second")
            logger.info(f"Batch statistics: {successful_batches} successful, {fallback_batches} with fallbacks")
            
            # Validate final results
            if len(translated_data) != len(input_data):
                logger.warning(f"Output size mismatch: expected {len(input_data)}, got {len(translated_data)}")
            
            # Save translated output
            try:
                # Create backup if output file already exists
                if os.path.exists(output_file):
                    backup_file = f"{output_file}.backup_{int(time.time())}"
                    logger.info(f"Creating backup of existing output file: {backup_file}")
                    os.rename(output_file, backup_file)
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(translated_data, f, ensure_ascii=False, indent=2)
                
                logger.info(f"Successfully saved {len(translated_data)} translated texts to: {output_file}")
                
                # Count successful translations vs fallbacks
                successful_translations = 0
                fallback_count = 0
                
                for i, (original, translated) in enumerate(zip(input_data, translated_data)):
                    if original.get('word') != translated.get('word'):
                        successful_translations += 1
                    else:
                        fallback_count += 1
                
                logger.info(f"Translation summary: {successful_translations} successful, {fallback_count} fallbacks")
                
                return translated_data
                
            except Exception as e:
                logger.error(f"Failed to save output file: {e}")
                # Try to save to alternative location
                alt_output = f"translated_output_{int(time.time())}.json"
                try:
                    with open(alt_output, 'w', encoding='utf-8') as f:
                        json.dump(translated_data, f, ensure_ascii=False, indent=2)
                    logger.info(f"Saved to alternative location: {alt_output}")
                    return translated_data
                except Exception as e2:
                    logger.error(f"Failed to save to alternative location: {e2}")
                    raise
            
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in input file: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to process file: {e}")
            raise

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Translate extracted English texts using Inference API with optimized batch processing")
    parser.add_argument("--input", "-i", default="extracted_english.json", 
                       help="Input file with extracted English texts (default: extracted_english.json)")
    parser.add_argument("--output", "-o", 
                       help="Output file for translated texts (default: input_translated.json)")
    parser.add_argument("--batch-size", "-b", type=int, default=10,
                       help="Batch size for API requests (default: 10)")
    parser.add_argument("--max-workers", "-w", type=int, default=3,
                       help="Maximum concurrent workers for batch processing (default: 3)")
    
    args = parser.parse_args()
    
    # Use context manager for proper resource cleanup
    with InferenceAPITranslator(INFERENCE_URL, INFERENCE_KEY, INFERENCE_MODEL_ID, args.max_workers) as translator:
        try:
            translated_texts = translator.translate_file(args.input, args.output, args.batch_size)
            
            if translated_texts:
                sample = translated_texts[0]
                logger.info(f"Sample translated text: ID={sample['id']}, Word={sample['word']}")
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()