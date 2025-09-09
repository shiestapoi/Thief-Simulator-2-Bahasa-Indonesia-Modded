#!/usr/bin/env python3
"""
Thief Simulator 2 - Indonesian Translation System
Optimized with PyTorch XPU support for Intel Arc GPUs

This script translates English game text (language ID 10) to Indonesian
while preserving all other language translations.

Features:
- PyTorch XPU support for Intel Arc GPU acceleration
- Batch processing for efficiency
- LRU caching for repeated translations
- Data sanitization for clean JSON output
- Fallback to CPU if XPU is not available
- Optimized for large files up to 6MB+
- Sentence case handling based on original text
- Improved corrections for inaccurate translations
"""

import json
import re
import argparse
import logging
import os
import sys
import time
from functools import lru_cache
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm

try:
    import torch
    from transformers import MarianMTModel, MarianTokenizer
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch or transformers not available. Please install requirements.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('translation_errors.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class IndonesianTranslator:
    """Indonesian translation system with PyTorch XPU optimization"""
    
    def __init__(self, model_name: str = "Helsinki-NLP/opus-mt-en-id", batch_size: int = 50):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = self._get_optimal_device()
        self.model = None
        self.tokenizer = None
        self.translation_cache = {}
        
        # Batch processing configuration
        self.max_cache_size = 10000
        
        # Progress tracking for large files
        self.processed_count = 0
        self.total_count = 0
        
        # Model configuration from Hugging Face
        self.model_config = {
            "bos_token_id": 0,
            "eos_token_id": 0,
            "forced_eos_token_id": 0,
            "pad_token_id": 54795,
            "max_length": 512,
            "num_beams": 4,
            "early_stopping": True
        }
        
        # Essential gaming terminology - minimal dictionary for critical terms only
        self.essential_gaming_terms = {
            "ability": "kemampuan",
            "accept": "terima", 
            "accepted": "diterima",
            "account": "akun",
            "activate": "aktifkan",
            "balance": "saldo",
            "task": "tugas",
            "experience": "pengalaman",
            "skill": "keahlian",
            "level": "tingkat",
            "quest": "misi",
            "inventory": "inventaris",
            "equipment": "perlengkapan",
            "character": "karakter",
            "player": "pemain",
            "PC": "PC",
            "computer": "komputer",
            "break into": "masuk ke",
            "get off": "matikan",
            "turn off": "matikan"
        }
        
        # Translation correction dictionary for common errors
        self.correction_dict = {
            'kemampuan': 'Kemampuan',
            'sempoa': 'Sempoa',
            'abakus': 'Sempoa',
            'kemampuan': 'Kemampuan',
            'keterampilan': 'Keterampilan',
            'pengetahuan': 'Pengetahuan',
            'pengalaman': 'Pengalaman',
            'prestasi': 'Prestasi',
            'pencapaian': 'Pencapaian',
            'kemampuan' : 'Kemampuan',
            'abstrak melukis': 'lukisan abstrak',
            'Abstrak melukis': 'Lukisan abstrak',
            'abstrak lukisan': 'lukisan abstrak',
            'Abstrak lukisan': 'Lukisan abstrak',
            'melukis abstrak': 'lukisan abstrak',
            'Melukis abstrak': 'Lukisan abstrak'
        }
        
        # Contextual phrase patterns for better translation
        self.phrase_patterns = {
            r"You'll need more experience before we break into (\d+)": "Kamu perlu lebih banyak pengalaman sebelum kita masuk ke {}",
            r"Accept the first task and get off the PC": "Terima dulu tugas pertama dan matikan PC-nya",
            r"get off the PC": "matikan PC-nya",
            r"turn off the PC": "matikan PC-nya",
            r"break into (\d+)": "masuk ke {}",
            r"You'll need": "Kamu perlu",
            r"You need": "Kamu perlu",
            r"before we": "sebelum kita",
            r"Abstract painting 'Planet'": "Lukisan abstrak 'Planet'",
            r"abstract painting 'planet'": "Lukisan abstrak 'Planet'",
            r"Abstract painting 'Nebula'": "Lukisan abstrak 'Nebula'",
            r"abstract painting 'nebula'": "Lukisan abstrak 'Nebula'"
        }
        
        logger.info(f"Initialized translator with device: {self.device}")
    
    def _get_optimal_device(self) -> str:
        """Determine the best available device for computation"""
        if not TORCH_AVAILABLE:
            return "cpu"
        
        # Check for Intel XPU (Arc GPU) support
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            logger.info("Intel XPU (Arc GPU) detected and available")
            return "xpu"
        
        # Fallback to CUDA if available
        if torch.cuda.is_available():
            logger.info("CUDA GPU detected and available")
            return "cuda"
        
        # Default to CPU
        logger.info("Using CPU for computation")
        return "cpu"
    
    def load_model(self):
        """Load the translation model with device optimization"""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch and transformers are required but not available")
        
        try:
            logger.info(f"Loading model {self.model_name} on {self.device}...")
            
            # Load tokenizer
            self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)
            
            # Load model
            self.model = MarianMTModel.from_pretrained(self.model_name)
            
            # Move model to optimal device
            if self.device != "cpu":
                self.model = self.model.to(self.device)
                logger.info(f"Model moved to {self.device}")
            
            # Set model to evaluation mode for inference
            self.model.eval()
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.info("Falling back to CPU...")
            self.device = "cpu"
            if self.model is not None:
                self.model = self.model.to("cpu")
            raise
    
    def sanitize_text(self, text: str) -> str:
        """Sanitize input text by removing redundant quotes and special characters"""
        if not isinstance(text, str):
            return str(text)
        
        # Remove redundant double quotes
        text = re.sub(r'""', '"', text)
        
        # Remove trailing carriage returns and newlines
        text = text.rstrip('\r\n')
        
        # Clean up extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def apply_contextual_patterns(self, text: str) -> str:
        """Apply contextual phrase patterns for better translation quality."""
        # Apply essential gaming terms first
        for english_term, indonesian_term in self.essential_gaming_terms.items():
            # Case-insensitive replacement while preserving original case pattern
            if english_term.lower() in text.lower():
                # Find the actual occurrence in the text
                pattern = re.compile(re.escape(english_term), re.IGNORECASE)
                text = pattern.sub(indonesian_term, text)
        
        # Apply phrase patterns
        for pattern, replacement in self.phrase_patterns.items():
            if '{}' in replacement:
                # Handle patterns with capture groups
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    if match.groups():
                        text = re.sub(pattern, replacement.format(*match.groups()), text, flags=re.IGNORECASE)
                    else:
                        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
            else:
                # Simple replacement
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def post_process_translation(self, original_text: str, translated_text: str) -> str:
        """Post-process translation to improve quality and naturalness."""
        improved_text = translated_text
        
        # Fix common translation issues
        improvements = {
            r'\bistirahat ke\b': 'masuk ke',
            r'\bMenerima\b': 'Terima',
            r'\bturun PC\b': 'matikan PC',
            r'\bAnda akan\b': 'Kamu akan',
            r'\bAnda perlu\b': 'Kamu perlu',
            r'\bkita istirahat\b': 'kita masuk',
            r'\bdan turun\b': 'dan matikan',
            r'\bPC\."$': 'PC-nya."',
            r'\bPC\.$': 'PC-nya.',
            r'\bsebelum kita istirahat ke (\d+)\b': r'sebelum kita masuk ke \1',
            r'\bMenerima tugas pertama dan turun PC\b': 'Terima dulu tugas pertama dan matikan PC-nya',
            r'\bAbstrak melukis\b': 'Lukisan abstrak',
            r'\babstrak melukis\b': 'lukisan abstrak',
            r'\bMelukis abstrak\b': 'Lukisan abstrak',
            r'\bmelukis abstrak\b': 'lukisan abstrak'
        }
        
        for pattern, replacement in improvements.items():
            improved_text = re.sub(pattern, replacement, improved_text)
        
        # Apply correction dictionary for translation errors
        for wrong, correct in self.correction_dict.items():
            # Case-sensitive replacement
            improved_text = improved_text.replace(wrong, correct)
        
        # Apply sentence case: Capitalize first letter if original starts with uppercase
        # Also handle multiple sentences
        if original_text and original_text[0].isupper():
            # Split into sentences and capitalize each
            sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', improved_text)
            capitalized_sentences = [s.strip().capitalize() if s.strip() else '' for s in sentences]
            improved_text = ' '.join(capitalized_sentences)
        
        return improved_text
    
    @lru_cache(maxsize=1000)
    def translate_with_cache(self, text: str) -> str:
        """Translate text with caching for improved performance."""
        if not text or not text.strip():
            return text
            
        # Check gaming dictionary first
        text_lower = text.lower().strip()
        if text_lower in self.gaming_dict:
            # Preserve case based on dict
            return self.gaming_dict[text_lower]
            
        # Check if already in cache
        if text in self.translation_cache:
            return self.translation_cache[text]
            
        try:
            # First try contextual patterns
            contextual_result = self.apply_contextual_patterns(text)
            if contextual_result != text:
                # Cache and return contextual translation
                self.translation_cache[text] = contextual_result
                return contextual_result
            
            # Sanitize input text
            clean_text = self.sanitize_text(text)
            
            # Tokenize and translate
            inputs = self.tokenizer(clean_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **self.model_config)
            
            translated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Post-process the translation
            translated = self.post_process_translation(text, translated)
            
            # Sanitize output
            translated = self.sanitize_text(translated)
            
            # Cache the result
            self.translation_cache[text] = translated
            
            return translated
            
        except Exception as e:
            logger.error(f"Translation error for text '{text}': {e}")
            return text  # Return original text if translation fails
    
    def _translate_single(self, text: str) -> str:
        """Translate a single text string"""
        if not text or not text.strip():
            return text
        
        # Sanitize input
        clean_text = self.sanitize_text(text)
        
        # Check gaming dictionary first
        lower_text = clean_text.lower()
        if lower_text in self.gaming_dict:
            return self.gaming_dict[lower_text]
        
        # Check cache
        if clean_text in self.translation_cache:
            return self.translation_cache[clean_text]
        
        try:
            # Tokenize input
            inputs = self.tokenizer(clean_text, return_tensors="pt", padding=True, truncation=True)
            
            # Move inputs to device
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate translation
            with torch.no_grad():
                if self.device == "xpu":
                    # Synchronize XPU operations
                    torch.xpu.synchronize()
                
                outputs = self.model.generate(**inputs, **self.model_config)
                
                if self.device == "xpu":
                    torch.xpu.synchronize()
            
            # Decode output
            translated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Post-process
            translated = self.post_process_translation(clean_text, translated)
            
            # Sanitize
            translated = self.sanitize_text(translated)
            
            # Cache the result
            self.translation_cache[clean_text] = translated
            
            return translated
            
        except Exception as e:
            logger.error(f"Translation failed for '{clean_text}': {e}")
            return clean_text  # Return original text if translation fails
    
    def translate_batch(self, texts: List[str]) -> List[str]:
        """
        Translate a batch of texts efficiently with caching
        """
        if not texts:
            return []
        
        # Check cache first and prepare uncached texts
        results = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            if not text or not text.strip():
                results.append("")
                continue
                
            # Check cache
            cache_key = hash(text.strip().lower())
            if cache_key in self.translation_cache:
                results.append(self.translation_cache[cache_key])
            else:
                results.append(None)  # Placeholder
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Translate uncached texts
        if uncached_texts:
            try:
                # Tokenize batch
                inputs = self.tokenizer(
                    uncached_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                
                # Move to device if available
                if self.device:
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Generate translations
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=512,
                        num_beams=4,
                        early_stopping=True,
                        do_sample=False
                    )
                
                # Decode translations
                translations = self.tokenizer.batch_decode(
                    outputs, 
                    skip_special_tokens=True
                )
                
                # Apply post-processing and cache results
                for i, translation in enumerate(translations):
                    processed_translation = self.post_process_translation(uncached_texts[i], translation)
                    original_text = uncached_texts[i]
                    cache_key = hash(original_text.strip().lower())
                    
                    # Cache management
                    if len(self.translation_cache) >= self.max_cache_size:
                        # Remove oldest entries (simple FIFO)
                        oldest_key = next(iter(self.translation_cache))
                        del self.translation_cache[oldest_key]
                    
                    self.translation_cache[cache_key] = processed_translation
                    results[uncached_indices[i]] = processed_translation
                
            except Exception as e:
                logger.error(f"Batch translation error: {e}")
                # Fallback to individual translation
                for i, text in enumerate(uncached_texts):
                    results[uncached_indices[i]] = self._translate_single(text)
        
        return results
    
    def process_localization_file(self, input_file: str, output_file: str = None) -> Dict[str, Any]:
        """Process localization file efficiently with batch translation and progress tracking"""
        if output_file is None:
            base_name = os.path.splitext(input_file)[0]
            output_file = f"{base_name}_Indonesian.json"
        
        logger.info(f"Processing {input_file}...")
        
        try:
            # Load and validate file
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'allText' not in data:
                raise ValueError("Invalid localization file format: missing 'allText' key")
            
            # Initialize progress tracking
            self.total_count = len(data['allText'])
            self.processed_count = 0
            
            # Collect all English texts for batch processing
            english_texts = []
            text_positions = []  # Track where each text belongs
            
            for entry_idx, entry in enumerate(data['allText']):
                if 'translations' in entry:
                    for trans_idx, translation in enumerate(entry['translations']):
                        if translation.get('language') == 10:  # English
                            original_text = translation.get('word', '')
                            if original_text and original_text.strip():
                                english_texts.append(original_text)
                                text_positions.append((entry_idx, trans_idx))
            
            logger.info(f"Processing {len(english_texts)} English texts from {self.total_count} entries")
            
            # Process in batches for efficiency
            translated_texts = []
            for i in range(0, len(english_texts), self.batch_size):
                batch_texts = english_texts[i:i + self.batch_size]
                batch_translations = self.translate_batch(batch_texts)
                translated_texts.extend(batch_translations)
                
                # Update progress
                self.processed_count = min(i + self.batch_size, len(english_texts))
                progress = (self.processed_count / len(english_texts)) * 100
                logger.info(f"Translation progress: {progress:.1f}% ({self.processed_count}/{len(english_texts)})")
            
            # Apply translations back to the data structure
            for i, (entry_idx, trans_idx) in enumerate(text_positions):
                if i < len(translated_texts):
                    data['allText'][entry_idx]['translations'][trans_idx]['word'] = translated_texts[i]
            
            # Save output file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Translation completed: {len(english_texts)} entries translated")
            logger.info(f"Output saved to: {output_file}")
            logger.info(f"Cache size: {len(self.translation_cache)}")
            
            return {
                'status': 'success',
                'input_file': input_file,
                'output_file': output_file,
                'total_entries': self.total_count,
                'translated_texts': len(english_texts),
                'cache_size': len(self.translation_cache)
            }
            
        except Exception as e:
            logger.error(f"Failed to process localization file: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def clear_cache(self):
        """
        Clear translation cache to free memory
        """
        self.translation_cache.clear()
        logger.info("Translation cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics for monitoring
        """
        return {
            'cache_size': len(self.translation_cache),
            'max_cache_size': self.max_cache_size,
            'cache_usage_percent': (len(self.translation_cache) / self.max_cache_size) * 100
        }

def main():
    """Main function to demonstrate efficient translation system"""
    parser = argparse.ArgumentParser(description="Translate Thief Simulator 2 to Indonesian")
    parser.add_argument("--input", "-i", default="MediumTestFile.json", 
                       help="Input localization file (default: MediumTestFile.json)")
    parser.add_argument("--output", "-o", 
                       help="Output file (default: input_Indonesian.json)")
    parser.add_argument("--batch-size", "-b", type=int, default=50,
                       help="Batch size for translation (default: 50)")
    parser.add_argument("--demo", "-d", action="store_true",
                       help="Run demonstration mode with test translations")
    
    args = parser.parse_args()
    
    # Initialize translator with larger default batch for optimization
    translator = IndonesianTranslator(batch_size=args.batch_size)
    
    try:
        # Load model
        translator.load_model()
        
        # Run demonstration if requested
        if args.demo:
            print("\n=== Efficient Indonesian Translation System ===")
            print(f"Device: {translator.device}")
            print(f"Model: {translator.model_name}")
            print(f"Batch size: {translator.batch_size}")
            print(f"Max cache size: {translator.max_cache_size}")
            
            # Test batch translation
            test_texts = [
                "Accept the first task and get off the PC.",
                "You'll need more experience before we break into 103.",
                "Ability activated",
                "Account balance", 
                "Abstract painting 'Nebula'",
                "ACCEPTED",
                "Player inventory",
                "Character level",
                "Quest completed",
                "Equipment upgraded"
            ]
            
            print("\n=== Batch Translation Test ===")
            translations = translator.translate_batch(test_texts)
            
            for original, translated in zip(test_texts, translations):
                print(f"'{original}' -> '{translated}'")
            
            # Show cache statistics
            cache_stats = translator.get_cache_stats()
            print(f"\n=== Cache Statistics ===")
            print(f"Cache size: {cache_stats['cache_size']}")
            print(f"Cache usage: {cache_stats['cache_usage_percent']:.1f}%")
        
        # Process file
        start_time = time.time()
        result = translator.process_localization_file(args.input, args.output)
        end_time = time.time()
        
        if result['status'] == 'success':
            logger.info(f"✅ Translation completed in {end_time - start_time:.2f} seconds")
            logger.info(f"📁 Output file: {result['output_file']}")
            logger.info(f"📊 Total entries: {result['total_entries']}")
            logger.info(f"🔤 Translated texts: {result['translated_texts']}")
            logger.info(f"💾 Final cache size: {result['cache_size']}")
        else:
            logger.error(f"❌ Translation failed: {result['error']}")
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()