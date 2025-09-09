#!/usr/bin/env python3
"""
Thief Simulator 2 - Indonesian Translation System
Optimized with PyTorch XPU support for Intel Arc GPUs
Using facebook/nllb-200-distilled-600M model

This script translates English game text (language ID 10) to Indonesian
while preserving all other language translations. Model is downloaded to ./models.

Features:
- Downloads NLLB-200-distilled-600M to ./models directory
- PyTorch XPU support for Intel Arc GPU acceleration
- Batch processing for efficiency
- LRU caching for repeated translations
- Data sanitization for clean JSON output
- Fallback to CPU if XPU/CUDA not available
- Optimized for large files up to 6MB+
- Sentence case handling based on original text
"""

import json
import re
import argparse
import logging
import os
import sys
from functools import lru_cache
from typing import List, Dict, Any
from tqdm import tqdm
import time

try:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Error: PyTorch or transformers not available. Please install: pip install torch transformers")
    sys.exit(1)

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
    """Indonesian translation system with NLLB-200 and PyTorch XPU optimization"""
    
    def __init__(self, model_name: str = "facebook/nllb-200-distilled-600M", 
                 model_dir: str = "./models", batch_size: int = 32):
        self.model_name = model_name
        self.model_dir = os.path.abspath(model_dir)  # Absolute path for model storage
        self.batch_size = batch_size
        self.device = self._get_optimal_device()
        self.model = None
        self.tokenizer = None
        self.translation_cache = {}
        self.source_lang = "eng_Latn"
        self.target_lang = "ind_Latn"
        
        # Model configuration for NLLB
        self.model_config = {
            "max_length": 512,
            "num_beams": 6,
            "early_stopping": True
        }
        
        logger.info(f"Initialized translator with device: {self.device}, model_dir: {self.model_dir}")

    def _get_optimal_device(self) -> str:
        """Determine the best available device for computation"""
        if not TORCH_AVAILABLE:
            return "cpu"
        
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            logger.info("Intel XPU (Arc GPU) detected and available")
            return "xpu"
        
        if torch.cuda.is_available():
            logger.info("CUDA GPU detected and available")
            return "cuda"
        
        logger.info("Using CPU for computation")
        return "cpu"
    
    def load_model(self):
        """Load or download the translation model to model_dir with device optimization"""
        try:
            logger.info(f"Loading model {self.model_name} to {self.model_dir} on {self.device}...")
            
            # Ensure model_dir exists
            os.makedirs(self.model_dir, exist_ok=True)
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.model_dir,
                use_fast=True
            )
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                cache_dir=self.model_dir
            )
            
            # Set source language
            self.tokenizer.src_lang = self.source_lang
            
            # Get BOS token ID for target language
            self.target_lang_bos_id = self.tokenizer.convert_tokens_to_ids(self.target_lang)
            if self.target_lang_bos_id is None:
                logger.error(f"Could not find BOS token ID for {self.target_lang}")
                raise ValueError(f"Invalid target language code: {self.target_lang}")
            
            # Move model to optimal device
            if self.device != "cpu":
                self.model = self.model.to(self.device)
                logger.info(f"Model moved to {self.device}")
            
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
        
        text = re.sub(r'""', '"', text)
        text = text.rstrip('\r\n')
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def post_process_translation(self, original_text: str, translated_text: str) -> str:
        """Post-process translation to handle sentence case only"""
        improved_text = translated_text
        
        # Apply sentence case
        if original_text and original_text[0].isupper():
            sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', improved_text)
            capitalized_sentences = []
            for s in sentences:
                s = s.strip()
                if s:
                    capitalized_sentences.append(s[0].upper() + s[1:])
                else:
                    capitalized_sentences.append('')
            improved_text = ' '.join(capitalized_sentences)
        
        return improved_text
    
    @lru_cache(maxsize=2000)
    def translate_with_cache(self, text: str) -> str:
        """Translate text with caching for improved performance"""
        if not text or not text.strip():
            return text
            
        if text in self.translation_cache:
            return self.translation_cache[text]
            
        try:
            clean_text = self.sanitize_text(text)
            
            inputs = self.tokenizer(clean_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                if self.device == "xpu":
                    torch.xpu.synchronize()
                outputs = self.model.generate(
                    **inputs,
                    **self.model_config,
                    forced_bos_token_id=self.target_lang_bos_id
                )
                if self.device == "xpu":
                    torch.xpu.synchronize()
            
            translated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            translated = self.post_process_translation(text, translated)
            translated = self.sanitize_text(translated)
            
            self.translation_cache[text] = translated
            return translated
            
        except Exception as e:
            logger.error(f"Translation error for text '{text}': {e}")
            return text
    
    def translate_batch(self, texts: List[str]) -> List[str]:
        """Translate a batch of texts for improved efficiency"""
        if not texts:
            return []
        
        clean_texts = [self.sanitize_text(text) for text in texts if text and text.strip()]
        if not clean_texts:
            return texts
        
        try:
            results = [None] * len(clean_texts)
            uncached_texts = []
            uncached_indices = []
            uncached_originals = []
            
            for i, text in enumerate(clean_texts):
                if text in self.translation_cache:
                    results[i] = self.translation_cache[text]
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
                    uncached_originals.append(text)
            
            if uncached_texts:
                translated_uncached = [None] * len(uncached_texts)
                for start in range(0, len(uncached_texts), self.batch_size):
                    end = start + self.batch_size
                    batch = uncached_texts[start:end]
                    batch_originals = uncached_originals[start:end]
                    
                    inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        if self.device == "xpu":
                            torch.xpu.synchronize()
                        outputs = self.model.generate(
                            **inputs,
                            **self.model_config,
                            forced_bos_token_id=self.target_lang_bos_id
                        )
                        if self.device == "xpu":
                            torch.xpu.synchronize()
                    
                    for j, output in enumerate(outputs):
                        translated = self.tokenizer.decode(output, skip_special_tokens=True)
                        original = batch_originals[j]
                        translated = self.post_process_translation(original, translated)
                        translated = self.sanitize_text(translated)
                        translated_uncached[start + j] = translated
                        self.translation_cache[batch[j]] = translated
                
                for idx, trans in zip(uncached_indices, translated_uncached):
                    results[idx] = trans
            
            return results
            
        except Exception as e:
            logger.error(f"Batch translation failed: {e}")
            return [self.translate_with_cache(text) for text in clean_texts]
    
    def process_localization_file(self, input_file: str, output_file: str = None) -> Dict[str, Any]:
        """Process the localization file and translate English entries while preserving all other languages"""
        if output_file is None:
            base_name = os.path.splitext(input_file)[0]
            output_file = f"{base_name}_Indonesian.json"
        
        logger.info(f"Processing {input_file}...")
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'allText' not in data:
                raise ValueError("Invalid localization file format: missing 'allText' key")
            
            english_texts = []
            positions = []
            for entry_idx, entry in enumerate(data['allText']):
                if 'translations' in entry:
                    for trans_idx, translation in enumerate(entry['translations']):
                        if translation.get('language') == 10:
                            english_texts.append(translation.get('word', ''))
                            positions.append((entry_idx, trans_idx))
            
            logger.info(f"Found {len(english_texts)} English entries to translate")
            
            translated_texts = []
            with tqdm(total=len(english_texts), desc="Translating") as pbar:
                for i in range(0, len(english_texts), self.batch_size):
                    batch = english_texts[i:i + self.batch_size]
                    batch_translations = self.translate_batch(batch)
                    translated_texts.extend(batch_translations)
                    pbar.update(len(batch))
            
            for (entry_idx, trans_idx), translated_word in zip(positions, translated_texts):
                data['allText'][entry_idx]['translations'][trans_idx]['word'] = translated_word
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Translation completed: {len(english_texts)} entries translated")
            logger.info(f"Output saved to: {output_file}")
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to process localization file: {e}")
            raise

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Translate Thief Simulator 2 to Indonesian")
    parser.add_argument("--input", "-i", default="LocalizationFile.json", 
                       help="Input localization file (default: LocalizationFile.json)")
    parser.add_argument("--output", "-o", 
                       help="Output file (default: input_Indonesian.json)")
    parser.add_argument("--batch-size", "-b", type=int, default=32,
                       help="Batch size for translation (default: 32)")
    parser.add_argument("--model-dir", "-m", default="./models",
                       help="Directory to store model files (default: ./models)")
    
    args = parser.parse_args()
    
    translator = IndonesianTranslator(
        model_name="facebook/nllb-200-distilled-600M",
        model_dir=args.model_dir,
        batch_size=args.batch_size
    )
    
    try:
        translator.load_model()
        start_time = time.time()
        result = translator.process_localization_file(args.input, args.output)
        end_time = time.time()
        
        logger.info(f"Translation completed in {end_time - start_time:.2f} seconds")
        logger.info(f"Cache size: {len(translator.translation_cache)} cached translations")
        
        if result['allText']:
            sample = result['allText'][0]
            for trans in sample['translations']:
                if trans['language'] == 10:
                    logger.info(f"Sample translation: {sample['ID']} -> {trans['word']}")
                    break
        
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()