#!/usr/bin/env python3
"""
Extract English text (language ID 10) from Thief Simulator 2 localization file
for external translation, preserving structure for reintegration.

This script extracts English 'word' entries from the JSON localization file
and saves them to a new JSON file with their positions (entry_idx, trans_idx)
for later reintegration after translation.

Features:
- Extracts English text (language ID 10) from 'allText' entries
- Preserves original file structure metadata
- Outputs to a JSON file for translation
- Handles large files efficiently
- Logs extraction process and errors
"""

import json
import argparse
import logging
import os
import sys
from typing import List, Dict, Any
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('extraction_errors.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class EnglishTextExtractor:
    """Extracts English text from localization files for external translation"""
    
    def __init__(self):
        self.extracted_texts = []
    
    def extract_english_texts(self, input_file: str, output_file: str = None) -> List[Dict[str, Any]]:
        """Extract English texts (language ID 10) and save their positions"""
        if output_file is None:
            base_name = os.path.splitext(input_file)[0]
            output_file = f"{base_name}_english_extracted.json"
        
        logger.info(f"Processing {input_file} for English text extraction...")
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'allText' not in data:
                raise ValueError("Invalid localization file format: missing 'allText' key")
            
            self.extracted_texts = []
            for entry_idx, entry in enumerate(tqdm(data['allText'], desc="Extracting English texts")):
                if 'translations' in entry:
                    for trans_idx, translation in enumerate(entry['translations']):
                        if translation.get('language') == 10:
                            word = translation.get('word', '')
                            if word and word.strip():
                                self.extracted_texts.append({
                                    'entry_idx': entry_idx,
                                    'trans_idx': trans_idx,
                                    'id': entry.get('ID', ''),
                                    'word': word
                                })
            
            logger.info(f"Extracted {len(self.extracted_texts)} English entries")
            
            # Save extracted texts to output file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.extracted_texts, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Extracted texts saved to: {output_file}")
            return self.extracted_texts
            
        except Exception as e:
            logger.error(f"Failed to process localization file: {e}")
            raise

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Extract English text from Thief Simulator 2 localization file")
    parser.add_argument("--input", "-i", default="LocalizationFile.json", 
                       help="Input localization file (default: LocalizationFile.json)")
    parser.add_argument("--output", "-o", 
                       help="Output file for extracted texts (default: input_english_extracted.json)")
    
    args = parser.parse_args()
    
    extractor = EnglishTextExtractor()
    
    try:
        extracted_texts = extractor.extract_english_texts(args.input, args.output)
        
        if extracted_texts:
            sample = extracted_texts[0]
            logger.info(f"Sample extracted text: ID={sample['id']}, Word={sample['word']}")
        
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()