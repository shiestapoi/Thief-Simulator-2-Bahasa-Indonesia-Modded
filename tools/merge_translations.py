#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script untuk menggabungkan terjemahan dari test_output_concurrent.json
ke dalam LocalizationFile.json dengan mengganti teks pada language 10 saja.
"""

import json
import sys
from typing import Dict, List, Any

def load_json_file(file_path: str) -> Any:
    """
    Memuat file JSON dengan penanganan error.
    
    Args:
        file_path: Path ke file JSON
        
    Returns:
        Data JSON yang telah dimuat
        
    Raises:
        FileNotFoundError: Jika file tidak ditemukan
        json.JSONDecodeError: Jika file JSON tidak valid
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File {file_path} tidak ditemukan.")
        raise
    except json.JSONDecodeError as e:
        print(f"Error: File {file_path} bukan JSON yang valid. {e}")
        raise
    except Exception as e:
        print(f"Error: Gagal membaca file {file_path}. {e}")
        raise

def save_json_file(data: Any, file_path: str) -> None:
    """
    Menyimpan data ke file JSON dengan penanganan error.
    
    Args:
        data: Data yang akan disimpan
        file_path: Path file tujuan
        
    Raises:
        Exception: Jika gagal menyimpan file
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"File berhasil disimpan ke {file_path}")
    except Exception as e:
        print(f"Error: Gagal menyimpan file {file_path}. {e}")
        raise

def create_translation_mapping(source_data: List[Dict]) -> Dict[str, str]:
    """
    Membuat mapping ID ke terjemahan dari data sumber.
    
    Args:
        source_data: Data dari test_output_concurrent.json
        
    Returns:
        Dictionary dengan ID sebagai key dan terjemahan sebagai value
    """
    translation_map = {}
    
    for item in source_data:
        if 'id' in item and 'word' in item:
            translation_map[item['id']] = item['word']
    
    print(f"Berhasil membuat mapping untuk {len(translation_map)} terjemahan")
    return translation_map

def merge_translations(target_data: Dict, translation_map: Dict[str, str]) -> tuple:
    """
    Menggabungkan terjemahan ke dalam data target dengan mengganti language 10 saja.
    
    Args:
        target_data: Data dari LocalizationFile.json
        translation_map: Mapping ID ke terjemahan
        
    Returns:
        Tuple berisi (data_yang_diupdate, jumlah_berhasil, jumlah_tidak_ditemukan)
    """
    updated_count = 0
    not_found_count = 0
    not_found_ids = []
    
    if 'allText' not in target_data:
        raise ValueError("Format file target tidak valid: tidak ada field 'allText'")
    
    for entry in target_data['allText']:
        if 'ID' not in entry:
            continue
            
        entry_id = entry['ID']
        
        if entry_id in translation_map:
            # Cari dan update translation untuk language 10
            if 'translations' in entry:
                language_10_found = False
                for translation in entry['translations']:
                    if translation.get('language') == 10:
                        old_word = translation.get('word', '')
                        new_word = translation_map[entry_id]
                        translation['word'] = new_word
                        language_10_found = True
                        updated_count += 1
                        print(f"Updated ID '{entry_id}': '{old_word}' -> '{new_word}'")
                        break
                
                if not language_10_found:
                    print(f"Warning: ID '{entry_id}' tidak memiliki translation untuk language 10")
            else:
                print(f"Warning: ID '{entry_id}' tidak memiliki field 'translations'")
        else:
            not_found_count += 1
            not_found_ids.append(entry_id)
    
    if not_found_ids:
        print(f"\nID yang tidak ditemukan dalam file sumber ({not_found_count} total):")
        for i, missing_id in enumerate(not_found_ids[:10]):  # Tampilkan maksimal 10 ID pertama
            print(f"  - {missing_id}")
        if len(not_found_ids) > 10:
            print(f"  ... dan {len(not_found_ids) - 10} ID lainnya")
    
    return target_data, updated_count, not_found_count

def main():
    """
    Fungsi utama untuk menjalankan proses penggabungan.
    """
    source_file = "test_output_concurrent.json"
    target_file = "LocalizationFile.json"
    output_file = "LocalizationFile_merged.json"
    
    try:
        print("=== Memulai proses penggabungan terjemahan ===")
        print(f"File sumber: {source_file}")
        print(f"File target: {target_file}")
        print(f"File output: {output_file}")
        print()
        
        # Memuat file sumber
        print("Memuat file sumber...")
        source_data = load_json_file(source_file)
        
        # Memuat file target
        print("Memuat file target...")
        target_data = load_json_file(target_file)
        
        # Membuat mapping terjemahan
        print("Membuat mapping terjemahan...")
        translation_map = create_translation_mapping(source_data)
        
        # Menggabungkan terjemahan
        print("Menggabungkan terjemahan...")
        merged_data, updated_count, not_found_count = merge_translations(target_data, translation_map)
        
        # Menyimpan hasil
        print("\nMenyimpan hasil...")
        save_json_file(merged_data, output_file)
        
        # Menampilkan ringkasan
        print("\n=== RINGKASAN PROSES ===")
        print(f"Total terjemahan dalam file sumber: {len(translation_map)}")
        print(f"Total entry berhasil diupdate: {updated_count}")
        print(f"Total ID tidak ditemukan: {not_found_count}")
        print(f"File hasil disimpan sebagai: {output_file}")
        
        if not_found_count > 0:
            print(f"\nPerhatian: {not_found_count} ID dari file target tidak ditemukan dalam file sumber.")
            print("ID-ID tersebut tetap menggunakan terjemahan asli.")
        
        print("\nProses selesai!")
        
    except Exception as e:
        print(f"\nError: Proses gagal. {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()