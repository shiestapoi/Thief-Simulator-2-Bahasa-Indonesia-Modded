#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script untuk memverifikasi ID dari file LocalizationFile_english_extracted.json
dengan membandingkannya terhadap file test_output_concurrent.json

Script ini akan:
1. Mengidentifikasi ID yang hilang (miss) antara kedua file
2. Mengidentifikasi ID yang duplikat dalam masing-masing file
3. Menghasilkan laporan yang jelas tentang perbedaan yang ditemukan
"""

import json
import sys
from collections import Counter
from pathlib import Path

def load_json_file(file_path):
    """Memuat file JSON dan mengembalikan data"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File {file_path} tidak ditemukan")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: File {file_path} bukan JSON yang valid - {e}")
        return None
    except Exception as e:
        print(f"Error: Gagal membaca file {file_path} - {e}")
        return None

def extract_ids(data):
    """Mengekstrak semua ID dari data JSON"""
    ids = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and 'id' in item:
                ids.append(item['id'])
    return ids

def find_duplicates(ids):
    """Menemukan ID yang duplikat dalam list"""
    counter = Counter(ids)
    duplicates = {id_val: count for id_val, count in counter.items() if count > 1}
    return duplicates

def find_missing_ids(ids1, ids2):
    """Menemukan ID yang hilang antara dua set ID"""
    set1 = set(ids1)
    set2 = set(ids2)
    
    missing_in_set2 = set1 - set2  # ID yang ada di set1 tapi tidak di set2
    missing_in_set1 = set2 - set1  # ID yang ada di set2 tapi tidak di set1
    
    return missing_in_set2, missing_in_set1

def generate_report(english_ids, indonesian_ids, english_duplicates, indonesian_duplicates, missing_in_indonesian, missing_in_english):
    """Menghasilkan laporan lengkap tentang perbedaan ID"""
    report = []
    report.append("=" * 80)
    report.append("LAPORAN VERIFIKASI ID LOCALIZATION FILES")
    report.append("=" * 80)
    report.append("")
    
    # Statistik umum
    report.append("STATISTIK UMUM:")
    report.append("-" * 40)
    report.append(f"Total ID dalam file English: {len(english_ids)}")
    report.append(f"Total ID unik dalam file English: {len(set(english_ids))}")
    report.append(f"Total ID dalam file Indonesian: {len(indonesian_ids)}")
    report.append(f"Total ID unik dalam file Indonesian: {len(set(indonesian_ids))}")
    report.append("")
    
    # ID Duplikat dalam file English
    report.append("1. ID DUPLIKAT DALAM FILE ENGLISH:")
    report.append("-" * 40)
    if english_duplicates:
        for id_val, count in sorted(english_duplicates.items()):
            report.append(f"   - '{id_val}' muncul {count} kali")
        report.append(f"\n   Total ID duplikat: {len(english_duplicates)}")
    else:
        report.append("   Tidak ada ID duplikat ditemukan")
    report.append("")
    
    # ID Duplikat dalam file Indonesian
    report.append("2. ID DUPLIKAT DALAM FILE INDONESIAN:")
    report.append("-" * 40)
    if indonesian_duplicates:
        for id_val, count in sorted(indonesian_duplicates.items()):
            report.append(f"   - '{id_val}' muncul {count} kali")
        report.append(f"\n   Total ID duplikat: {len(indonesian_duplicates)}")
    else:
        report.append("   Tidak ada ID duplikat ditemukan")
    report.append("")
    
    # ID yang hilang dalam file Indonesian
    report.append("3. ID YANG HILANG DALAM FILE INDONESIAN:")
    report.append("-" * 40)
    if missing_in_indonesian:
        report.append(f"   Total ID yang hilang: {len(missing_in_indonesian)}")
        report.append("   ID yang hilang:")
        for id_val in sorted(missing_in_indonesian):
            report.append(f"   - '{id_val}'")
    else:
        report.append("   Semua ID dari file English ada dalam file Indonesian")
    report.append("")
    
    # ID yang ada dalam file Indonesian tapi tidak dalam English
    report.append("4. ID TAMBAHAN DALAM FILE INDONESIAN:")
    report.append("-" * 40)
    if missing_in_english:
        report.append(f"   Total ID tambahan: {len(missing_in_english)}")
        report.append("   ID tambahan:")
        for id_val in sorted(missing_in_english):
            report.append(f"   - '{id_val}'")
    else:
        report.append("   Tidak ada ID tambahan dalam file Indonesian")
    report.append("")
    
    # Ringkasan
    report.append("RINGKASAN:")
    report.append("-" * 40)
    total_issues = len(english_duplicates) + len(indonesian_duplicates) + len(missing_in_indonesian) + len(missing_in_english)
    
    if total_issues == 0:
        report.append("✅ Tidak ada masalah ditemukan! Kedua file memiliki ID yang konsisten.")
    else:
        report.append(f"⚠️  Total masalah ditemukan: {total_issues}")
        if english_duplicates:
            report.append(f"   - {len(english_duplicates)} ID duplikat dalam file English")
        if indonesian_duplicates:
            report.append(f"   - {len(indonesian_duplicates)} ID duplikat dalam file Indonesian")
        if missing_in_indonesian:
            report.append(f"   - {len(missing_in_indonesian)} ID hilang dalam file Indonesian")
        if missing_in_english:
            report.append(f"   - {len(missing_in_english)} ID tambahan dalam file Indonesian")
    
    report.append("")
    report.append("=" * 80)
    
    return "\n".join(report)

def main():
    """Fungsi utama untuk menjalankan verifikasi ID"""
    # Path file
    english_file = "d:/Project/thiefs2-to-indo/LocalizationFile_english_extracted.json"
    indonesian_file = "d:/Project/thiefs2-to-indo/test_output_concurrent.json"
    
    print("Memuat file JSON...")
    
    # Memuat data dari kedua file
    english_data = load_json_file(english_file)
    if english_data is None:
        return 1
    
    indonesian_data = load_json_file(indonesian_file)
    if indonesian_data is None:
        return 1
    
    print("Mengekstrak ID dari kedua file...")
    
    # Ekstrak ID dari kedua file
    english_ids = extract_ids(english_data)
    indonesian_ids = extract_ids(indonesian_data)
    
    if not english_ids:
        print("Error: Tidak ada ID ditemukan dalam file English")
        return 1
    
    if not indonesian_ids:
        print("Error: Tidak ada ID ditemukan dalam file Indonesian")
        return 1
    
    print("Menganalisis perbedaan...")
    
    # Cari duplikat dalam masing-masing file
    english_duplicates = find_duplicates(english_ids)
    indonesian_duplicates = find_duplicates(indonesian_ids)
    
    # Cari ID yang hilang
    missing_in_indonesian, missing_in_english = find_missing_ids(english_ids, indonesian_ids)
    
    # Generate laporan
    report = generate_report(
        english_ids, indonesian_ids,
        english_duplicates, indonesian_duplicates,
        missing_in_indonesian, missing_in_english
    )
    
    # Tampilkan laporan
    print(report)
    
    # Simpan laporan ke file
    report_file = "d:/Project/thiefs2-to-indo/id_verification_report.txt"
    try:
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\nLaporan telah disimpan ke: {report_file}")
    except Exception as e:
        print(f"\nError: Gagal menyimpan laporan - {e}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())