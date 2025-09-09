# Setup GPU Intel ARC A380 untuk Thiefs2-to-Indo

Panduan lengkap untuk mengoptimalkan sistem terjemahan dengan GPU Intel ARC A380.

## Persyaratan Sistem

### Hardware
- GPU Intel ARC A380 atau seri ARC lainnya
- RAM minimal 8GB (16GB direkomendasikan)
- Storage minimal 10GB untuk model ML

### Software
- Windows 10/11 dengan driver Intel ARC terbaru
- Python 3.8 atau lebih baru
- Intel oneAPI Base Toolkit (opsional, untuk performa optimal)

## Instalasi

### 1. Install PyTorch 2.5+ dengan Dukungan XPU Built-in

```bash
# Install PyTorch dengan dukungan Intel XPU (official PyTorch 2.5+)
pip install torch>=2.5.0 torchvision>=0.20.0 torchaudio>=2.5.0 --index-url https://download.pytorch.org/whl/xpu

# Install dependensi lainnya
pip install transformers>=4.21.0
pip install tqdm>=4.64.0
pip install sentencepiece>=0.1.97
pip install sacremoses>=0.0.53
```

### 2. Install Intel Extension for PyTorch (Opsional)

```bash
# Opsional: untuk optimasi tambahan
pip install intel-extension-for-pytorch>=2.0.0
pip install mkl>=2023.0.0
```

**Catatan:** PyTorch 2.5+ memiliki dukungan Intel XPU built-in. Intel Extension for PyTorch sekarang opsional dan hanya diperlukan untuk optimasi tambahan.

### 3. Verifikasi Driver GPU

Pastikan driver Intel ARC terbaru terinstall:
- Download dari [Intel Driver & Support Assistant](https://www.intel.com/content/www/us/en/support/detect.html)
- Atau gunakan Intel Arc Control untuk update otomatis

## Penggunaan

### Test Kompatibilitas

Jalankan script test untuk memverifikasi setup:

```bash
python test_arc_gpu.py
```

Output yang diharapkan:
```
✓ Intel Extension for PyTorch tersedia
✓ Intel XPU tersedia: 1 perangkat
  - XPU 0: Intel(R) Arc(TM) A380 Graphics
✓ Model berhasil dimuat dalam X.XX detik
✓ GPU Intel ARC terdeteksi dan digunakan
✓ Optimasi XPU aktif
```

### Menjalankan Terjemahan

Sistem akan otomatis mendeteksi dan menggunakan GPU ARC:

```bash
# Terjemahan file JSON
python translate.py --input input.json --output output.json

# Terjemahan teks langsung
python translate.py --text "Hello world"
```

## Optimasi Performa

### 1. Konfigurasi Memory

Sistem secara otomatis mengoptimalkan penggunaan memori untuk ARC A380:
- Batch size disesuaikan dengan VRAM yang tersedia
- Beam search dikurangi untuk efisiensi (1 beam vs 2 beam)
- Max sequence length dioptimalkan (128 vs 256 token)

### 2. Monitoring Performa

Sistem menyediakan monitoring real-time:
```
Intel XPU optimization: batch_size=10, max_length=128
XPU Memory allocated: 1024.5 MB
```

### 3. Fallback Otomatis

Jika GPU ARC tidak tersedia atau mengalami masalah:
1. Sistem akan otomatis fallback ke CUDA (jika tersedia)
2. Jika CUDA tidak tersedia, fallback ke CPU
3. Tidak ada intervensi manual yang diperlukan

## Troubleshooting

### GPU Tidak Terdeteksi

**Gejala:**
```
✗ Intel XPU tidak tersedia
Using CPU device
```

**Solusi:**
1. Pastikan driver Intel ARC terbaru terinstall
2. Restart sistem setelah install driver
3. Verifikasi GPU terdeteksi di Device Manager
4. Install Intel Extension for PyTorch:
   ```bash
   pip install intel-extension-for-pytorch --upgrade
   ```

### Error "Cannot copy out of meta tensor"

**Gejala:**
```
Warning: Meta tensor issue detected but continuing...
```

**Solusi:**
Sistem sudah menangani error ini secara otomatis. Warning ini normal dan tidak mempengaruhi fungsionalitas.

### Performa Lambat

**Gejala:**
Throughput rendah (<1 teks/detik)

**Solusi:**
1. Pastikan GPU ARC terdeteksi dengan benar
2. Tutup aplikasi lain yang menggunakan GPU
3. Increase virtual memory jika diperlukan
4. Monitor temperature GPU (gunakan Intel Arc Control)

### Out of Memory Error

**Gejala:**
```
RuntimeError: out of memory
```

**Solusi:**
1. Sistem akan otomatis mengurangi batch size
2. Jika masih error, akan fallback ke CPU
3. Untuk file besar, gunakan parameter `--batch-size` yang lebih kecil

## Perbandingan Performa

| Device | Throughput | Memory Usage | Power Consumption |
|--------|------------|--------------|-------------------|
| Intel ARC A380 | ~15-25 teks/detik | ~2-4GB VRAM | ~75W |
| NVIDIA RTX 3060 | ~20-30 teks/detik | ~3-5GB VRAM | ~170W |
| CPU (Intel i5) | ~5-8 teks/detik | ~4-8GB RAM | ~65W |

*Performa aktual dapat bervariasi tergantung sistem dan konfigurasi.

## Fitur Khusus ARC A380

### 1. Optimasi Otomatis
- Deteksi otomatis kapabilitas GPU
- Penyesuaian parameter berdasarkan VRAM
- Memory management yang efisien

### 2. Monitoring Real-time
- Tracking penggunaan memori VRAM
- Monitoring throughput dan latency
- Pelaporan error dan warning

### 3. Fallback Cerdas
- Deteksi error GPU secara real-time
- Fallback otomatis ke device alternatif
- Recovery otomatis setelah error

## Dukungan dan Kontribusi

Jika mengalami masalah atau ingin berkontribusi:
1. Buat issue di GitHub repository
2. Sertakan output dari `python test_arc_gpu.py`
3. Sertakan informasi sistem (OS, driver version, dll)

## Referensi

- [Intel Extension for PyTorch Documentation](https://intel.github.io/intel-extension-for-pytorch/)
- [Intel ARC GPU Developer Guide](https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html)
- [PyTorch XPU Backend](https://pytorch.org/docs/stable/notes/extending.html)