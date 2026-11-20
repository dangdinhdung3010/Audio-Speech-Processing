import os
import sys
import numpy as np
import scipy.signal as signal
import soundfile as sf
import pandas as pd
import matplotlib.pyplot as plt

# Ensure UTF-8 output on Windows terminal
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


# ---------------------------------------------------------
# Step 1: Text Normalization
# ---------------------------------------------------------
DIGIT = {
    "0": "không", "1": "một", "2": "hai", "3": "ba", "4": "bốn",
    "5": "năm", "6": "sáu", "7": "bảy", "8": "tám", "9": "chín"
}

def normalize_queue_text(text: str) -> list[str]:
    """
    Chuẩn hóa văn bản trong miền hẹp 'thông báo xếp hàng':
    - Lowercase
    - Bỏ dấu chấm (.), phẩy (,)
    - Tách token và đọc chuỗi số theo từng chữ số
    """
    text = text.lower().replace(".", "").replace(",", "")
    out = []
    for tok in text.split():
        if tok.isdigit():
            out.extend(DIGIT[c] for c in tok)
        else:
            out.append(tok)
    return out

# ---------------------------------------------------------
# Step 2: Source-Filter Speech Synthesizer for 16 Words
# ---------------------------------------------------------

WORD_SPECS = {
    "moi": {
        "word": "mời",
        "type": "voiced_diphthong",
        "consonant": "m",
        "f_start": [500, 1350, 2400, 3500],
        "f_end":   [360, 2100, 2800, 3500],
        "tone": "huyen",
        "base_dur": 0.32
    },
    "khach": {
        "word": "khách",
        "type": "unvoiced_stop_coda",
        "consonant": "kh",
        "f_start": [750, 1350, 2500, 3500],
        "f_end":   [600, 1800, 2600, 3500],
        "coda": "ch",
        "tone": "sac",
        "base_dur": 0.30
    },
    "hang": {
        "word": "hàng",
        "type": "voiced_nasal_coda",
        "consonant": "h",
        "f_start": [780, 1250, 2500, 3500],
        "f_end":   [650, 1100, 2400, 3500],
        "coda": "ng",
        "tone": "huyen",
        "base_dur": 0.34
    },
    "so": {
        "word": "số",
        "type": "fricative_vowel",
        "consonant": "s",
        "f_start": [520, 920, 2400, 3400],
        "f_end":   [480, 880, 2400, 3400],
        "tone": "sac",
        "base_dur": 0.31
    },
    "khong": {
        "word": "không",
        "type": "voiced_nasal_coda",
        "consonant": "kh",
        "f_start": [480, 880, 2350, 3400],
        "f_end":   [400, 800, 2300, 3400],
        "coda": "ng",
        "tone": "ngang",
        "base_dur": 0.33
    },
    "mot": {
        "word": "một",
        "type": "voiced_stop_coda",
        "consonant": "m",
        "f_start": [520, 960, 2400, 3400],
        "f_end":   [450, 1400, 2400, 3400],
        "coda": "t",
        "tone": "nang",
        "base_dur": 0.25
    },
    "hai": {
        "word": "hai",
        "type": "voiced_diphthong",
        "consonant": "h",
        "f_start": [760, 1280, 2500, 3500],
        "f_end":   [360, 2150, 2800, 3500],
        "tone": "ngang",
        "base_dur": 0.32
    },
    "ba": {
        "word": "ba",
        "type": "voiced_stop_onset",
        "consonant": "b",
        "f_start": [780, 1260, 2450, 3500],
        "f_end":   [760, 1220, 2450, 3500],
        "tone": "ngang",
        "base_dur": 0.30
    },
    "bon": {
        "word": "bốn",
        "type": "voiced_stop_onset_nasal_coda",
        "consonant": "b",
        "f_start": [500, 920, 2400, 3400],
        "f_end":   [450, 1450, 2450, 3400],
        "coda": "n",
        "tone": "sac",
        "base_dur": 0.30
    },
    "nam": {
        "word": "năm",
        "type": "voiced_nasal_coda",
        "consonant": "n",
        "f_start": [750, 1300, 2450, 3500],
        "f_end":   [600, 1100, 2400, 3500],
        "coda": "m",
        "tone": "ngang",
        "base_dur": 0.32
    },
    "sau": {
        "word": "sáu",
        "type": "fricative_vowel",
        "consonant": "s",
        "f_start": [740, 1320, 2450, 3500],
        "f_end":   [450, 850, 2300, 3400],
        "tone": "sac",
        "base_dur": 0.31
    },
    "bay": {
        "word": "bảy",
        "type": "voiced_diphthong",
        "consonant": "b",
        "f_start": [720, 1320, 2450, 3500],
        "f_end":   [380, 2050, 2750, 3500],
        "tone": "hoi",
        "base_dur": 0.32
    },
    "tam": {
        "word": "tám",
        "type": "stop_onset_nasal_coda",
        "consonant": "t",
        "f_start": [780, 1260, 2450, 3500],
        "f_end":   [600, 1100, 2400, 3500],
        "coda": "m",
        "tone": "sac",
        "base_dur": 0.30
    },
    "chin": {
        "word": "chín",
        "type": "stop_onset_nasal_coda",
        "consonant": "ch",
        "f_start": [320, 2250, 3050, 3700],
        "f_end":   [300, 1850, 2800, 3600],
        "coda": "n",
        "tone": "sac",
        "base_dur": 0.30
    },
    "den": {
        "word": "đến",
        "type": "voiced_stop_onset_nasal_coda",
        "consonant": "d",
        "f_start": [520, 1850, 2600, 3600],
        "f_end":   [480, 1750, 2600, 3600],
        "coda": "n",
        "tone": "sac",
        "base_dur": 0.30
    },
    "quay": {
        "word": "quầy",
        "type": "voiced_diphthong",
        "consonant": "qu",
        "f_start": [420, 800, 2300, 3400],
        "f_end":   [380, 2050, 2750, 3500],
        "tone": "huyen",
        "base_dur": 0.33
    }
}

def generate_glottal_excitation(n_samples: int, f0_contour: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Sinh chuỗi xung glottal (Rosenberg glottal pulse) với F0 contour biến thiên."""
    glottal = np.zeros(n_samples, dtype=np.float32)
    phase = 0.0
    for i in range(n_samples):
        f0 = f0_contour[i]
        t0 = sr / max(f0, 40.0)
        # Rosenberg pulse within period
        p = phase % t0
        n1 = 0.40 * t0
        n2 = 0.16 * t0
        if p < n1:
            val = 0.5 * (1.0 - np.cos(np.pi * p / n1))
        elif p < n1 + n2:
            val = np.cos(np.pi * (p - n1) / (2.0 * n2))
        else:
            val = 0.0
        glottal[i] = val
        phase += 1.0
    # Lip radiation: +6 dB/octave tilt (differentiation)
    glottal_diff = np.diff(glottal, prepend=glottal[0])
    # Add tiny aspiration noise for natural voice texture
    noise = np.random.normal(0, 0.015, n_samples)
    return glottal_diff + noise

def apply_formant_resonator(x: np.ndarray, f_c: float, bw: float, sr: int = 16000) -> np.ndarray:
    """Áp dụng 2-pole resonant filter cho formant f_c với bandwidth bw."""
    r = np.exp(-np.pi * bw / sr)
    theta = 2.0 * np.pi * f_c / sr
    a1 = -2.0 * r * np.cos(theta)
    a2 = r * r
    b0 = 1.0 - r
    return signal.lfilter([b0], [1.0, a1, a2], x)

def synthesize_word_audio(spec: dict, candidate_idx: int, sr: int = 16000) -> np.ndarray:
    """
    Tổng hợp một mẫu từ theo mô hình source-filter với biến thiên giữa 3 candidate:
    - candidate 1 (idx=0): duration chuẩn, F0 ~ 142 Hz, target RMS ~ 0.080
    - candidate 2 (idx=1): duration +12%, F0 ~ 155 Hz, target RMS ~ 0.088
    - candidate 3 (idx=2): duration -10%, F0 ~ 136 Hz, target RMS ~ 0.072
    - Đệm 20 ms silence (320 samples) ở hai đầu.
    """
    dur_factors = [1.0, 1.14, 0.88]
    f0_offsets = [0.0, 12.0, -8.0]
    rms_targets = [0.080, 0.088, 0.072]
    
    dur = spec["base_dur"] * dur_factors[candidate_idx]
    f0_base = 142.0 + f0_offsets[candidate_idx]
    rms_target = rms_targets[candidate_idx]
    
    n_active = int(dur * sr)
    t = np.linspace(0, 1, n_active)
    
    # 1. Pitch contour based on Vietnamese tone
    tone = spec["tone"]
    if tone == "ngang":
        f0_contour = f0_base * (1.0 + 0.02 * np.sin(2 * np.pi * t))
    elif tone == "huyen":
        f0_contour = f0_base * (1.05 - 0.20 * t)
    elif tone == "sac":
        f0_contour = f0_base * (0.95 + 0.25 * (t ** 1.3))
    elif tone == "nang":
        # Sharp drop & glottal constriction near end
        f0_contour = f0_base * (1.0 - 0.28 * (t ** 0.8))
    elif tone == "hoi":
        # Dipping tone: down then up
        f0_contour = f0_base * (1.0 - 0.18 * np.sin(np.pi * t))
    else:
        f0_contour = np.full(n_active, f0_base)
        
    # 2. Source excitation
    glottal = generate_glottal_excitation(n_active, f0_contour, sr=sr)
    
    # Smooth temporal envelope for vocalic core
    env = signal.windows.tukey(n_active, alpha=0.25)
    if tone == "nang":
        # Abrupt cutoff
        env[int(n_active * 0.75):] *= np.linspace(1.0, 0.0, n_active - int(n_active * 0.75)) ** 2
        
    voiced_src = glottal * env
    
    # 3. Formant trajectories
    f_start = np.array(spec["f_start"], dtype=float)
    f_end = np.array(spec["f_end"], dtype=float)
    # Bandwidths for F1-F4
    bws = [80.0, 110.0, 160.0, 220.0]
    
    # Time-varying formant filtering via frame processing (overlap-add, 10ms hop)
    frame_len = int(0.025 * sr)
    hop_len = int(0.010 * sr)
    n_frames = max(1, (n_active - frame_len) // hop_len + 1)
    
    filtered_speech = np.zeros(n_active + frame_len)
    window = signal.windows.hann(frame_len)
    
    for i in range(n_frames):
        start = i * hop_len
        end = start + frame_len
        frame = voiced_src[start:min(end, n_active)]
        if len(frame) < frame_len:
            frame = np.pad(frame, (0, frame_len - len(frame)))
            
        rel_pos = (start + frame_len / 2) / n_active
        f_cur = (1.0 - rel_pos) * f_start + rel_pos * f_end
        
        # Cascade formant filtering
        y = frame.copy()
        for fc, bw in zip(f_cur, bws):
            y = apply_formant_resonator(y, fc, bw, sr=sr)
            
        filtered_speech[start:end] += y * window
        
    speech_core = filtered_speech[:n_active]
    
    # 4. Consonant onsets / codas (fricatives, stops, bursts)
    consonant = spec.get("consonant", "")
    if consonant == "s":
        # High-frequency frication noise for ~70ms
        n_c = int(0.070 * sr)
        noise = np.random.normal(0, 0.4, n_c)
        # Bandpass 4000 - 7500 Hz
        b_bp, a_bp = signal.butter(4, [4000 / (sr/2), 7500 / (sr/2)], btype='band')
        fric = signal.lfilter(b_bp, a_bp, noise) * np.linspace(0.8, 0.1, n_c)
        speech_core[:n_c] = speech_core[:n_c] * 0.3 + fric * 0.7
    elif consonant == "kh":
        # Velar frication noise 1200 - 3500 Hz for ~50ms
        n_c = int(0.050 * sr)
        noise = np.random.normal(0, 0.35, n_c)
        b_bp, a_bp = signal.butter(4, [1200 / (sr/2), 3500 / (sr/2)], btype='band')
        fric = signal.lfilter(b_bp, a_bp, noise) * np.linspace(0.9, 0.1, n_c)
        speech_core[:n_c] = speech_core[:n_c] * 0.4 + fric * 0.6
    elif consonant == "h":
        # Glottal aspiration noise for ~40ms
        n_c = int(0.040 * sr)
        noise = np.random.normal(0, 0.25, n_c)
        speech_core[:n_c] = speech_core[:n_c] * 0.5 + noise * np.linspace(0.5, 0.05, n_c)
    elif consonant in ["t", "ch", "d", "b"]:
        # Stop burst / onset
        n_c = int(0.025 * sr)
        burst_freq = 4200.0 if consonant in ["t", "ch"] else 1500.0
        burst = np.random.normal(0, 0.3, n_c) * np.exp(-np.linspace(0, 6, n_c))
        b_bp, a_bp = signal.butter(2, min(burst_freq, 7000.0) / (sr / 2.0), btype='high')
        burst = signal.lfilter(b_bp, a_bp, burst)
        speech_core[:n_c] = speech_core[:n_c] * 0.3 + burst * 0.7
        
    coda = spec.get("coda", "")
    if coda == "ch":
        # Palatal stop / unreleased burst at end
        n_end = int(0.030 * sr)
        burst = np.random.normal(0, 0.25, n_end) * np.exp(-np.linspace(6, 0, n_end))
        speech_core[-n_end:] = speech_core[-n_end:] * 0.4 + burst * 0.6
    elif coda in ["m", "n", "ng"]:
        # Nasal murmur tail
        n_end = int(0.045 * sr)
        speech_core[-n_end:] *= np.linspace(1.0, 0.4, n_end)
        
    # Scale to desired candidate RMS
    current_rms = np.sqrt(np.mean(speech_core ** 2) + 1e-12)
    speech_core = speech_core * (rms_target / current_rms)
    speech_core = np.clip(speech_core, -0.95, 0.95)
    
    # 5. Add 20ms silence padding at both ends (20ms * 16000 / 1000 = 320 samples)
    n_silence = int(0.020 * sr)
    silence = np.zeros(n_silence, dtype=np.float32)
    final_audio = np.concatenate([silence, speech_core.astype(np.float32), silence])
    
    return final_audio

# ---------------------------------------------------------
# Step 3: Required Standard Audio Normalization Functions
# ---------------------------------------------------------

def load_unit(path: str, sr: int = 16000) -> np.ndarray:
    """Đọc file float32, chuyển mono nếu cần, assert fs == sr."""
    x, fs = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    assert fs == sr, f"Sampling rate mismatch: {fs} != {sr}"
    return x

def rms(x: np.ndarray) -> float:
    """Tính Root Mean Square energy."""
    return float(np.sqrt(np.mean(x * x) + 1e-12))

def match_rms(x: np.ndarray, target: float = 0.08) -> np.ndarray:
    """Chuẩn hóa RMS về target (mặc định 0.08), giới hạn clip (-0.99, 0.99)."""
    y = x * (target / rms(x))
    return np.clip(y, -0.99, 0.99)

# ---------------------------------------------------------
# Step 4: Verification & Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    print("=== KIỂM CHỨNG BƯỚC 1 - LAB 4 ===")
    
    # 1. Test Text Normalization
    test_text = "Mời khách hàng số 23 đến quầy 5."
    tokens = normalize_queue_text(test_text)
    print(f"Câu gốc: \"{test_text}\"")
    print(f"Tokens chuẩn hóa: {tokens}")
    expected = ["mời", "khách", "hàng", "số", "hai", "ba", "đến", "quầy", "năm"]
    assert tokens == expected, f"Lỗi normalization: {tokens} != {expected}"
    print("-> Test normalize_queue_text: PASSED!\n")
    
    # 2. Sinh Inventory mẫu nếu chưa có
    units_dir = "lab4_data/units"
    os.makedirs(units_dir, exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    
    generated_count = 0
    words = list(WORD_SPECS.keys())
    print(f"Khởi tạo / Kiểm tra Inventory ({len(words)} từ, 3 candidates/từ = 48 units)...")
    
    for word_key in words:
        spec = WORD_SPECS[word_key]
        for c_idx in range(3):
            cand_id = f"0{c_idx+1}"
            filename = f"{word_key}_{cand_id}.wav"
            filepath = os.path.join(units_dir, filename)
            if not os.path.exists(filepath):
                audio = synthesize_word_audio(spec, c_idx, sr=16000)
                sf.write(filepath, audio, 16000, subtype="PCM_16")
                generated_count += 1
                
    print(f"-> Đã sinh mới {generated_count} file WAV mẫu.")
    
    # 3. Quét inventory và xuất metadata.csv
    records = []
    wav_files = sorted([f for f in os.listdir(units_dir) if f.endswith(".wav")])
    print(f"Đang quét {len(wav_files)} file trong {units_dir}...")
    
    for fname in wav_files:
        # e.g. moi_01.wav -> word_key='moi', cand_id='01'
        parts = fname[:-4].rsplit("_", 1)
        word_key = parts[0]
        cand_id = parts[1]
        vn_label = WORD_SPECS[word_key]["word"] if word_key in WORD_SPECS else word_key
        
        fpath = os.path.join(units_dir, fname).replace("\\", "/")
        x = load_unit(fpath, sr=16000)
        dur = len(x) / 16000.0
        r_raw = rms(x)
        x_norm = match_rms(x, target=0.08)
        r_norm = rms(x_norm)
        
        records.append({
            "label": vn_label,
            "candidate_id": cand_id,
            "speaker": "spk1",
            "duration": round(dur, 4),
            "rms_raw": round(r_raw, 6),
            "rms_norm": round(r_norm, 6),
            "file_path": fpath
        })
        
    df = pd.DataFrame(records)
    csv_path = "lab4_data/metadata.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"-> Đã xuất metadata: {csv_path} ({len(df)} dòng)")
    print(df.head(10))
    
    # 4. Vẽ Hình 1 bắt buộc: figures/fig1_inventory_stats.png
    print("\nĐang tạo Hình 1: figures/fig1_inventory_stats.png...")
    plt.figure(figsize=(12, 5), dpi=300)
    
    # Subplot 1: Duration Distribution
    plt.subplot(1, 2, 1)
    n_dur, bins_dur, _ = plt.hist(df["duration"] * 1000, bins=12, color="#2b5c8f", edgecolor="black", alpha=0.85)
    mean_dur = df["duration"].mean() * 1000
    plt.axvline(mean_dur, color="#d9534f", linestyle="--", linewidth=2, label=f"Mean: {mean_dur:.1f} ms")
    plt.title("Phân bố Duration của các Unit", fontsize=13, fontweight="bold", pad=10)
    plt.xlabel("Duration (ms)", fontsize=11)
    plt.ylabel("Số lượng Unit", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, facecolor="white", loc="upper right")
    
    # Subplot 2: RMS Distribution (Raw vs Normalized)
    plt.subplot(1, 2, 2)
    plt.hist(df["rms_raw"], bins=10, color="#f0ad4e", edgecolor="black", alpha=0.7, label=f"Raw RMS (Mean: {df['rms_raw'].mean():.4f})")
    plt.hist(df["rms_norm"], bins=10, color="#5cb85c", edgecolor="black", alpha=0.7, label=f"Norm RMS (Mean: {df['rms_norm'].mean():.4f})")
    plt.axvline(0.08, color="#d9534f", linestyle="--", linewidth=2.0, label="Target RMS (0.08)")
    plt.title("Phân bố RMS trước và sau match_rms()", fontsize=13, fontweight="bold", pad=10)
    plt.xlabel("RMS Energy", fontsize=11)
    plt.ylabel("Số lượng Unit", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, facecolor="white", loc="upper left")
    
    plt.tight_layout()
    fig1_path = "figures/fig1_inventory_stats.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"-> Đã lưu Hình 1: {fig1_path}")
    print("=== HOÀN TẤT BƯỚC 1 THÀNH CÔNG ===")
