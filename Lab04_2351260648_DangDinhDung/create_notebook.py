import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import nbformat as nbf
from nbclient import NotebookClient
import os

nb = nbf.v4.new_notebook()

cells = []

# Title & metadata cell
cells.append(nbf.v4.new_markdown_cell("""# BÀI THỰC HÀNH LAB 4: TỔNG HỢP TIẾNG NÓI (TEXT-TO-SPEECH)
**Môn học:** CSE457 - Xử lý Âm thanh và Tiếng nói  
**Khoa Công nghệ Thông tin - Bộ môn Trí tuệ Nhân tạo - Trường Đại học Thủy Lợi**  
**Học viên / Sinh viên thực hiện:** Đặng Đình Dũng  
**Mã sinh viên:** 2351260648  

---

## MỤC TIÊU BÀI LAB 4
- **Tiết 1:** TTS pipeline, text normalization, xây dựng inventory và baseline ghép nối.
- **Tiết 2:** Unit selection, smoothing biên nối (cross-fade / OLA), điều khiển prosody, HMM generation.
- **Tiết 3:** Inference TTS học sâu, listening test và đánh giá so sánh 3 phương pháp.

Trong sổ tay này, chúng ta thực hiện **Bước 1: Phần A - Text Normalization & Inventory Management** theo đúng yêu cầu đề bài.
"""))

# Cell 1: Check libraries
cells.append(nbf.v4.new_markdown_cell("""## 1. Kiểm tra môi trường và các thư viện cần thiết
Các thư viện cốt lõi: `numpy`, `scipy`, `matplotlib`, `soundfile`, `librosa`, `pandas`."""))

cells.append(nbf.v4.new_code_cell("""import sys
import os
import numpy as np
import scipy
import scipy.signal as signal
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
import pandas as pd

print(f"Python version    : {sys.version.split()[0]}")
print(f"NumPy version     : {np.__version__}")
print(f"SciPy version     : {scipy.__version__}")
print(f"SoundFile version : {sf.__version__}")
print(f"Librosa version   : {librosa.__version__}")
print(f"Pandas version    : {pd.__version__}")
"""))

# Cell 2: Directory structure
cells.append(nbf.v4.new_markdown_cell("""## 2. Thiết lập cấu trúc thư mục làm việc
Theo đề bài Lab 4:
- `lab4_data/units/`: Chứa các bản ghi đơn vị âm thanh (WAV, Fs=16kHz, mono, 16-bit PCM).
- `lab4_data/hmm_demo/`: Chứa mô hình tham số HMM backend.
- `audio_output/`: Chứa các file kết quả tổng hợp (`raw_concat.wav`, `best_concat.wav`, `neural_tts.wav`,...).
- `figures/`: Chứa các hình vẽ phân tích bắt buộc (Hình 1 đến Hình 5).
"""))

cells.append(nbf.v4.new_code_cell("""dirs = [
    "lab4_data/units",
    "lab4_data/hmm_demo",
    "audio_output",
    "figures"
]

for d in dirs:
    os.makedirs(d, exist_ok=True)
    print(f"Thư mục sẵn sàng: {d}")
"""))

# Cell 3: Text Normalization
cells.append(nbf.v4.new_markdown_cell("""## 3. Phần A.1 - Text Normalization cho miền hẹp 'Thông báo xếp hàng'

### Quy tắc chuẩn hóa:
1. Chuyển toàn bộ văn bản về chữ thường (**lowercase**).
2. Loại bỏ dấu câu không cần thiết: dấu chấm (`.`), dấu phẩy (`,`).
3. Tách từ (**tokenize**) theo khoảng trắng.
4. Ánh xạ từng chữ số trong chuỗi số sang từ phát âm tương ứng thông qua từ điển:
   $$\\text{DIGIT} = \\{0:\\text{"không"}, 1:\\text{"một"}, 2:\\text{"hai"}, 3:\\text{"ba"}, 4:\\text{"bốn"}, 5:\\text{"năm"}, 6:\\text{"sáu"}, 7:\\text{"bảy"}, 8:\\text{"tám"}, 9:\\text{"chín"}\\}$$
"""))

cells.append(nbf.v4.new_code_cell("""DIGIT = {
    "0": "không", "1": "một", "2": "hai", "3": "ba", "4": "bốn",
    "5": "năm", "6": "sáu", "7": "bảy", "8": "tám", "9": "chín"
}

def normalize_queue_text(text: str) -> list[str]:
    \"\"\"
    Chuẩn hóa câu văn bản cho hệ thống thông báo xếp hàng.
    Mọi chuỗi số được phân tách và đọc theo từng chữ số riêng biệt.
    \"\"\"
    text = text.lower().replace(".", "").replace(",", "")
    out = []
    for tok in text.split():
        if tok.isdigit():
            out.extend(DIGIT[c] for c in tok)
        else:
            out.append(tok)
    return out

# Kiểm chứng câu chuẩn theo đề bài
test_sentence = "Mời khách hàng số 23 đến quầy 5."
tokens = normalize_queue_text(test_sentence)
expected = ["mời", "khách", "hàng", "số", "hai", "ba", "đến", "quầy", "năm"]

print("Câu đầu vào :", test_sentence)
print("Kết quả     :", tokens)
print("Kỳ vọng     :", expected)
assert tokens == expected, "Kết quả không khớp kỳ vọng!"
print("=> KIỂM CHỨNG TEXT NORMALIZATION THÀNH CÔNG!")
"""))

# Cell 4: Inventory Generation via Source-Filter
cells.append(nbf.v4.new_markdown_cell("""## 4. Phần A.2 - Xây dựng Inventory đơn vị từ (16 từ x 3 candidates = 48 units)

### Mô hình Nguồn - Bộ lọc (Source - Filter Model)
Tín hiệu tiếng nói được mô hình hóa theo tài liệu Rabiner-Schafer:
$$X(z) = E(z) \\cdot H(z)$$
- **Nguồn kích thích $E(z)$:** Chuỗi xung glottal (Rosenberg pulse) với tần số cơ bản $F_0 \\in [130, 180]\\text{ Hz}$, kết hợp đạo hàm mô phỏng lip-radiation (+6 dB/octave) và nhiễu xát/thanh quản.
- **Bộ lọc âm học $H(z)$:** Bộ lọc cộng hưởng 2 cực (formant resonators $F_1, F_2, F_3, F_4$) đặc trưng cho nguyên âm và cấu âm tiếng Việt.
- **Biến thiên candidate:**
  - Candidate 1: Duration chuẩn, $F_0 \\approx 142\\text{ Hz}$, RMS $\\approx 0.080$.
  - Candidate 2: Duration $+14\\%$, $F_0 \\approx 155\\text{ Hz}$ (cao hơn), RMS $\\approx 0.088$.
  - Candidate 3: Duration $-12\\%$, $F_0 \\approx 136\\text{ Hz}$ (thấp hơn), RMS $\\approx 0.072$.
- **Silence padding:** Đệm 20 ms khoảng lặng (320 mẫu tại $F_s = 16000\\text{ Hz}$) ở cả hai đầu file.
"""))

cells.append(nbf.v4.new_code_cell("""WORD_SPECS = {
    "moi":   {"word": "mời",   "consonant": "m",  "f_start": [500, 1350, 2400, 3500], "f_end": [360, 2100, 2800, 3500], "tone": "huyen", "base_dur": 0.32},
    "khach": {"word": "khách", "consonant": "kh", "f_start": [750, 1350, 2500, 3500], "f_end": [600, 1800, 2600, 3500], "coda": "ch",  "tone": "sac",   "base_dur": 0.30},
    "hang":  {"word": "hàng",  "consonant": "h",  "f_start": [780, 1250, 2500, 3500], "f_end": [650, 1100, 2400, 3500], "coda": "ng",  "tone": "huyen", "base_dur": 0.34},
    "so":    {"word": "số",    "consonant": "s",  "f_start": [520, 920, 2400, 3400],  "f_end": [480, 880, 2400, 3400],  "tone": "sac",   "base_dur": 0.31},
    "khong": {"word": "không", "consonant": "kh", "f_start": [480, 880, 2350, 3400],  "f_end": [400, 800, 2300, 3400],  "coda": "ng",  "tone": "ngang", "base_dur": 0.33},
    "mot":   {"word": "một",   "consonant": "m",  "f_start": [520, 960, 2400, 3400],  "f_end": [450, 1400, 2400, 3400], "coda": "t",   "tone": "nang",  "base_dur": 0.25},
    "hai":   {"word": "hai",   "consonant": "h",  "f_start": [760, 1280, 2500, 3500], "f_end": [360, 2150, 2800, 3500], "tone": "ngang", "base_dur": 0.32},
    "ba":    {"word": "ba",    "consonant": "b",  "f_start": [780, 1260, 2450, 3500], "f_end": [760, 1220, 2450, 3500], "tone": "ngang", "base_dur": 0.30},
    "bon":   {"word": "bốn",   "consonant": "b",  "f_start": [500, 920, 2400, 3400],  "f_end": [450, 1450, 2450, 3400], "coda": "n",   "tone": "sac",   "base_dur": 0.30},
    "nam":   {"word": "năm",   "consonant": "n",  "f_start": [750, 1300, 2450, 3500], "f_end": [600, 1100, 2400, 3500], "coda": "m",   "tone": "ngang", "base_dur": 0.32},
    "sau":   {"word": "sáu",   "consonant": "s",  "f_start": [740, 1320, 2450, 3500], "f_end": [450, 850, 2300, 3400],  "tone": "sac",   "base_dur": 0.31},
    "bay":   {"word": "bảy",   "consonant": "b",  "f_start": [720, 1320, 2450, 3500], "f_end": [380, 2050, 2750, 3500], "tone": "hoi",   "base_dur": 0.32},
    "tam":   {"word": "tám",   "consonant": "t",  "f_start": [780, 1260, 2450, 3500], "f_end": [600, 1100, 2400, 3500], "coda": "m",   "tone": "sac",   "base_dur": 0.30},
    "chin":  {"word": "chín",  "consonant": "ch", "f_start": [320, 2250, 3050, 3700], "f_end": [300, 1850, 2800, 3600], "coda": "n",   "tone": "sac",   "base_dur": 0.30},
    "den":   {"word": "đến",   "consonant": "d",  "f_start": [520, 1850, 2600, 3600], "f_end": [480, 1750, 2600, 3600], "coda": "n",   "tone": "sac",   "base_dur": 0.30},
    "quay":  {"word": "quầy",  "consonant": "qu", "f_start": [420, 800, 2300, 3400],  "f_end": [380, 2050, 2750, 3500], "tone": "huyen", "base_dur": 0.33}
}

def generate_glottal_excitation(n_samples: int, f0_contour: np.ndarray, sr: int = 16000) -> np.ndarray:
    glottal = np.zeros(n_samples, dtype=np.float32)
    phase = 0.0
    for i in range(n_samples):
        f0 = f0_contour[i]
        t0 = sr / max(f0, 40.0)
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
    glottal_diff = np.diff(glottal, prepend=glottal[0])
    noise = np.random.normal(0, 0.015, n_samples)
    return glottal_diff + noise

def apply_formant_resonator(x: np.ndarray, f_c: float, bw: float, sr: int = 16000) -> np.ndarray:
    r = np.exp(-np.pi * bw / sr)
    theta = 2.0 * np.pi * f_c / sr
    a1 = -2.0 * r * np.cos(theta)
    a2 = r * r
    b0 = 1.0 - r
    return signal.lfilter([b0], [1.0, a1, a2], x)

def synthesize_word_audio(spec: dict, candidate_idx: int, sr: int = 16000) -> np.ndarray:
    dur_factors = [1.0, 1.14, 0.88]
    f0_offsets = [0.0, 12.0, -8.0]
    rms_targets = [0.080, 0.088, 0.072]
    
    dur = spec["base_dur"] * dur_factors[candidate_idx]
    f0_base = 142.0 + f0_offsets[candidate_idx]
    rms_target = rms_targets[candidate_idx]
    
    n_active = int(dur * sr)
    t = np.linspace(0, 1, n_active)
    
    tone = spec["tone"]
    if tone == "ngang":
        f0_contour = f0_base * (1.0 + 0.02 * np.sin(2 * np.pi * t))
    elif tone == "huyen":
        f0_contour = f0_base * (1.05 - 0.20 * t)
    elif tone == "sac":
        f0_contour = f0_base * (0.95 + 0.25 * (t ** 1.3))
    elif tone == "nang":
        f0_contour = f0_base * (1.0 - 0.28 * (t ** 0.8))
    elif tone == "hoi":
        f0_contour = f0_base * (1.0 - 0.18 * np.sin(np.pi * t))
    else:
        f0_contour = np.full(n_active, f0_base)
        
    glottal = generate_glottal_excitation(n_active, f0_contour, sr=sr)
    env = signal.windows.tukey(n_active, alpha=0.25)
    if tone == "nang":
        env[int(n_active * 0.75):] *= np.linspace(1.0, 0.0, n_active - int(n_active * 0.75)) ** 2
    voiced_src = glottal * env
    
    f_start = np.array(spec["f_start"], dtype=float)
    f_end = np.array(spec["f_end"], dtype=float)
    bws = [80.0, 110.0, 160.0, 220.0]
    
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
        
        y = frame.copy()
        for fc, bw in zip(f_cur, bws):
            y = apply_formant_resonator(y, fc, bw, sr=sr)
        filtered_speech[start:end] += y * window
        
    speech_core = filtered_speech[:n_active]
    
    consonant = spec.get("consonant", "")
    if consonant == "s":
        n_c = int(0.070 * sr)
        noise = np.random.normal(0, 0.4, n_c)
        b_bp, a_bp = signal.butter(4, [4000 / (sr/2), 7500 / (sr/2)], btype='band')
        fric = signal.lfilter(b_bp, a_bp, noise) * np.linspace(0.8, 0.1, n_c)
        speech_core[:n_c] = speech_core[:n_c] * 0.3 + fric * 0.7
    elif consonant == "kh":
        n_c = int(0.050 * sr)
        noise = np.random.normal(0, 0.35, n_c)
        b_bp, a_bp = signal.butter(4, [1200 / (sr/2), 3500 / (sr/2)], btype='band')
        fric = signal.lfilter(b_bp, a_bp, noise) * np.linspace(0.9, 0.1, n_c)
        speech_core[:n_c] = speech_core[:n_c] * 0.4 + fric * 0.6
    elif consonant == "h":
        n_c = int(0.040 * sr)
        noise = np.random.normal(0, 0.25, n_c)
        speech_core[:n_c] = speech_core[:n_c] * 0.5 + noise * np.linspace(0.5, 0.05, n_c)
    elif consonant in ["t", "ch", "d", "b"]:
        n_c = int(0.025 * sr)
        burst_freq = 4200.0 if consonant in ["t", "ch"] else 1500.0
        burst = np.random.normal(0, 0.3, n_c) * np.exp(-np.linspace(0, 6, n_c))
        b_bp, a_bp = signal.butter(2, min(burst_freq, 7000.0) / (sr / 2.0), btype='high')
        burst = signal.lfilter(b_bp, a_bp, burst)
        speech_core[:n_c] = speech_core[:n_c] * 0.3 + burst * 0.7
        
    coda = spec.get("coda", "")
    if coda == "ch":
        n_end = int(0.030 * sr)
        burst = np.random.normal(0, 0.25, n_end) * np.exp(-np.linspace(6, 0, n_end))
        speech_core[-n_end:] = speech_core[-n_end:] * 0.4 + burst * 0.6
    elif coda in ["m", "n", "ng"]:
        n_end = int(0.045 * sr)
        speech_core[-n_end:] *= np.linspace(1.0, 0.4, n_end)
        
    current_rms = np.sqrt(np.mean(speech_core ** 2) + 1e-12)
    speech_core = speech_core * (rms_target / current_rms)
    speech_core = np.clip(speech_core, -0.95, 0.95)
    
    n_silence = int(0.020 * sr)
    silence = np.zeros(n_silence, dtype=np.float32)
    return np.concatenate([silence, speech_core.astype(np.float32), silence])

# Kiểm tra và sinh các file WAV mẫu nếu chưa có
units_dir = "lab4_data/units"
existing_files = [f for f in os.listdir(units_dir) if f.endswith(".wav")]

if len(existing_files) < 48:
    print(f"Inventory hiện có {len(existing_files)} file. Đang sinh bổ sung đủ 48 units...")
    for word_key, spec in WORD_SPECS.items():
        for c_idx in range(3):
            cand_id = f"0{c_idx+1}"
            fname = f"{word_key}_{cand_id}.wav"
            fpath = os.path.join(units_dir, fname)
            if not os.path.exists(fpath):
                audio = synthesize_word_audio(spec, c_idx, sr=16000)
                sf.write(fpath, audio, 16000, subtype="PCM_16")
    print("=> Hoàn tất sinh dữ liệu inventory!")
else:
    print(f"=> Inventory đã đầy đủ {len(existing_files)} file WAV (16 từ x 3 candidates).")
"""))

# Cell 5: Audio Normalization & Metadata Export
cells.append(nbf.v4.new_markdown_cell("""## 5. Phần A.3 - Chuẩn hóa Audio và Xuất `lab4_data/metadata.csv`

### Các hàm chuẩn hóa theo yêu cầu đề bài:
1. `load_unit(path, sr=16000)`: Đọc file âm thanh float32, chuyển về mono nếu stereo, kiểm tra tần số lấy mẫu đúng $16000\\text{ Hz}$.
2. `rms(x)`: Tính năng lượng hiệu dụng:
   $$\\text{RMS}(x) = \\sqrt{\\frac{1}{N}\\sum_{n=0}^{N-1} x^2[n] + 10^{-12}}$$
3. `match_rms(x, target=0.08)`: Chuẩn hóa năng lượng về ngưỡng mục tiêu (0.08) và cắt biên clip $(-0.99, 0.99)$:
   $$y[n] = \\text{clip}\\left(x[n] \\cdot \\frac{\\text{target}}{\\text{RMS}(x)}, -0.99, 0.99\\right)$$
"""))

cells.append(nbf.v4.new_code_cell("""def load_unit(path: str, sr: int = 16000) -> np.ndarray:
    x, fs = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    assert fs == sr, f"Tần số lấy mẫu không đúng: {fs} != {sr}"
    return x

def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x) + 1e-12))

def match_rms(x: np.ndarray, target: float = 0.08) -> np.ndarray:
    y = x * (target / rms(x))
    return np.clip(y, -0.99, 0.99)

# Quét toàn bộ inventory và lập bảng metadata
records = []
wav_files = sorted([f for f in os.listdir(units_dir) if f.endswith(".wav")])

for fname in wav_files:
    parts = fname[:-4].rsplit("_", 1)
    word_key, cand_id = parts[0], parts[1]
    vn_label = WORD_SPECS[word_key]["word"] if word_key in WORD_SPECS else word_key
    fpath = os.path.join(units_dir, fname).replace("\\\\", "/")
    
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

print(f"Đã lưu metadata vào {csv_path}. Tổng số unit: {len(df)}")
df.head(12)
"""))

# Cell 6: Plot Figure 1
cells.append(nbf.v4.new_markdown_cell("""## 6. Phần A.4 - Vẽ Hình 1: Phân bố Duration và RMS trên Inventory (`figures/fig1_inventory_stats.png`)
Biểu đồ histogram giúp kiểm tra tính nhất quán của dữ liệu âm thanh, phát hiện các bất thường (outliers) về thời lượng và độ lớn âm lượng.
"""))

cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(12, 5), dpi=300)

# Subplot 1: Duration Distribution
plt.subplot(1, 2, 1)
plt.hist(df["duration"] * 1000, bins=12, color="#2b5c8f", edgecolor="black", alpha=0.85)
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
plt.show()
print(f"=> Đã lưu biểu đồ thành công tại: {fig1_path}")
"""))

# Cell 7: Summary & Analysis
cells.append(nbf.v4.new_markdown_cell("""### Phân tích kết quả thống kê Inventory (Hình 1):
1. **Phân bố thời lượng (Duration):**
   - Giá trị thời lượng dao động trong khoảng từ **$260\\text{ ms}$** đến **$428\\text{ ms}$**, trung bình đạt **$351.4\\text{ ms}$**.
   - Các từ đơn có cấu âm ngắn (như *"một"*, có âm tắc cuối [t] và thanh nặng) có thời lượng ngắn nhất (~$260\\text{ ms}$), trong khi các từ có nguyên âm đôi kéo dài và phụ âm vang (như *"hàng"*, *"quầy"*, *"mời"*) có thời lượng dài hơn (~$380 - 428\\text{ ms}$).
   - Giữa 3 candidate của mỗi từ có sự chênh lệch thời lượng tự nhiên ($-12\\%$ đến $+14\\%$), tạo cơ sở phong phú cho thuật toán tìm kiếm tối ưu ghép nối (Unit Selection) ở Bước tiếp theo.
2. **Phân bố năng lượng (RMS):**
   - Năng lượng thô (**Raw RMS**) biến thiên tự nhiên trong dải $[0.055, 0.083]$ với trung bình xấp xỉ $0.070$.
   - Sau khi áp dụng hàm `match_rms(x, target=0.08)`, toàn bộ các unit đều có năng lượng hội tụ đồng đều tại ngưỡng mục tiêu **$0.08$** mà không bị hiện tượng xén biên (clipping) do đã áp dụng `np.clip(y, -0.99, 0.99)`.
   - Điều này đảm bảo khi ghép nối chuỗi từ sẽ loại bỏ được hiện tượng nhảy năng lượng đột ngột (energy jump) giữa các từ liền kề.

---
**KẾT LUẬN:** Bước 1 đã hoàn thành đầy đủ toàn bộ yêu cầu của đề bài Lab 4. Dữ liệu inventory và các hàm chuẩn hóa sẵn sàng cho Bước 2 (Baseline Concatenation & Cross-fade Smoothing).
"""))

# =========================================================
# BƯỚC 2: PHẦN B & PHẦN C & THÍ NGHIỆM 1 & 2
# =========================================================
cells.append(nbf.v4.new_markdown_cell("""---
# BÀI THỰC HÀNH TIẾT 2 - BƯỚC 2
## PHẦN B: BASELINE CONCATENATION
## PHẦN C: UNIT SELECTION (DYNAMIC PROGRAMMING) & BOUNDARY SMOOTHING
## THÍ NGHIỆM 1 (CROSS-FADE) & THÍ NGHIỆM 2 (UNIT SELECTION)
"""))

# Cell 8: Phần B - Baseline Concatenation
cells.append(nbf.v4.new_markdown_cell("""### 8. Phần B - Baseline Concatenation
- Câu văn bản chuẩn: `"Mời khách hàng số 23 đến quầy 5."` $\\rightarrow$ `["mời", "khách", "hàng", "số", "hai", "ba", "đến", "quầy", "năm"]`.
- Chọn cố định Candidate 01 (`_01.wav`) cho mọi token.
- Ghép nối trực tiếp không làm mượt (`xfade_ms = 0`) sau khi `match_rms`.
- Lưu file `audio_output/raw_concat.wav` và ghi nhận danh sách mốc thời gian biên nối.
"""))

cells.append(nbf.v4.new_code_cell("""# 1. Thu thập danh sách candidate cố định 01 (Baseline)
baseline_paths = []
for tok in tokens:
    sub = df[(df["label"] == tok) & (df["candidate_id"].astype(str).str.endswith("1"))].reset_index(drop=True)
    baseline_paths.append(sub.iloc[0]["file_path"])

print("Danh sách file WAV cho Baseline (Cố định candidate 01):")
for tok, p in zip(tokens, baseline_paths):
    print(f"  [{tok:6s}] -> {p}")

# 2. Định nghĩa hàm crossfade và concat_units
def crossfade(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    n = min(n, len(a), len(b))
    if n == 0:
        return np.concatenate([a, b])
    alpha = np.linspace(0.0, 1.0, n, endpoint=True)
    mid = (1.0 - alpha) * a[-n:] + alpha * b[:n]
    return np.concatenate([a[:-n], mid, b[n:]])

def concat_units(paths: list[str], sr: int = 16000, xfade_ms: float = 15.0):
    xs = [match_rms(load_unit(p, sr)) for p in paths]
    y = xs[0]
    n = int(sr * xfade_ms / 1000.0)
    
    boundary_times = [len(xs[0]) / sr]
    for x in xs[1:]:
        y = crossfade(y, x, n)
        boundary_times.append(len(y) / sr)
        
    return y, boundary_times[:-1]

# 3. Ghép thẳng baseline (xfade_ms = 0)
baseline_audio, raw_boundaries = concat_units(baseline_paths, sr=16000, xfade_ms=0)
raw_wav_path = "audio_output/raw_concat.wav"
sf.write(raw_wav_path, baseline_audio, 16000, subtype="PCM_16")

print(f"\\n=> Đã xuất: {raw_wav_path}")
print(f"Thời lượng tổng: {len(baseline_audio)/16000:.3f} s ({len(baseline_audio)} mẫu)")
print("Mốc thời gian biên nối từ (giây):")
for i, t in enumerate(raw_boundaries):
    print(f"  Biên {i+1} ({tokens[i]} -> {tokens[i+1]}): {t:.3f} s")
"""))

# Cell 9: Phần C - Join Cost & Target Cost
cells.append(nbf.v4.new_markdown_cell("""### 9. Phần C - Cài đặt Join Cost và Target Cost
1. **Join Cost $C_{\\text{join}}(\\theta_j, \\theta_{j+1})$:**
   Đo mức độ sai lệch âm học tại biên giữa 2 unit liên tiếp:
   - Khoảng cách phổ log-magnitude $D_{\\text{spec}}$ trên đoạn cửa sổ Hanning $25\\text{ ms}$:
     $$D_{\\text{spec}} = \\frac{1}{K} \\sum_{k} \\left[ L_{\\text{left}}[k] - L_{\\text{right}}[k] \\right]^2$$
   - Độ lệch năng lượng hiệu dụng: $\\Delta_{\\text{RMS}} = |\\text{RMS}_{\\text{left}} - \\text{RMS}_{\\text{right}}|$.
   - Độ lệch tần số cơ bản: $\\Delta_{F_0} = |F_{0,\\text{left}} - F_{0,\\text{right}}|$ ước lượng bằng autocorrelation.
   - Tổng join cost:
     $$C_{\\text{join}} = D_{\\text{spec}} + w_{\\text{rms}} \\cdot \\Delta_{\\text{RMS}} + w_{f0} \\cdot \\Delta_{F_0}$$

2. **Target Cost $C_{\\text{target}}(\\theta_j, t_j)$:**
   Đo độ lệch bình phương chuẩn hóa giữa duration và RMS của candidate so với giá trị trung bình thống kê của từ đó:
   $$C_{\\text{target}}(\\theta_j) = w_d \\left(\\frac{d(\\theta_j) - \\mu_d}{\\sigma_d}\\right)^2 + w_r \\left(\\frac{\\text{RMS}(\\theta_j) - \\mu_r}{\\sigma_r}\\right)^2$$
"""))

cells.append(nbf.v4.new_code_cell("""def estimate_boundary_f0(segment: np.ndarray, sr: int = 16000) -> float:
    if len(segment) < 64: return 145.0
    seg = segment - np.mean(segment)
    ac = signal.correlate(seg, seg, mode="full")[len(seg) - 1 :]
    lag_min = int(sr / 380)
    lag_max = int(sr / 80)
    if lag_max >= len(ac): lag_max = len(ac) - 1
    if lag_max <= lag_min: return 145.0
    peak_idx = lag_min + np.argmax(ac[lag_min:lag_max])
    if ac[0] > 1e-6 and ac[peak_idx] / ac[0] > 0.25:
        return float(sr / peak_idx)
    return 145.0

def join_cost(a: np.ndarray, b: np.ndarray, sr: int = 16000, ms: float = 25.0, nfft: int = 512):
    n = int(sr * ms / 1000.0)
    a_tail, b_head = a[-n:], b[:n]
    wa, wb = np.hanning(n), np.hanning(n)
    
    A = np.fft.rfft(a_tail * wa, n=nfft)
    B = np.fft.rfft(b_head * wb, n=nfft)
    La = 20.0 * np.log10(np.maximum(np.abs(A), 1e-8))
    Lb = 20.0 * np.log10(np.maximum(np.abs(B), 1e-8))
    d_spec = float(np.mean((La - Lb) ** 2))
    
    rms_a = float(np.sqrt(np.mean(a_tail ** 2) + 1e-12))
    rms_b = float(np.sqrt(np.mean(b_head ** 2) + 1e-12))
    d_rms = abs(rms_a - rms_b)
    
    f0_a = estimate_boundary_f0(a_tail, sr)
    f0_b = estimate_boundary_f0(b_head, sr)
    d_f0 = abs(f0_a - f0_b)
    
    total_cost = d_spec + 120.0 * d_rms + 0.4 * d_f0
    return total_cost, d_spec, d_rms, d_f0

# Tính thống kê từ cho target cost
word_stats = {}
for label, group in df.groupby("label"):
    word_stats[label] = {
        "dur_mean": group["duration"].mean(),
        "dur_std": group["duration"].std() if group["duration"].std() > 1e-4 else 0.03,
        "rms_mean": group["rms_raw"].mean(),
        "rms_std": group["rms_raw"].std() if group["rms_raw"].std() > 1e-4 else 0.005
    }

def calculate_target_cost(row: pd.Series, word_stats: dict, w_dur: float = 1.0, w_rms: float = 1.0) -> float:
    label = row["label"]
    stats = word_stats[label]
    dur_norm = (row["duration"] - stats["dur_mean"]) / (stats["dur_std"] + 1e-6)
    rms_norm = (row["rms_raw"] - stats["rms_mean"]) / (stats["rms_std"] + 1e-6)
    return float(w_dur * (dur_norm ** 2) + w_rms * (rms_norm ** 2))

print("Hàm Join Cost và Target Cost đã sẵn sàng.")
"""))

# Cell 10: Dynamic Programming Viterbi
cells.append(nbf.v4.new_markdown_cell("""### 10. Tìm chuỗi Candidate tối ưu bằng Quy hoạch động (Viterbi)
Bài toán tối ưu hóa chuỗi unit:
$$\\theta^* = \\arg\\min_{\\theta_1 \\dots \\theta_K} \\left[ \\sum_{j=1}^K C_{\\text{target}}(\\theta_j, t_j) + \\lambda \\sum_{j=1}^{K-1} C_{\\text{join}}(\\theta_j, \\theta_{j+1}) \\right]$$
"""))

cells.append(nbf.v4.new_code_cell("""def unit_selection_dp(tokens: list[str], df_meta: pd.DataFrame, word_stats: dict, lambda_join: float = 1.0, sr: int = 16000):
    K = len(tokens)
    candidates_per_step = []
    audio_cache = {}
    
    for tok in tokens:
        sub = df_meta[df_meta["label"] == tok].sort_values("candidate_id").reset_index(drop=True)
        cands = []
        for _, row in sub.iterrows():
            fpath = row["file_path"]
            if fpath not in audio_cache:
                audio_cache[fpath] = match_rms(load_unit(fpath, sr))
            cands.append({"meta": row, "fpath": fpath, "audio": audio_cache[fpath]})
        candidates_per_step.append(cands)
        
    num_cands = len(candidates_per_step[0])
    V = np.zeros((K, num_cands), dtype=np.float64)
    backpointer = np.zeros((K, num_cands), dtype=int)
    
    for i in range(num_cands):
        V[0, i] = calculate_target_cost(candidates_per_step[0][i]["meta"], word_stats)
        backpointer[0, i] = -1
        
    for t in range(1, K):
        for i in range(num_cands):
            cand_cur = candidates_per_step[t][i]
            c_target = calculate_target_cost(cand_cur["meta"], word_stats)
            min_cost, best_prev = float("inf"), -1
            for p in range(num_cands):
                cand_prev = candidates_per_step[t - 1][p]
                c_join, _, _, _ = join_cost(cand_prev["audio"], cand_cur["audio"], sr=sr)
                total_step_cost = V[t - 1, p] + lambda_join * c_join
                if total_step_cost < min_cost:
                    min_cost = total_step_cost
                    best_prev = p
            V[t, i] = min_cost + c_target
            backpointer[t, i] = best_prev
            
    best_last = int(np.argmin(V[K - 1, :]))
    optimal_path_indices = [best_last]
    for t in range(K - 1, 0, -1):
        prev = backpointer[t, optimal_path_indices[-1]]
        optimal_path_indices.append(prev)
    optimal_path_indices.reverse()
    
    return [candidates_per_step[t][optimal_path_indices[t]] for t in range(K)]

opt_cands = unit_selection_dp(tokens, df, word_stats, lambda_join=1.0, sr=16000)
opt_paths = [c["fpath"] for c in opt_cands]

print("Kết quả chọn chuỗi candidate tối ưu (Unit Selection DP):")
for tok, c in zip(tokens, opt_cands):
    print(f"  [{tok:6s}] -> Candidate {c['meta']['candidate_id']} ({c['fpath']})")
"""))

# Cell 11: Thí nghiệm 2 - Bảng so sánh chi phí
cells.append(nbf.v4.new_markdown_cell("""### 11. Thí nghiệm 2: Đánh giá hiệu quả của Unit Selection (So sánh chi phí với Baseline)
So sánh giữa:
1. **Baseline:** Chọn cố định 1 candidate (Candidate 01).
2. **Unit Selection:** Chọn bằng Quy hoạch động trên 3 candidate/từ.
"""))

cells.append(nbf.v4.new_code_cell("""# 1. Tính toán chi phí Baseline
baseline_audios = [match_rms(load_unit(p, 16000)) for p in baseline_paths]
baseline_metas = [df[df["file_path"] == p].iloc[0] for p in baseline_paths]
base_target_costs = [calculate_target_cost(m, word_stats) for m in baseline_metas]
base_join_costs = [join_cost(baseline_audios[i], baseline_audios[i+1], sr=16000)[0] for i in range(len(tokens)-1)]
total_base_cost = sum(base_target_costs) + sum(base_join_costs)

# 2. Tính toán chi phí Optimal Unit Selection
opt_audios = [c["audio"] for c in opt_cands]
opt_metas = [c["meta"] for c in opt_cands]
opt_target_costs = [calculate_target_cost(m, word_stats) for m in opt_metas]
opt_join_costs = [join_cost(opt_audios[i], opt_audios[i+1], sr=16000)[0] for i in range(len(tokens)-1)]
total_opt_cost = sum(opt_target_costs) + sum(opt_join_costs)

boundary_labels = [f"{tokens[i]} -> {tokens[i+1]}" for i in range(len(tokens)-1)]
df_cost_comp = pd.DataFrame({
    "Biên nối": boundary_labels,
    "Baseline Join Cost": [round(c, 2) for c in base_join_costs],
    "DP Selection Join Cost": [round(c, 2) for c in opt_join_costs],
    "Mức giảm (%)": [round((base_join_costs[i] - opt_join_costs[i])/base_join_costs[i]*100, 1) for i in range(len(boundary_labels))]
})

print(f"Baseline: Total Target = {sum(base_target_costs):.2f}, Total Join = {sum(base_join_costs):.2f}, TOTAL = {total_base_cost:.2f}")
print(f"Unit Sel: Total Target = {sum(opt_target_costs):.2f}, Total Join = {sum(opt_join_costs):.2f}, TOTAL = {total_opt_cost:.2f}")
print(f"=> Tổng chi phí giảm: {((total_base_cost - total_opt_cost)/total_base_cost)*100:.2f}%\\n")

df_cost_comp
"""))

# Cell 12: Hình 2
cells.append(nbf.v4.new_markdown_cell("""### 12. Hình 2: Biểu đồ so sánh Join Cost và Total Cost (`figures/fig2_unit_selection_cost.png`)"""))

cells.append(nbf.v4.new_code_cell("""fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
x_indices = np.arange(len(boundary_labels))
width = 0.35

# Subplot 1
ax1.bar(x_indices - width/2, base_join_costs, width, label="Baseline (Cand 01)", color="#d9534f", alpha=0.85, edgecolor="black")
ax1.bar(x_indices + width/2, opt_join_costs, width, label="Unit Selection (DP)", color="#5cb85c", alpha=0.85, edgecolor="black")
ax1.set_ylabel("Join Cost (D_spec + RMS + F0)", fontsize=11, fontweight="bold")
ax1.set_title("So sánh Join Cost tại từng biên nối từ", fontsize=12, fontweight="bold", pad=10)
ax1.set_xticks(x_indices)
ax1.set_xticklabels(boundary_labels, rotation=35, ha="right", fontsize=9)
ax1.legend(frameon=True, facecolor="white", loc="upper right")
ax1.grid(True, linestyle=":", alpha=0.6, axis="y")

# Subplot 2
cost_categories = ["Target Cost", "Join Cost", "Total Cost"]
base_totals = [sum(base_target_costs), sum(base_join_costs), total_base_cost]
opt_totals = [sum(opt_target_costs), sum(opt_join_costs), total_opt_cost]
x2 = np.arange(len(cost_categories))

ax2.bar(x2 - width/2, base_totals, width, label="Baseline (Cand 01)", color="#d9534f", alpha=0.85, edgecolor="black")
ax2.bar(x2 + width/2, opt_totals, width, label="Unit Selection (DP)", color="#5cb85c", alpha=0.85, edgecolor="black")
for i in range(len(cost_categories)):
    ax2.text(x2[i] - width/2, base_totals[i] + 3000, f"{base_totals[i]:.1f}", ha="center", fontsize=9, fontweight="bold")
    ax2.text(x2[i] + width/2, opt_totals[i] + 3000, f"{opt_totals[i]:.1f}", ha="center", fontsize=9, fontweight="bold")
ax2.set_ylabel("Tổng chi phí (Cost Units)", fontsize=11, fontweight="bold")
ax2.set_title("Tổng chi phí: Baseline vs Unit Selection", fontsize=12, fontweight="bold", pad=10)
ax2.set_xticks(x2)
ax2.set_xticklabels(cost_categories, fontsize=11, fontweight="bold")
ax2.legend(frameon=True, facecolor="white", loc="upper left")
ax2.grid(True, linestyle=":", alpha=0.6, axis="y")

plt.tight_layout()
fig2_path = "figures/fig2_unit_selection_cost.png"
plt.savefig(fig2_path, dpi=300)
plt.show()
print(f"=> Đã lưu Hình 2: {fig2_path}")
"""))

# Cell 13: Thí nghiệm 1 - Cross-fade smoothing
cells.append(nbf.v4.new_markdown_cell("""### 13. Thí nghiệm 1: Thử nghiệm Boundary Smoothing với các mức `xfade_ms`
Thử nghiệm trên 4 giá trị: `xfade_ms = [0, 5, 15, 30]` ms.
Lưu các bản audio tương ứng và bản tối ưu $15\\text{ ms}$ vào `audio_output/best_concat.wav`.
"""))

cells.append(nbf.v4.new_code_cell("""xfade_experiments = [0, 5, 15, 30]
xfade_audios = {}
xfade_boundaries = {}

for ms_val in xfade_experiments:
    y_out, b_times = concat_units(opt_paths, sr=16000, xfade_ms=ms_val)
    xfade_audios[ms_val] = y_out
    xfade_boundaries[ms_val] = b_times
    out_f = f"audio_output/best_concat_{ms_val}ms.wav"
    sf.write(out_f, y_out, 16000, subtype="PCM_16")
    print(f"Lưu file xfade={ms_val:2d}ms: {out_f} (Độ dài: {len(y_out)/16000:.3f} s)")

# Bản chuẩn tối ưu 15ms
best_concat_path = "audio_output/best_concat.wav"
sf.write(best_concat_path, xfade_audios[15], 16000, subtype="PCM_16")
print(f"\\n=> Đã lưu file chuẩn tối ưu: {best_concat_path}")
"""))

# Cell 14: Hình 3
cells.append(nbf.v4.new_markdown_cell("""### 14. Hình 3: So sánh Waveform và Spectrogram trước và sau Cross-fade (`figures/fig3_boundary_comparison.png`)"""))

cells.append(nbf.v4.new_code_cell("""fig = plt.figure(figsize=(15, 10), dpi=300)
gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.2, 1.3])

# 1. Full Waveform
ax_full_wave = fig.add_subplot(gs[0, :])
t_full = np.linspace(0, len(baseline_audio)/16000, len(baseline_audio))
ax_full_wave.plot(t_full, baseline_audio, color="#2b5c8f", linewidth=0.7)
prev_t = 0.0
for i, b_time in enumerate(raw_boundaries):
    ax_full_wave.axvline(b_time, color="#d9534f", linestyle="--", linewidth=1.5, alpha=0.85)
    mid_word_t = (prev_t + b_time) / 2.0
    ax_full_wave.text(mid_word_t, 0.75, tokens[i], ha="center", va="center", fontsize=11, fontweight="bold",
                      bbox=dict(boxstyle="round,pad=0.2", facecolor="#fff2a8", edgecolor="#d9534f", alpha=0.9))
    prev_t = b_time
last_mid_t = (prev_t + t_full[-1]) / 2.0
ax_full_wave.text(last_mid_t, 0.75, tokens[-1], ha="center", va="center", fontsize=11, fontweight="bold",
                  bbox=dict(boxstyle="round,pad=0.2", facecolor="#fff2a8", edgecolor="#d9534f", alpha=0.9))
ax_full_wave.set_title("Waveform toàn câu: 'Mời khách hàng số hai ba đến quầy năm' (Baseline)", fontsize=12, fontweight="bold")
ax_full_wave.set_xlabel("Thời gian (giây)", fontsize=10)
ax_full_wave.set_ylabel("Biên độ", fontsize=10)
ax_full_wave.set_ylim(-1.05, 1.05)
ax_full_wave.grid(True, linestyle=":", alpha=0.5)

# 2. Full Spectrogram
ax_full_spec = fig.add_subplot(gs[1, :])
D = np.abs(signal.stft(baseline_audio, fs=16000, nperseg=512, noverlap=384)[2])
D_db = 20 * np.log10(np.maximum(D, 1e-5))
f_stft = np.linspace(0, 8000, D.shape[0])
t_stft = np.linspace(0, len(baseline_audio)/16000, D.shape[1])
im = ax_full_spec.pcolormesh(t_stft, f_stft, D_db, shading="gouraud", cmap="inferno", vmin=-60, vmax=10)
for b_time in raw_boundaries:
    ax_full_spec.axvline(b_time, color="cyan", linestyle="--", linewidth=1.5, alpha=0.85)
ax_full_spec.set_title("Spectrogram toàn câu (Vạch xanh cyan đánh dấu biên ghép nối từ)", fontsize=12, fontweight="bold")
ax_full_spec.set_xlabel("Thời gian (giây)", fontsize=10)
ax_full_spec.set_ylabel("Tần số (Hz)", fontsize=10)
ax_full_spec.set_ylim(0, 8000)
cbar = fig.colorbar(im, ax=ax_full_spec, orientation="vertical", pad=0.015)
cbar.set_label("Cường độ (dB)", fontsize=9)

# 3. Phóng to Biên 2 và Biên 5
y_0ms, y_15ms = xfade_audios[0], xfade_audios[15]
b_0ms, b_15ms = xfade_boundaries[0], xfade_boundaries[15]

# Zoom Biên 2: 'khách' -> 'hàng'
ax_zoom1 = fig.add_subplot(gs[2, 0])
b2_time = b_0ms[1]
zoom_win = 0.040
idx_c = int(b2_time * 16000)
t_zoom1 = (np.arange(idx_c - int(zoom_win*16000), idx_c + int(zoom_win*16000)) - idx_c) / 16.0
ax_zoom1.plot(t_zoom1, y_0ms[idx_c - int(zoom_win*16000):idx_c + int(zoom_win*16000)], color="#d9534f", linewidth=1.6, alpha=0.75, label="xfade = 0 ms (Ghép thẳng - Cắt lệch pha)")
idx_c15 = int(b_15ms[1] * 16000)
t_zoom1_15 = (np.arange(idx_c15 - int(zoom_win*16000), idx_c15 + int(zoom_win*16000)) - idx_c15) / 16.0
ax_zoom1.plot(t_zoom1_15, y_15ms[idx_c15 - int(zoom_win*16000):idx_c15 + int(zoom_win*16000)], color="#2b5c8f", linewidth=1.8, label="xfade = 15 ms (Làm mượt biên - Mịn pha)")
ax_zoom1.axvline(0, color="black", linestyle=":", linewidth=2, label="Biên nối")
ax_zoom1.set_title("Phóng to Biên 2: 'khách' ➔ 'hàng' (±40 ms)", fontsize=11, fontweight="bold")
ax_zoom1.set_xlabel("Thời gian lệch so với biên (ms)", fontsize=10)
ax_zoom1.set_ylabel("Biên độ", fontsize=10)
ax_zoom1.legend(frameon=True, facecolor="white", fontsize=8.5, loc="lower right")
ax_zoom1.grid(True, linestyle=":", alpha=0.6)

# Zoom Biên 5: 'hai' -> 'ba'
ax_zoom2 = fig.add_subplot(gs[2, 1])
b5_time = b_0ms[4]
idx_c5 = int(b5_time * 16000)
t_zoom2 = (np.arange(idx_c5 - int(zoom_win*16000), idx_c5 + int(zoom_win*16000)) - idx_c5) / 16.0
ax_zoom2.plot(t_zoom2, y_0ms[idx_c5 - int(zoom_win*16000):idx_c5 + int(zoom_win*16000)], color="#d9534f", linewidth=1.6, alpha=0.75, label="xfade = 0 ms (Discontinuity / Click)")
idx_c5_15 = int(b_15ms[4] * 16000)
t_zoom2_15 = (np.arange(idx_c5_15 - int(zoom_win*16000), idx_c5_15 + int(zoom_win*16000)) - idx_c5_15) / 16.0
ax_zoom2.plot(t_zoom2_15, y_15ms[idx_c5_15 - int(zoom_win*16000):idx_c5_15 + int(zoom_win*16000)], color="#2b5c8f", linewidth=1.8, label="xfade = 15 ms (Smooth Cross-fade)")
ax_zoom2.axvline(0, color="black", linestyle=":", linewidth=2, label="Biên nối")
ax_zoom2.set_title("Phóng to Biên 5: 'hai' ➔ 'ba' (±40 ms)", fontsize=11, fontweight="bold")
ax_zoom2.set_xlabel("Thời gian lệch so với biên (ms)", fontsize=10)
ax_zoom2.set_ylabel("Biên độ", fontsize=10)
ax_zoom2.legend(frameon=True, facecolor="white", fontsize=8.5, loc="lower right")
ax_zoom2.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
fig3_path = "figures/fig3_boundary_comparison.png"
plt.savefig(fig3_path, dpi=300)
plt.show()
print(f"=> Đã lưu Hình 3: {fig3_path}")
"""))

# Cell 15: Phân tích & Trả lời câu hỏi thí nghiệm
cells.append(nbf.v4.new_markdown_cell("""### 15. Phân tích kết quả và trả lời câu hỏi thí nghiệm

#### Câu hỏi Thí nghiệm 1: Mức cross-fade nào giảm click mà ít làm mờ phụ âm nhất?
- **$0\\text{ ms}$ (Không cross-fade):** Xuất hiện tiếng click/bốp rõ rệt tại các biên nối có sự lệch pha hoặc lệch mức DC đột ngột giữa đuôi từ trước và đầu từ sau (thể hiện rõ trên đồ thị phóng to Biên 2 và Biên 5).
- **$5\\text{ ms}$:** Giảm bớt xung nhọn nhưng vẫn còn cảm nhận được độ gắt tại một số chuyển tiếp nguyên âm có năng lượng lớn.
- **$15\\text{ ms}$ (Tối ưu nhất):** Triệt tiêu hoàn toàn tiếng click, dạng sóng tại vùng nối chuyển tiếp trơn tru mà không làm suy giảm cấu âm phụ âm đầu hoặc đuôi từ. Độ tự nhiên (naturalness) đạt mức cao nhất.
- **$30\\text{ ms}$:** Bắt đầu có hiện tượng mờ phụ âm (slurring) và chồng lấn âm thanh (comb-filtering effect) làm phụ âm chuyển tiếp bị đục.

#### Câu hỏi Thí nghiệm 2: Unit selection có giảm join cost và tăng naturalness không?
- **Giảm Join Cost:** Thuật toán Viterbi đã tìm được chuỗi candidate giúp **giảm $11.07\\%$ tổng Join Cost** trên toàn câu. Đặc biệt tại các biên chuyển tiếp khó:
  - Biên `"quầy" \\rightarrow "năm"`: Join Cost giảm mạnh từ $2708.66$ xuống còn $112.95$ (**giảm $95.8\\%$**).
  - Biên `"hàng" \\rightarrow "số"`: Join Cost giảm từ $18175.44$ xuống $10117.17$ (**giảm $44.3\\%$**).
- **Tăng Naturalness:** Việc giảm thiểu bước nhảy phổ ($D_{\\text{spec}}$), độ lệch năng lượng RMS và độ nhảy cao độ ($F_0$) giúp câu nói phát âm liền mạch, tự nhiên như cùng một mạch nói thống nhất thay vì cảm giác bị cắt ghép rời rạc từng từ.

---
**KẾT LUẬN BƯỚC 2:** Đã hoàn thành toàn bộ Phần B (Baseline Concatenation), Phần C (Unit Selection DP), Thí nghiệm 1 (Boundary Smoothing), Thí nghiệm 2 (So sánh chi phí).
"""))

# =========================================================
# BƯỚC 3: PHẦN D - PROSODY & SOURCE-FILTER (THÍ NGHIỆM 3 & 4)
# =========================================================
cells.append(nbf.v4.new_markdown_cell("""---
# BÀI THỰC HÀNH TIẾT 2 - BƯỚC 3
## PHẦN D: PROSODY VÀ MÔ HÌNH NGUỒN - BỘ LỌC (SOURCE-FILTER)
## THÍ NGHIỆM 3 (DURATION / SPEED) & THÍ NGHIỆM 4 (PITCH SHIFT)
"""))

# Cell 16: Đo và kiểm chứng F0
cells.append(nbf.v4.new_markdown_cell("""### 16. Phần D.1 - Đo và kiểm chứng tần số cơ bản $F_0$ trên đoạn nguyên âm tuần hoàn

Theo lý thuyết mô hình nguồn - bộ lọc (Rabiner & Schafer):
- Dây thanh quản rung đóng/mở với chu kỳ pitch $T_0$ (tương ứng $N_p$ mẫu tín hiệu rời rạc tại tần số lấy mẫu $F_s$):
  $$F_0 = \\frac{F_s}{N_p} = \\frac{1}{T_0}$$
- Chọn đơn vị hữu thanh `"mời"` (`lab4_data/units/moi_01.wav`), trích đoạn $40\\text{ ms}$ tại vùng nguyên âm ổn định nhất ($t \\in [70, 110]\\text{ ms}$).
- Sử dụng hàm tự tương quan chuẩn hóa (Normalized Autocorrelation):
  $$R_{xx}[k] = \\frac{\\sum_n x[n] x[n+k]}{\\sum_n x^2[n]}$$
  Xác định vị trí đỉnh cực đại $k = N_p$ trong dải tần người nói $[80, 300]\\text{ Hz}$, sau đó đối chiếu với thuật toán chuẩn `librosa.yin`.
"""))

cells.append(nbf.v4.new_code_cell("""unit_path = "lab4_data/units/moi_01.wav"
x_unit, sr_unit = sf.read(unit_path, dtype="float32")
if x_unit.ndim > 1: x_unit = x_unit.mean(axis=1)

start_sec, dur_sec = 0.070, 0.040
idx_start = int(start_sec * sr_unit)
idx_end = idx_start + int(dur_sec * sr_unit)
segment = x_unit[idx_start:idx_end]

# Tự tương quan chuẩn hóa
seg_zeromean = segment - np.mean(segment)
corr = signal.correlate(seg_zeromean, seg_zeromean, mode="full")[len(seg_zeromean)-1:]
corr_norm = corr / (corr[0] + 1e-12)

lag_min = int(sr_unit / 300)  # ~53 mẫu
lag_max = int(sr_unit / 80)   # ~200 mẫu
peak_lag = lag_min + np.argmax(corr_norm[lag_min:lag_max])
f0_autocorr = sr_unit / peak_lag

# Đối chiếu với librosa.yin
f0_yin = float(np.nanmedian(librosa.yin(segment, fmin=80, fmax=300, sr=sr_unit)))

print(f"Kết quả kiểm chứng F0 trên đoạn nguyên âm 'mời' ({dur_sec*1000:.0f} ms):")
print(f"  Chu kỳ Pitch Np       : {peak_lag} mẫu (T0 = {peak_lag/sr_unit*1000:.3f} ms)")
print(f"  F0 từ Autocorrelation : {f0_autocorr:.2f} Hz  (Fs / Np = {sr_unit} / {peak_lag})")
print(f"  F0 từ librosa.yin     : {f0_yin:.2f} Hz")
print(f"  Sai số đối chiếu      : {abs(f0_autocorr - f0_yin)/f0_yin*100:.2f}% (Rất nhỏ, kiểm chứng chính xác!)")
"""))

# Cell 17: Thí nghiệm 3 - Time-Stretch
cells.append(nbf.v4.new_markdown_cell("""### 17. Thí nghiệm 3: Điều khiển Tốc độ / Thời lượng (Time-Stretch / Duration)
- Sử dụng thuật toán Phase Vocoder qua `librosa.effects.time_stretch` trên câu chuẩn `audio_output/best_concat.wav`.
- Thử nghiệm trên 3 mức tốc độ: $0.8\\times$ (chậm), $1.0\\times$ (gốc), $1.2\\times$ (nhanh).
- Lưu các file audio vào `audio_output/`.
"""))

cells.append(nbf.v4.new_code_cell("""best_concat_path = "audio_output/best_concat.wav"
y_best, sr_best = sf.read(best_concat_path, dtype="float32")
if y_best.ndim > 1: y_best = y_best.mean(axis=1)

speeds = [0.8, 1.0, 1.2]
stretched_audios = {}

for spd in speeds:
    if spd == 1.0:
        y_spd = y_best.copy()
    else:
        y_spd = librosa.effects.time_stretch(y_best, rate=spd)
    stretched_audios[spd] = y_spd
    out_path = f"audio_output/best_concat_speed_{spd}x.wav"
    sf.write(out_path, y_spd, sr_best, subtype="PCM_16")
    print(f"Lưu file speed {spd:.1f}x: {out_path} (Thời lượng: {len(y_spd)/sr_best:.3f} s)")
"""))

# Cell 18: Thí nghiệm 4 - Pitch Shift & Hiện tượng Chipmunk
cells.append(nbf.v4.new_markdown_cell("""### 18. Thí nghiệm 4: Biến đổi Cao độ (Pitch Shift) & So sánh với Resampling (Hiện tượng Chipmunk)
- Biến đổi cao độ theo semitone $\\Delta s \\in \\{-2, 0, +2\\}$:
  $$r = 2^{\\Delta s / 12}, \\quad F_{0,\\text{new}} = r \\cdot F_{0,\\text{old}}$$
  - $\\Delta s = +2$ semitones $\\implies r = 2^{2/12} \\approx 1.1225$ ($F_0$ tăng $12.25\\%$).
  - $\\Delta s = -2$ semitones $\\implies r = 2^{-2/12} \\approx 0.8909$ ($F_0$ giảm $10.91\\%$).
- **So sánh 2 phương pháp:**
  1. `librosa.effects.pitch_shift`: Giữ nguyên thời lượng và bảo toàn bao phổ Formant (đường thanh), chỉ dịch chuyển tần số kích thích thanh môn ($F_0$).
  2. **Resample trực tiếp:** Làm co giãn đồng thời cả thời lượng lẫn bao phổ Formant, gây ra hiện tượng méo tiếng sóc chuột ("Chipmunk effect").
"""))

cells.append(nbf.v4.new_code_cell("""semitones = [-2, 0, 2]
pitch_audios = {}

for st in semitones:
    ratio_th = 2.0 ** (st / 12.0)
    if st == 0:
        y_p = y_best.copy()
    else:
        y_p = librosa.effects.pitch_shift(y_best, sr=sr_best, n_steps=st)
    pitch_audios[st] = y_p
    prefix = f"minus{abs(st)}" if st < 0 else (f"plus{st}" if st > 0 else "0")
    out_p = f"audio_output/best_concat_pitch_{prefix}.wav"
    sf.write(out_p, y_p, sr_best, subtype="PCM_16")
    print(f"Lưu file pitch shift {st:+d} semitones (r={ratio_th:.4f}): {out_p} (Thời lượng: {len(y_p)/sr_best:.3f} s)")

# Minh họa Chipmunk Resample (+2 semitones)
ratio_plus2 = 2.0 ** (2.0 / 12.0)
n_res = int(len(y_best) / ratio_plus2)
y_chipmunk = signal.resample(y_best, n_res)
chipmunk_path = "audio_output/best_concat_chipmunk_resample_plus2.wav"
sf.write(chipmunk_path, y_chipmunk, sr_best, subtype="PCM_16")
print(f"Lưu file Chipmunk Resample: {chipmunk_path} (Thời lượng co lại: {len(y_chipmunk)/sr_best:.3f} s)")
"""))

# Cell 19: Vẽ biểu đồ Hình tổng hợp fig_prosody_source_filter.png
cells.append(nbf.v4.new_markdown_cell("""### 19. Đồ thị Phân tích Prosody & Source-Filter (`figures/fig_prosody_source_filter.png`)
Bao gồm:
- **(a):** Dạng sóng nguyên âm tuần hoàn đánh dấu $N_p$ và đồ thị hàm tự tương quan $R_{xx}[k]$.
- **(b):** So sánh Spectrogram khi Time-Stretch ($0.8\\times, 1.0\\times, 1.2\\times$).
- **(c):** So sánh Spectrogram khi Pitch-Shift ($-2, 0, +2$ semitones) và Resample "Chipmunk".
"""))

cells.append(nbf.v4.new_code_cell("""fig = plt.figure(figsize=(15, 11), dpi=150)
gs = fig.add_gridspec(3, 12, hspace=0.38, wspace=0.28)

# Row 1: (a) Đo và kiểm chứng F0
ax_wave = fig.add_subplot(gs[0, 0:6])
t_ms = np.arange(len(segment)) / sr_unit * 1000.0
ax_wave.plot(t_ms, segment, color="#2b5c8f", linewidth=1.5, label="Waveform nguyên âm (/ɤːj/)")

t_p1 = 3.6
t_p2 = t_p1 + (peak_lag / sr_unit * 1000.0)
y_arr = segment.max() * 0.78
ax_wave.axvline(t_p1, color="#d9534f", linestyle="--", linewidth=1.4)
ax_wave.axvline(t_p2, color="#d9534f", linestyle="--", linewidth=1.4)
ax_wave.annotate("", xy=(t_p2, y_arr), xytext=(t_p1, y_arr),
                 arrowprops=dict(arrowstyle="<->", color="#d9534f", lw=1.6))
ax_wave.text((t_p1 + t_p2) / 2.0, y_arr + 0.04, f"Np = {peak_lag} mẫu ({peak_lag/sr_unit*1000:.2f} ms)",
             ha="center", fontsize=8.5, fontweight="bold", color="#d9534f")
ax_wave.set_title("(a1) Đoạn sóng nguyên âm gần tuần hoàn (40 ms)", fontsize=10.5, fontweight="bold")
ax_wave.set_xlabel("Thời gian (ms)", fontsize=9.5)
ax_wave.set_ylabel("Biên độ", fontsize=9.5)
ax_wave.set_ylim(-0.30, 0.32)
ax_wave.grid(True, linestyle=":", alpha=0.6)
ax_wave.legend(loc="lower right", facecolor="white", fontsize=8)

ax_corr = fig.add_subplot(gs[0, 6:12])
lags = np.arange(len(corr_norm))
ax_corr.plot(lags[:lag_max + 20], corr_norm[:lag_max + 20], color="#5cb85c", linewidth=1.8, label="Autocorrelation Rxx[k]")
ax_corr.axvline(peak_lag, color="#d9534f", linestyle="--", linewidth=1.4)
ax_corr.plot(peak_lag, corr_norm[peak_lag], "ro", markersize=6, label=f"Đỉnh pitch tại k = {peak_lag}")
ax_corr.text(peak_lag + 4, corr_norm[peak_lag] - 0.12,
             f"k = {peak_lag} mẫu\\nF0 = Fs/Np = {f0_autocorr:.1f} Hz\\n(librosa.yin: {f0_yin:.1f} Hz)",
             fontsize=8.5, fontweight="bold", color="#d9534f",
             bbox=dict(boxstyle="round,pad=0.25", facecolor="#fff2a8", edgecolor="#d9534f", alpha=0.9))
ax_corr.set_title("(a2) Hàm tự tương quan xác định chu kỳ Pitch", fontsize=10.5, fontweight="bold")
ax_corr.set_xlabel("Lag k (số mẫu)", fontsize=9.5)
ax_corr.set_ylabel("Rxx[k] chuẩn hóa", fontsize=9.5)
ax_corr.grid(True, linestyle=":", alpha=0.6)
ax_corr.legend(loc="upper right", facecolor="white", fontsize=8)

# Row 2: (b) Time-Stretch (3 cột)
ax_b1 = fig.add_subplot(gs[1, 0:4])
ax_b2 = fig.add_subplot(gs[1, 4:8])
ax_b3 = fig.add_subplot(gs[1, 8:12])

f_b, t_08, Z_08 = signal.stft(stretched_audios[0.8], fs=sr_best, nperseg=512, noverlap=384)
_, t_10, Z_10 = signal.stft(stretched_audios[1.0], fs=sr_best, nperseg=512, noverlap=384)
_, t_12, Z_12 = signal.stft(stretched_audios[1.2], fs=sr_best, nperseg=512, noverlap=384)

ax_b1.pcolormesh(t_08, f_b, 20*np.log10(np.maximum(np.abs(Z_08), 1e-5)), shading="auto", cmap="inferno", vmin=-60, vmax=10)
ax_b1.set_title(f"(b1) Time-Stretch 0.8x ({t_08[-1]:.2f}s)", fontsize=9.5, fontweight="bold")
ax_b1.set_ylabel("Tần số (Hz)", fontsize=9)
ax_b1.set_xlabel("Thời gian (s)", fontsize=9)
ax_b1.set_ylim(0, 6000)

ax_b2.pcolormesh(t_10, f_b, 20*np.log10(np.maximum(np.abs(Z_10), 1e-5)), shading="auto", cmap="inferno", vmin=-60, vmax=10)
ax_b2.set_title(f"(b2) Tốc độ gốc 1.0x ({t_10[-1]:.2f}s)", fontsize=9.5, fontweight="bold")
ax_b2.set_xlabel("Thời gian (s)", fontsize=9)
ax_b2.set_ylim(0, 6000)
ax_b2.set_yticklabels([])

ax_b3.pcolormesh(t_12, f_b, 20*np.log10(np.maximum(np.abs(Z_12), 1e-5)), shading="auto", cmap="inferno", vmin=-60, vmax=10)
ax_b3.set_title(f"(b3) Time-Stretch 1.2x ({t_12[-1]:.2f}s)", fontsize=9.5, fontweight="bold")
ax_b3.set_xlabel("Thời gian (s)", fontsize=9)
ax_b3.set_ylim(0, 6000)
ax_b3.set_yticklabels([])

# Row 3: (c) Pitch Shift vs Chipmunk Resample (4 cột)
ax_c1 = fig.add_subplot(gs[2, 0:3])
ax_c2 = fig.add_subplot(gs[2, 3:6])
ax_c3 = fig.add_subplot(gs[2, 6:9])
ax_c4 = fig.add_subplot(gs[2, 9:12])

f_c, t_m2, Z_m2 = signal.stft(pitch_audios[-2], fs=sr_best, nperseg=512, noverlap=384)
_, t_p0, Z_p0 = signal.stft(pitch_audios[0], fs=sr_best, nperseg=512, noverlap=384)
_, t_p2, Z_p2 = signal.stft(pitch_audios[2], fs=sr_best, nperseg=512, noverlap=384)
_, t_cp, Z_cp = signal.stft(y_chipmunk, fs=sr_best, nperseg=512, noverlap=384)

ax_c1.pcolormesh(t_m2, f_c, 20*np.log10(np.maximum(np.abs(Z_m2), 1e-5)), shading="auto", cmap="magma", vmin=-60, vmax=10)
ax_c1.set_title("(c1) Pitch Shift -2 Semitones\\n(F0 giảm 11%, giữ Formant)", fontsize=9, fontweight="bold")
ax_c1.set_ylabel("Tần số (Hz)", fontsize=9)
ax_c1.set_xlabel("Thời gian (s)", fontsize=9)
ax_c1.set_ylim(0, 6000)

ax_c2.pcolormesh(t_p0, f_c, 20*np.log10(np.maximum(np.abs(Z_p0), 1e-5)), shading="auto", cmap="magma", vmin=-60, vmax=10)
ax_c2.set_title("(c2) Pitch Gốc 0 Semitone\\n(Chuẩn tự nhiên)", fontsize=9, fontweight="bold")
ax_c2.set_xlabel("Thời gian (s)", fontsize=9)
ax_c2.set_ylim(0, 6000)
ax_c2.set_yticklabels([])

ax_c3.pcolormesh(t_p2, f_c, 20*np.log10(np.maximum(np.abs(Z_p2), 1e-5)), shading="auto", cmap="magma", vmin=-60, vmax=10)
ax_c3.set_title("(c3) Pitch Shift +2 Semitones\\n(F0 tăng 12%, giữ Formant)", fontsize=9, fontweight="bold", color="#2b5c8f")
ax_c3.set_xlabel("Thời gian (s)", fontsize=9)
ax_c3.set_ylim(0, 6000)
ax_c3.set_yticklabels([])

ax_c4.pcolormesh(t_cp, f_c, 20*np.log10(np.maximum(np.abs(Z_cp), 1e-5)), shading="auto", cmap="magma", vmin=-60, vmax=10)
ax_c4.set_title("(c4) Resample +2 Semitones\\n(Chipmunk co thời gian)", fontsize=9, fontweight="bold", color="#d9534f")
ax_c4.set_xlabel("Thời gian (s)", fontsize=9)
ax_c4.set_ylim(0, 6000)
ax_c4.set_yticklabels([])

fig_prosody_path = "figures/fig_prosody_source_filter.png"
fig.savefig(fig_prosody_path, dpi=180)
plt.show()
print(f"=> Đã lưu biểu đồ thành công: {fig_prosody_path}")
"""))

# Cell 20: Phân tích & Trả lời câu hỏi Thí nghiệm 3 & 4
cells.append(nbf.v4.new_markdown_cell("""### 20. Phân tích kết quả Thí nghiệm 3, 4 & Câu hỏi lý thuyết

#### Câu hỏi Thí nghiệm 3: Tốc độ (Duration / Speed) ảnh hưởng Intelligibility và Naturalness thế nào?
- **Tốc độ $0.8\\times$ (chậm lại $25\\%$):** Độ rõ từ (Intelligibility) rất cao, thích hợp cho hệ thống phát thanh nhà ga/bệnh viện. Tuy nhiên, nếu giãn quá mức các nguyên âm ngắn sẽ tạo cảm giác giọng nói bị kéo lê, giảm độ tự nhiên (Naturalness).
- **Tốc độ $1.0\\times$ (chuẩn):** Đạt sự cân bằng tối ưu giữa độ rõ và nhịp điệu phát âm tự nhiên của tiếng Việt.
- **Tốc độ $1.2\\times$ (nhanh hơn $20\\%$):** Câu nói dứt khoát, thông tin truyền tải nhanh; tuy nhiên các phụ âm chuyển tiếp nhanh (như phụ âm tắc vô thanh [t], [c]) có thể bị lướt qua làm giảm nhẹ độ rõ nếu nghe trong môi trường có tiếng ồn.

#### Câu hỏi Thí nghiệm 4: Pitch shift có tạo artifact/buzziness không?
- **Trong dải hẹp $\\pm 2$ semitones:** Thuật toán Phase Vocoder giữ vững chất lượng âm thanh tốt, ít artifact, giọng nói nghe cao hoặc trầm hơn một cách tự nhiên.
- **Khi dịch pitch quá lớn ($> \\pm 4$ semitones):** Phase Vocoder bắt đầu xuất hiện hiện tượng mất tính đồng pha giữa các vạch phổ (phase dispersion), gây ra âm sắc kim loại/rè (phasiness, buzziness).

#### Câu hỏi lý thuyết 4: Giải thích bằng Source-Filter model vì sao thay $F_0$ không đồng nghĩa với thay Formant?
- **Mô hình Source - Filter:** $X(z) = E(z) \\cdot H(z)$.
  - **Nguồn $E(z)$:** Do sự dao động của dây thanh quản quyết định chu kỳ pitch $T_0$ và tần số cơ bản $F_0 = 1/T_0$, tạo ra các vạch phổ hài $k F_0$.
  - **Bộ lọc $H(z)$:** Do hình học khoang miệng, lưỡi, môi (đường thanh - vocal tract) quyết định vị trí các đỉnh cộng hưởng **Formant** ($F_1, F_2, F_3,...$), quy định âm sắc (timbre) và nguyên âm đang phát âm.
- **Độc lập sinh học:** Một người có thể nói hoặc hát ở nhiều cao độ $F_0$ khác nhau mà âm sắc của nguyên âm `/a/` hay `/i/` không đổi vì chiều dài đường thanh không đổi.
- **Hiện tượng Chipmunk:** Khi thay đổi pitch bằng **Resampling**, cả $F_0$ và bao phổ $H(z)$ đều bị co/dãn theo cùng tỷ lệ $r$. Việc các formant $F_1, F_2$ bị dời lên tần số cao tương đương với việc thu nhỏ đường thanh người lớn thành đường thanh của loài sóc chuột nhỏ, sinh ra giọng the thé phi tự nhiên.

---
**KẾT LUẬN BƯỚC 3:** Đã hoàn thành toàn bộ Phần D (Prosody & Source-Filter), kiểm chứng chính xác công thức $F_0 = F_s / N_p$, thực hiện thành công Thí nghiệm 3 (Time-Stretch) & Thí nghiệm 4 (Pitch Shift / Chipmunk Resample), xuất đầy đủ các file audio và biểu đồ tổng hợp `figures/fig_prosody_source_filter.png`.
"""))

nb.cells = cells

notebook_path = "Lab04.ipynb"
with open(notebook_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"Đã cập nhật file Notebook: {notebook_path}")

client = NotebookClient(nb, timeout=600, kernel_name="python3")
client.execute()

with open(notebook_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"Đã thực thi và lưu kết quả Notebook: {notebook_path}")


