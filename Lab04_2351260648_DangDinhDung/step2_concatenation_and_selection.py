import os
import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import scipy.signal as signal
import soundfile as sf
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Step 1 Modules: Load audio, RMS, match_rms
# ---------------------------------------------------------
def load_unit(path: str, sr: int = 16000) -> np.ndarray:
    x, fs = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    assert fs == sr, f"Sampling rate mismatch: {fs} != {sr}"
    return x

def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x) + 1e-12))

def match_rms(x: np.ndarray, target: float = 0.08) -> np.ndarray:
    y = x * (target / rms(x))
    return np.clip(y, -0.99, 0.99)

def crossfade(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    n = min(n, len(a), len(b))
    if n == 0:
        return np.concatenate([a, b])
    alpha = np.linspace(0.0, 1.0, n, endpoint=True)
    mid = (1.0 - alpha) * a[-n:] + alpha * b[:n]
    return np.concatenate([a[:-n], mid, b[n:]])

def concat_units(paths: list[str], sr: int = 16000, xfade_ms: float = 15.0) -> tuple[np.ndarray, list[float]]:
    xs = [match_rms(load_unit(p, sr)) for p in paths]
    y = xs[0]
    n = int(sr * xfade_ms / 1000.0)
    
    # Track boundary timestamps (in seconds)
    boundary_times = []
    current_time = len(xs[0]) / sr
    boundary_times.append(current_time)
    
    for x in xs[1:]:
        y = crossfade(y, x, n)
        overlap_offset = n / sr
        current_time = (len(y) / sr)
        boundary_times.append(current_time)
        
    return y, boundary_times[:-1]

# ---------------------------------------------------------
# Step 2: Join Cost and Target Cost
# ---------------------------------------------------------
def estimate_boundary_f0(segment: np.ndarray, sr: int = 16000) -> float:
    """Ước lượng F0 ở đoạn biên bằng autocorrelation."""
    if len(segment) < 64:
        return 145.0
    seg = segment - np.mean(segment)
    ac = signal.correlate(seg, seg, mode="full")[len(seg) - 1 :]
    lag_min = int(sr / 380)  # max ~380 Hz
    lag_max = int(sr / 80)   # min ~80 Hz
    if lag_max >= len(ac):
        lag_max = len(ac) - 1
    if lag_max <= lag_min:
        return 145.0
    peak_idx = lag_min + np.argmax(ac[lag_min:lag_max])
    if ac[0] > 1e-6 and ac[peak_idx] / ac[0] > 0.25:
        return float(sr / peak_idx)
    return 145.0

def join_cost(a: np.ndarray, b: np.ndarray, sr: int = 16000, ms: float = 25.0, nfft: int = 512) -> tuple[float, float, float, float]:
    """
    Tính join cost giữa 2 unit a và b theo công thức Lab 4:
    - Log-magnitude spectrum mismatch (D_spec) của 20-30 ms ở hai phía biên.
    - Cộng thêm độ lệch RMS (|RMS_left - RMS_right|) và F0 (|F0_left - F0_right|) sau chuẩn hóa.
    """
    n = int(sr * ms / 1000.0)
    a_tail = a[-n:]
    b_head = b[:n]
    wa = np.hanning(n)
    wb = np.hanning(n)
    
    A = np.fft.rfft(a_tail * wa, n=nfft)
    B = np.fft.rfft(b_head * wb, n=nfft)
    La = 20.0 * np.log10(np.maximum(np.abs(A), 1e-8))
    Lb = 20.0 * np.log10(np.maximum(np.abs(B), 1e-8))
    
    d_spec = float(np.mean((La - Lb) ** 2))
    
    # RMS difference
    rms_a = float(np.sqrt(np.mean(a_tail ** 2) + 1e-12))
    rms_b = float(np.sqrt(np.mean(b_head ** 2) + 1e-12))
    d_rms = abs(rms_a - rms_b)
    
    # F0 difference
    f0_a = estimate_boundary_f0(a_tail, sr)
    f0_b = estimate_boundary_f0(b_head, sr)
    d_f0 = abs(f0_a - f0_b)
    
    # Trọng số chuẩn hóa để cân bằng độ lớn (d_spec ~ 20-80, d_rms ~ 0.01, d_f0 ~ 5-25 Hz)
    total_cost = d_spec + 120.0 * d_rms + 0.4 * d_f0
    return total_cost, d_spec, d_rms, d_f0

def calculate_target_cost(row: pd.Series, word_stats: dict, w_dur: float = 1.0, w_rms: float = 1.0) -> float:
    """
    Target cost: độ lệch chuẩn hóa bình phương của duration và RMS
    so với giá trị trung bình của từ đó.
    """
    label = row["label"]
    stats = word_stats[label]
    
    dur_norm = (row["duration"] - stats["dur_mean"]) / (stats["dur_std"] + 1e-6)
    rms_norm = (row["rms_raw"] - stats["rms_mean"]) / (stats["rms_std"] + 1e-6)
    
    return float(w_dur * (dur_norm ** 2) + w_rms * (rms_norm ** 2))

# ---------------------------------------------------------
# Step 3: Dynamic Programming / Viterbi for Unit Selection
# ---------------------------------------------------------
def unit_selection_dp(tokens: list[str], df_meta: pd.DataFrame, word_stats: dict, lambda_join: float = 1.0, sr: int = 16000):
    """
    Tìm chuỗi candidate tối ưu qua thuật toán Quy hoạch động (Viterbi):
    theta* = arg min sum(C_target) + lambda * sum(C_join)
    """
    K = len(tokens)
    
    # Lấy danh sách candidate và nạp sẵn audio
    candidates_per_step = []
    audio_cache = {}
    
    for tok in tokens:
        sub = df_meta[df_meta["label"] == tok].sort_values("candidate_id").reset_index(drop=True)
        cands = []
        for _, row in sub.iterrows():
            fpath = row["file_path"]
            if fpath not in audio_cache:
                audio_cache[fpath] = match_rms(load_unit(fpath, sr))
            cands.append({
                "meta": row,
                "fpath": fpath,
                "audio": audio_cache[fpath]
            })
        candidates_per_step.append(cands)
        
    # Khởi tạo ma trận chi phí Viterbi
    # V[t, i] = chi phí tích lũy nhỏ nhất đến bước t tại candidate i
    num_cands = len(candidates_per_step[0])
    V = np.zeros((K, num_cands), dtype=np.float64)
    backpointer = np.zeros((K, num_cands), dtype=int)
    
    # Bước t = 0 (bước đầu tiên chỉ có target cost)
    for i in range(num_cands):
        cand = candidates_per_step[0][i]
        c_t = calculate_target_cost(cand["meta"], word_stats)
        V[0, i] = c_t
        backpointer[0, i] = -1
        
    # Bước t = 1 đến K-1
    for t in range(1, K):
        for i in range(num_cands):
            cand_cur = candidates_per_step[t][i]
            c_target = calculate_target_cost(cand_cur["meta"], word_stats)
            
            min_cost = float("inf")
            best_prev = -1
            
            for p in range(num_cands):
                cand_prev = candidates_per_step[t - 1][p]
                c_join, _, _, _ = join_cost(cand_prev["audio"], cand_cur["audio"], sr=sr)
                total_step_cost = V[t - 1, p] + lambda_join * c_join
                
                if total_step_cost < min_cost:
                    min_cost = total_step_cost
                    best_prev = p
                    
            V[t, i] = min_cost + c_target
            backpointer[t, i] = best_prev
            
    # Truy vết đường đi tối ưu (backtracking)
    best_last = int(np.argmin(V[K - 1, :]))
    optimal_path_indices = [best_last]
    for t in range(K - 1, 0, -1):
        prev = backpointer[t, optimal_path_indices[-1]]
        optimal_path_indices.append(prev)
    optimal_path_indices.reverse()
    
    optimal_candidates = [candidates_per_step[t][optimal_path_indices[t]] for t in range(K)]
    return optimal_candidates, V, backpointer

# ---------------------------------------------------------
# Step 4: Execution & Evaluation
# ---------------------------------------------------------
if __name__ == "__main__":
    print("=== BẮT ĐẦU BƯỚC 2: PHẦN B, C, THÍ NGHIỆM 1 & 2 ===")
    
    # 1. Đọc metadata và tính thống kê từng từ
    metadata_path = "lab4_data/metadata.csv"
    assert os.path.exists(metadata_path), "File metadata.csv chưa tồn tại!"
    df_meta = pd.read_csv(metadata_path)
    
    word_stats = {}
    for label, group in df_meta.groupby("label"):
        word_stats[label] = {
            "dur_mean": group["duration"].mean(),
            "dur_std": group["duration"].std() if group["duration"].std() > 1e-4 else 0.03,
            "rms_mean": group["rms_raw"].mean(),
            "rms_std": group["rms_raw"].std() if group["rms_raw"].std() > 1e-4 else 0.005
        }
        
    test_sentence = "Mời khách hàng số 23 đến quầy 5."
    # Text normalization
    from step1_inventory import normalize_queue_text
    tokens = normalize_queue_text(test_sentence)
    print(f"Câu kiểm chứng: '{test_sentence}'")
    print(f"Tokens: {tokens}")
    
    # ---------------------------------------------------------
    # PHẦN B: Baseline Concatenation (Cố định candidate 01, xfade_ms=0)
    # ---------------------------------------------------------
    print("\n--- PHẦN B: BASELINE CONCATENATION ---")
    baseline_paths = []
    for tok in tokens:
        sub = df_meta[(df_meta["label"] == tok) & (df_meta["candidate_id"] == 1)].reset_index(drop=True)
        if len(sub) == 0:
            # fallback if candidate_id is string "01"
            sub = df_meta[(df_meta["label"] == tok) & (df_meta["candidate_id"].astype(str).str.endswith("1"))].reset_index(drop=True)
        baseline_paths.append(sub.iloc[0]["file_path"])
        
    print("Baseline candidate paths (cố định cand 01):")
    for tok, p in zip(tokens, baseline_paths):
        print(f"  [{tok}] -> {p}")
        
    # Ghép thẳng không làm mượt (xfade_ms=0)
    sr = 16000
    os.makedirs("audio_output", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    
    baseline_audio, raw_boundaries = concat_units(baseline_paths, sr=sr, xfade_ms=0)
    raw_wav_path = "audio_output/raw_concat.wav"
    sf.write(raw_wav_path, baseline_audio, sr, subtype="PCM_16")
    print(f"-> Đã lưu Baseline audio: {raw_wav_path}")
    print(f"  Độ dài audio: {len(baseline_audio)/sr:.3f} s ({len(baseline_audio)} samples)")
    print(f"  Vị trí các biên ghép (timestamps in sec): {[round(t, 3) for t in raw_boundaries]}")
    
    # ---------------------------------------------------------
    # PHẦN C: Unit Selection bằng Dynamic Programming (Viterbi)
    # ---------------------------------------------------------
    print("\n--- PHẦN C: UNIT SELECTION (DYNAMIC PROGRAMMING) ---")
    opt_cands, V, _ = unit_selection_dp(tokens, df_meta, word_stats, lambda_join=1.0, sr=sr)
    opt_paths = [c["fpath"] for c in opt_cands]
    
    print("Optimal candidates từ Dynamic Programming:")
    for tok, c in zip(tokens, opt_cands):
        cid = c["meta"]["candidate_id"]
        p = c["fpath"]
        print(f"  [{tok}] -> Cand {cid} ({p})")
        
    # Tính chi phí chi tiết cho cả 2 phương pháp (Thí nghiệm 2)
    # Chi phí Baseline
    baseline_audios = [match_rms(load_unit(p, sr)) for p in baseline_paths]
    baseline_metas = [df_meta[df_meta["file_path"] == p].iloc[0] for p in baseline_paths]
    
    base_target_costs = [calculate_target_cost(m, word_stats) for m in baseline_metas]
    base_join_costs = []
    base_spec_costs = []
    for i in range(len(tokens) - 1):
        jc, d_spec, _, _ = join_cost(baseline_audios[i], baseline_audios[i+1], sr=sr)
        base_join_costs.append(jc)
        base_spec_costs.append(d_spec)
        
    total_base_cost = sum(base_target_costs) + sum(base_join_costs)
    
    # Chi phí Optimal (DP)
    opt_audios = [c["audio"] for c in opt_cands]
    opt_metas = [c["meta"] for c in opt_cands]
    
    opt_target_costs = [calculate_target_cost(m, word_stats) for m in opt_metas]
    opt_join_costs = []
    opt_spec_costs = []
    for i in range(len(tokens) - 1):
        jc, d_spec, _, _ = join_cost(opt_audios[i], opt_audios[i+1], sr=sr)
        opt_join_costs.append(jc)
        opt_spec_costs.append(d_spec)
        
    total_opt_cost = sum(opt_target_costs) + sum(opt_join_costs)
    
    print("\n--- KẾT QUẢ SO SÁNH CHI PHÍ (THÍ NGHIỆM 2) ---")
    print(f"Baseline: Total Target Cost = {sum(base_target_costs):.2f}, Total Join Cost = {sum(base_join_costs):.2f}, TOTAL = {total_base_cost:.2f}")
    print(f"Unit Sel: Total Target Cost = {sum(opt_target_costs):.2f}, Total Join Cost = {sum(opt_join_costs):.2f}, TOTAL = {total_opt_cost:.2f}")
    print(f"-> Giảm chi phí tổng: {((total_base_cost - total_opt_cost)/total_base_cost)*100:.2f}%")
    print(f"-> Giảm Join Cost biên: {((sum(base_join_costs) - sum(opt_join_costs))/sum(base_join_costs))*100:.2f}%")
    
    # Bảng chi tiết từng biên
    boundary_labels = [f"{tokens[i]} -> {tokens[i+1]}" for i in range(len(tokens)-1)]
    df_cost_comp = pd.DataFrame({
        "Biên nối": boundary_labels,
        "Baseline Join Cost": [round(c, 2) for c in base_join_costs],
        "DP Selection Join Cost": [round(c, 2) for c in opt_join_costs],
        "Mức giảm (%)": [round((base_join_costs[i] - opt_join_costs[i])/base_join_costs[i]*100, 1) for i in range(len(boundary_labels))]
    })
    print("\nBảng so sánh Join Cost tại từng biên nối:")
    print(df_cost_comp.to_string(index=False))
    df_cost_comp.to_csv("audio_output/unit_selection_cost_comparison.csv", index=False, encoding="utf-8-sig")
    
    # ---------------------------------------------------------
    # THÍ NGHIỆM 1: Boundary Smoothing với xfade = 0, 5, 15, 30 ms
    # ---------------------------------------------------------
    print("\n--- THÍ NGHIỆM 1: BOUNDARY SMOOTHING (CROSS-FADE) ---")
    xfade_experiments = [0, 5, 15, 30]
    xfade_audios = {}
    xfade_boundaries = {}
    
    for ms_val in xfade_experiments:
        y_out, b_times = concat_units(opt_paths, sr=sr, xfade_ms=ms_val)
        xfade_audios[ms_val] = y_out
        xfade_boundaries[ms_val] = b_times
        out_f = f"audio_output/best_concat_{ms_val}ms.wav"
        sf.write(out_f, y_out, sr, subtype="PCM_16")
        print(f"  Lưu audio xfade={ms_val:2d}ms: {out_f} (Độ dài: {len(y_out)/sr:.3f}s)")
        
    # Lưu bản 15ms vào best_concat.wav theo đúng chuẩn đề bài
    best_concat_path = "audio_output/best_concat.wav"
    sf.write(best_concat_path, xfade_audios[15], sr, subtype="PCM_16")
    print(f"-> Đã lưu bản tối ưu (xfade=15ms) vào: {best_concat_path}")
    
    # ---------------------------------------------------------
    # HÌNH 2: figures/fig2_unit_selection_cost.png
    # ---------------------------------------------------------
    print("\nĐang vẽ Hình 2: figures/fig2_unit_selection_cost.png...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    
    x_indices = np.arange(len(boundary_labels))
    width = 0.35
    
    # Subplot 1: Join Cost tại từng biên nối
    rects1 = ax1.bar(x_indices - width/2, base_join_costs, width, label="Baseline (Cand 01)", color="#d9534f", alpha=0.85, edgecolor="black")
    rects2 = ax1.bar(x_indices + width/2, opt_join_costs, width, label="Unit Selection (DP)", color="#5cb85c", alpha=0.85, edgecolor="black")
    
    ax1.set_ylabel("Join Cost (D_spec + RMS + F0)", fontsize=11, fontweight="bold")
    ax1.set_title("So sánh Join Cost tại từng biên nối từ", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(boundary_labels, rotation=35, ha="right", fontsize=9)
    ax1.legend(frameon=True, facecolor="white", loc="upper right")
    ax1.grid(True, linestyle=":", alpha=0.6, axis="y")
    
    # Subplot 2: Tổng hợp chi phí (Target Cost, Join Cost, Total Cost)
    cost_categories = ["Target Cost", "Join Cost", "Total Cost"]
    base_totals = [sum(base_target_costs), sum(base_join_costs), total_base_cost]
    opt_totals = [sum(opt_target_costs), sum(opt_join_costs), total_opt_cost]
    
    x2 = np.arange(len(cost_categories))
    ax2.bar(x2 - width/2, base_totals, width, label="Baseline (Cand 01)", color="#d9534f", alpha=0.85, edgecolor="black")
    ax2.bar(x2 + width/2, opt_totals, width, label="Unit Selection (DP)", color="#5cb85c", alpha=0.85, edgecolor="black")
    
    for i in range(len(cost_categories)):
        ax2.text(x2[i] - width/2, base_totals[i] + 3, f"{base_totals[i]:.1f}", ha="center", fontsize=9, fontweight="bold")
        ax2.text(x2[i] + width/2, opt_totals[i] + 3, f"{opt_totals[i]:.1f}", ha="center", fontsize=9, fontweight="bold")
        
    ax2.set_ylabel("Tổng chi phí (Cost Units)", fontsize=11, fontweight="bold")
    ax2.set_title("Tổng chi phí: Baseline vs Unit Selection", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(cost_categories, fontsize=11, fontweight="bold")
    ax2.legend(frameon=True, facecolor="white", loc="upper left")
    ax2.grid(True, linestyle=":", alpha=0.6, axis="y")
    
    plt.tight_layout()
    fig2_path = "figures/fig2_unit_selection_cost.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"-> Đã lưu Hình 2: {fig2_path}")
    
    # ---------------------------------------------------------
    # HÌNH 3: figures/fig3_boundary_comparison.png
    # ---------------------------------------------------------
    print("\nĐang vẽ Hình 3: figures/fig3_boundary_comparison.png...")
    # Tạo layout chuyên nghiệp:
    # Hàng 1: Full Waveform của toàn câu (đánh dấu biên các từ)
    # Hàng 2: Full Spectrogram của toàn câu (đánh dấu biên các từ)
    # Hàng 3: Phóng to (Zoom-in) tại 2 biên nối: Biên 2 (khách -> hàng) và Biên 5 (hai -> ba)
    # So sánh trước cross-fade (0ms) và sau cross-fade (15ms)
    
    fig = plt.figure(figsize=(15, 10), dpi=300)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.2, 1.3])
    
    # 1. Full Waveform
    ax_full_wave = fig.add_subplot(gs[0, :])
    t_full = np.linspace(0, len(baseline_audio)/sr, len(baseline_audio))
    ax_full_wave.plot(t_full, baseline_audio, color="#2b5c8f", linewidth=0.7, label="Waveform (raw_concat)")
    
    # Vẽ vạch đứt các biên từ và ghi nhãn từ
    prev_t = 0.0
    for i, b_time in enumerate(raw_boundaries):
        ax_full_wave.axvline(b_time, color="#d9534f", linestyle="--", linewidth=1.5, alpha=0.85)
        mid_word_t = (prev_t + b_time) / 2.0
        ax_full_wave.text(mid_word_t, 0.75, tokens[i], ha="center", va="center", fontsize=11, fontweight="bold",
                          bbox=dict(boxstyle="round,pad=0.2", facecolor="#fff2a8", edgecolor="#d9534f", alpha=0.9))
        prev_t = b_time
    # Từ cuối cùng
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
    D = np.abs(signal.stft(baseline_audio, fs=sr, nperseg=512, noverlap=384)[2])
    D_db = 20 * np.log10(np.maximum(D, 1e-5))
    f_stft = np.linspace(0, sr/2, D.shape[0])
    t_stft = np.linspace(0, len(baseline_audio)/sr, D.shape[1])
    
    im = ax_full_spec.pcolormesh(t_stft, f_stft, D_db, shading="gouraud", cmap="inferno", vmin=-60, vmax=10)
    for b_time in raw_boundaries:
        ax_full_spec.axvline(b_time, color="cyan", linestyle="--", linewidth=1.5, alpha=0.85)
    ax_full_spec.set_title("Spectrogram toàn câu (Vạch xanh cyan đánh dấu biên ghép nối từ)", fontsize=12, fontweight="bold")
    ax_full_spec.set_xlabel("Thời gian (giây)", fontsize=10)
    ax_full_spec.set_ylabel("Tần số (Hz)", fontsize=10)
    ax_full_spec.set_ylim(0, 8000)
    cbar = fig.colorbar(im, ax=ax_full_spec, orientation="vertical", pad=0.015)
    cbar.set_label("Cường độ (dB)", fontsize=9)
    
    # 3. Phóng to Biên nối 2 (khách -> hàng) và Biên nối 5 (hai -> ba)
    # Lấy tín hiệu 0ms và 15ms của Unit Selection
    y_0ms = xfade_audios[0]
    y_15ms = xfade_audios[15]
    b_0ms = xfade_boundaries[0]
    b_15ms = xfade_boundaries[15]
    
    # Zoom tại Biên 2: 'khách' -> 'hàng'
    ax_zoom1 = fig.add_subplot(gs[2, 0])
    b2_time = b_0ms[1]
    zoom_win = 0.040  # 40ms quanh biên
    idx_center = int(b2_time * sr)
    idx_start = max(0, idx_center - int(zoom_win * sr))
    idx_end = min(len(y_0ms), idx_center + int(zoom_win * sr))
    
    t_zoom1_0ms = (np.arange(idx_start, idx_end) - idx_center) / sr * 1000.0  # ms quanh biên
    ax_zoom1.plot(t_zoom1_0ms, y_0ms[idx_start:idx_end], color="#d9534f", linewidth=1.6, alpha=0.75, label="xfade = 0 ms (Ghép thẳng - Cắt lệch pha)")
    
    # Cho 15ms
    idx_center_15 = int(b_15ms[1] * sr)
    idx_start_15 = max(0, idx_center_15 - int(zoom_win * sr))
    idx_end_15 = min(len(y_15ms), idx_center_15 + int(zoom_win * sr))
    t_zoom1_15ms = (np.arange(idx_start_15, idx_end_15) - idx_center_15) / sr * 1000.0
    ax_zoom1.plot(t_zoom1_15ms, y_15ms[idx_start_15:idx_end_15], color="#2b5c8f", linewidth=1.8, label="xfade = 15 ms (Làm mượt biên - Mịn pha)")
    
    ax_zoom1.axvline(0, color="black", linestyle=":", linewidth=2, label="Biên nối")
    ax_zoom1.set_title("Phóng to Biên 2: 'khách' ➔ 'hàng' (±40 ms)", fontsize=11, fontweight="bold")
    ax_zoom1.set_xlabel("Thời gian lệch so với biên (ms)", fontsize=10)
    ax_zoom1.set_ylabel("Biên độ", fontsize=10)
    ax_zoom1.legend(frameon=True, facecolor="white", fontsize=8.5, loc="lower right")
    ax_zoom1.grid(True, linestyle=":", alpha=0.6)
    
    # Zoom tại Biên 5: 'hai' -> 'ba'
    ax_zoom2 = fig.add_subplot(gs[2, 1])
    b5_time = b_0ms[4]
    idx_center5 = int(b5_time * sr)
    idx_start5 = max(0, idx_center5 - int(zoom_win * sr))
    idx_end5 = min(len(y_0ms), idx_center5 + int(zoom_win * sr))
    
    t_zoom2_0ms = (np.arange(idx_start5, idx_end5) - idx_center5) / sr * 1000.0
    ax_zoom2.plot(t_zoom2_0ms, y_0ms[idx_start5:idx_end5], color="#d9534f", linewidth=1.6, alpha=0.75, label="xfade = 0 ms (Discontinuity / Click)")
    
    idx_center5_15 = int(b_15ms[4] * sr)
    idx_start5_15 = max(0, idx_center5_15 - int(zoom_win * sr))
    idx_end5_15 = min(len(y_15ms), idx_center5_15 + int(zoom_win * sr))
    t_zoom2_15ms = (np.arange(idx_start5_15, idx_end5_15) - idx_center5_15) / sr * 1000.0
    ax_zoom2.plot(t_zoom2_15ms, y_15ms[idx_start5_15:idx_end5_15], color="#2b5c8f", linewidth=1.8, label="xfade = 15 ms (Smooth Cross-fade)")
    
    ax_zoom2.axvline(0, color="black", linestyle=":", linewidth=2, label="Biên nối")
    ax_zoom2.set_title("Phóng to Biên 5: 'hai' ➔ 'ba' (±40 ms)", fontsize=11, fontweight="bold")
    ax_zoom2.set_xlabel("Thời gian lệch so với biên (ms)", fontsize=10)
    ax_zoom2.set_ylabel("Biên độ", fontsize=10)
    ax_zoom2.legend(frameon=True, facecolor="white", fontsize=8.5, loc="lower right")
    ax_zoom2.grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    fig3_path = "figures/fig3_boundary_comparison.png"
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"-> Đã lưu Hình 3: {fig3_path}")
    print("=== HOÀN TẤT BƯỚC 2 THÀNH CÔNG ===")
