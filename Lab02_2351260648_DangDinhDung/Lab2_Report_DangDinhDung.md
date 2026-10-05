# BÁO CÁO THỰC HÀNH LAB 2: XỬ LÝ ÂM THANH VÀ TIẾNG NÓI
**Chủ đề:** Đặc trưng tiếng nói và nhận dạng bằng DTW
**Sinh viên thực hiện:** Đặng Đình Dũng
**Chuyên ngành:** Trí tuệ Nhân tạo và Khoa học Dữ liệu

---

## 1. Bảng cấu hình tham số Baseline
| Tham số | Giá trị | Ý nghĩa |
| :--- | :--- | :--- |
| **$F_s$** | 16,000 Hz | Chuẩn hóa sampling rate cho toàn bộ file |
| **Frame / Hop** | 25 ms / 10 ms | Giả thiết tín hiệu gần dừng trong khung ngắn |
| **Window** | Hamming | Giảm hiện tượng rò rỉ phổ (spectral leakage) |
| **Pre-emphasis** | $\alpha = 0.97$ | Nhấn mạnh các thành phần tần số cao |
| **NFFT** | 512 | Số điểm lấy FFT cho mỗi frame |
| **Mel filters** | 24 | Số lượng bộ lọc mô phỏng thính giác |
| **MFCC** | 13 hệ số | Vector đặc trưng nhỏ gọn đại diện cho mỗi frame |
| **Local distance**| Euclidean | Tính khoảng cách không gian giữa 2 frame MFCC |
| **DTW cost** | Tổng / path length| Chuẩn hóa chi phí để công bằng với các file dài |

---

## 2. Thí nghiệm mở rộng có kiểm soát
**Thí nghiệm E1: Không trim vs. Có trim endpoint**
*   **Kết quả:** Khi không cắt khoảng lặng, thuật toán DTW mất rất nhiều chi phí để căn chỉnh phần silence nền ở hai đầu file, dẫn đến `DTW_norm` tăng cao sai lệch và Accuracy giảm mạnh. 
*   **Kết luận:** Endpoint detection là bước tiền xử lý bắt buộc để DTW tập trung đối sánh đúng đặc trưng âm học của giọng nói.

**Thí nghiệm E2: 13 MFCC vs. 13 MFCC + $\Delta$ (Delta features)**
*   **Kết quả:** Việc bổ sung đạo hàm bậc 1 ($\Delta$) giúp hệ thống nắm bắt được sự biến thiên động của phổ âm thanh theo thời gian, qua đó cải thiện khả năng phân biệt các từ có cấu trúc nguyên âm tĩnh khá giống nhau.

---

## 3. Trả lời câu hỏi học thuật
**Câu 1:** Dùng toàn bộ waveform làm template rất dễ sai lệch vì tín hiệu miền thời gian phụ thuộc lớn vào pha và tốc độ nói. Waveform của cùng một từ đọc 2 lần sẽ không bao giờ khớp nhau từng mẫu (sample).

**Câu 2:** Short-time energy có biên độ bình phương, nhạy cảm với các âm hữu thanh (Voiced - năng lượng cao). ZCR đo mức độ đổi dấu, nhạy cảm với âm vô thanh (Unvoiced - năng lượng thấp nhưng tần số cao). Kết hợp cả hai giúp tìm chính xác ranh giới đầu/cuối của từ mà không cắt phạm vào các phụ âm.

**Câu 3:** Thang Mel mô phỏng đặc tính phi tuyến của ốc tai người: rất nhạy bén với sự thay đổi ở dải tần số thấp (nên các bộ lọc hẹp và sát nhau), nhưng kém nhạy ở dải tần số cao (nên các bộ lọc trải rộng dần).

**Câu 4:** Phép Log nén dải động (dynamic range) của năng lượng, giúp hệ thống không bị phụ thuộc quá nhiều vào âm lượng to/nhỏ của người nói. Phép biến đổi DCT biến phổ log-energy thành các hệ số Cepstral, giúp giải tương quan (decorrelate) các đặc trưng để dùng tốt cho các bộ đối sánh.

**Câu 5:** 
*   Bước chéo: Căn chỉnh khớp 1-1 thời gian giữa 2 utterance.
*   Bước ngang/dọc: Xảy ra hiện tượng nén hoặc giãn thời gian (một frame của test file phải khớp với nhiều frame của template hoặc ngược lại).

**Câu 6:** Khi một utterance bị phát âm chậm, số lượng frame tăng lên kéo theo đường Optimal Path dài ra. Nếu không chia trung bình cho chiều dài đường đi (path length), các từ nói dài sẽ luôn bị cộng dồn chi phí cao hơn một cách bất công.

**Câu 7:** Ba nguyên nhân làm thay đổi MFCC của cùng một từ: (1) Tốc độ nói (nhịp điệu) khác nhau; (2) Mức độ nhấn nhá/âm lượng (Dynamics) thay đổi; (3) Nhiễu môi trường hoặc khoảng cách đến micro bị thay đổi.

**Câu 8:** Dựa trên ma trận nhầm lẫn, các từ dễ nhầm thường chia sẻ chung cấu trúc nguyên âm hoặc có thời lượng rất ngắn. Việc cắt biên thiếu khoảng đệm (margin) có thể làm mất âm đầu/cuối, biến đặc trưng âm học của chúng trở nên tương đồng.

**Câu 9:** DTW hoạt động theo cơ chế "khớp mẫu" (Template Matching) nên rất nhạy cảm với biến thiên đặc trưng cá nhân (giọng nam/nữ, âm sắc). Khi gặp người nói mới hoàn toàn (cross-speaker), khoảng cách DTW sẽ tăng vọt. Để khắc phục, ta cần sử dụng Mô hình Markov Ẩn (HMM) ở Chương 3 để mô hình hóa phân phối xác suất thống kê thay vì so sánh cứng nhắc.
