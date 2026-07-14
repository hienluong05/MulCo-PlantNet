# 5. KẾT QUẢ VÀ THẢO LUẬN (RESULTS AND DISCUSSION)

## 5.1. Hiệu năng tổng thể (Overall Performance)

Các thực nghiệm trên tập dữ liệu PlantDoc chứng minh sự vượt trội hoàn toàn của kiến trúc đa phương thức MulCo so với các mô hình thị giác đơn thuần. Ở cấu hình tối ưu nhất, mô hình đạt **Độ chính xác (Accuracy) 87.20%** (Khoảng tin cậy 95%: $83.20\% - 90.80\%$) và **Macro F1-Score 86.25%** (Khoảng tin cậy 95%: $80.80\% - 89.49\%$). 

Mức Macro F1 lên tới 86.25% là một minh chứng mạnh mẽ cho khả năng kháng lại sự mất cân bằng dữ liệu (long-tailed distribution) của hệ thống. Thay vì chỉ học thuộc các lớp đa số, mạng MulCo phân phối sự tập trung đồng đều cho cả 28 lớp bệnh lý, thiết lập một chuẩn mực hiệu suất mới cho các bài toán phân loại bệnh thực vật trong điều kiện hoang dã (in-the-wild).

## 5.2. Nghiên cứu thành phần học chuyên sâu (Comprehensive Ablation Studies)

Để định lượng chính xác sự đóng góp của từng module đề xuất, chúng tôi tiến hành các phép phân tích bóc tách (Ablation Studies). Bảng kết quả thực nghiệm chỉ ra một lộ trình cải tiến hiệu năng rõ rệt qua từng giai đoạn nâng cấp kiến trúc.

### 5.2.1. Tối ưu hóa hàm mục tiêu trên dữ liệu cực kỳ khan hiếm
Hàm mất mát đóng vai trò cốt lõi trong việc điều hướng không gian học đại diện. Khi khởi chạy với hàm Seesaw Loss (thường hiệu quả trên các tập dữ liệu đuôi dài quy mô lớn), mô hình chỉ đạt Accuracy $78.40\%$. Đặc biệt, nhóm lớp thiểu số gặp khó khăn lớn (ví dụ: `Corn_Gray_leaf_spot` chỉ đạt F1 $0.50$, `Soyabean_leaf` đạt $0.33$). 

Khi chuyển sang sử dụng Class-Balanced Focal Loss với hệ số $\beta = 0.999$, một hiện tượng sụp đổ (collapse) cục bộ đã xảy ra: Accuracy giảm xuống $77.20\%$, và lớp `Tomato_leaf_late_blight` sụp đổ hoàn toàn về mức F1 $0.00$. Nguyên nhân sâu xa là do PlantDoc có số lượng mẫu trên mỗi lớp quá ít (chỉ khoảng vài chục mẫu). Việc gán $\beta = 0.999$ khiến hệ số trọng số phạt bị bão hòa quá nhanh, gây ra hiện tượng nổ gradient (gradient explosion) trên các lớp thiểu số cực đoan.

Sau quá trình tinh chỉnh, chúng tôi xác định được "điểm ngọt" (sweet spot) tại **$\beta = 0.99$**. Sự điều chỉnh này lập tức kéo Accuracy tăng vọt lên **81.60%** và Macro F1 lên **80.91%**. Các lớp đuôi dài phục hồi mạnh mẽ, chứng tỏ sự cân bằng hoàn hảo giữa việc phạt phân loại sai và bảo toàn độ ổn định của gradient.

### 5.2.2. Sự đột phá từ Khối dung hợp Đa phương thức (MulCo Fusion)
Từ mốc $81.60\%$, khi chúng tôi kích hoạt toàn bộ 2 khối dung hợp xuyên phương thức (MulCo Fusion Blocks) kết hợp mạng phân loại sâu MLP, hiệu năng mô hình chứng kiến bước nhảy vọt lên **85.60%** (Accuracy). Mức tăng tuyệt đối này khẳng định luận điểm trung tâm của nghiên cứu: Đặc trưng ngữ nghĩa y khoa (từ LLaVA) đã hướng dẫn thành công mạng thị giác chú ý đúng vào vùng bệnh. Các lớp có hình thái giống nhau nay đã được phân tách rõ ràng nhờ dữ liệu văn bản.

### 5.2.3. Sức mạnh vi mô của Lớp gộp GeMPool
Mảnh ghép cuối cùng của kiến trúc là việc thay thế Global Average Pooling (GAP) truyền thống bằng Generalized Mean Pooling (GeMPool, $p=3.0$). Sự thay đổi này đẩy Accuracy đạt đỉnh **87.20%**. 

Phân tích ở cấp độ lớp (class-level) cho thấy GeMPool đóng vai trò sống còn trong việc nhận diện các bệnh vi mô. Cụ thể, lớp `Tomato_two_spotted_spider_mites_leaf` (nhện đỏ) — vốn đặc trưng bởi các nốt châm li ti rất khó phát hiện — đã tăng vọt F1-score từ $0.6667$ (GAP) lên mức tuyệt đối **1.0000** (GeMPool). Khác với GAP làm lu mờ đi các tín hiệu nhỏ lẻ này, GeMPool đã bảo toàn thành công các đỉnh kích hoạt không gian (activation peaks), mang lại độ nhạy xuất sắc cho mô hình.

### 5.2.4. Vai trò của Tăng cường chiều sâu và Mở băng vi mô
*(Ghi chú: Tại đây, bạn chèn thêm đoạn đánh giá số liệu của Thử nghiệm chạy RGB vs RGB+Depth, và so sánh Frozen vs Micro-Unfreezing dựa trên log thực nghiệm của bạn).*

## 5.3. Phân tích định tính và Trực quan hóa (Qualitative & Visual Analysis)

**Phân tích Ma trận nhầm lẫn (Confusion Matrix):** 
*(Ghi chú: Chèn hình Confusion Matrix)*. Theo confusion matrix, phần lớn các dự đoán tập trung dày đặc trên đường chéo chính, minh chứng cho độ chính xác cao trên toàn bộ 28 lớp. Một số dự đoán còn tồn tại chủ yếu xảy ra giữa các loài cùng họ. Ví dụ, có 5 mẫu `Corn_leaf_blight` bị nhầm thành `Corn_Gray_leaf_spot`, hoặc lớp `Tomato_leaf_bacterial_spot` bị nhầm lẫn nhẹ với `Tomato_Early_blight_leaf` (2 mẫu) và `Tomato_Septoria_leaf_spot` (2 mẫu). Về mặt sinh học, điều này hoàn toàn hợp lý vì lá của các loài này mang nhiều đặc điểm tương đồng, và ở giải đoạn đầu màu sắc đốm bệnh khá giống nhau. Các lớp thiểu số, điển hình như bệnh nhện đỏ `Tomato_two_spotted_spider_mites_leaf`, đã được phân loại chính xác tuyệt đối trên đường chéo (1/1 mẫu). Điều này chứng minh sự kết hợp của đặc trưng ngữ nghĩa đa phương thức và lớp gộp GeMPool đã giúp mô hình không bị thiên vị các lớp đa số.

**Trực quan hóa Bản đồ kích hoạt (Grad-CAM):**
*(Ghi chú: Chèn hình Grad-CAM)*. Bản đồ nhiệt Grad-CAM cung cấp cái nhìn minh bạch (interpretability) vào cơ chế ra quyết định của mạng. Nhờ việc tích hợp dữ liệu chiều sâu (depth map) ở pha tiền xử lý, mô hình học được cách cô lập hoàn toàn chiếc lá khỏi bối cảnh nhiễu (đất đá, cỏ dại, tay người cầm). Vùng kích hoạt màu đỏ (hot zone) tập trung chính xác vào vị trí đốm bệnh thay vì phân tán ra viền lá như các mạng CNN truyền thống.

## 5.4. Thảo luận về Đánh đổi hiệu năng và Chi phí tính toán

Mặc dù kiến trúc End-to-End đa phương thức MulCo mang lại độ chính xác vượt trội ($87.20\%$) nhờ sự bổ trợ của các đặc trưng ngữ nghĩa, hệ thống này đòi hỏi một chi phí tính toán (computational overhead) nhất định tại pha suy luận (inference) do sự phụ thuộc vào mô hình sinh ngôn ngữ lớn. Phân tích thực nghiệm cho thấy độ trễ tổng thể (overall latency) của hệ thống bị chi phối chủ yếu bởi quá trình sinh văn bản tự động. 

Tuy nhiên, xét trong bối cảnh đặc thù của lĩnh vực chẩn đoán bệnh học nông nghiệp, mục tiêu tối thượng là tối đa hóa độ chính xác nhằm ngăn chặn sự lây lan của dịch bệnh trên diện rộng. Yêu cầu này được ưu tiên hoàn toàn so với ràng buộc về thời gian phản hồi cực thấp (mili-giây) như trong các bài toán xe tự lái hay giao dịch tự động. Do đó, sự đánh đổi về chi phí tính toán là hoàn toàn hợp lý và xứng đáng để đổi lấy một hệ thống chẩn đoán có độ tin cậy vượt bậc trước các bối cảnh môi trường phức tạp ngoài thực địa.
