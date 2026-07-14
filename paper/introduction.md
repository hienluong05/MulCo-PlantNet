# Động lực nghiên cứu

Bệnh thực vật là mối đe dọa nghiêm trọng với an ning lương thực toàn cầu, gây ra **tổn thất kinh tế200 tỷ USD mỗi năm** và làm **mất 10-16% sản lượng cây trồng chủ lực** trên toàn cầu. **A Review on Sustainable Plant Disease Management through Integrated Approaches** **Emerging infectious diseases threatening food security and economies in Africa**

Từ đó đặt ra yêu cầu cấp thiết là nông dân cần phát hiện bệnh sớm tại đồng ruộng. Tuy nhiên, việc này rất khó khăn vì đồng ruộng là nơi có điều kiện môi trường phức tạp. Có nhiều yếu tố tác động đến việc phát hiện bệnh như độ ẩm, nhiệt độ, ánh sáng hay tính chât khác nhau của từng loại đất. Các hiện tượng thời tiết như mưa gió cũng làm bệnh lan nhanh hoặc thay đổi triệu chứng.

Trí tuệ nhân tạo và học sâu đang nổi lên như một công nghệ hiệu quả trong nông nghiệp, giúp phát hiện bệnh sớm và chính xác ngay tại đồng ruộng.

# Hạn chế của các nghiên cứu tiền nhiệm

Các mô hình **CNN** và **ViT** hiện tại phần lớn được huấn luyện trên **ảnh chụp trong phòng thí nghiệm** với nền trơn, ánh sáng đồng đều, không nhiễu. Tuy nhiên, trên thực tế, ảnh chụp cây trồng có thể bị nhiễu nghiêm trọng bởi các yếu tố tự nhiên. Vết bệnh có thể bị bóng râm che khuất, màu sắc lá có thể bị ánh sáng làm thay đổi. Nhiễu nền và đất đá xung quanh cũng dễ lấn át đặc điểm bệnh, khiến mô hình học nhầm. Vì vậy khi áp dụng vào thực tế, độ chính xác của các mô hình này thường giảm mạnh.

Không chỉ nhiễu môi trường mà đặc điểm vết bệnh cũng là thách thức. Đặc biệt ở giai đoạn khởi phát, các vết bệnh thường khá mờ nhạt và nhỏ so với diện tích lá. Các kiến trúc phân loại thông thường sử dụng Global Average Pooling (như resnet) hay các lớp tích chập để giảm độ phân giải không gian, có thể làm các triệu chứng bệnh bị trộn lẫn với vùng lá khỏe mạnh xung quanh, từ đó không nhận diện được bệnh.

Bên cạnh thách thức về thị giác, các mô hình còn gặp khó khăn với tình trạng mất cân bằng nghiêm trọng của các tập dữ liệu thực tế. Điển hình là tập PlantDoc. Sự chênh lệch lớn về số lượng mẫu giữa các lớp bệnh phổ biến và các lớp bệnh hiếm gặp tạo ra hiện tượng phân phối đuôi dài (**long-tailed distribution**). Điều này khiến mô hình thiên vị các lớp đa số và ngó lơ các lớp thiểu số, hay các lớp bệnh hiếm.

Để khắc phục hạn chế của các mô hình chỉ dựa trên thị giácVision-Language Models (VLMs) đang trở thành xu hướng mới. Phương pháp này được kì vọng sẽ bổ sung thêm thông tin chi tiết về đặc điểm vết bệnh và môi trường để mô hình được định hướng tốt hơn so với chỉ có dữ liệu hình ảnh. Tuy nhiên hướng tiếp cận này vấp phải rào cản lớn. Việc gán nhãn thủ công các mô tả cho hình ảnh đòi hỏi chi phí lớn cho các chuyên gia nông nghiệp.

# Đóng góp của bài báo

Nghiên cứu đề xuất MulCo - một framwork deep learning đa phương thức giải quyết đồng thời 3 bài toán: nhiễu ngoài tự nhiên, phát hiện đốm bệnh kích thước nhỏ và thiên lệch do mất cân bằng dữ liệu. Các đóng góp cốt lõi của nghiên cứu bao gồm:

### Xử lý ảnh theo chiều sâu

Nghiên cứu thực hiện xử lý ảnh theo chiều sâu trước khi đưa vào huấn luyện. Quá trình này tách nhiễu nền ra khỏi lá bệnh, định hướng sự tập trung của mô hình vào vùng quan trọng mà không làm phát sinh thêm bất kỳ chi phí tính toán nào trong quá trình dự đoán thực tế.

### **Cơ chế dung hợp và kỹ thuật tinh chỉnh vi mô (Micro Unfreezing)**

- Về vision, chúng tôi tích hợp mạng **ConvNeXt** với mô-đun chú ý **CBAM** nhằm khai thác đồng thời các đặc trưng không gian và kênh sâu, bảo toàn các đốm bệnh siêu nhỏ trước các lớp gộp không gian.
- Về dung hợp, chúng tôi phát triển khối **MulCoFusion** kết hợp giữa *Cross-Attention* (hướng dẫn thị giác bằng ngữ cảnh) và khối *Restormer* cải tiến (sử dụng *Multi-Dconv Head Transposed Attention - MDTA* và *Gated-Dconv Feed-Forward Network - GDFN*) nhằm học mối liên hệ giữa 2 nhánh dữ liệu.
- Về ngôn ngữ, chúng tôi áp dụng kỹ thuật **Micro Unfreezing** trên bộ mã hóa **RoBERTa**, chỉ giải phóng một tỷ lệ nhỏ các trọng số chuyên biệt để học thuật ngữ nông nghiệp bản địa, giúp mô hình nhạy bén với thông tin bệnh học mà không làm mất đi tri thức tổng quát sẵn có hoặc gây overfitting.

### Tối ưu hóa hàm loss cho dữ liệu long-tail

Chúng tôi sử dụng hàm loss tối ưu cho số mẫu hiệu dụng và độ khó của từng lớp dữ liệu. Hàm loss này dành trọng số phạt lớn hơn cho các lớp bệnh hiếm hoặc khó học, tạo sự cân bằng lớp trong quá trình huấn luyện, nâng cao đáng kể độ chính xác trên toàn bộ tập dữ liệu.

### Sử dụng cơ chế pooling cải tiến

Nghiên cứu sử dụng Genearlized Mean Pooling (GeM Pooling) thay cho các phương pháp Pooling truyền thống như Global Max Pooling (GMP) hay Global Average Pooling (GAP). Bằng cách điều chỉnh tham số mũ có thể học (learnable parameter), GeM Pooling dung hòa ưu điểm của cả hai phương pháp trên: vừa có khả năng học đặc trưng tổng thể của GAP, vữa giữ được các đặc trưng nổi bật đại diện cho các đốm bện nhỏ như GMP.
