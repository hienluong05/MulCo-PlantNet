# Khung dán nhãn ngoại tuyến

Để cung cấp luồng thông tin văn bản đầu vào cho kiến trúc MulCo, nghiên cứu này áp dụng phương pháp Sinh mô tả hình ảnh tự động (Automated Image Description Generation - AIDG) [1] nhằm chuyển đổi tập dữ liệu đơn phương thức PlantDoc. Cụ thể, hệ thống sử dụng mô hình ngôn ngữ lớn đa phương thức LLaVA để tự động tạo ra các đoạn mô tả tự nhiên đa câu cho từng bức ảnh. Quá trình này đóng vai trò là một module tiền xử lý ngoại tuyến (offline preprocessing) độc lập.

Để đảm bảo các mô tả bám sát đúng logic chẩn đoán bệnh học, một cấu trúc gợi ý theo Chuỗi suy luận (Zero-Shot Chain-of-Thought - CoT) được thiết lập theo phương pháp của [1]. Biểu mẫu này định hướng LLaVA tuân theo quy trình lập luận từng bước: nhận diện loại cây, định vị tổn thương, phân tích sự biến đổi màu sắc của lá, cho đến đánh giá các triệu chứng hình thái học (ví dụ: hoại tử, đốm vòng).

Về mặt hình thức toán học [1], với một ảnh đầu vào $I_i$ thuộc tập dữ liệu ảnh gốc $\mathcal{D}_{\text{img}}$, mô tả văn bản có cấu trúc tương ứng $T_i$ được sinh ra thông qua hàm sinh của mô hình đa phương thức được dẫn hướng bằng CoT (ký hiệu là $\mathcal{G}_{\text{CoT}}$):

$T_i = \mathcal{G}_{\text{CoT}}(I_i)$

Sau khi quá trình sinh văn bản hoàn tất, mỗi hình ảnh gốc được kết nối với mô tả của nó để tạo thành một tập dữ liệu giả nhãn (pseudo-labeled dataset) [1]. Tập dữ liệu này được lưu trữ dưới định dạng JSON để đảm bảo sự đồng bộ về mặt ngữ nghĩa và được định nghĩa toán học như sau:

$\mathcal{D}_{\text{pair}} = \{(I_i, T_i)\}_{i=1}^{N}$

Trong đó, $N$ là tổng số lượng mẫu hình ảnh của tập PlantDoc. Tập dữ liệu chất lượng cao $\mathcal{D}_{\text{pair}}$ này cung cấp trực tiếp các tín hiệu ngữ nghĩa (semantic cues) hỗ trợ đắc lực cho mạng phân loại đa phương thức MulCo (downstream multimodal training) ở các giai đoạn tiếp theo.

# Tiền xử lý dữ liệu

PlantDoc là tập dữ liệu nhỏ, chỉ chứa khoảng hơn 2500 bức ảnh, được chia thành 27 - 30 lớp khác nhau, bao gồm cả các triệu chứng bệnh trên lá và trạng thái lá khỏe mạnh.

Phiên bản được sử dụng trong nghiên cứu này có tập train chứa 2337 ảnh, được chia thành 28 lớp với phân bố như sau:

| Lớp                                  | Số lượng ảnh |
| ------------------------------------ | ------------ |
| Apple_leaf                           | 69           |
| Apple_rust_leaf                      | 84           |
| Apple_Scab_Leaf                      | 73           |
| Bell_pepper_leaf                     | 29           |
| Bell_pepper_leaf_spot                | 65           |
| Blueberry_leaf                       | 93           |
| Cherry_leaf                          | 41           |
| Corn_Gray_leaf_spot                  | 55           |
| Corn_leaf_blight                     | 160          |
| Corn_rust_leaf                       | 94           |
| grape_leaf                           | 55           |
| grape_leaf_black_rot                 | 62           |
| Peach_leaf                           | 90           |
| Potato_leaf_early_blight             | 138          |
| Potato_leaf_late_blight              | 176          |
| Raspberry_leaf                       | 98           |
| Soyabean_leaf                        | 50           |
| Squash_Powdery_mildew_leaf           | 109          |
| Strawberry_leaf                      | 77           |
| Tomato_Early_blight_leaf             | 69           |
| Tomato_leaf                          | 38           |
| Tomato_leaf_bacterial_spot           | 88           |
| Tomato_leaf_late_blight              | 88           |
| Tomato_leaf_mosaic_virus             | 38           |
| Tomato_leaf_yellow_virus             | 196          |
| Tomato_mold_leaf                     | 74           |
| Tomato_Septoria_leaf_spot            | 127          |
| Tomato_two_spotted_spider_mites_leaf | 1            |

Số lượng ảnh trong mỗi lớp không vượt quá 196 ảnh.

Với tập dữ liệu nhỏ như vậy, mô hình khó có thể khái quát hóa các đặc trưng chung, là một trong những nguyên nhân hàng đầu dẫn đến hiện tượng overfitting.

Vấn đề này đặt ra yêu cầu tiền xử lý và tăng cường dữ liệu, giúp mô hình được huấn luyện trên tập dữ liệu lớn hơn, đa dạng hơn, giảm thiểu khả năng overfitting, nâng cao khả năng tổng quát hóa của mô hình để hoạt động tốt trong môi trường thực tế.

Ảnh gốc được chuẩn hóa về kích thước 224 x 224 - kích thước chuẩn hóa tối ưu của mạng backbone như ConvNext-CBAM. Mỗi hình ảnh trong tập huấn luyện được đưa vào mô hình Depth Anything V2 để dự đoán bản đồ độ sâu tương ứng với mỗi pixel. Lá cây hay những vùng bị bệnh thường nằm gần camera hơn so với nhiễu nền như đất, đá, cỏ dại và các lá cây khác. Từ bản đồ độ sâu, chọn một ngưỡng gần/xa. Dựa trên ngưỡng này có thể sinh mặt nạ nền bằng phân ngưỡng - những điểm ảnh có giá trị lớn hơn giá trị ngưỡng được gán là 1, ngược lại nhỏ hơn giá trị ngưỡng được gán là 0. Quy ước này tương đương với những điểm ảnh có độ sâu lớn hơn ngưỡng được coi là xa, còn lại được coi là gần. Nhân mặt nạ nền với ảnh gốc để xóa nền. Kết quả là tạo ra một phiên bản hình ảnh mới với nhiễu nền được loại bỏ hoàn toàn, giúp mô hình tập trung vào những vùng ảnh chứa đặc trưng lá cây hoặc bệnh, có ảnh hưởng lớn đến quyết định phân loại.

Hình … cho thấy bản đồ độ sâu giúp loại bỏ nhiễu nền hiệu quả trên hình ảnh được lấy từ lớp bệnh trên cây …

<aside>
💡

Chèn ảnh và bản đồ độ sâu vào đây

</aside>

Tăng lượng mẫu trên tập huấn luyện

Giúp mô hình tập trung vào vùng cần thiết

Giải quyết vấn đề domain shift

Tập huấn luyện mới được tạo bằng cách đặt ảnh đã xóa nhiễu nền bên cạnh ảnh gốc, giúp lượng mẫu của mỗi lớp tăng gấp đôi, giải quyết tình trạng khan hiếm mẫu của tập dữ liệu PlantDoc. Mô hình được huấn luyện trên cả ảnh đã xóa nhiễu nền và ảnh gốc, giúp mô hình vừa học cách tập trung vào những vùng lá chứa đặc trưng bệnh, giảm ảnh hưởng của nhiều nền, vừa không bị giảm hiệu suất khi làm việc trên những bức ảnh chụp trong điều điện môi trường phức tạp ngoài phòng thí nghiệm. Quá trình làm giàu dữ liệu được thực hiện hoàn toàn ở bước tiền xử lý, quy trình huấn luận và dự đoán không chứa bước chạy mô hình Depth Anything V2 nên không kéo dài thời gian huấn luyện.

Dữ liệu trong tập huấn luyện mới còn trải qua các phép biến đổi dữ liệu để mô phỏng những yếu tố tự nhiên trong môi trường thực địa như che khuất, nhòe nét, thay đổi ánh sáng, hướng chụp ảnh ngẫu nhiên của người nông dân trên cánh đồng. Bảng … trình bày cụ thể tác động và ý nghĩa của từng phép biến đổi.

| Phép biến đổi                                                                  | Tác dụng kỹ thuật                                                                                                                           | Ý nghĩa thực tế                                                                                                                                                                  |
| ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| transforms.RandomHorizontalFlip(p=0.5) và transforms.RandomVerticalFlip(p=0.5) | Lật ảnh ngẫu nhiên theo chiều ngang/dọc với xác suất 50%                                                                                    | Trên thực tế chiếc lá bị bệnh có thể mọc quay sang trái, sang phải, dựng lên hoặc rủ xuống. Việc lật ảnh giúp mô hình học được đặc điểm lá và đốm bệnh bất kể góc nhìn/hướng lá. |
| transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2))    | Thực hiện đồng thời 3 biến đổi: xoay $\pm 15^\circ$, dịch tâm ảnh tối đa 10% theo chiều ngang/dọc, thu phóng ảnh trong khoảng 80% đến 120%. | Mô phỏng hành vi chụp ảnh của con người:                                                                                                                                         |

• Xoay: chụp nghiêng góc
• Dịch chuyển: chiếc lá không nằm giữa khung hình
• Thu phóng: khoảng cách từ camera đến lá có thể gần/xa |
| transforms.ColorJitter(brightness=0.15, contrast=0.15) | Thay đổi ngẫu nhiên độ sáng và độ tương phản trong khoảng $\pm 15^\circ$ | Ánh sáng trong môi trường tự nhiên thay đổi theo thời gian trong ngày hoặc theo thời tiết. |
| transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.2) | Làm mờ Gauss với xác suất 20% | Mô phỏng hiện tượng ảnh bị nhòe nét do camera không lấy được nét. |
| transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0) | Xóa ngẫu nhiên một vùng hình chữ nhật trên ảnh với xác suất 20%. Diện tích vùng xóa khoảng 2% đến 20% toàn bộ bức ảnh. | Mô phỏng hiện tượng lá bị che khuất, hoặc lá cây mọc chen chúc lên nhau trong tự nhiên. |

# Mạng trích xuất đặc trưng

### Hình ảnh

ConvNext tích hợp nhiều đặc điểm của Vision Tranformer như chia ảnh thành các patch, cho phép học tương quan toàn cục, không bị giới hạn trong vùng lân cận bởi các phép tích chập như CNN. Về bản chất ConvNext vẫn là CNN được cải tiến trên nền ResNet-50 nên giữ lại được ưu điểm của CNN. Nhờ vậy ConvNext vừa học được hình thái tổng thể lá, vừa học được chi tiết vùng lá chứa đặc điểm bệnh.

Cơ chế attention từ CBAM giúp mô hình tập trung vào vùng thông tin quan trọng nhất, chẳng hạn như các đốm bệnh, bỏ qua các vùng không liên quan như phần lá khỏe mạnh và nhiễu nền.

Những vector đặc trưng sinh ra bởi mô hình ConvNext-CBAM tạo tiền đề vững chắc cho quá trình tích hợp đa phương thức, nâng cao hiệu năng của toàn bộ thiết kế.

### Text

Nhờ cơ chế self-attention, mô hình RoBERTa có khả năng hiểu ngữ cảnh một cách sâu sắc, từ đó trích xuất chính xác những từ, cụm từ chứa mô tả tình trạng bệnh trên chiếc lá. RoBERTa được pre-training trên tập dữ liệu khổng lồ, đa dạng, giúp mô hình học được các cấu trúc câu và kiến thức tổng quát về thế giới. Do đó, mô hình có khả năng tranfer learning mạnh mẽ, dễ dàng được tinh chỉnh cho các bài toán cụ thể, kể cả những lĩnh vực đặc thù như nông nghiệp.

Những đặc trưng ngữ nghĩa được RoBERTa trích xuất mang tính biểu diễn cao, phản ánh trung thực mức độ nghiêm trọng và bản chất của từng loại bệnh thông qua không gian vector dày đặc. Các vector này không chỉ nắm bắt được thông tin của từng token mà còn chứa ngữ cảnh của toàn bộ đoạn văn, tạo nên khác biệt giữa các nhãn bệnh khác nhau. Điều này cho phép đầu phân loại phía sau hội tụ nhanh hơn và đạt độ chính xác vượt trội.

# Head phân loại

Bài toán fine-grained đặc trưng bởi khác biệt rất nhỏ giữa các lớp, ví dụ hình dạng lá, vết bệnh, mảng màu, hoặc cấu trúc chi tiết cục bộ. Ở khối phân loại cuối cùng, việc lựa chọn cơ chế gộp không gian (Spatial Pooling) ảnh hưởng trực tiếp đến khả năng giữ lại các thông tin vi mô này.

Global Average Pooling lấy trung bình tất cả các giá trị trong kernel, có thể làm lu mờ giá trị của các đốm bệnh nhỏ. Global Max Pooling lấy giá trị lớn nhất, giúp mô hình tập trung vào đốm bệnh, nhưng cũng khiến mô hình không học được hình thái tổng quát của chiếc lá.

Nghiên cứu sử dụng Generalized Mean Pooling (GeM Pooling) để cân bằng giữa hai phương pháp pooling trên, bằng cách sử dụng một tham số điều chỉnh có thể học được. GeM Pooling hoạt động như một bộ lọc thông minh, vừa khuếch đại các tín hiệu kích hoạt mạnh mang tính quyết định (hoạt động giống GMP), vừa không vứt bỏ hoàn toàn các điểm ảnh bối cảnh (hoạt động giống GAP).

Vector đặc trưng F_GEM sau khi nén sẽ được làm phẳng và truyền qua một mạng Multi-Layer Perceptron (MLP) bao gồm các lớp Chuẩn hóa Batch (BatchNorm1d), kích hoạt phi tuyến GELU, và Dropout (20%) trước khi dự đoán xác suất cho 28 lớp bệnh.

<aside>
💡

Chèn sơ đồ kiến trúc head phân loại

</aside>

# Các khối dung hợp

Nghiên cứu sử dụng khối dung hợp 2 nhánh đặc trưng bao gồm 2 thành phần: một khối cross attention và một khối restormer. Đặc trưng ảnh và văn bản được đưa vào khối cross attention, thu được kết quả là đặc trưng ảnh đã được thêm thông tin từ văn bản. Vector này tiếp tục đi qua khối Restormer, trở thành đặc trưng ảnh đầu ra cuối cùng sau khi được tinh chỉnh.

Giả sử $F_{img}$ là ma trận đặc trưng ảnh đầu vào, $F_{txt}$ là ma trận đặc trưng văn bản đầu vào. Qu á trình xử lý của khối dung hợp gồm những công thức sau:

$\mathbf{F}{guided} = \mathbf{F}{img} + \text{CrossAttention}(\mathbf{F}{img}, \mathbf{F}{txt})$

$\mathbf{F}{refined} = \text{RestormerBlock}(\mathbf{F}{guided})$

Với $\mathbf{F}_{guided}$ là đặc trưng ảnh sau khi đã hấp thụ thông tin từ văn bản, $\mathbf{F}_{refined}$ là đặc trưng ảnh đầu ra cuối cùng sau khi được tinh chỉnh.

# Tổng quan kiến trúc

Hình … thể hiện kiến trúc tổng thể của mô hình mạng trong nghiên cứu. Toàn bộ kiến trúc gồm 4 giai đoạn chính: trích xuất đặc trưng, chiếu về cùng một số chiều, dung hợp đa phương thức, phân loại và dự đoán.

### Giai đoạn 1: Trích xuất đặc trưng đơn phương thức

Hai nhánh dữ liệu được trích xuất độc lập bằng cách backbone ConvNeXt-CBAM cho nhánh ảnh và RoBERTa cho nhánh văn bản. Đầu ra của nhánh ảnh có 1024 kênh, có kích thước $(1024, H, W)$. Đầu ra của nhánh văn bản là các vector ngữ nghĩa đại diện cho từng token với kích thước $(L, 768)$ với L là chiều dài chuỗi văn bản.

### Giai đoạn 2: Đồng nhất số chiều

Nhằm giải quyết sự khác biệt về mặt cấu trúc của 2 nhánh dữ liệu, các lớp chiếu được sử dụng để đưa cả đặc trưng ảnh và văn bản về cùng một không gian chung, hay về cùng một số chiều là 512. 2 nhánh sử dụng lớp chiếu đặc trưng ảnh và lớp chiếu đặc trưng văn bản độc lập, và có cùng số chiều đầu ra là 512.

### Giai đoạn 3: Dung hợp đa phương thức

Sau khi đồng nhất chiều, các đặc trưng này được đưa qua chuỗi 2 khối dung hợp xếp chồng.
// ảnh

Tại đây, mạng áp dụng cơ chế text-guided: đặc trưng văn bản được giữ nguyên cấu trúc để làm mỏ neo ngữ nghĩa, liên tục định hướng và tinh chỉnh bản đồ đặc trưng hình ảnh qua từng lớp. Sự xếp chồng này giúp mô hình liên kết các mối quan hệ đa phương thức ở cấp độ trừu tượng cao hơn.

### **Giai đoạn 4: Phân loại và Dự đoán (Prediction)**

Cuối cùng, đặc trưng đa phương thức sau dung hợp được nén lại bằng lớp GeM Pooling và đưa qua bộ phân loại MLP để dự đoán phân phối xác suất trên 28 lớp đầu ra.

# Phương pháp huấn luyện

### Mất cân bằng lớp trong tập dữ liệu PlantDoc

PlantDoc là tập dữ liệu bị mất cân bằng lớp trầm trọng. Các lớp phổ biến với đặc trưng dễ nhận diện như thường đạt độ chính xác trên 90%. Ví dụ, lớp train/Potato_leaf_early_blight và train/Potato_leaf_late_blight có lần lượt 79 và 71 ảnh, có thể dễ dàng đạt kết quả cao ngay từ những phiên bản đầu tiên.

Các lớp thiểu số (có ít hơn 50 hình ảnh) hoặc có đặc trưng khó nhận diện lại có hiệu suất rất thấp. Điển hình là lớp train/Tomato_two_spotted_spider_mites_leaf chỉ có duy nhất 1 ảnh. **Việc cải thiện f1 score cần rất nhiều bước cải tiến và tinh chỉnh**.

### Focal Loss

Focal Loss là phương pháp hiệu quả để huấn luyện trên tập dữ liệu mất cân bằng lớp. Nghiên cứu này sử dụng Focal Loss với beta = 0.99. Với thiết lập này, ngưỡng bão hòa thông tin là khoảng 100 mẫu. Ví dụ lớp Tomato*leaf_yellow_virus có 196 mẫu, số mẫu hiệu quả là $E*{196} = \frac{1 - 0.99^{196}}{1 - 0.99} \approx 86.0$

Lớp Bell*pepper_leaf có 29 mẫu, số mẫu hiệu quả là số mẫu hiệu quả $E*{29} \approx 25$.

Lớp Tomato_two_spotted_spider_mites_leaf có 1 mẫu, số mẫu hiệu quả $E_1 = 1.0$. Như vậy, Focal Loss với beta = 0.99 đã giảm sự chênh lệch mẫu từ 196 lần xuống 86 lần.

Nếu dùng beta = 0.9, tập dữ liệu được giả định là bão hõa ở 10 ảnh. Như vậy lớp 196 ảnh và lớp 29 ảnh gần như giống nhau. Điều này khiến Focal Loss phản tác dụng khi lớp đa số và thiểu số lại bị phạt với cùng một trọng số như nhau.

Nếu dùng beta = 0.999, ngưỡng bão hòa 1000 ảnh. Ngưỡng này là quá muộn vì lớp lớn nhất trên tập PlantDoc chỉ có 196 ảnh. Lúc này Focal Loss quay trở lại thành Cross Entropy thông thường.

Kiến trúc trong nghiên cứu này được kết hợp từ ba thành phần hoàn toàn khác nhau, nên không thể dùng chung tốc độ học cho toàn bộ mạng.

### Differential Learning Rate

Nghiên cứu sử dụng tốc độ học khác nhau cho từng nhóm tham số của mô hình. Tốc độ học cơ sở là 1e-4, từ đó nhân hệ số cho từng nhóm tham số.

Tham số mô hình được chia làm ba nhóm.

| Nhóm                                                                                     | Tốc độ học áp dụng    | Nguyên nhân                                                                                                                                         |
| ---------------------------------------------------------------------------------------- | --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| Các lớp mới khởi tạo: **Image/Text Projection Head, Fusion Layers, Classification Head** | Tốc độ học cơ sở      | Trọng số ban đầu của chúng chỉ là rác nên cần learning rate lớn để học thật nhanh cách trích xuất và kết hợp thông tin.                             |
| Textual Backbone: RoBERTa                                                                | 1/10 tốc độ học cơ sở | RoBERTa đã được tiền huấn luyện trên hàng tỷ văn bản. Nếu dùng tốc độ học quá lớn, các trọng số cũ sẽ bị phá hỏng.                                  |
| **Visual Backbone: ConvNext**                                                            | 1/50 tốc độ học cơ sở | ConvNeXt đã trích xuất rất tốt các đặc trưng hình ảnh. Chỉ cẩn dử dụng tốc độ học cực nhỏ để tinh chỉnh thêm để tập trung vào đúng các đốm bệnh nhỏ |

### **Cosine Annealing Learning Rate**

Tốc độ học giảm dần theo hàm cosine qua các epoch. Phương pháp này đặc biệt hữu ích với tập dữ liệu phức tạp như PlantDoc. Việc bắt đầu với tốc độ học lớn giúp mô hình nhanh chóng thoát khỏi vùng tham số xấu. Khi mô hình đã học tốt hơn, việc giảm dần tốc độ học giúp mô hình tinh chỉnh nhẹ về cuối quá trình huấn luyện và hội tụ ổn định.

### Mở băng một phần backbone

Mạng ConvNext được thiết kế theo dạng phân cấp. Các tầng đầu tiên (Stage 1, Stage 2) dùng để trích xuất các đặc trưng thị giác cơ bản như cạnh, góc, màu sắc và kết cấu bề mặt. Các tầng cuối cùng (Stage 3, Stage 4) sẽ tổng hợp chúng thành các đặc trưng ngữ nghĩa cao cấp hơn (hình dáng tổn thương, cấu trúc lá). Hình ảnh trong PlantDoc được chụp ngoài tự nhiên với môi trường cực kỳ phức tạp: nhiều lá cây, đất đá, ánh sáng thay đổi phụ thuộc vào thời tiết, chất lượng camera… Nếu mở băng hoàn toàn, mô hình dễ dàng bị quên thảm họa khi tập trung vào các yếu tố nhiễu nền ngay từ những tầng đầu tiên.

Việc chỉ mở băng các lớp ở tầng cao giúp mô hình giữ lại bộ trích xuất đặc trưng cơ bản đã được tối ưu hóa trên hàng triệu ảnh ImageNet, vừa thích nghi với những đốm bệnh nhỏ trong tập PlantDoc.

RoBERTa gồm nhiều tầng Transformer xếp chồng lên nhau. Việc chỉ mở băng một vài tầng Transformer cuối cùng giúp RoBERTa giữ nguyên khả năng từ vựng, ngữ pháp và cách diễn đạt của ngôn ngữ tự nhiên, đồng thời nhanh chóng học cách biểu diễn các thuật ngữ chuyên ngành nông nghiệp và mô tả triệu chứng bệnh học từ văn bản đầu vào.
