# Depthwise separable convolution

**Depthwise Separable Convolution** là một kỹ thuật giúp **cắt giảm đáng kể số lượng tham số và khối lượng tính toán** của mạng tích chập. Ý tưởng chính là thay vì làm phép tích chập chuẩn trong một bước, ta tách nó ra thành hai bước nhỏ hơn: Depthwise Convolution và Pointwise Convolution.

// hình ảnh

Hình … minh họa phép tích chập chuẩn. Có thể thấy, cách làm này tiêu tốn khá nhiều phép tính. Cụ thể, với đầu vào kích thước Hf × Wf × D, kernel kích thước Hk × Wk × D và N kênh đầu ra, tổng số phép tính cần thực hiện sẽ là Hf × Wf × Hk × Wk × D × N, còn số tham số là Hf × Wf × Hk × Wk × D + Hf' × Wf' × N. Con số này tăng lên rất nhanh khi mạng sâu hoặc số kênh lớn.

// hình ảnh depthwise conv

Depthwise Convolution (Hình 2.2) xử lý vấn đề trên theo một hướng khác. Thay vì dùng một bộ lọc trải đều trên toàn bộ các kênh đầu vào, mỗi kênh sẽ có riêng một bộ lọc kích thước Hk × Wk. Nói cách khác, nếu đầu vào có D kênh thì cũng sẽ có D bộ lọc làm việc song song và hoàn toàn độc lập với nhau, không trộn lẫn thông tin giữa các kênh. Kết quả thu được vẫn giữ nguyên D kênh như ban đầu.

// hình ảnh pointwise conv

Sau đó, Pointwise Convolution (Hình …), hay phép tích chập 1 × 1, ghép thông tin giữa các kênh lại với nhau. Tại mỗi vị trí trên ảnh, bộ lọc 1 × 1 sẽ lấy giá trị từ toàn bộ các kênh đầu vào rồi tổ hợp tuyến tính chúng, nhờ vậy mạng có thể học được mối liên hệ giữa các đặc trưng mà bước Depthwise trước đó bỏ qua.

Tổng chi phí tính toán của Depthwise + Pointwise chỉ:

Hf′⋅Wf′⋅D⋅(Hk⋅Wk+N)H'\_f \cdot W'\_f \cdot D \cdot (H_k \cdot W_k + N)Hf′⋅Wf′⋅D⋅(Hk⋅Wk+N)

Con số này nhỏ hơn rất nhiều so với tích chập chuẩn, và kích thước bộ lọc (Hk, Wk) càng lớn thì chênh lệch càng rõ.

Nhờ ưu thế về số phép tính lẫn số tham số, Depthwise Separable Convolution thường được sử dụng trong nhiều kiến trúc như Mobile, IoT, FPGA. **HEMNet: A Hardware-Efficient MobileNet for Gastrointestinal Pathological Findings Classification** Do yêu cầu khối lượng tính toán thấp hơn nhiều nên thường được sử dụng để tối ưu các mô hình trên thiết bị di động và thiết bị nhúng. Không dừng lại ở đó, các biến thể CNN gần đây như EfficientNet hay ConvNeXt cũng tận dụng kỹ thuật này để cân bằng giữa độ chính xác và tốc độ, đặc biệt trong những bài toán thị giác máy tính đòi hỏi tính toán trong thời gian thực

# Layer Normalization

**Layer Normalization (LayerNorm)** là một kỹ thuật chuẩn hóa trong mạng neural sâu, giúp ổn định phân phối của các kích hoạt trong quá trình huấn luyện và cải thiện tốc độ hội tụ, độ ổn định và khả năng tổng quát hóa của mô hình. Understanding and Improving Layer Normalization

!image.png

Ảnh: BatchNorm

Nếu ở chế độ huấn luyện, Batch Norm tính toán trung bình và phương sai cho từng kênh C độc lập trên toàn bộ batch:

$\mu_c = \frac{1}{N \times H \times W} \sum_{n=1}^{N} \sum_{h=1}^{H} \sum_{w=1}^{W} x_{n, c, h, w}$

Sau đó dữ liệu được chuẩn hóa theo công thức $\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$. Điều này khiến Batch Norm phụ thuộc vào Batch Size, và không phù hợp với Batch Size nhỏ vì tính không ổn định thống kê của nó.

!image.png

Ảnh: Layer Norm

Thay vì tính trên toàn bộ batch, Layer Normalization tính trung bình và phương sai cho từng ảnh độc lập tại *t*ừng vị trí pixel (H, W).

$\mu_{n, h, w} = \frac{1}{C} \sum_{c=1}^{C} x_{n, c, h, w}$

Công thức chuẩn hóa dữ liệu giống với Batch Norm: $\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$

Phương pháp tiếp cận này giúp việc chuẩn hóa dữ liệu không phụ thuộc vào batch size. Trong những trường hợp batch size nhỏ hay batch size thay đổi, phương pháp vẫn hoạt động linh hoạt và phù hợp. LayerNorm cũng được thiết kế tương thích với optimizer **AdamW** và các kỹ thuật training hiện đại. Pre-RMSNorm and Pre-CRMSNorm Transformers:
Equivalent and Efficient Pre-LN Transformers

# Inverted bottleneck

Inverted Bottleneck là một kĩ thuật thiết kế khối tích chập để tạo cho mô hình khả năng học các đặc trưng phong phú ở lớp giữa và tối ưu chi phí tính toán.

Bottleneck truyền thống thường thường nén số chiều thông tin ở lớp giữa, trong khi đó inverted bottle mở rộng số chiều lên một hệ số lớn, ví dụ gấp 6 lần ở các mô hình MobileNetV2 Memory Efficient 3D U-Net with Reversible Mobile Inverted Bottlenecks for Brain Tumor Segmentation. EfficientNet kế thừa trực tiếp khối inverted bottleneck từ MobileNetV2 và cũng sử dụng hệ số mở rộng **6** làm giá trị mặc định. **Practical Analysis on Architecture of EfficientNet**

Ảnh đầu vào khi đi vào khối inverted bottleneck được đưa vào một lớp tích chập 1x1 để tăng số lượng kênh lên gấp 6 lần, sau đó đi qua lớp tích chập depthwise 3x3 không làm thay đổi số kênh, cuối cùng là một lớp tích chập 1x1 để đưa số kênh về lại ban đầu.

!image.png

Ảnh: Khối inverted bottleneck của MobilenetV2

# **Cosine Annealing Learning Rate**

**Cosine annealing learning rate** là một chiến lược điều chỉnh learning rate trong quá trình huấn luyện mô hình deep learning, trong đó learning rate giảm dần theo hàm cosine, từ giá trị thiết lập ban đầu đến xấp xỉ 0.

Công thức:

$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right)$

trong đó:

- $\eta_t$: learning rate tại bước t
- $T_{cur}$: số bước đã thực hiện trong chu kỳ hiện tại
- $T_{max}$: tổng số bước trong một chu kỳ **Enhancing Gastrointestinal Disease Diagnosis Using Fine-Tuned MobileNetV2**

Việc giảm learning rate mượt mà theo hàm cosine giúp mô hình hội tụ tốt hơn. Kĩ thuật này được dùng phổ biến trong huấn luyện các mô hình như MobileNetV2, YOLO, Transformer để cải thiện độ chính xác và khả năng tổng quát hóa, đồng thời tránh overfitting. **Enhancing Gastrointestinal Disease Diagnosis Using Fine-Tuned MobileNetV2** **Advanced Training Algorithms in Sigma-Delta Spiking YOLO for Energy-Efficient Object Detection on Neuromorphic Hardware**

# ConvNext

ConNext là một mạng tích chập hiện đại hóa, được cải tiến từ kiến trúc ResNet truyền thống theo hướng tiệm cận của mô hình Transformer, nhưng vẫn giữ nguyên bản chất là một mạng tích chập. **A ConvNet for the 2020s**

Trước đó, các mô hình transformer phân cấp sử dụng kiến trúc lai, tận dụng kĩ thuật cửa sổ trượt của CNN (như Swin Transformer) đã vượt trội hơn CNN tiêu chuẩn trên nhiều nhiệm vụ thị giác máy tính. **A ConvNet for the 2020s**

Dù hoàn toàn dựa trên mạng tích chập, ConvNext được xây dựng lại từ đầu với nhiều cải tiến đáng kể nhằm bắt kịp các kiến trúc Transformer về độ chính xác, khả năng mở rộng và độ bền. **A ConvNet for the 2020s** Các cải tiến đáng chú ý bao gồm **Depthwise convolution thay thế convolution tiêu chuẩn, Inverted bottleneck, Layer Normalization, hàm kích hoạt GELU cùng một loạt các kĩ thuật huấn luyện hiện đại như Cosine Annealing Learning Rate Schedule hay Stochastic Depth.**

!image.png

Ảnh: Sơ đồ kiến trúc ConvNeXt

ConvNext có cấu trúc phân cấp gồm 4 tầng, mỗi tầng gồm số lượng khối ConvNext nhất định.

Mô hình sử dụng số khối ConvNeXt lần lượt là (3, 3, 27, 3), tổng cộng 36 khối.

Ở giai đoạn stem, ConvNeXt nhận đầu vào là ảnh kích thước H x W x 3. Tại đây, ảnh được đưa qua một lớp tích chập **kích thước 4x4 với stride bằng 4, không chồng lấn, được gọi** là **Patchify Stem - phương pháp tích chập mô phỏng cách ViT chia ảnh thành các patch riêng biệt. Qua lớp tích chập này, độ phân giải ảnh giảm xuống H/4, W/4, số kênh tăng lên 96. Sau lớp tích chập là một lớp Layer Normalization chuẩn hóa dữ liệu theo chiều kênh, đảm bảo sự ổn định cho quá trình huấn luyện. Các stage 1, 2, 3, 4 có số kênh đầu vào và đầu ra bằng nhau, lần lượt là 96, 192, 384, 768. Do đó, ngoại trừ stage 1 do số kênh đã được stem stage giảm trước đó, các stage còn lại đều có lớp downsample đứng trước để chỉnh số kênh tương thích với đầu vào của stage.**

Khối ConvNext là trái tim của kiến trúc, được thiết kế lại hoàn toàn so với khối ResNet.

// ảnh convnext block

Mỗi khối đều được thiết kế theo cấu trúc Inverted Bottleneck, gồm 6 thành phần: trích xuất đặc trưng (tích chập depthwise 7x7), chuẩn hóa dữ liệu (permute x: N, C, H, W → N, H, W, C, Layer Normalization), mở rộng chiều kênh (Pointwise convolution), kích hoạt GELU, nén chiều kênh (Pointwise convolution), điều chuẩn và kết nối tắt (Layer Scale, permute x: N, H, W, C → N, C, H, W, Drop Path). Drop Path là kĩ thuật Stochastic Depth, đóng vai trò điều chuẩn (regularization) bằng cách thỉnh thoảng tắt toàn bộ nhánh mạng này để chống overfitting.

Inverted bottleneck mở rộng số kênh đầu vào lên 4C, cho phép mô hình học các đặc trưng phong phú hơn.

Tích chập 3x3 thông thường được thay thế bằng tích chập depthwise kích thước lớn 7x7. Bằng cách này, ConvNeXt giảm thiểu đáng kể chi phí tính toán, đồng thời tận dụng các **kết nối tắt** để bảo toàn dữ liệu gốc và tối ưu hóa quá trình lan truyền gradient. Kiến trúc này mang lại sự cân bằng vượt trội giữa độ chính xác và hiệu năng vận hành, đặc biệt hiệu quả trong các bài toán phức tạp như nhận diện bệnh lý thực vật - nơi việc phân tích triệu chứng đòi hỏi sự chi tiết và đa dạng cao về đặc trưng. Ngoài ra, việc tích hợp cấu trúc **inverted bottleneck** với kích thước kernel mở rộng (lên đến 7×7) giúp ConvNeXt nới rộng trường tiếp nhận (receptive field). Nhờ vậy, mô hình có thể khai thác thông tin không gian tốt hơn nhiều so với giới hạn kernel 3×3 của kiến trúc ResNet.

ConvNeXt cũng sử dụng Layer Normalization thay cho Batch Normalization, đặt ở ngay sau lớp tích chập depthwise 7x7 trong mỗi khối ConvNeXt. **Hybrid Deep Learning with ConvNeXt and Self-Attention for Pulmonary Disease Classification** Các transformer như ViT, BERT, GPT luôn dùng LayerNorm vì không phụ thuộc vào batch size và tương thích với optimizer **AdamW** và các kỹ thuật training hiện đại. Pre-RMSNorm and Pre-CRMSNorm Transformers:
Equivalent and Efficient Pre-LN Transformers ConvNeXt được cải tiến theo nhiều đặc điểm của transformer nên cũng dùng LayerNorm thay cho BatchNorm. LayerNorm giúp ConvNeXt đạt **accuracy cao hơn** và **training ổn định hơn** so với dùng BatchNorm. **A ConvNet for the 2020s**

ConvNeXt sử dụng hàm kích hoạt tối ưu GELU thay cho ReLU. So với ReLU, GELU có các ưu thế như khả vi ở mọi nơi **RCR-AF: Enhancing Model Generalization via Rademacher Complexity Reduction Activation Function**, giữ thông tin âm, giúp giảm hiện tượng dead neurons - hiện tượng thường gặp khi sử dụng hàm kích hoạt ReLU. **Spoken-Intent Classification using Hybrid Activation Function** **RCR-AF: Enhancing Model Generalization via Rademacher Complexity Reduction Activation Function**. Hơn nữa, GELU được thiết kế để tương thích với các mô hình transformers, là hàm kích hoạt mặc định trong BERT, GPT, ViT. Do chi phí tính toán của GELU cao hơn ReLU, ConvNeXt chỉ sử dụng duy nhất một hàm GELU trong mỗi khối ConNeXt. **A ConvNet for the 2020s**

ConvNeXt cũng được áp dụng các kĩ thuật huấn luyện hiện đại từ Swin Transformer như **Stochastic Depth bỏ ngẫu nhiên một số lớp trong quá trình huấn luyện, Cosine annealing learning rate - giảm dần learning rate theo hàm cosine giúp khám phá tốt hơn ở giai đoạn đầu và hội tụ tốt hơn ở giai đoạn cuối, hay Label Smoothing thay vì huấn luyện mô hình để dự đoán gần 1.0** cho lớp đúng và **gần 0.0** cho lớp sai thì phân phối lại một phần xác suất từ lớp đúng sang lớp sai.

ConvNeXt là minh chứng cho việc một mạng tích chập vẫn đủ sức cạnh tranh với Transformer nếu được thiết kế hợp lý. Kiến trúc này giữ được thế mạnh vốn có của CNN truyền thống, đồng thời học hỏi nhiều ý tưởng hay từ Transformer hiện đại, nhờ đó vừa duy trì chi phí tính toán tối ưu, vừa đạt hiệu năng ấn tượng trong các bài toán thị giác máy tính hiện đại.

# **Multi-Dconv Head Transposed Attention (**MDTA)

Multi-Dconv Head Transposed Attention (MDTA) là một khối cơ chế attention được cải tiến, thường sử dụng cho các mô hình transformers trong các tác vụ xử lý hình ảnh. MDTA được sử dụng làm khối chính trong mô hình **Restormer** (một mô hình phục hồi hình ảnh đạt hiệu suất cao tại CVPR 2022).**Restormer: Efficient Transformer for High-Resolution Image Restoration**

Giả sử ảnh đầu vào có kích thước H x W x C, self-attention có độ phức tạp tính toán là $\mathcal{O}(H^2 W^2 C)$. Chỉ cần tăng gấp đôi kích thước ảnh, số phép nhân đã tăng khoảng 16 lần. phép nhân. Do đó rất khó áp dụng self-attention cho ảnh có độ phân giải cao. MDTA được thiết kế để khắc phục hạn chế này.

Thay vì tính attention theo chiều không gian H x W, MDTA tính theo chiều kênh C, tức là tạo ra ma trận attention có kích thước C x C thay vì H x W, giúp giảm đáng kể chi phí tính toán.

// ảnh mdta

Trước hết, MDTA tạo ra ma trận Q, K, V bằng cách đưa ảnh đầu vào lần lượt đi qua lớp tích chập 1x1 và tích chập depwise 3x3. Đầu ra được chia thành 3 phần bằng nhau, thu được Q, K, V với kích thước $C \times H \times W$. Các vector Q, K, V này tiếp tục được chia nhỏ thành kích thước $C_h \times H \times W$ để sử dụng cho các head, với $C_h = C / head_{num}$. Trong self attention tiêu chuẩn, kích thước Q, K, V trong 1 head là $H \times W \times C_h$, tức là số pixel nằm ở chiều dọc, còn trong MDTA, số pixel được đẩy sang chiều ngang.

Q và K sau đó được chuẩn hóa theo chiều không gian, tức là chiều $H \times W$, theo công thức $Q = \frac{Q}{\|Q\|_2}$ và $K = \frac{K}{\|K\|_2}$. Q, K đã chuẩn hóa được đưa vào tính toán ma trận attention theo công thức $Attention = \text{Softmax}\left(Q \cdot K^T \cdot \tau \right)$, trong đó $\tau$ (temperature) là một tham số có thể học được, dùng để điều chỉnh độ sắc nét của phân phối Softmax trước khi đưa qua hàm Softmax. Ma trận attention có kích thước $C_h \times C_h$, được nhân với ma trận $V$ $(C_h \times H \times W)$ để thu được ma trận đầu ra kích thước $C_h \times H \times W$. Ma trận đầu ra của các head được nối lại với nhau, sau đó đi qua một lớp Conv 1x1 để trộn thông tin giữa các head.

Với ảnh đầu vào kích thước H x W x C, MDTA chỉ có độ phức tạp tính toán tuyến tính $\mathcal{O}(HW C^2)$. Điều này cho phép mạng xử lý được những bức ảnh có độ phân giải siêu cao (4K).

# Gated-Dconv Feed-Forward Network (GDFN)

GDFN là một module được sử dụng trong mạng Restormer, thay thế cho Feed-Forward Network truyền thống trong Transformers để cải thiện khả năng phục hồi chi tiết ảnh. **Pureformer: Transformer-Based Image Denoising**. GDFN gồm hai thành phần chính là cơ chế gated và tích chập depthwise.

Trong GDFN, dữ liệu được tăng số chiều để học được nhiều đặc trưng phong phú hơn. Số chiều của dữ liệu được nhân với một hệ số gọi là expansion factor. Expansion factor được sử dụng trong GDFN là 2.66.

Dữ liệu đầu vào được đưa vào lớp tích chập để tăng số kênh đầu vào $C$ lên $C \times expansion factor \times 2$. Lý do phải nhân 2 là vì dữ liệu sẽ được chia làm 2 nhánh cho cơ chế Gating ở bước sau. Theo sau đó là một lớp tích chập Depthwise 3x3, cho phép mô hình học thêm đặc trưng cục bộ xung quanh pixel. Số kênh đi qua lớp này vẫn giữ nguyên là $C \times expansion factor \times 2$, sau đó được chia thành 2 nhánh, mỗi nhánh có $C \times expansion factor$ kênh. 2 nhánh được được vào cơ chế gating có công thức $Y = \text{GELU}(X_1) \odot X_2$. Nhánh $X_1$ đi qua hàm GELU, tạo ra một ma trận trọng số chứa các giá trị từ 0 đến 1. Việc nhân từng phần tử của ma trận này với nhánh $X_2$ giúp giữ lại những đặc trưng quan trọng của $X_2$, triệt tiêu những thông tin không cần thiết. Lớp tích chập 1x1 cuối cùng đưa số kênh trở lại như ban đầu.

Nhờ lớp tích chập Depthwise 3x3, GDFN rất mạnh trong trích xuất các đặc trưng cục bộ, có thể kết hợp với tranformers vốn có khả năng trích xuất đặc trung toàn cục tốt nhờ vào attention. Cơ chế Gating giúp mô hình linh hoạt hơn trong việc lựa chọn thông tin để học.

# Restormer block

Khối Restormer là thành phần cốt lõi của mô hình Restormer, đóng vai trò tương tự như một khối Transformer trong các mô hình Vision Transformers, kết hợp 2 khối là MDTA và GDFN để học đặc trưng toàn cục và đặc trưng cục bộ. Trong nghiên cứu này, khối Restormer được sử dụng làm thành phần chính trong khối dung hợp.

Khối Restormer là sự kết hợp của 2 thành phần đã được giới thiệu ở trên - MDTA để học các mối liên hệ toàn cục theo chiều kênh và GDFN để học các đặc trưng cục bộ theo chiều không gian. Khối Restormer nhận đầu vào là vector đặc trưng đã dung hợp thu được từ bước trước, hay $\mathbf{F}{refined}$ (gọi ngắn gọn là $X$) lần lượt đi qua khối MDTA và GDFN. Công thức toán học của khối Restormer bao gồm hai giai đoạn:
$\mathbf{X} = \mathbf{X} + \text{MDTA}\big(\text{LN}(\mathbf{X})\big)$

$\mathbf{X''} = \mathbf{X'} + \text{GDFN}\big(\text{LN}(\mathbf{X'})\big)$

$\mathbf{X''}$ là đầu ra cuối cùng của khối dung hợp.

# Focal Loss

Focal Loss là một loại **loss trong machine learning**. Focal loss dùng để giảm độ quan trọng cho những dữ liệu thuật toán đã học tốt rồi, và **tập trung học những dữ liệu khó học hơn**.

Gọi $y \in \{0, 1\}$ là nhãn thực tế và $p \in [0, 1]$ là xác suất dự đoán của mô hình cho lớp $y = 1$. Ta định nghĩa $p_t$ như sau: $p_t = \begin{cases} p & \text{nếu } y = 1 \\ 1 - p & \text{ngược lại} \end{cases}$

Từ định nghĩa $p_t$, công thức của Cross Entropy Loss được viết gọn lại thành: $CE(p_t) = -\log(p_t)$.

So với Cross Entropy Loss, Focal Loss có thêm thành phần điều chỉnh là $-(1 - p_t)^\gamma$. Công thức như sau:

$FL(p_t) = -(1 - p_t)^\gamma \log(p_t)$

Thành phần này điều chỉnh giá trị loss của mẫu dựa vào độ khó học của lớp đó. Nếu một mẫu quá dễ đoán (ví dụ pt = 0.95), thì (1 - 0.95) tiến gần về 0, làm loss triệt tiêu, mô hình không tốn công học những thứ nó đã biết rõ nữa. Nếu một mẫu rất khó đoán (pt nhỏ, ví dụ 0.1), thì (1 - 0.1) tiến gần về 1, giữ nguyên loss ở mức cao ép AI phải học. $\gamma$ tham số hội tụ, thường dao động từ 0 đến 5 do người dùng điều chỉnh. $\gamma$ càng lớn, mức độ phớt lờ các mẫu dễ đoán càng mạnh. $\gamma = 0$ thì Focal Loss sẽ trở thành Cross Entropy Loss thông thường.
Focal Loss còn một biến thể khác là Class Balanced Focal Loss, được phát triển để giải quyết đồng thời cả hai vấn đề: mẫu dễ/khó (bằng $\gamma$) và mất cân bằng số lượng giữa các lớp (bằng $\alpha$).
$FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$

$\alpha_t$ là trọng số lớp, được tính bằng nghịch đảo của số mẫu hiệu dụng: $\frac{1}{E_n}$. Số mẫu hiệu dụng được tính bằng công thức:

$E_n = \frac{1 - \beta^n}{1 - \beta}$

Trong đó:

n: Số lượng mẫu thực tế của lớp đó ($n > 0$).

$\beta$: Tham số đại diện cho mức độ trùng lặp của các mẫu trong một lớp, nằm trong khoảng $[0, 1)$.

Khi n đủ lớn, số mẫu hiệu dụng sẽ tiệm cận một giá trị cụ thể. Khi đó lớp đạt trạng thái bão hòa, có thêm mẫu mới vào lớp thì số mẫu hiệu dụng cũng không tăng nữa. $\beta$ càng gần 1 thì lớp bão hòa càng nhanh. Do đó, $\beta$ còn được xem là tham số điều chỉnh tốc độ bão hòa thông tin.

Lớp có số mẫu hiệu dụng càng lớn thì trọng số lớp $\alpha_t$ càng nhỏ, do đó giá trị loss của lớp càng nhỏ, tức là lớp chứa nhiều thông tin nên mô hình sẽ không tập trung học nữa. Không dùng trực tiếp $\frac{1}{N}$ vì không phải mọi mẫu đều mang lại thông tin hữu ích.

# Automated image description generation (AIDG)

Phương pháp này dùng để tự động tạo dữ liệu văn bản cho tập dữ liệu chỉ chứa dữ liệu hình ảnh. Từng hình ảnh được đưa vào LLMs để tạo mô tả bằng ngôn ngữ tự nhiên. LLMs có thể sinh mô tả phong phú, tự nhiên, bám ngữ cảnh tốt. Toàn bộ quy trình vận hành hoàn toàn tự động, không cần dữ liệu huấn luyện, không cần sự giám sát của con người. Mô tả được tạo ra sẽ đượcbiên soạn thành cặp hình ảnh - văn bản, lưu trữ dưới dạng json. So với phương pháp image captionin thủ công, phương pháp này đạt hiệu suất cao hơn đáng kể. Phương pháp này giúp giảm cả thời gian lẫn chi phí chuẩn bị dữ liệu bằng cách loại bỏ việc mô tả thủ công.

Việc tự động sinh ra tập dữ liệu mới gồm các cặp hình ảnh - văn bản từ tập dữ liệu gốc chỉ chứa hình ảnh tạo nền tảng vững chắc cho việc sử dụng các mô hình đa phương thức trong phân loại hình ảnh. Phương pháp này có thể sử dụng để phát triển các bộ dữ liệu đa phương thức chất lượng cho nhiều lĩnh vực một cách đơn giản và nhanh chóng, giúp mở rộng phạm vi ứng dụng của các mô hình đa phương thức.

# **Zero-shot Chain of Thought**

Chain of Thought (CoT) Prompting là một kỹ thuật thiết kế prompt giúp LLMs suy luận tốt hơn trong các tác vụ phức tạp như giải toán, tư duy logic hay phân tích tình huống. Thay vì yêu cầu trả lời trực tiếp, kỹ thuật này hướng dẫn mô hình trình bày từng bước suy nghĩ, giống như việc con người diễn đạt suy nghĩ thành lời. Ví dụ, thêm chỉ dẫn như “giải thích từng bước” vào câu hỏi để hướng dẫn mô hình diễn giải quá trình lập luận.

Phương pháp này hiệu quả vì nó mô phỏng quá trình lập luận giống con người bằng cách chia nhỏ các vấn đề phức tạp thành các bước trung gian dễ quản lý, dẫn đến câu trả lời cuối cùng một cách tuần tự.

Zero-shot Chain of Thought là một biến thể của Zero-shot Chain of Thought. \*\*\*\*Biến thể này cho phép mô hình tự suy luận mà không cần ví dụ mẫu hay huấn luyện bổ sung. Mô hình tận dụng kiến thức có sẵn để phân tích và đưa ra các bước suy luận logic. Zero-shot CoT đặc biệt hữu ích trong những tình huống mà dữ liệu huấn luyện chuyên biệt không có sẵn.

# Generalized Mean Pooling

**Generalized Mean Pooling (GeM)** là một phương pháp pooling tổng quát hóa, kết hợp ưu điểm của cả Global Average Pooling (GAP) và Global Max Pooling (GMP). GAP lấy giá trị trung bình trên toàn bộ cửa sổ, giúp mô hình nắm bắt được cấu trúc tổng thể và bối cảnh (global context), nhưng làm suy yếu đi các giá trị kích hoạt cực đại, từ đó làm lu mờ các đốm bệnh vi mô. Ngược lại, GMP chỉ trích xuất giá trị lớn nhất, giúp giữ lại các đốm bệnh, nhưng lại làm mất hoàn toàn bối cảnh tổng quan của chiếc lá. Để khắc phụ hạn chế của 2 phương pháp này, GeM cho phép điều chỉnh mức độ "nhấn mạnh" vào các đặc trưng nổi bật thông qua một tham số có thể học được. **Fine-Grained Image Classification Based on Attention Mechanism and GeM Feature Fusion**

$f = \left( \frac{1}{H \times W} \sum_{i=1}^{H \times W} x_i^p \right)^{\frac{1}{p}}$

Trong đó $x_i$ là giá trị pixel chứ i trên feature map, H x W là kích thước không gian cảu feature map và p là tham số điều chỉnh có thể học được. Tính linh hoạt của GeM thể hiện ở: Khi $p \to 1$, GeM tiệm cận trở thành GAP; và khi $p \to \infty$, GeM hội tụ về GMP.
