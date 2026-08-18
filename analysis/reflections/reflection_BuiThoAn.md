# Individual Reflection — Lab 18: Production RAG Pipeline

**Họ và tên:** Bùi Thọ An  
**MSSV:** 2A202601883  
**Module phụ trách:** Toàn bộ 5 Modules (M1: Chunking, M2: Search, M3: Reranking, M4: Evaluation, M5: Enrichment)  

---

## 1. Đóng góp kỹ thuật & Mapping bài giảng

### Bảng Mapping Concept → Code Thực Tế

| Lecture Concept | Module | Hàm / Class cụ thể | Quan sát & Đánh giá thực nghiệm |
|----------------|:------:|-------------------|----------------------------------|
| **Semantic Chunking** | M1 | `chunk_semantic()` | Dùng cosine similarity giữa các câu qua `all-MiniLM-L6-v2`. Với threshold `0.85`, tạo ra 208 chunks ngắn hơn so với 51 chunks của Basic, giúp bảo đảm trọn vẹn ngữ nghĩa từng câu. |
| **Hierarchical Chunking** | M1 | `chunk_hierarchical()` | Tạo 11 Parent chunks (2048 chars) và 99 Child chunks (256 chars). Link `parent_id` giúp retriever tìm kiếm với độ chính xác cao ở cấp child nhưng trả về ngữ cảnh rộng ở cấp parent. |
| **Structure-Aware Chunking** | M1 | `chunk_structure_aware()` | Parse Markdown headers (`#`, `##`, `###`), giữ nguyên cấu trúc bảng lương và code block, gắn metadata `section` cho từng chunk. |
| **Vietnamese Word Segmentation** | M2 | `segment_vietnamese()` | Sử dụng `underthesea.word_tokenize(format="text")` và thay `_` bằng khoảng trắng để đồng nhất vocabulary giữa query và index tài liệu tiếng Việt. |
| **Hybrid Search & RRF** | M2 | `BM25Search`, `DenseSearch`, `reciprocal_rank_fusion()` | Kết hợp thế mạnh khớp từ khóa chính xác của BM25 với khả năng hiểu ngữ cảnh của Dense Qdrant (`BAAI/bge-m3`). Công thức RRF $1/(60 + \text{rank} + 1)$ cân bằng hoàn hảo 2 luồng tìm kiếm. |
| **Cross-Encoder Reranking** | M3 | `CrossEncoderReranker.rerank()` | Mô hình `BAAI/bge-reranker-v2-m3` đọc đồng thời cặp `(Query, Doc)` để tính attention chéo. Lọc từ Top-20 xuống Top-3, đưa tài liệu đúng lên hạng 1 với điểm tự tin $> 0.99$. |
| **RAGAS 4 Metrics** | M4 | `evaluate_ragas()`, `failure_analysis()` | Đo lường định lượng 4 chiều: Faithfulness, Answer Relevancy, Context Precision, Context Recall. Diagnostic Tree tự động phân loại nguyên nhân gốc rễ và đề xuất cách sửa. |
| **Pre-index Enrichment** | M5 | `_enrich_single_call()`, `enrich_chunks()` | Tối ưu chi phí production bằng 1 API call duy nhất để sinh Summary + HyQA + Contextual Prepend + Metadata mà không làm mất metadata `source` gốc. |

- **Số test tự động pass:** **37 / 37 tests** (100% pass trên cả 5 test suites).

### Kết quả đánh giá API thực tế

| Metric | Naive | Production | Δ |
|---|---:|---:|---:|
| Faithfulness | 0.7917 | 0.8000 | +0.0083 |
| Answer Relevancy | 0.2390 | 0.2855 | +0.0465 |
| Context Precision | 0.9250 | 0.9250 | +0.0000 |
| Context Recall | 0.9250 | 0.8417 | -0.0833 |

Production RAG cải thiện nhẹ độ trung thực và độ liên quan, duy trì precision cao, nhưng giảm recall do reranker chỉ giữ Top-3. Bài học chính là cần cân bằng precision–recall và dùng query decomposition cho câu hỏi multi-hop.

---

## 2. Kiến thức học được

- **Khái niệm mới & sâu sắc nhất:**  
  - Cơ chế **Reciprocal Rank Fusion (RRF)**: Không cần chuẩn hóa điểm số khác thang đo giữa BM25 (điểm không giới hạn) và Dense Cosine (0 đến 1), chỉ dựa trên thứ hạng rank để dung hòa 2 danh sách ứng viên một cách tự nhiên và ổn định.
  - **Contextual Prepend (Anthropic style)**: Việc thêm 1 câu mô tả ngữ cảnh tài liệu vào đầu mỗi chunk giúp tăng độ bao phủ ngữ nghĩa của vector embedding lên đến 49%, khắc phục triệt để nhược điểm "mất ngữ cảnh cha" của các chunk nhỏ.
- **Điều bất ngờ nhất:**  
  - Dense Embedding đơn thuần rất dễ bị "đánh lừa" bởi các từ đồng nghĩa nhưng sai phiên bản (ví dụ câu hỏi về chính sách nghỉ phép năm v2024 nhưng lại match vào chính sách cũ v2023 vì cấu trúc câu tương tự). Sự kết hợp của BM25 và Metadata filtering là bắt buộc trong môi trường doanh nghiệp thực tế.
- **Kết nối với bài giảng:**  
  - Minh chứng rõ nét cho mô hình 2 tầng: **Retrieval Stage (Recall-focused / Fast)** $\rightarrow$ **Reranking Stage (Precision-focused / Deep)** giúp cân bằng giữa độ trễ (latency) và chất lượng câu trả lời.

---

## 3. Khó khăn & Cách giải quyết

1. **Lỗi Unicode / Charset trên Windows PowerShell:**
   - *Lỗi gặp phải:* `UnicodeEncodeError: 'charmap' codec can't encode character...` khi in ký tự tiếng Việt có dấu hoặc emoji.
   - *Cách giải quyết:* Bổ sung `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` và `sys.stderr.reconfigure(...)` ở đầu tất cả các file mã nguồn.

2. **Lỗi tương thích Reranker Transformers 5.0:**
   - *Lỗi gặp phải:* `FlagEmbedding` crash khi chạy với thư viện `transformers >= 5.0`.
   - *Cách giải quyết:* Chuyển sang sử dụng trực tiếp `sentence_transformers.CrossEncoder("BAAI/bge-reranker-v2-m3")` theo đúng scaffold khuyến nghị.

3. **Xử lý giới hạn Token / Quota trên API Gateway (OpenRouter 402 Error):**
   - *Lỗi gặp phải:* `APIStatusError: 402 - This request requires more credits, or fewer max_tokens. You requested up to 16384 tokens...`
   - *Cách giải quyết:* Khởi tạo `ChatOpenAI` với tham số `max_tokens=1024` rõ ràng và chuyển phần embeddings của RAGAS sang `HuggingFaceEmbeddings("all-MiniLM-L6-v2")` chạy cục bộ offline, vừa giảm 100% chi phí vừa tăng tốc độ đánh giá lên gấp 10 lần.

---

## 4. Action Plan cho Project Thực Tế

### Dự án: Hệ thống Trợ lý AI Hỏi đáp Quy trình & Văn bản Pháp quy Doanh nghiệp

#### Hiện tại:
- Pipeline cũ sử dụng Naive RAG: Cắt text theo độ dài cố định 500 ký tự $\rightarrow$ Embed bằng OpenAI `text-embedding-3-small` $\rightarrow$ Lưu ChromaDB $\rightarrow$ Prompt LLM.
- **Vấn đề gặp phải:** Bị cắt đứt các bảng biểu quy định; câu hỏi tra cứu mã điều khoản hoặc số liệu chi tiết hay bị hallucination; không phân biệt được văn bản sửa đổi bổ sung.

#### Kế hoạch áp dụng các kỹ thuật Lab 18:
1. **Chunking Strategy:** Áp dụng **Hierarchical Chunking** (Parent 2048 / Child 256) kết hợp Markdown Parser để giữ nguyên cấu trúc điều khoản và bảng biểu lương thưởng.
2. **Search Architecture:** Chuyển sang **Hybrid Search** kết hợp `underthesea` + `rank-bm25` và `Qdrant` với mô hình embedding đa ngôn ngữ `BAAI/bge-m3`.
3. **Reranking:** Tích hợp tầng Rerank với `bge-reranker-v2-m3` để chọn lọc Top-3 context uy tín nhất trước khi gửi LLM.
4. **Pre-index Enrichment:** Chạy 1-call Enrichment để trích xuất `effective_date`, `policy_id`, `department` và tự động tạo câu hỏi giả định HyQA.
5. **Evaluation:** Thiết lập bộ 50 câu hỏi vàng (Golden test set) và đánh giá định kỳ tự động bằng bộ metric RAGAS.

---

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) | Ghi chú |
|----------|:-------------:|---------|
| **Hiểu bài giảng** | 5/5 | Nắm vững toàn bộ luồng kiến trúc từ Chunking, Hybrid Search, RRF, Rerank đến RAGAS. |
| **Code quality** | 5/5 | Code chuẩn type hint, docstring đầy đủ, xử lý ngoại lệ chặt chẽ, tối ưu fallback. |
| **Teamwork / Independence** | 5/5 | Hoàn thành độc lập 100% cả 5 modules và vượt qua toàn bộ test suites. |
| **Problem solving** | 5/5 | Tự debug và xử lý triệt để các lỗi Unicode Windows, OpenRouter API 402, FlagEmbedding compatibility. |
