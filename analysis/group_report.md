# Group Report — Lab 18: Production RAG

**Học viên:** Bùi Thọ An (MSSV: 2A202601883)
**Ngày thực hiện:** 18/08/2026
**Dự án:** Production RAG Pipeline (Vietnamese Enterprise Policies)

---

## Thành viên & Phân công

| Tên | Module | Hoàn thành | Tests pass |
|-----|--------|:----------:|:----------:|
| Bùi Thọ An | M1: Advanced Chunking Strategies (Semantic, Hierarchical, Structure-Aware) | ✅ | 13/13 |
| Bùi Thọ An | M2: Vietnamese Hybrid Search (BM25 + Dense Qdrant + RRF) | ✅ | 5/5 |
| Bùi Thọ An | M3: Cross-Encoder Reranking (bge-reranker-v2-m3 + Flashrank) | ✅ | 5/5 |
| Bùi Thọ An | M4: RAGAS Evaluation & Failure Analysis | ✅ | 4/4 |
| Bùi Thọ An | M5: Pre-index Enrichment Pipeline (Summarize, HyQA, Contextual, Metadata) | ✅ | 10/10 |

---

## Kết quả RAGAS

| Metric | Naive Baseline (Dense only, Basic Chunk) | Production RAG Pipeline | Δ |
|--------|:---------------------------------------:|:-----------------------:|:---:|
| **Faithfulness** | 0.7917 | 0.8000 | **+0.0083** |
| **Answer Relevancy** | 0.2390 | 0.2855 | **+0.0465** |
| **Context Precision** | 0.9250 | 0.9250 | **+0.0000** |
| **Context Recall** | 0.9250 | 0.8417 | **-0.0833** |

---

## Key Findings

1. **Biggest improvement (Cải thiện lớn nhất):**
   - **Answer Relevancy (+0.0465):** Hybrid Search và Cross-Encoder giúp câu trả lời bám câu hỏi tốt hơn baseline. Mức tăng còn khiêm tốn, cho thấy retrieval đã tốt nhưng prompt sinh câu trả lời vẫn cần tối ưu.

2. **Biggest challenge (Thách thức lớn nhất):**
   - **Xung đột tài liệu đa phiên bản (Multi-version Policy Conflict):** Khi hệ thống tồn tại cả chính sách cũ (v2023, v1.0) và chính sách mới (v2024, v2.0), retriever thuần túy dựa vào độ tương đồng vector thường ưu tiên câu từ ngắn gọn của bản cũ thay vì bản mới. Việc bảo toàn metadata `source` và `version` qua các tầng chunking và enrichment là chìa khóa để giải quyết vấn đề này.

3. **Surprise finding (Phát hiện bất ngờ):**
   - **Context Precision giữ nguyên 0.9250 nhưng Context Recall giảm 0.0833.** Reranker Top-3 lọc nhiễu tốt nhưng có thể loại mất context cần thiết cho câu hỏi multi-hop. Cần query decomposition hoặc tăng số context cho truy vấn phức hợp.

---

## Presentation Notes (5 phút)

1. **RAGAS scores (Naive vs Production):**
   - Production tăng Faithfulness từ 0.7917 lên 0.8000 và Answer Relevancy từ 0.2390 lên 0.2855.
   - Context Precision giữ ở 0.9250; Context Recall giảm xuống 0.8417. Có 3/4 metrics đạt ít nhất 0.70, tương ứng mức 10 điểm mục RAGAS theo rubric.

2. **Biggest win — Module nào, tại sao:**
   - **M2 (Hybrid Search) + M3 (Reranker):** BM25 tiếng Việt, Dense Search và Cross-Encoder duy trì Context Precision ở mức cao 0.9250. Kết quả hiện tại chưa đủ bằng chứng để tuyên bố tỷ lệ Top-1 95%.

3. **Case study — 1 failure, Error Tree walkthrough:**
   - Phân tích câu hỏi về *Nhân viên Senior 9 năm thâm niên* (Multi-hop). Nguyên nhân thất bại đến từ việc query quá dài làm loãng search vector, dẫn đến giải pháp đề xuất là **Query Decomposition**.

4. **Next optimization nếu có thêm 1 giờ:**
   - Triển khai **HyDE (Hypothetical Document Embeddings)** hoặc **Sub-query Generator** để giải quyết triệt để các câu hỏi so sánh và tính toán đa văn bản.
   - Thêm bộ lọc metadata thời gian thực (`filter={"is_latest": True}`) trước khi rank.
