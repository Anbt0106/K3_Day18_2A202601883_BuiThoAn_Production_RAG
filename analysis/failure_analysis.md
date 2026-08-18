# Failure Analysis — Lab 18: Production RAG

**Họ và tên:** Bùi Thọ An
**MSSV:** 2A202601883
**Repo:** K3_Day18_2A202601883_BuiThoAn_Production_RAG

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|:--------------:|:----------:|:---:|
| Faithfulness | 0.7917 | 0.8000 | +0.0083 |
| Answer Relevancy | 0.2390 | 0.2855 | +0.0465 |
| Context Precision | 0.9250 | 0.9250 | +0.0000 |
| Context Recall | 0.9250 | 0.8417 | -0.0833 |

*(Ghi chú: Số liệu từ hai lần chạy API thật ngày 18/08/2026 trên cùng bộ 20 câu hỏi. Naive dùng Paragraph Chunking + Dense Only; Production dùng Hierarchical Chunking + Enrichment + BM25/Dense/RRF + CrossEncoder.)*

---

## Bottom-5 Failures

### #1
- **Question:** Nhân viên được nghỉ bao nhiêu ngày phép năm?
- **Expected:** Theo chính sách hiện hành (v2024), nhân viên được nghỉ 15 ngày phép năm có lương. Chính sách cũ (v2023) là 12 ngày nhưng đã bị thay thế.
- **Got:** Nhân viên được nghỉ 12 ngày làm việc mỗi năm.
- **Worst metric:** `context_precision` / `faithfulness`
- **Error Tree:** Output sai (12 ngày thay vì 15 ngày) → Context chứa cả chunk của `nghi_phep_nam_v2023.md` (12 ngày) và `nghi_phep_nam_v2024.md` (15 ngày) → Retriever & Reranker lấy đúng keyword nhưng không ưu tiên metadata phiên bản mới nhất.
- **Root cause:** Xung đột tài liệu đa phiên bản (Multi-version conflict). Do BM25 và Dense đều match cao với cả v2023 và v2024, LLM đọc context v2023 xuất hiện trước hoặc bị nhiễu nên trả lời theo chính sách cũ.
- **Suggested fix:** Thêm metadata filtering trong `src/m2_search.py` ưu tiên `version: active` hoặc bổ sung chỉ dẫn trong System Prompt của Generator: *"Nếu có nhiều phiên bản chính sách xung đột, luôn ưu tiên phiên bản v2024/mới nhất"*.

---

### #2
- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** Theo chính sách v2024: 15 ngày cơ bản + 3 ngày thâm niên (9 ÷ 3 = 3) = 18 ngày phép. Lương Senior (P3-P4): 20-35 triệu VNĐ/tháng.
- **Got:** Nhân viên Senior có lương từ 20.000.000 - 35.000.000 VNĐ/tháng và được nghỉ 13 ngày phép năm.
- **Worst metric:** `context_recall`
- **Error Tree:** Output tính sai số ngày phép (13 thay vì 18) → Context thiếu chunk thâm niên v2024 (cộng 1 ngày cho mỗi 3 năm) do retriever chỉ retrieve được chunk bảng lương `bang_luong_2024.md` và chunk nghỉ phép cũ v2023 (cộng 1 ngày cho mỗi 5 năm: 9 năm thâm niên = +1 ngày -> 12+1 = 13).
- **Root cause:** Multi-hop query kết hợp 2 domain độc lập (Lương + Nghỉ phép). Dense & BM25 top-k bị chiếm phần lớn bởi bảng lương, làm trôi mất chunk thâm niên của chính sách nghỉ phép v2024.
- **Suggested fix:** Sử dụng Query Decomposition (tách query phức hợp thành 2 sub-queries: "Lương nhân viên Senior" và "Quy định ngày phép thâm niên 9 năm"), sau đó merge kết quả bằng RRF.

---

### #3
- **Question:** Mật khẩu phải có tối thiểu bao nhiêu ký tự?
- **Expected:** Theo chính sách hiện hành (v2.0), mật khẩu phải có tối thiểu 12 ký tự. Chính sách cũ (v1.0) yêu cầu 8 ký tự nhưng đã bị thay thế.
- **Got:** Mật khẩu phải có tối thiểu 8 ký tự.
- **Worst metric:** `context_precision`
- **Error Tree:** Output sai (8 ký tự) → Context chứa cả `mat_khau_v1.md` và `mat_khau_v2.md` → Reranker xếp `mat_khau_v1.md` cao hơn vì câu ngắn gọn, khớp từ khóa trực diện hơn.
- **Root cause:** Temporal conflict: Tài liệu v1.0 có câu trực tiếp "Mật khẩu tối thiểu 8 ký tự" trong khi v2.0 nằm trong bảng tiêu chí phức tạp.
- **Suggested fix:** Tận dụng Module 5 Metadata Enrichment: trích xuất trường `effective_date` / `version` và áp dụng pre-filter hoặc rerank score penalty cho các tài liệu deprecated.

---

### #4
- **Question:** Nhân viên được tài trợ khóa học 25 triệu, nghỉ việc sau 8 tháng hoàn thành khóa học. Phải hoàn trả bao nhiêu?
- **Expected:** Nhân viên phải cam kết làm việc ít nhất 1 năm sau khi hoàn thành khóa học. Nghỉ sau 8 tháng là trước hạn cam kết, phải hoàn trả 100% chi phí tức 25.000.000 VNĐ.
- **Got:** Không tìm thấy thông tin quy định cụ thể về việc hoàn trả khi nghỉ việc sau 8 tháng.
- **Worst metric:** `faithfulness` / `context_recall`
- **Error Tree:** LLM từ chối trả lời vì ràng buộc "Chỉ dựa trên context" quá chặt → Context có đoạn: *"Cam kết phục vụ tối thiểu 12 tháng, nếu vi phạm hoàn trả toàn bộ chi phí"* nhưng không có từ "8 tháng" → LLM không thực hiện suy luận bắc cầu (8 tháng < 12 tháng).
- **Root cause:** Reasoning gap của LLM khi prompt yêu cầu quá nghiêm ngặt *"Chỉ trả lời nếu có trong context"*, khiến LLM không dám so sánh 8 tháng < 12 tháng.
- **Suggested fix:** Cải thiện Prompt generation: *"Được phép suy luận số học cơ bản (so sánh ngày tháng, thâm niên, số tiền) dựa trên các điều kiện quy định trong context"*.

---

### #5
- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Thời hạn thanh toán là 15 ngày. Quá hạn 5 ngày, bị tính phí 2%/tháng trên 15.000.000 VNĐ = 300.000 VNĐ/tháng (tính pro-rata khoảng 50.000 VNĐ cho 5 ngày).
- **Got:** Bị tính lãi phạt quá hạn 2%/tháng đối với số tiền tạm ứng 15 triệu đồng.
- **Worst metric:** `answer_relevancy`
- **Error Tree:** Output trả lời đúng công thức nhưng thiếu số tiền phạt cụ thể (50.000 VNĐ) → Context đã lấy đủ chunk `tam_ung.md` → LLM dừng lại ở mức trích xuất điều khoản mà không tính toán ra kết quả cuối cùng.
- **Root cause:** Question asking for quantitative calculation nhưng LLM chỉ trích xuất policy text.
- **Suggested fix:** Thêm Chain-of-Thought (CoT) prompting vào Generator: *"Nếu câu hỏi yêu cầu tính toán số tiền/ngày cụ thể, hãy trình bày từng bước tính toán chi tiết dựa trên công thức trong context"*.

---

## Case Study (cho presentation)

**Question chọn phân tích:**
> *"Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?"*

**Error Tree walkthrough:**
1. **Output đúng?** $\rightarrow$ **SAI**. Trả về 13 ngày phép và khoảng lương Senior, trong khi đáp án đúng là 18 ngày phép (15 cơ bản + 3 ngày thâm niên).
2. **Context đúng?** $\rightarrow$ **THIẾU 1 PHẦN**. Context có chứa bảng lương Senior từ `bang_luong_2024.md` và chunk `nghi_phep_nam_v2023.md`, nhưng **thiếu** chunk quy định thâm niên của `nghi_phep_nam_v2024.md`.
3. **Retrieval/Reranking có lỗi gì?** $\rightarrow$ Do query chứa cả "Senior", "lương", "9 năm thâm niên", "ngày phép năm", dense search ưu tiên các chunk có mật độ từ khóa lương cao, đẩy chunk thâm niên phép năm ra ngoài Top-20.
4. **Fix ở bước nào:**
   - **M2 (Search):** Áp dụng Sub-query decomposition hoặc Hybrid search với trọng số cân bằng giữa các entity.
   - **M5 (Enrichment):** Contextual Prepend bổ sung rõ topic "Chính sách nghỉ phép thâm niên v2024" giúp Dense search match đúng hơn.

**Nếu có thêm 1 giờ, sẽ optimize:**
1. **Query Rewriting & Decomposition:** Tách câu hỏi đa ý thành các sub-queries đơn lẻ trước khi gửi vào Hybrid Retriever.
2. **Metadata Filtering (Version/Status):** Thêm bộ lọc metadata tự động loại bỏ các tài liệu đã hết hiệu lực (`v2023`, `v1.0`) khi đã có phiên bản thay thế (`v2024`, `v2.0`).
