# Day 17 Report: Memory Systems for AI Agent

Sinh viên: Đỗ Trung Tuyến (2A202602427)

Báo cáo này trình bày phần triển khai trong `src/`, kết quả benchmark và phần phân tích trade-off (Bước 8 trong `Guide.md`), cùng các phần bonus (Bước 9).

## 1. Kiến trúc đã triển khai

| Lớp memory | Baseline | Advanced | Nằm ở đâu |
|---|---|---|---|
| Short-term (trong thread) | Toàn bộ message của thread | Message gần nhất (`compact_keep_messages`) | `SessionState` / `CompactMemoryManager` |
| Persistent (`User.md`) | Không có | `state/profiles/<user>/User.md` | `UserProfileStore` |
| Compact | Không có | Summary có giới hạn (tối đa 6 bullet) cho phần lịch sử cũ | `CompactMemoryManager`, `summarize_messages()` |

Luồng một lượt của Advanced (`AdvancedAgent._reply_offline`):

```
message → extract_profile_candidates()  (theo câu, có confidence)
        → Profile.apply()               (threshold, conflict, decay) → User.md
        → CompactMemoryManager.append() (compact khi vượt ngưỡng)
        → prompt = system + User.md digest + summary + recent messages
        → trả lời (recall từ User.md) → cập nhật bộ đếm token
```

Hai agent dùng chung một bộ trả lời offline (`offline_responder.py`). Khác biệt duy nhất là **fact mà mỗi agent nhìn thấy**: Baseline chỉ đọc được các message trong thread hiện tại, còn Advanced đọc `User.md`. Nhờ vậy phép so sánh công bằng: Baseline vẫn trả lời được trong cùng thread, nhưng sang thread mới thì quên.

**Chế độ chạy:** mặc định là offline deterministic, không cần API key. Đặt `LAB_LIVE=1` cùng cấu hình provider (xem `.env.example`) để chạy live bằng LangChain `create_agent` + `InMemorySaver`. Ở chế độ live, Advanced có thêm tool `read_user_memory`/`save_user_fact`, `dynamic_prompt` để inject `User.md`, và `SummarizationMiddleware` để compact. Hỗ trợ 6 provider: `openai`, `custom`, `gemini`, `anthropic`, `ollama`, `openrouter`.

## 2. Kết quả benchmark

Lệnh chạy: `python src/benchmark.py` (offline, `compact_threshold_tokens=800`, `compact_keep_messages=4`, `profile_confidence_threshold=0.6`). Output đầy đủ nằm trong `results/benchmark_results.md`.

### Standard Benchmark (`data/conversations.json`, 10 phiên, 14 câu recall)

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---|---|---|---|---|---|
| Baseline | 2006 | 18654 | 0.00 | 0.30 | 0 | 0 |
| Advanced | 2227 | 34160 | 1.00 | 1.00 | 979 | 0 |

### Long-Context Stress Benchmark (`data/advanced_long_context.json`, 16 lượt dài, 3 câu recall)

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---|---|---|---|---|---|
| Baseline | 389 | 23017 | 0.00 | 0.30 | 0 | 0 |
| Advanced | 469 | 11412 | 1.00 | 1.00 | 546 | 4 |

Prompt tokens theo từng lượt trong stress test:

| Lượt | 1 | 4 | 5 | 8 | 11 | 14 | 16 |
|---|---|---|---|---|---|---|---|
| Baseline | 212 | 720 | 895 | 1392 | 1803 | 2283 | 2591 |
| Advanced | 291 | 837 | **513** | **604** | **550** | **570** | 878 |
| Số lần compact (cộng dồn) | 0 | 0 | 1 | 2 | 3 | 4 | 4 |

Prompt của Baseline tăng tuyến tính theo độ dài thread. Prompt của Advanced có dạng răng cưa và bị chặn dưới khoảng 900 token: mỗi lần chạm ngưỡng, phần lịch sử cũ được nén lại.

## 3. Phân tích

### 3.1. Vì sao Advanced có recall tốt hơn Baseline?

Các câu recall luôn được hỏi ở **thread mới**. Baseline chỉ có `SessionState` theo `thread_id`, nên sang thread mới nó không còn gì để đọc, recall = 0. Đây là hành vi đúng thiết kế, không phải lỗi. Test `test_cross_session_recall` kiểm chứng rằng Baseline vẫn nhớ trong cùng thread nhưng quên khi sang thread mới.

Advanced ghi các fact ổn định vào `User.md` ngay khi người dùng nói ra. File này nằm trên đĩa nên sống qua thread mới, và cả khi khởi tạo lại agent (mô phỏng restart process). Advanced trả lời đúng cả các case khó:

- **correction:** nơi ở Đà Nẵng → Huế (Standard) và Huế → Đà Nẵng (Stress); nghề backend → MLOps engineer.
- **nhiễu:** "product manager" chỉ là câu đùa, nên candidate có confidence 0.05 và bị loại. "Hà Nội chỉ là nơi đi họp" không sinh candidate nào.

### 3.2. Vì sao Advanced tốn hơn ở hội thoại ngắn?

Ở Standard Benchmark, Advanced tốn **+83% prompt tokens** và **+11% agent tokens**:

- Mỗi lượt đều phải mang theo `User.md` digest và một system prompt dài hơn. Đó là chi phí cố định cho mỗi lượt.
- Mỗi phiên chỉ khoảng 10 câu ngắn, tổng chưa tới 800 token, nên **compact không bao giờ kích hoạt** (Compactions = 0). Advanced chịu toàn bộ overhead mà chưa nhận được lợi ích nào về token.
- Agent tokens cao hơn vì Advanced báo lại những gì đã lưu ("Đã lưu vào User.md: …").

Cái giá đó mua được recall từ 0 lên 1. Sweep ở mục 3.5 cho thấy khi tắt compact (ngưỡng 4000), Advanced còn tốn hơn Baseline ngay trên stress test (25851 so với 23017). Như vậy `User.md` tự nó **không tiết kiệm token**: lớp này tối ưu cho recall, còn lớp compact mới tối ưu cho token.

### 3.3. Vì sao compact giúp Advanced có lợi thế ở hội thoại dài?

Gọi chi phí prompt của mỗi lượt là P:

- Baseline: P(n) = system + tổng n message, tức tăng O(n). Tổng qua cả hội thoại là O(n²).
- Advanced: P(n) ≤ system + `User.md` + summary (tối đa 6 bullet) + `keep` message. P bị chặn trên bởi khoảng ngưỡng compact, nên tổng chỉ tăng O(n).

Với 16 lượt dài, mức chênh đã là **−50.4% prompt tokens**. Thread càng dài thì mức tiết kiệm càng tiến gần 1 − (ngưỡng / độ dài thread).

Compact tối ưu chủ yếu **prompt tokens processed**, không phải agent tokens. Phần model sinh ra gần như giữ nguyên, thậm chí Advanced còn nhiều hơn 20%. Thứ được cắt giảm là ngữ cảnh phải gửi lại ở mỗi lượt. Đây cũng là phần chiếm phần lớn chi phí và độ trễ khi gọi API thật.

Compact cũng có cái giá của nó: summary là lossy. Summary cuối của stress test chỉ còn 6 bullet gần nhất, và các chi tiết như "X-59 đạt Mach 1.1" đã bị rơi. Recall vẫn đạt 1.0 vì các fact quan trọng đã được đẩy sang `User.md` trước khi bị nén. Hai lớp này bổ sung cho nhau: fact ổn định đi vào persistent memory, ngữ cảnh tạm thời đi vào summary có thể mất.

### 3.4. File memory tăng trưởng ra sao và có rủi ro gì?

- `User.md` tăng 979 bytes sau 10 phiên Standard và 546 bytes sau stress test. File tăng theo **số fact khác nhau**, không theo số lượt. Fact trùng chỉ tăng `seen`, correction thay thế giá trị cũ thay vì thêm dòng mới. Style là tập LRU tối đa 6 item, interests tối đa 8 item (có decay).
- Prompt chỉ nhận **digest đã làm sạch**, không có metadata comment và chỉ lấy top 4 interests. Nhờ vậy chi phí prompt của `User.md` gần như cố định (khoảng 80–100 token).

Rủi ro:

1. **Lưu sai fact:** extraction dùng heuristic/regex, có thể bỏ sót cách nói mới hoặc hiểu sai câu phức. Một fact sai trong `User.md` sẽ sống qua mọi phiên, khác với lỗi trong short-term memory vốn sẽ tự biến mất.
2. **Phình to theo thời gian:** nếu bỏ giới hạn style/interests thì digest dài dần và ăn vào phần tiết kiệm của compact.
3. **Dữ liệu cá nhân:** `User.md` là PII lưu ở dạng plain text. Production cần mã hóa, có quyền xóa ("quên tôi đi") và audit.
4. **Compact thrashing** (xem sweep bên dưới): nếu riêng `keep` message đã vượt ngưỡng thì lượt nào cũng compact.

### 3.5. Sweep tham số compact (Advanced, stress test)

| `threshold` | `keep` | Prompt tokens processed | Compactions | Recall |
|---|---|---|---|---|
| 400 | 2 | 6910 | 14 | 1.00 |
| 400 | 4 | 9027 | **27** | 1.00 |
| 800 | 2 | 10569 | 4 | 1.00 |
| 800 | 4 | 11412 | 4 | 1.00 |
| 1600 | 2 | 15708 | 1 | 1.00 |
| 4000 | 4 | 25851 | 0 | 1.00 |
| *(Baseline)* | | *23017* | *0* | *0.00* |

- Ngưỡng càng thấp thì càng rẻ, nhưng summary càng bị nén nhiều, follow-up trong thread càng dễ mất ngữ cảnh.
- Với ngưỡng 400 và keep 4: 4 message dài đã vượt 400 token, nên hệ thống compact ở gần như mọi lần append (27 lần). Production nên đặt `keep` theo token thay vì số message, hoặc thêm hysteresis (compact xuống mức thấp hơn hẳn ngưỡng).
- Recall không đổi trên mọi cấu hình vì nó phụ thuộc vào `User.md`, không phụ thuộc summary.

## 4. Bonus

| Bonus | Giải quyết vấn đề gì | Cải thiện | Rủi ro mới |
|---|---|---|---|
| **Confidence threshold**: mỗi candidate có confidence theo câu. Bị trừ điểm khi là câu đùa, giả định ("nếu", "giả sử"), "tạm thời" hoặc câu hỏi. Được cộng khi có "đính chính", "thực ra"… Chỉ ghi khi ≥ 0.6. | Lưu nhầm nhiễu: "chuyển sang product manager… chỉ là câu đùa" | Recall đúng ở câu hỏi nhiễu của stress test, `User.md` không bị bẩn | Ngưỡng cao quá sẽ bỏ sót fact thật. Trọng số hedge được chỉnh tay theo dataset |
| **Conflict handling**: fact đơn trị bị thay thế, giá trị cũ chỉ còn trong metadata `prev=`, không bao giờ vào prompt. Bỏ qua match bị phủ định ("chứ không còn ở Đà Nẵng", "không còn làm backend"). Trong một câu thì match sau thắng | Giữ đồng thời fact cũ và fact mới | Trả lời đúng Huế/MLOps (Standard) và Đà Nẵng (Stress). Có test `test_correction_replaces_old_fact` | Câu kể lại quá khứ mà không có từ phủ định có thể bị hiểu nhầm thành fact mới |
| **Memory decay**: interests có `priority = conf × 0.5^(age/40) × (1 + log2(seen))`, bị cắt khi vượt `PROFILE_MAX_INTERESTS`. Style dùng LRU | `User.md` phình to, sở thích cũ chiếm chỗ | File và prompt digest bị chặn kích thước. Có test `test_interest_decay_prunes_stale_items` | Sở thích lâu dài nhưng ít được nhắc có thể bị quên |
| **Entity extraction có cấu trúc**: `User.md` gồm các field có tên (`name`, `location`, `profession`, `favorite_drink`, `favorite_food`, `pet`, `response_style`) cùng metadata `conf/seen/turn/prev` | Markdown tự do khó cập nhật và khó so sánh | Cập nhật đúng từng field, có audit trail. Recall trả lời theo field | Field cố định nên fact ngoài schema không được lưu |
| **Không lưu khi người dùng hỏi**: bỏ qua câu kết thúc bằng "?" hoặc mở đầu bằng "nhắc lại", "bạn có biết"… | "Bạn có thể nhắc lại tên mình không?" bị hiểu thành fact | Có test `test_noise_and_questions_are_not_stored` | Câu khẳng định mở đầu bằng "nhắc lại" sẽ bị bỏ qua |

## 5. Test

`pytest src/test_agents.py -v` cho kết quả **11 passed**:

- `User.md`: read/write/edit, sanitize path (chặn `../`), Advanced ghi file thật.
- compact: trigger, summary có giới hạn, giữ nguyên message gần nhất cho follow-up.
- cross-session recall: Baseline nhớ trong thread nhưng quên ở thread mới. Advanced nhớ qua thread mới và cả khi khởi tạo lại agent.
- prompt load: Advanced compact và tổng prompt nhỏ hơn 70% của Baseline. Prompt mỗi lượt bị chặn.
- trade-off: hội thoại ngắn thì Advanced tốn prompt hơn, compactions = 0, nhưng recall cao hơn.
- correction, nhiễu/câu hỏi, decay, alias provider.

## 6. Hạn chế

- Kết quả trên là của **chế độ offline deterministic**. Extraction và trả lời là heuristic được thiết kế quanh dataset tiếng Việt của lab, nên recall 1.0 không có nghĩa là extraction đã tổng quát. Ở chế độ live, chất lượng phụ thuộc vào model và tool calling. Khi có judge model, benchmark dùng LLM-as-judge thay cho `heuristic_quality`.
- `estimate_tokens` ước lượng theo số ký tự chia 4, không dùng tokenizer thật. Con số tuyệt đối chỉ mang tính tương đối, nhưng tỉ lệ giữa hai agent vẫn có ý nghĩa.
- `heuristic_quality` khá dễ dãi: Advanced được 1.0 vì câu trả lời đủ fact, ngắn và có bullet. Thang này không đo được độ tự nhiên của câu văn.
