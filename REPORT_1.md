# Các công việc đã hoàn thành:
1. Có tập dữ liệu huấn luyện lớn và đúng quy chuẩn: PhoMT (hơn 3 triệu câu)
2. Tìm hiểu về các mô hình dịch máy:
    - Rule-based Machine Translation (RBMT)
    - Statistical machine translation (SMT)
    - Neural machine translation (NMT)
    - Transformer
3. Xây dựng và huấn luyện được mô hình dịch máy cơ bản dùng mạng RNN (cụ thể là GRU)
    - Vì số lượng câu rất lớn nên số lượng từ vựng cũng lớn, nên chỉ sử dụng từ vựng gồm 6000 từ xuất hiện nhiều nhất
    - Giai đoạn huấn luyện, ước tính có 100 epochs:
      - Dùng 200.000 cặp câu để huấn luyện 20 epochs đầu tiên
      - Dùng 300.000 cặp câu để huấn luyện 20 epochs tiếp theo
      - Dùng 400.000 cặp câu để huấn luyện 20 epochs tiếp theo
      - dùng 500.000 cặp câu để huấn luyện 20 epochs tiếp theo
      - Dùng 600.000 cặp câu để huấn luyện 20 epochs cuối cùng
    - Validation data: dùng 2000 cặp câu không nằm trong tập huấn luyện 
4. Kết quả:
    - Training
    - Accuracy
    - Attention
5. Dịch một số câu khác:

# Những công việc cần làm tiếp theo
h