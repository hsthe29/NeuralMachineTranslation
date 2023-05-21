# Các công việc đã hoàn thành:
1. Có tập dữ liệu huấn luyện lớn và đúng quy chuẩn: PhoMT (hơn 3 triệu câu)
2. Tìm hiểu về các mô hình dịch máy:
    - Rule-based Machine Translation (RBMT)
    - Statistical machine translation (SMT)
    - Neural machine translation (NMT)
      - E-D (Encoder-Decoder)
      - Transformer
3. Xây dựng và huấn luyện được mô hình dịch máy cơ bản dùng mạng RNN (cụ thể là GRU)
    - Vì số lượng câu rất lớn nên số lượng từ vựng rất lớn, nên chỉ sử dụng từ vựng gồm 6000 từ xuất hiện nhiều nhất
    - Giai đoạn huấn luyện, ước tính có 100 epochs:
      - Dùng 300.000 cặp câu để huấn luyện (training) và 30.000 cặp câu để kiếm thử (validation) sau mỗi epoch
      - Mô hình theo dõi độ chính xác trên tập kiểm thử và lưu lại mô hình tương ứng với độ chính xác lớn nhất
      - Dùng Early Stopping để ngắn quá trình huấn luyện nếu `val_masked_loss` không có sự cải thiện
    - Validation data: dùng 30000 cặp câu không nằm trong tập huấn luyện 
4. Kết quả:
    - Training:
    - Accuracy:
    - Attention: 
      - ![img.png](result/attention/attention.png)
      - ![img.png](result/attention/attention1.png)
      - ![img.png](result/attention/attention2.png)
      - ![img.png](result/attention/attention3.png)
               
5. Kết quả khi dịch một số câu:
   

# Những công việc cần làm tiếp theo

1. Thử các mô hình khác và chọn mô hình hiệu quả hơn (E-D dùng Multi-head Attention, Transformer)
2. Dịch từ Tiếng Việt -> Tiếng Anh
3. Xử lý các từ không nằm trong từ điển
4. Xử lý các câu văn bản tự nhiên trước khi đưa vào mô hình