REPORT 3

Hướng phát triển:
- Em không phát triển mô hình theo hướng dùng Transformer
- Em tập trung tìm kiến trúc khác của mô hình Encoder-Decoder dùng RNN dựa trên mô hình cũ
- Em đã có một số thay đổi của mô hình:

Mô hình cũ | Mô hình mới
:---: | :---:
1 lớp Bidirectional LSTM ở encoder | `n` lớp Bidirectional LSTM ở encoder
`LSTM -> Multi-head Attention` | `n` khối `LSTM -> Additive Attention`
Độ dài câu không cố đinh | Độ dài câu cố định (Padding thêm vào câu để đủ độ dài)

Kết quả:
- Mô hình cũ: [REPORT 1](REPORT_1.md:31)
```
  history = model.fit(train_ds.repeat(), 
                      epochs=50, 
                      steps_per_epoch = 2500, 
                      validation_data=val_ds, 
                      callbacks=[early_stopping, checkpoint])
  ```
- In training phase:
  - Loss
![loss.png](pictures/loss.png)
  - Accuracy
![accuracy.png](pictures/accuracy.png)

## Đánh giá:
- Nhìn kết quả trên learning phase thì mô hình mới có kết quả tốt hơn mô hình cũ
- BLEU Score: Khi kết quả hoàn thiện em sẽ cập nhật sau 
## Inference
- Do máy em mới sửa nên em chưa thực hiện phần này