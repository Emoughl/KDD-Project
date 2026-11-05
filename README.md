# Trans2Vec và Hui2Vec cho phân loại giao dịch trên dữ liệu Food

## Cài đặt

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Chạy

Mở thư mục `notebooks/` trong Jupyter và chạy lần lượt:

| Notebook | Nội dung | Thời gian (ước tính) |
|---|---|---|
| `01_Data_Prep` | Đọc 2 dataset, lấy utility thật cho Food, gán nhãn Foodmart, chia train/test | < 1 phút |
| `02_Trans2Vec` | Trans2Vec trên 2 dataset, tái thực nghiệm Bảng 2 của bài Trans2Vec | 12–20 phút |
| `03_Hui2Vec` | Hui2Vec trên 2 dataset | 10–20 phút |
| `04_Comparison` | So sánh Trans2Vec với Hui2Vec, đóng góp mới, kết luận | 2–5 phút |

Bắt buộc chạy `01` trước. `02` và `03` độc lập với nhau. `04` cần kết quả của cả `02` và `03`.

Mỗi thực nghiệm lặp 10 lần như bài báo. Muốn chạy thử nhanh để kiểm tra code, đặt biến môi trường trước khi mở Jupyter (kết quả khi đó không có ý nghĩa):

```bash
KDD_N_RUNS=1 KDD_EPOCHS=2 jupyter notebook
```

## Thư mục

```
data/
  food/          food.txt + food_*_fis_*.txt   dataset của bài Trans2Vec, FI gốc từ repo tác giả
  utility/       foodmart.txt                   dataset của bài Hui2Vec (SPMF)
  foodmart_db/   sales.csv, customer.csv, ...   CSDL gốc FoodMart (repo neo4j-examples/neo4j-foodmart-dataset)
notebooks/
  utils.py       hàm dùng chung
  processed/     file sinh ra bởi notebook 01, 02, 03
  figures/       biểu đồ cho báo cáo
  results.csv    toàn bộ kết quả
papers/
```

## Cấu hình (theo bài báo và code gốc)

PV-DBOW, d = 128, `min_count = 0`, 50 epoch; Linear SVM C = 1; accuracy và F1-macro trung bình 10 lần chạy; Food chia 3.200/800 theo thứ tự cố định, Foodmart chia ngẫu nhiên 3.313/828 (seed 0); δ và minutil chọn bằng elbow method.

## Đóng góp mới

**Nhãn của dataset Food được quyết định bởi số lần khách đi mua hàng** (notebook 04, mục 2).

- Chỉ một feature là độ dài giỏ hàng (LENGTH-ONLY) đã phân loại tốt hơn mọi embedding trên Food.
- Nhóm chỉ ra cơ chế: nhãn `yes` nghĩa là "có ít nhất một lượt mua chứa hàng khuyến mãi", và mỗi lượt mua trong FoodMart có hàng khuyến mãi với xác suất gần như cố định p ≈ 0,26. Mô hình P(yes | n) = 1 − (1 − p)ⁿ, chỉ với một tham số, khớp với thực tế đến khoảng 0,02 ở mọi nhóm số lượt mua.
- Khách đi mua càng nhiều lần thì giỏ gộp càng dài, nên độ dài giỏ thay thế được cho số lượt mua.
- Kiểm chứng trên 20 cách chia train/test ngẫu nhiên: LENGTH-ONLY so với Trans2Vec-JOI.

Phần này dựa vào việc nối `food.txt` về CSDL FoodMart gốc (notebook 01). Bước nối này cũng cho utility thật của Food để chạy Hui2Vec trên Food.

## Điểm khác bài báo 
- Nhãn Foodmart: giỏ giá trị cao (tổng utility ≥ trung vị), do bài Hui2Vec không công bố nhãn (notebook 01, mục 3).
- HUI trên Food giới hạn tối đa 2 item, vì giỏ quá dài làm số HUI bùng nổ (notebook 03).
- Elbow method cụ thể hoá thành: ngưỡng đầu tiên (từ cao xuống thấp) mà số pattern tăng gấp đôi trở lên.
- Token FI/HUI được thêm tiền tố `p` (`p1, p2, …`) để không trùng tên với mã item. File FI của repo Trans2Vec đánh số FI từ 1, trùng dải số với item.
- Dùng gensim 4 thay cho gensim 3.4, nên số liệu có thể lệch nhẹ so với bài.
