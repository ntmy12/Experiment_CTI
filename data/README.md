# COCO 2014 Dataset Setup

Thí nghiệm 1 cần dữ liệu COCO 2014 validation:
1. Thư mục ảnh: `val2014/` (chứa các file dạng `COCO_val2014_000000xxxxxx.jpg`)
2. File instances annotations: `instances_val2014.json`
3. File captions annotations: `captions_val2014.json`
4. File synonyms: `synonyms.txt` (đã có sẵn trong thư mục này)

### Tải dữ liệu chính thức:
```bash
# Tải annotations:
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip annotations_trainval2014.zip
# Di chuyển instances_val2014.json và captions_val2014.json vào data/

# Tải ảnh val2014 (khoảng 6.2GB):
wget http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip -d data/
```

### Trên Kaggle:
Notebook `notebooks/run_kaggle.ipynb` có sẵn ô lệnh tự động tải annotations hoặc mount từ Kaggle Dataset có sẵn.
