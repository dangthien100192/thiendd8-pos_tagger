# POS Tagging with Brown Corpus (NLTK)

## Giới thiệu
Project này thực hiện bài toán **Part-of-Speech (POS) Tagging** sử dụng **Brown Corpus** trong NLTK.

Hệ thống sử dụng 2 mô hình:
- UnigramTagger
- BigramTagger (có backoff Unigram)

Và đánh giá bằng:
- Precision
- Recall
- Macro-F1

---

## Tính năng
- Tự động download Brown corpus (1 lần)
- Train 2 POS tagger
- Đánh giá với Precision / Recall / Macro-F1
- So sánh mô hình
- Demo gán nhãn cho 1 câu cụ thể

---

## Cài đặt
pip install -r requirements.txt

## Chạy chuương trình
python main.py