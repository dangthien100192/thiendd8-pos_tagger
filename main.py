import os
import nltk
from nltk.corpus import brown
from nltk.tag import UnigramTagger, BigramTagger
from sklearn.metrics import precision_recall_fscore_support


# =========================================================
# Cấu hình thư mục lưu NLTK data trong thư mục gốc project
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NLTK_DATA_DIR = os.path.join(BASE_DIR, "nltk_data")

os.makedirs(NLTK_DATA_DIR, exist_ok=True)

if NLTK_DATA_DIR not in nltk.data.path:
    nltk.data.path.insert(0, NLTK_DATA_DIR)


# =========================================================
# Download dataset 1 lần, nếu đã có thì bỏ qua
# =========================================================
def ensure_nltk_resources():
    resources = [
        ("corpora/brown.zip", "brown"),
        ("taggers/universal_tagset.zip", "universal_tagset"),
    ]

    for resource_path, resource_name in resources:
        try:
            nltk.data.find(resource_path)
            print(f"[OK] {resource_name} đã tồn tại.")
        except LookupError:
            print(f"[DOWNLOAD] Đang tải {resource_name} vào {NLTK_DATA_DIR} ...")
            nltk.download(resource_name, download_dir=NLTK_DATA_DIR, quiet=False)

    print("[INFO] Hoàn tất kiểm tra dữ liệu.\n")


# =========================================================
# Load Brown corpus với universal tagset
# =========================================================
def load_dataset():
    tagged_sents = brown.tagged_sents(tagset="universal")
    return tagged_sents


# =========================================================
# Chia train/test
# =========================================================
def split_dataset(tagged_sents, train_ratio=0.9):
    split_idx = int(len(tagged_sents) * train_ratio)
    train_sents = tagged_sents[:split_idx]
    test_sents = tagged_sents[split_idx:]
    return train_sents, test_sents


# =========================================================
# Train 2 POS tagger
# =========================================================
def train_taggers(train_sents):
    unigram_tagger = UnigramTagger(train_sents)
    bigram_tagger = BigramTagger(train_sents, backoff=unigram_tagger)
    return unigram_tagger, bigram_tagger


# =========================================================
# Đánh giá: precision / recall / macro-F1
# =========================================================
def evaluate_tagger(tagger, test_sents, tagger_name):
    y_true = []
    y_pred = []

    for sent in test_sents:
        words = [word for word, tag in sent]
        gold_tags = [tag for word, tag in sent]

        pred_sent = tagger.tag(words)
        pred_tags = [tag if tag is not None else "X" for _, tag in pred_sent]

        y_true.extend(gold_tags)
        y_pred.extend(pred_tags)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    return {
        "tagger": tagger_name,
        "precision": precision,
        "recall": recall,
        "macro_f1": f1,
    }


# =========================================================
# In bảng kết quả
# =========================================================
def print_results_table(results):
    print("=" * 78)
    print(f"{'POS TAGGER':35} {'PRECISION':>12} {'RECALL':>12} {'MACRO-F1':>12}")
    print("=" * 78)

    for r in results:
        print(
            f"{r['tagger']:35} "
            f"{r['precision']:12.4f} "
            f"{r['recall']:12.4f} "
            f"{r['macro_f1']:12.4f}"
        )

    print("=" * 78)


# =========================================================
# So sánh POS tagging trên 1 câu cụ thể
# =========================================================
def compare_single_sentence(unigram_tagger, bigram_tagger, sentence):
    print("\n=== SO SÁNH TRÊN 1 CÂU CỤ THỂ ===\n")

    tokens = sentence.split()

    uni_tags = unigram_tagger.tag(tokens)
    bi_tags = bigram_tagger.tag(tokens)

    print(f"Câu: {sentence}\n")

    print(f"{'WORD':15} {'UNIGRAM':10} {'BIGRAM':10} {'DIFF'}")
    print("-" * 50)

    for (w1, t1), (_, t2) in zip(uni_tags, bi_tags):
        diff = "<<<" if t1 != t2 else ""
        print(f"{w1:15} {str(t1):10} {str(t2):10} {diff}")
# =========================================================
# Main
# =========================================================
def main():
    print("=== BÀI TOÁN POS TAGGING TRÊN BROWN CORPUS ===\n")

    # 1. Download dataset 1 lần
    ensure_nltk_resources()

    # 2. Load dữ liệu
    print("[INFO] Đang load Brown corpus...")
    tagged_sents = load_dataset()
    print(f"[INFO] Tổng số câu: {len(tagged_sents)}")

    # 3. Chia train/test
    train_sents, test_sents = split_dataset(tagged_sents, train_ratio=0.9)
    print(f"[INFO] Số câu train: {len(train_sents)}")
    print(f"[INFO] Số câu test : {len(test_sents)}\n")

    # 4. Train 2 POS tagger
    print("[INFO] Đang huấn luyện UnigramTagger và BigramTagger...")
    unigram_tagger, bigram_tagger = train_taggers(train_sents)

    # 5. Đánh giá
    print("[INFO] Đang đánh giá mô hình...\n")
    results = [
        evaluate_tagger(unigram_tagger, test_sents, "UnigramTagger"),
        evaluate_tagger(bigram_tagger, test_sents, "BigramTagger + backoff Unigram"),
    ]

    # 6. In bảng kết quả
    print_results_table(results)

    # 7. Nhận xét nhanh
    best_f1 = max(results, key=lambda x: x["macro_f1"])
    best_precision = max(results, key=lambda x: x["precision"])
    best_recall = max(results, key=lambda x: x["recall"])

    print("\n=== SO SÁNH TỔNG KẾT ===\n")

    print(f"[BEST PRECISION] : {best_precision['tagger']} ({best_precision['precision']:.4f})")
    print(f"[BEST RECALL   ] : {best_recall['tagger']} ({best_recall['recall']:.4f})")
    print(f"[BEST MACRO-F1 ] : {best_f1['tagger']} ({best_f1['macro_f1']:.4f})")

    # 8. So sánh 1 câu cụ thể
    test_sentence = "I saw her duck"
    compare_single_sentence(unigram_tagger, bigram_tagger, test_sentence)

if __name__ == "__main__":
    main()