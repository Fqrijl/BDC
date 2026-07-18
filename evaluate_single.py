import os
import sys
import csv
from typing import Dict, List, Tuple, Set

# =====================================================================
# KONFIGURASI PATH FILE
# Silakan ganti path di bawah ini sesuai dengan file yang ingin dievaluasi
# =====================================================================
SUBMISSION_FILE = "submission-1.csv"  # Ganti dengan path file submission Anda
SOLUTION_FILE = "solution.csv"        # Ganti dengan path file solution Anda
# =====================================================================

def load_csv(filepath: str) -> Dict[str, str]:
    """Membaca file CSV dan mengembalikan dictionary dengan format {id: prediksi}."""
    if not os.path.exists(filepath):
        print(f"Error: File tidak ditemukan di '{filepath}'")
        sys.exit(1)
        
    data = {}
    with open(filepath, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            print(f"Error: File '{filepath}' kosong.")
            sys.exit(1)
            
        # Cari indeks kolom ID dan Target/Predicted
        id_idx = -1
        pred_idx = -1
        for idx, col in enumerate(header):
            col_lower = col.strip().lower()
            if col_lower == 'id':
                id_idx = idx
            elif col_lower in ['predicted', 'target', 'label', 'class', 'prediksi']:
                pred_idx = idx
                
        # Jika header tidak cocok secara eksak, gunakan fallback (kolom 0 untuk ID, kolom 1 untuk prediksi)
        if id_idx == -1 or pred_idx == -1:
            if len(header) >= 2:
                id_idx = 0
                pred_idx = 1
            else:
                print(f"Error: Tidak dapat mengidentifikasi kolom ID dan Prediksi di '{filepath}'.")
                print(f"Header yang ditemukan: {header}")
                sys.exit(1)
                
        for row_num, row in enumerate(reader, start=2):
            if not row:
                continue
            if len(row) <= max(id_idx, pred_idx):
                print(f"Warning: Baris {row_num} di '{filepath}' kekurangan kolom. Lewati baris ini.")
                continue
            row_id = row[id_idx].strip()
            row_pred = row[pred_idx].strip()
            data[row_id] = row_pred
            
    return data

def calculate_metrics(y_true: List[str], y_pred: List[str]) -> Tuple[float, float, Dict[str, Dict[str, float]], Dict[str, int], Dict[str, int]]:
    """Menghitung akurasi, F1 Macro, dan metrik per kelas."""
    classes: Set[str] = set(y_true).union(set(y_pred))
    classes_list = sorted(list(classes))
    
    total = len(y_true)
    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    accuracy = correct / total if total > 0 else 0.0
    
    true_dist = {c: y_true.count(c) for c in classes_list}
    pred_dist = {c: y_pred.count(c) for c in classes_list}
    
    class_metrics = {}
    f1_sum = 0.0
    
    for c in classes_list:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp == c)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != c and yp == c)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp != c)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        class_metrics[c] = {
            'precision': precision,
            'recall': recall,
            'f1-score': f1,
            'support': true_dist[c]
        }
        f1_sum += f1
        
    macro_f1 = f1_sum / len(classes_list) if classes_list else 0.0
    return accuracy, macro_f1, class_metrics, true_dist, pred_dist

def main():
    print("=" * 60)
    print(" EVALUASI F1 MACRO - SINGLE FILE")
    print("=" * 60)
    print(f"File Solution   : {SOLUTION_FILE}")
    print(f"File Submission : {SUBMISSION_FILE}")
    print("-" * 60)
    
    # Load data
    solution_data = load_csv(SOLUTION_FILE)
    submission_data = load_csv(SUBMISSION_FILE)
    
    sol_ids = set(solution_data.keys())
    sub_ids = set(submission_data.keys())
    
    missing_in_sub = sol_ids - sub_ids
    extra_in_sub = sub_ids - sol_ids
    common_ids = sorted(list(sol_ids.intersection(sub_ids)))
    
    if not common_ids:
        print("Error: Tidak ada ID yang cocok sama sekali antara solution dan submission.")
        sys.exit(1)
        
    if missing_in_sub:
        print(f"Peringatan: {len(missing_in_sub)} ID di solution tidak ada di submission.")
        print(f"5 ID pertama yang hilang: {list(missing_in_sub)[:5]}")
    if extra_in_sub:
        print(f"Peringatan: {len(extra_in_sub)} ID di submission tidak ada di solution (akan diabaikan).")
        
    y_true = [solution_data[i] for i in common_ids]
    y_pred = [submission_data[i] for i in common_ids]
    
    print(f"Total baris yang cocok dievaluasi: {len(common_ids)}")
    print("-" * 60)
    
    # Coba gunakan scikit-learn jika terinstall
    sklearn_used = False
    try:
        from sklearn.metrics import classification_report, f1_score, accuracy_score
        
        accuracy = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        report = classification_report(y_true, y_pred, digits=6)
        
        print(f"Accuracy : {accuracy:.6f}")
        print(f"F1 Macro : {macro_f1:.6f}")
        print("\nClassification Report (scikit-learn):")
        print(report)
        sklearn_used = True
    except ImportError:
        pass
        
    if not sklearn_used:
        # Fallback menggunakan pure Python
        accuracy, macro_f1, class_metrics, true_dist, pred_dist = calculate_metrics(y_true, y_pred)
        
        print(f"Accuracy : {accuracy:.6f}")
        print(f"F1 Macro : {macro_f1:.6f}")
        print("\nClassification Report (Pure Python):")
        print(f"{'Class':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
        print("-" * 57)
        for c in sorted(class_metrics.keys()):
            metrics = class_metrics[c]
            print(f"{c:<10} | {metrics['precision']:<10.6f} | {metrics['recall']:<10.6f} | {metrics['f1-score']:<10.6f} | {metrics['support']:<8}")
        print("-" * 57)
        
        # Perbandingan Distribusi
        print("\nPerbandingan Distribusi Kelas:")
        print(f"{'Class':<10} | {'Jumlah Solution':<15} | {'Jumlah Submission':<17}")
        print("-" * 48)
        for c in sorted(set(true_dist.keys()).union(set(pred_dist.keys()))):
            t_count = true_dist.get(c, 0)
            p_count = pred_dist.get(c, 0)
            print(f"{c:<10} | {t_count:<15} | {p_count:<17}")
            
    print("=" * 60)

if __name__ == "__main__":
    main()
