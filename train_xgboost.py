import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import RobustScaler, OrdinalEncoder
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc, roc_curve, f1_score
from xgboost import XGBClassifier
import optuna
import joblib
import shap
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. VERİ YÜKLEME VE ÖN İŞLEME (Bölüm 3.3 & 3.4)
# ==========================================
print("Veri yükleniyor...")
df = pd.read_csv("clinvar_conflicting.csv")

# Veri Kalitesi: Duplicate kayıtların silinmesi
df = df.drop_duplicates()
features_only = df.drop(columns=["CLASS"])
df = df.drop_duplicates(subset=features_only.columns, keep=False) # Tutarsızlıkları sil

X = df.drop(columns=["CLASS", "CHROM", "POS", "REF", "ALT", "CLNHGVS", "Allele", "SYMBOL", "Feature"]) 
y = df["CLASS"]

numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

# Sıfır veya sıfıra yakın varyanslı kolonları ele
variances = X[numeric_cols].var()
low_var_cols = variances[variances < 1e-5].index.tolist()
X = X.drop(columns=low_var_cols)
numeric_cols = [c for c in numeric_cols if c not in low_var_cols]

# Kategorik Verileri XGBoost için Sayısallaştırma (Ordinal Encoding)
for col in categorical_cols:
    X[col] = X[col].astype(str).fillna("missing")

encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X[categorical_cols] = encoder.fit_transform(X[categorical_cols])

# Sayısal kolonlarda Robust Scaler uygulaması (Aykırı Değer Direnci)
scaler = RobustScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols].fillna(X[numeric_cols].median()))

# Eğitim ve Test Ayrımı
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Sınıf Dengesizliği için Ölçekleme Katsayısı (Scale Pos Weight)
neg_class = (y_train == 0).sum()
pos_class = (y_train == 1).sum()
scale_weight = neg_class / pos_class

# ==========================================
# 2. OPTUNA İLE HİPERPARAMETRE OPTİMİZASYONU (Bölüm 4.1)
# ==========================================
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True), # L2 Regularizasyonu
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'tree_method': 'hist',    # XGBoost için GPU ayarlamaları
        'device': 'cuda',         # GPU KULLANIMI İÇİN KRİTİK KOMUT!
        'scale_pos_weight': scale_weight,
        'eval_metric': 'logloss',
        'random_state': 42
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_scores = []

    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model = XGBClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        
        preds = model.predict(X_val)
        f1_scores.append(f1_score(y_val, preds))

    return np.mean(f1_scores)

print("Optuna ile XGBoost (GPU) optimizasyonu başlatılıyor...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=15) # Süreye göre n_trials artırılabilir
print(f"En İyi Parametreler: {study.best_params}")

# ==========================================
# 3. EN İYİ MODELİN EĞİTİLMESİ VE DİNAMİK EŞİK (Bölüm 3.5 & 4.2)
# ==========================================
best_params = study.best_params
best_params.update({
    'tree_method': 'hist',
    'device': 'cuda', # GPU
    'scale_pos_weight': scale_weight,
    'eval_metric': 'logloss',
    'random_state': 42
})

final_model = XGBClassifier(**best_params)

print("En iyi XGBoost modeli GPU'da eğitiliyor...")
final_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=50)

# Olasılık değerlerinin alınması
y_probs = final_model.predict_proba(X_test)[:, 1]

# Dinamik Threshold Optimizasyonu (Risk Perspektifi)
precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
best_threshold = thresholds[np.argmax(f1_scores)]
print(f"\nModelin F1 Skorunu Maksimize Eden Karar Eşiği (Threshold): {best_threshold:.3f}")

y_pred = (y_probs >= best_threshold).astype(int)

# ==========================================
# 4. DEĞERLENDİRME METRİKLERİ VE GRAFİKLER (Bölüm 4.2)
# ==========================================
print("\n--- Sınıflandırma Raporu ---")
print(classification_report(y_test, y_pred))

# Grafik Çizimleri
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], annot_kws={"size": 14})
axes[0].set_title(f"Confusion Matrix (Threshold: {best_threshold:.2f})")
axes[0].set_xlabel("Tahmin Edilen")
axes[0].set_ylabel("Gerçek Sınıf")

# Precision-Recall Eğrisi
pr_auc = auc(recalls, precisions)
axes[1].plot(recalls, precisions, label=f"PR-AUC: {pr_auc:.3f}", color='darkorange', lw=2)
axes[1].set_title("Precision-Recall Eğrisi")
axes[1].set_xlabel("Duyarlılık (Recall)")
axes[1].set_ylabel("Hassasiyet (Precision)")
axes[1].legend()

# ROC Eğrisi
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)
axes[2].plot(fpr, tpr, label=f"ROC-AUC: {roc_auc:.3f}", color='green', lw=2)
axes[2].plot([0, 1], [0, 1], linestyle='--', color='gray')
axes[2].set_title("ROC Eğrisi")
axes[2].set_xlabel("False Positive Rate")
axes[2].set_ylabel("True Positive Rate")
axes[2].legend()

plt.tight_layout()
plt.savefig("xgboost_degerlendirme_grafikleri.png", dpi=300)
print("Grafikler 'xgboost_degerlendirme_grafikleri.png' olarak kaydedildi.")

# ==========================================
# 5. MODELİN KAYDEDİLMESİ VE SHAP AÇIKLANABİLİRLİK (Bölüm 4.4)
# ==========================================
joblib.dump(final_model, "xgboost_gpu_model.pkl")
joblib.dump(scaler, "robust_scaler.pkl")
joblib.dump(encoder, "ordinal_encoder.pkl")
print("Model, Scaler ve Encoder başarıyla kaydedildi.")

# SHAP Analizi (Açıklanabilirlik)
print("SHAP Analizi Grafiği oluşturuluyor...")
explainer = shap.TreeExplainer(final_model)
# SHAP analizi GPU'da çalışırken tüm veriyi verirseniz bazen bellek şişebilir, 
# test setinden rastgele 1000 örnek seçerek hızlıca çizdiriyoruz.
X_test_sampled = X_test.sample(n=min(1000, len(X_test)), random_state=42)
shap_values = explainer.shap_values(X_test_sampled)

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test_sampled, show=False)
plt.savefig("shap_aciklanabilirlik_xgboost.png", dpi=300, bbox_inches='tight')
print("SHAP grafiği kaydedildi. (shap_aciklanabilirlik_xgboost.png)")