import os
import random
import json
import numpy as np

# Если вы будете использовать torch / tf позже — сразу заложим задел
try:
    import torch
except ImportError:
    torch = None

try:
    import tensorflow as tf
except ImportError:
    tf = None

# --- Подключение Google Drive (работает только в Colab) ---
try:
    from google.colab import drive  # type: ignore
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    drive.mount("/content/drive")
    BASE_DIR = "/content/drive/MyDrive/digital_marketing_conversion"
else:
    # Локальный fallback — можно поменять под себя
    BASE_DIR = os.path.abspath("./digital_marketing_conversion")

DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

for d in [BASE_DIR, DATA_DIR, MODELS_DIR, PLOTS_DIR, REPORTS_DIR, ARTIFACTS_DIR]:
    os.makedirs(d, exist_ok=True)

print("BASE_DIR:", BASE_DIR)
print("DATA_DIR:", DATA_DIR)
print("MODELS_DIR:", MODELS_DIR)
print("PLOTS_DIR:", PLOTS_DIR)
print("REPORTS_DIR:", REPORTS_DIR)
print("ARTIFACTS_DIR:", ARTIFACTS_DIR)

# --- Функция для установки случайных зерен ---
def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Дополнительно чуть жёстче фиксируем детерминизм
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if tf is not None:
        try:
            tf.random.set_seed(seed)
        except Exception:
            pass

    print(f"[INFO] Глобальное зерно случайности установлено: {seed}")

set_global_seed(42)

# --- Проверка наличия GPU ---
if torch is not None and torch.cuda.is_available():
    print("[INFO] PyTorch видит GPU:", torch.cuda.get_device_name(0))
elif tf is not None and len(tf.config.list_physical_devices("GPU")) > 0:
    print("[INFO] TensorFlow видит GPU:", tf.config.list_physical_devices("GPU"))
else:
    print("[WARN] GPU не обнаружен (или библиотеки DL ещё не установлены).")
    print("       Но для датасета на 8000 строк это не критично.")

# Базовые библиотеки анализа данных и визуализации
!pip install -q pandas numpy matplotlib seaborn

# ML-стек (дальше пригодится, но уже сейчас можно поставить)
!pip install -q scikit-learn imbalanced-learn xgboost lightgbm shap umap-learn

# Работа с Kaggle API и конфигами
!pip install -q kaggle pyyaml

import os
import zipfile
from pathlib import Path

if IN_COLAB:
    from google.colab import files  # type: ignore

    print(
        "Пожалуйста, загрузите kaggle.json (из профиля Kaggle → Account → Create New Token)."
    )
    uploaded = files.upload()  # Откроется диалог выбора файла

    if "kaggle.json" not in uploaded:
        raise FileNotFoundError(
            "Файл kaggle.json не найден среди загруженных. "
            "Убедитесь, что выбрали правильный файл."
        )

    # Настройка Kaggle API
    kaggle_dir = os.path.join(os.path.expanduser("~"), ".kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)

    with open("kaggle.json", "r") as f_in:
        token = json.load(f_in)

    kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")
    with open(kaggle_json_path, "w") as f_out:
        json.dump(token, f_out)

    os.chmod(kaggle_json_path, 0o600)
    print("[INFO] kaggle.json настроен.")

    # Скачивание датасета
    print("[INFO] Скачиваю датасет 'Predict Conversion in Digital Marketing'...")
    !kaggle datasets download -d rabieelkharoua/predict-conversion-in-digital-marketing-dataset -p {DATA_DIR} --unzip

    # Проверим, какие CSV появились
    csv_files = list(Path(DATA_DIR).glob("*.csv"))
    print("[INFO] Найдены CSV-файлы в DATA_DIR:")
    for p in csv_files:
        print("   -", p.name)

else:
    print(
        "[WARN] Не в Colab: этот блок рассчитан на Google Colab и может не работать локально."
    )

import yaml

dataset_registry_path = os.path.join(DATA_DIR, "dataset_registry.yaml")

# Попробуем автоматически найти основной CSV
from glob import glob

csv_files = glob(os.path.join(DATA_DIR, "*.csv"))

main_dataset_path = None
for candidate in csv_files:
    name = os.path.basename(candidate).lower()
    if "digital_marketing_campaign" in name:
        main_dataset_path = candidate
        break

if main_dataset_path is None and csv_files:
    main_dataset_path = csv_files[0]  # fallback — первый найденный CSV

dataset_registry = {
    "main_conversion_dataset": {
        "source": "kaggle",
        "kaggle_dataset": "rabieelkharoua/predict-conversion-in-digital-marketing-dataset",
        "path": main_dataset_path,
        "description": "Predict Conversion in Digital Marketing Dataset (основной датасет для экспериментов)",
    },
    # Сюда позже можно дописать ещё один датасет (stuffmart, др.)
}

with open(dataset_registry_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(dataset_registry, f, allow_unicode=True)

print("[INFO] dataset_registry.yaml сохранён по пути:", dataset_registry_path)
print("       Основной датасет:", main_dataset_path)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("default")
sns.set_theme(style="whitegrid")

# Грузим registry, чтобы взять путь к датасету
with open(dataset_registry_path, "r", encoding="utf-8") as f:
    dataset_registry = yaml.safe_load(f)

main_dataset_info = dataset_registry["main_conversion_dataset"]
csv_path = main_dataset_info["path"]

if csv_path is None or not os.path.exists(csv_path):
    raise FileNotFoundError(
        f"CSV-файл основного датасета не найден. Ожидался путь: {csv_path}"
    )

print("[INFO] Загружаю основной датасет из:", csv_path)
df = pd.read_csv(csv_path)

print("\n[SHAPE] Размерность датасета:", df.shape)
print("\n[D_TYPES] Типы данных колонок:")
print(df.dtypes)

print("\n[HEAD] Первые 5 строк:")
display(df.head())

print("\n[NA] Количество пропусков в каждом столбце:")
print(df.isna().sum())

print("\n[DESCRIBE] Базовая статистика по числовым признакам:")
display(df.describe().T)

# --- Data Dictionary / Feature Catalog ---
feature_catalog = pd.DataFrame(
    {
        "feature_name": df.columns,
        "dtype": [str(dt) for dt in df.dtypes],
        "n_unique": [df[col].nunique() for col in df.columns],
        "n_missing": [df[col].isna().sum() for col in df.columns],
        # поле для ручного заполнения в отчёте при необходимости
        "description_ru": ["" for _ in df.columns],
        "example_value": [df[col].iloc[0] for col in df.columns],
    }
)

feature_catalog_path = os.path.join(DATA_DIR, "feature_catalog.csv")
feature_catalog.to_csv(feature_catalog_path, index=False, encoding="utf-8-sig")
print("\n[INFO] Каталог признаков сохранён в:", feature_catalog_path)
display(feature_catalog)

TARGET_COL = "Conversion"

if TARGET_COL not in df.columns:
    raise KeyError(
        f"Ожидалась колонка таргета '{TARGET_COL}', но её нет в df.columns: {df.columns.tolist()}"
    )

target_counts = df[TARGET_COL].value_counts().sort_index()
target_probs = df[TARGET_COL].value_counts(normalize=True).sort_index()

print("\n[INFO] Распределение целевой переменной (сырые значения):")
print(target_counts)
print("\n[INFO] Распределение целевой переменной (доли):")
print(target_probs)

fig, ax = plt.subplots(figsize=(5, 4))
sns.barplot(
    x=target_counts.index.astype(str),
    y=target_counts.values,
    ax=ax,
)
ax.set_xlabel("Conversion (0 = не конвертировался, 1 = конвертировался)")
ax.set_ylabel("Количество наблюдений")
ax.set_title("Распределение целевой переменной Conversion")
for i, v in enumerate(target_counts.values):
    ax.text(i, v + 0.01 * max(target_counts.values), str(v), ha="center", va="bottom")
plt.tight_layout()

target_plot_path = os.path.join(PLOTS_DIR, "target_distribution_conversion.png")
plt.savefig(target_plot_path, dpi=150)
plt.show()
plt.close()

print("[INFO] График распределения таргета сохранён в:", target_plot_path)

conversion_summary_rows = []

for col in categorical_cols:
    # Считаем среднюю конверсию по категориям
    group_df = (
        df.groupby(col)[TARGET_COL]
        .agg(["mean", "count"])
        .rename(columns={"mean": "conversion_rate", "count": "n"})
        .sort_values("conversion_rate", ascending=False)
    )

    print(f"\n[SUMMARY] Конверсия по признаку {col}:")
    display(group_df)

    # Сохраним в список для дальнейшего объединения
    tmp = group_df.reset_index()
    tmp.insert(0, "feature", col)
    conversion_summary_rows.append(tmp)

    # Визуализация
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(
        data=group_df.reset_index(),
        x=col,
        y="conversion_rate",
        ax=ax,
    )
    ax.set_title(f"Средняя конверсия по категориям признака {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Conversion rate")

    for i, (category, row) in enumerate(group_df.reset_index()[[col, "conversion_rate"]].values):
        ax.text(i, row + 0.01 * group_df["conversion_rate"].max(), f"{row:.2f}", ha="center", va="bottom")

    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    conv_plot_path = os.path.join(PLOTS_DIR, f"conversion_by_{col}.png")
    plt.savefig(conv_plot_path, dpi=150)
    plt.show()
    plt.close()

    print(f"[INFO] Сохранён график конверсии по признаку {col}: {conv_plot_path}")

# Объединяем все summary в один DataFrame и сохраняем
if conversion_summary_rows:
    conv_summary_df = pd.concat(conversion_summary_rows, ignore_index=True)
    conv_summary_path = os.path.join(DATA_DIR, "conversion_summary_by_categorical.csv")
    conv_summary_df.to_csv(conv_summary_path, index=False, encoding="utf-8-sig")
    print(
        "\n[INFO] Таблица с конверсией по категориальным признакам сохранена в:",
        conv_summary_path,
    )
    display(conv_summary_df.head())
else:
    print("[WARN] Нет категориальных признаков для расчёта конверсии.")

corr_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print("\n[INFO] Числовые признаки для корреляции:", corr_cols)

corr_matrix = df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    corr_matrix,
    annot=False,
    cmap="coolwarm",
    center=0.0,
    square=True,
    cbar_kws={"shrink": 0.8},
    ax=ax,
)
ax.set_title("Матрица корреляций числовых признаков")
plt.tight_layout()

corr_plot_path = os.path.join(PLOTS_DIR, "correlation_matrix_numeric.png")
plt.savefig(corr_plot_path, dpi=150)
plt.show()
plt.close()

print("[INFO] Матрица корреляций сохранена в:", corr_plot_path)

# --- Пара интересных пар признаков для скеттерплотов / jointplot ---
candidate_pairs = [
    ("AdSpend", "ClickThroughRate"),
    ("WebsiteVisits", "ConversionRate"),
    ("PagesPerVisit", "TimeOnSite"),
    ("PreviousPurchases", "LoyaltyPoints"),
]

for x_col, y_col in candidate_pairs:
    if x_col in df.columns and y_col in df.columns:
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.scatterplot(data=df, x=x_col, y=y_col, hue=TARGET_COL, alpha=0.6, ax=ax)
        ax.set_title(f"Связь между {x_col} и {y_col} (цвет = Conversion)")
        plt.tight_layout()

        pair_plot_path = os.path.join(PLOTS_DIR, f"pair_{x_col}_vs_{y_col}.png")
        plt.savefig(pair_plot_path, dpi=150)
        plt.show()
        plt.close()

        print(
            f"[INFO] Сохранён график связи {x_col} vs {y_col}:",
            pair_plot_path,
        )
    else:
        print(
            f"[WARN] Пара ({x_col}, {y_col}) пропущена: один из признаков отсутствует в df.columns."
        )

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# На всякий случай ещё раз зададим некоторые базовые объекты
plt.style.use("default")
sns.set_theme(style="whitegrid")

# Если вы меняли TARGET_COL выше — обязательно синхронизируйте здесь
TARGET_COL = "Conversion"

def add_business_features(df_in: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Добавляет бизнес-ориентированные производные признаки.
    Возвращает новый DataFrame и список созданных колонок.
    """
    df_out = df_in.copy()
    derived_cols: list[str] = []
    EPS = 1e-6

    # 1) Метрики e-mail вовлечённости
    if {"EmailClicks", "EmailOpens"}.issubset(df_out.columns):
        df_out["EmailEngagementRate"] = df_out["EmailClicks"] / (df_out["EmailOpens"] + 1.0)
        derived_cols.append("EmailEngagementRate")

    # 2) Метрики социальных сетей
    if {"SocialShares", "WebsiteVisits"}.issubset(df_out.columns):
        df_out["SocialShareRate"] = df_out["SocialShares"] / (df_out["WebsiteVisits"] + 1.0)
        derived_cols.append("SocialShareRate")

    # 3) Стоимость контакта/визита
    if {"AdSpend", "WebsiteVisits"}.issubset(df_out.columns):
        df_out["CostPerVisit"] = df_out["AdSpend"] / (df_out["WebsiteVisits"] + 1.0)
        derived_cols.append("CostPerVisit")

    if {"AdSpend", "EmailClicks"}.issubset(df_out.columns):
        df_out["CostPerEmailClick"] = df_out["AdSpend"] / (df_out["EmailClicks"] + 1.0)
        derived_cols.append("CostPerEmailClick")

    # 4) Показатель лояльности
    if {"LoyaltyPoints", "PreviousPurchases"}.issubset(df_out.columns):
        df_out["LoyaltyPerPurchase"] = df_out["LoyaltyPoints"] / (df_out["PreviousPurchases"] + 1.0)
        derived_cols.append("LoyaltyPerPurchase")

    # 5) Интегральный индекс вовлечённости
    engagement_components = [
        c
        for c in [
            "WebsiteVisits",
            "PagesPerVisit",
            "TimeOnSite",
            "SocialShares",
            "EmailOpens",
            "EmailClicks",
            "ClickThroughRate",
            "ConversionRate",
        ]
        if c in df_out.columns
    ]
    if engagement_components:
        # Простейшая нормализация через ранги (0..1) и среднее
        df_out["EngagementScore"] = (
            df_out[engagement_components]
            .rank(pct=True)
            .mean(axis=1)
        )
        derived_cols.append("EngagementScore")

    print("\n[INFO] Добавлены производные признаки:")
    for col in derived_cols:
        print("   -", col)

    return df_out, derived_cols


# Применяем функцию к исходному df
df_fe, DERIVED_FEATURES = add_business_features(df)

print("\n[SHAPE] После добавления признаков размерность:", df_fe.shape)
print("[INFO] Список производных признаков:", DERIVED_FEATURES)

import os
import json
import numpy as np
import pandas as pd

# Колонка ID (если есть)
ID_COL = "CustomerID" if "CustomerID" in df_fe.columns else None
if ID_COL:
    print(f"\n[INFO] Обнаружена ID-колонка: {ID_COL}")
else:
    print("\n[WARN] Явная ID-колонка не найдена (CustomerID отсутствует).")

# --- Удаляем константные признаки (все значения одинаковые) ---
nunique = df_fe.nunique(dropna=False)
constant_cols = [
    c for c in df_fe.columns
    if (nunique[c] == 1) and (c not in [ID_COL, TARGET_COL])
]

if constant_cols:
    print("\n[INFO] Найдены константные признаки, будем удалять их из признакового пространства:")
    for c in constant_cols:
        print("   -", c)
    df_fe = df_fe.drop(columns=constant_cols)
else:
    print("\n[INFO] Константные признаки не обнаружены.")

# Числовые и категориальные признаки
numeric_cols_all = df_fe.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols_all = [c for c in numeric_cols_all if c not in [TARGET_COL, ID_COL]]

categorical_cols_all = df_fe.select_dtypes(include=["object", "category"]).columns.tolist()

print("\n[INFO] Числовые признаки (кандидаты для масштабирования):")
print(numeric_cols_all)
print("\n[INFO] Категориальные признаки (кандидаты для one-hot):")
print(categorical_cols_all)

# Мягкая обработка выбросов: усечение по 1-му и 99-му перцентилям
winsor_config: dict[str, dict[str, float]] = {}
LOW_Q = 0.01
HIGH_Q = 0.99

for col in numeric_cols_all:
    q_low = df_fe[col].quantile(LOW_Q)
    q_high = df_fe[col].quantile(HIGH_Q)

    outlier_mask = (df_fe[col] < q_low) | (df_fe[col] > q_high)
    outlier_pct = float(outlier_mask.mean() * 100)

    print(
        f"[OUTLIERS] {col}: {outlier_pct:.2f}% наблюдений за пределами "
        f"[{LOW_Q:.2f}; {HIGH_Q:.2f}] перцентилей ({q_low:.3f}, {q_high:.3f})"
    )

    df_fe[col] = df_fe[col].clip(lower=q_low, upper=q_high)

    winsor_config[col] = {
        "q_low": float(q_low),
        "q_high": float(q_high),
        "outlier_pct": outlier_pct,
    }

winsor_config_path = os.path.join(DATA_DIR, "winsorization_config.json")
with open(winsor_config_path, "w", encoding="utf-8") as f:
    json.dump(winsor_config, f, indent=2, ensure_ascii=False)

print("\n[INFO] Конфигурация усечения выбросов сохранена в:", winsor_config_path)

# Финальные списки признаков
NUMERIC_FEATURES = numeric_cols_all.copy()
CATEGORICAL_FEATURES = categorical_cols_all.copy()

print("\n[FEATURES] Числовые признаки для препроцессинга:")
print(NUMERIC_FEATURES)
print("\n[FEATURES] Категориальные признаки для препроцессинга:")
print(CATEGORICAL_FEATURES)

from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import sklearn

# подбираем правильный аргумент для OneHotEncoder в зависимости от версии sklearn
skl_version = sklearn.__version__
major, minor, *_ = skl_version.split(".")
major = int(major)
minor = int(minor)

onehot_kwargs = dict(
    handle_unknown="ignore",
    drop="first",
)

# новые версии (≈>=1.2) используют sparse_output, старые — sparse
if major > 1 or (major == 1 and minor >= 2):
    onehot_kwargs["sparse_output"] = False
else:
    onehot_kwargs["sparse"] = False

print(f"[INFO] Версия scikit-learn: {skl_version}")
print("[INFO] Параметры OneHotEncoder:", onehot_kwargs)

numeric_transformer = Pipeline(
    steps=[
        ("scaler", RobustScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("onehot", OneHotEncoder(**onehot_kwargs)),
    ]
)

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ]
)

# Описание пространства признаков для Flask и отчёта
feature_space = {
    "id_col": ID_COL,
    "target_col": TARGET_COL,
    "numeric_features": NUMERIC_FEATURES,
    "categorical_features": CATEGORICAL_FEATURES,
    "derived_features": DERIVED_FEATURES,
    "constant_dropped": constant_cols,
    "winsorization": winsor_config,
    "scaler": "RobustScaler",
    "encoder": {
        "class": "OneHotEncoder",
        "params": onehot_kwargs,
    },
}

feature_space_path = os.path.join(DATA_DIR, "feature_space.json")
with open(feature_space_path, "w", encoding="utf-8") as f:
    json.dump(feature_space, f, indent=2, ensure_ascii=False)

print("\n[INFO] Описание пространства признаков сохранено в:", feature_space_path)

# Быстрая проверка работы препроцессора
sample_size = min(1000, len(df_fe))
X_sample = df_fe[NUMERIC_FEATURES + CATEGORICAL_FEATURES].iloc[:sample_size]
print(f"\n[DEBUG] Пробный препроцессинг на {sample_size} наблюдениях...")
X_sample_transformed = preprocess.fit_transform(X_sample)
print("   Исходная форма X_sample:", X_sample.shape)
print("   Форма после preprocess:", X_sample_transformed.shape)

from sklearn.model_selection import train_test_split

feature_cols_for_model = NUMERIC_FEATURES + CATEGORICAL_FEATURES

X = df_fe[feature_cols_for_model].copy()
y = df_fe[TARGET_COL].astype(int)

print("\n[SHAPE] Размерности X и y перед разбиением:")
print("   X:", X.shape)
print("   y:", y.shape)

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_trainval,
    y_trainval,
    test_size=0.25,  # 0.25 от 0.8 = 0.2 общей выборки
    random_state=42,
    stratify=y_trainval,
)

print("\n[SHAPE] Итоговые размерности выборок:")
print("   X_train:", X_train.shape, "| y_train:", y_train.shape)
print("   X_valid:", X_valid.shape, "| y_valid:", y_valid.shape)
print("   X_test :", X_test.shape,  "| y_test :", y_test.shape)

def print_class_balance(name: str, y_part: pd.Series) -> None:
    vc = y_part.value_counts(normalize=True).sort_index()
    print(f"   [{name}] распределение классов:")
    for cls, p in vc.items():
        print(f"      класс {cls}: {p*100:.2f}%")

print()
print_class_balance("train", y_train)
print_class_balance("valid", y_valid)
print_class_balance("test", y_test)

splits_indices = {
    "train": X_train.index.tolist(),
    "valid": X_valid.index.tolist(),
    "test": X_test.index.tolist(),
}

split_indices_path = os.path.join(DATA_DIR, "split_indices.json")
with open(split_indices_path, "w", encoding="utf-8") as f:
    json.dump(splits_indices, f, indent=2, ensure_ascii=False)

print("\n[INFO] Индексы разбиений train/valid/test сохранены в:", split_indices_path)

if ID_COL is not None:
    for name, idx in splits_indices.items():
        subset = df_fe.loc[idx, [ID_COL, TARGET_COL]]
        out_path = os.path.join(DATA_DIR, f"{name}_id_target.csv")
        subset.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[INFO] Для {name} сохранён файл ID+таргет:", out_path)
else:
    print("\n[WARN] ID-колонка не задана, отдельные файлы ID+таргет не сохранялись.")

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
)

plt.style.use("default")

def _compute_basic_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Считает набор стандартных метрик бинарной классификации.
    y_proba — вероятности класса 1.
    """
    y_pred = (y_proba >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    roc = roc_auc_score(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "threshold": threshold,
        "accuracy": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "roc_auc": roc,
        "pr_auc": pr_auc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "support_neg": int(tn + fp),
        "support_pos": int(fn + tp),
    }
    return metrics


def plot_roc_pr_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str,
    split_name: str,
    plots_dir: str,
) -> dict:
    """
    Строит ROC- и PR-кривые, сохраняет их в файлы и возвращает пути.
    """
    model_key = model_name.lower().replace(" ", "_")
    split_key = split_name.lower()

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"{model_name}")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Случайный классификатор")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title(f"ROC-кривая ({model_name}, {split_name})")
    ax.legend()
    plt.tight_layout()
    roc_path = os.path.join(plots_dir, f"roc_{model_key}_{split_key}.png")
    plt.savefig(roc_path, dpi=150)
    plt.show()
    plt.close()

    # PR
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"{model_name}")
    base_rate = (y_true == 1).mean()
    ax.hlines(base_rate, 0, 1, linestyles="--", label=f"Базовая линия (p={base_rate:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall-кривая ({model_name}, {split_name})")
    ax.legend()
    plt.tight_layout()
    pr_path = os.path.join(plots_dir, f"pr_{model_key}_{split_key}.png")
    plt.savefig(pr_path, dpi=150)
    plt.show()
    plt.close()

    return {"roc_path": roc_path, "pr_path": pr_path}


def evaluate_binary_model(
    model: Pipeline,
    model_name: str,
    X_train,
    y_train,
    X_valid,
    y_valid,
    X_test,
    y_test,
    plots_dir: str,
    threshold: float = 0.5,
) -> dict:
    """
    Обучает модель и считает метрики на train / valid / test.
    Возвращает словарь с метриками и путями к картинкам ROC/PR для теста.
    """
    print(f"\n[INFO] Обучение модели: {model_name}")
    model.fit(X_train, y_train)

    # Предсказания вероятностей (класс 1)
    if hasattr(model, "predict_proba"):
        proba_train = model.predict_proba(X_train)[:, 1]
        proba_valid = model.predict_proba(X_valid)[:, 1]
        proba_test = model.predict_proba(X_test)[:, 1]
    else:
        # На всякий случай fallback через decision_function
        decision_train = model.decision_function(X_train)
        decision_valid = model.decision_function(X_valid)
        decision_test = model.decision_function(X_test)
        # нормируем до 0..1 (логистика)
        proba_train = 1 / (1 + np.exp(-decision_train))
        proba_valid = 1 / (1 + np.exp(-decision_valid))
        proba_test = 1 / (1 + np.exp(-decision_test))

    metrics_train = _compute_basic_metrics(y_train, proba_train, threshold)
    metrics_valid = _compute_basic_metrics(y_valid, proba_valid, threshold)
    metrics_test = _compute_basic_metrics(y_test, proba_test, threshold)

    print(f"[METRICS] {model_name} — validation:")
    for k, v in metrics_valid.items():
        if k in ["tn", "fp", "fn", "tp", "support_neg", "support_pos"]:
            continue
        print(f"   {k:10s}: {v:.4f}" if isinstance(v, float) else f"   {k:10s}: {v}")

    # ROC/PR для теста
    plot_paths = plot_roc_pr_curves(
        y_true=y_test,
        y_proba=proba_test,
        model_name=model_name,
        split_name="test",
        plots_dir=plots_dir,
    )

    return {
        "train": metrics_train,
        "valid": metrics_valid,
        "test": metrics_test,
        "plots": plot_paths,
    }


# Словарь, куда будем складывать результаты по моделям
baseline_results: dict[str, dict] = {}

dummy_clf = DummyClassifier(strategy="most_frequent")

dummy_pipeline = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("clf", dummy_clf),
    ]
)

baseline_results["Dummy_most_frequent"] = evaluate_binary_model(
    model=dummy_pipeline,
    model_name="Dummy (most frequent)",
    X_train=X_train,
    y_train=y_train,
    X_valid=X_valid,
    y_valid=y_valid,
    X_test=X_test,
    y_test=y_test,
    plots_dir=PLOTS_DIR,
    threshold=0.5,
)

# 1) Логистическая регрессия без учёта дисбаланса
logreg_plain = LogisticRegression(
    penalty="l2",
    C=1.0,
    solver="liblinear",
    max_iter=1000,
    n_jobs=-1,
)

logreg_plain_pipeline = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("clf", logreg_plain),
    ]
)

baseline_results["LogReg_plain"] = evaluate_binary_model(
    model=logreg_plain_pipeline,
    model_name="LogReg (plain)",
    X_train=X_train,
    y_train=y_train,
    X_valid=X_valid,
    y_valid=y_valid,
    X_test=X_test,
    y_test=y_test,
    plots_dir=PLOTS_DIR,
    threshold=0.5,
)

# 2) Логистическая регрессия с учётом дисбаланса классов
logreg_bal = LogisticRegression(
    penalty="l2",
    C=1.0,
    solver="liblinear",
    max_iter=1000,
    n_jobs=-1,
    class_weight="balanced",
)

logreg_bal_pipeline = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("clf", logreg_bal),
    ]
)

baseline_results["LogReg_balanced"] = evaluate_binary_model(
    model=logreg_bal_pipeline,
    model_name="LogReg (balanced)",
    X_train=X_train,
    y_train=y_train,
    X_valid=X_valid,
    y_valid=y_valid,
    X_test=X_test,
    y_test=y_test,
    plots_dir=PLOTS_DIR,
    threshold=0.5,
)

# Преобразуем baseline_results в таблицу
rows = []
for model_name, res in baseline_results.items():
    for split_name in ["train", "valid", "test"]:
        metrics = res[split_name]
        row = {
            "model": model_name,
            "split": split_name,
        }
        row.update(metrics)
        rows.append(row)

metrics_df = pd.DataFrame(rows)

print("\n[INFO] Сводная таблица метрик базовых моделей:")
display(metrics_df)

# Сохраняем в CSV и JSON
os.makedirs(REPORTS_DIR, exist_ok=True)

metrics_csv_path = os.path.join(REPORTS_DIR, "metrics_baselines.csv")
metrics_df.to_csv(metrics_csv_path, index=False, encoding="utf-8-sig")

metrics_json_path = os.path.join(REPORTS_DIR, "metrics_baselines.json")
with open(metrics_json_path, "w", encoding="utf-8") as f:
    json.dump(baseline_results, f, indent=2, ensure_ascii=False)

print("\n[INFO] Метрики базовых моделей сохранены в:")
print("   CSV :", metrics_csv_path)
print("   JSON:", metrics_json_path)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.inspection import PartialDependenceDisplay

# XGBoost
from xgboost import XGBClassifier

plt.style.use("default")

advanced_results: dict[str, dict] = {}

neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
neg_ratio = neg_count / (neg_count + pos_count)
pos_ratio = pos_count / (neg_count + pos_count)

print(f"[INFO] Обучающая выборка: всего {len(y_train)}")
print(f"   Нулевой класс (0): {neg_count} ({neg_ratio*100:.2f}%)")
print(f"   Единичный класс (1): {pos_count} ({pos_ratio*100:.2f}%)")

# Для XGBoost будем использовать scale_pos_weight = n_neg / n_pos,
# чтобы относительно усилить вклад редкого класса (0)
scale_pos_weight = neg_count / pos_count
print(f"[INFO] Рекомендуемый scale_pos_weight для XGBoost: {scale_pos_weight:.4f}")

rf_base = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",   # компенсируем дисбаланс
    random_state=42,
    n_jobs=-1,
)

rf_pipeline = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("clf", rf_base),
    ]
)

rf_param_distributions = {
    "clf__n_estimators": [150, 200, 300],
    "clf__max_depth": [None, 5, 8, 12],
    "clf__min_samples_split": [2, 5, 10],
    "clf__min_samples_leaf": [1, 2, 4],
    "clf__max_features": ["sqrt", "log2", 0.5],
}

rf_search = RandomizedSearchCV(
    estimator=rf_pipeline,
    param_distributions=rf_param_distributions,
    n_iter=15,
    scoring="average_precision",   # PR-AUC чувствительна к дисбалансу
    n_jobs=-1,
    cv=3,
    verbose=1,
    random_state=42,
)

print("\n[INFO] Запуск RandomizedSearchCV для Random Forest...")
rf_search.fit(X_train, y_train)

print("\n[INFO] Лучшие параметры для Random Forest:")
print(rf_search.best_params_)
print(f"[INFO] Лучший score (cv average_precision): {rf_search.best_score_:.4f}")

rf_best_pipeline = rf_search.best_estimator_

# Оцениваем с помощью общей функции из этапа 3
advanced_results["RandomForest"] = evaluate_binary_model(
    model=rf_best_pipeline,
    model_name="RandomForest",
    X_train=X_train,
    y_train=y_train,
    X_valid=X_valid,
    y_valid=y_valid,
    X_test=X_test,
    y_test=y_test,
    plots_dir=PLOTS_DIR,
    threshold=0.5,
)

xgb_base = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",    # быстрый и экономичный по памяти
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
)

xgb_pipeline = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("clf", xgb_base),
    ]
)

xgb_param_distributions = {
    "clf__n_estimators": [200, 300, 400],
    "clf__max_depth": [3, 4, 5],
    "clf__learning_rate": [0.03, 0.05, 0.1],
    "clf__subsample": [0.7, 0.85, 1.0],
    "clf__colsample_bytree": [0.7, 0.85, 1.0],
}

xgb_search = RandomizedSearchCV(
    estimator=xgb_pipeline,
    param_distributions=xgb_param_distributions,
    n_iter=15,
    scoring="average_precision",
    n_jobs=-1,
    cv=3,
    verbose=1,
    random_state=42,
)

print("\n[INFO] Запуск RandomizedSearchCV для XGBoost...")
xgb_search.fit(X_train, y_train)

print("\n[INFO] Лучшие параметры для XGBoost:")
print(xgb_search.best_params_)
print(f"[INFO] Лучший score (cv average_precision): {xgb_search.best_score_:.4f}")

xgb_best_pipeline = xgb_search.best_estimator_

advanced_results["XGBoost"] = evaluate_binary_model(
    model=xgb_best_pipeline,
    model_name="XGBoost",
    X_train=X_train,
    y_train=y_train,
    X_valid=X_valid,
    y_valid=y_valid,
    X_test=X_test,
    y_test=y_test,
    plots_dir=PLOTS_DIR,
    threshold=0.5,
)

rows = []
for model_name, res in advanced_results.items():
    for split_name in ["train", "valid", "test"]:
        metrics = res[split_name]
        row = {
            "model": model_name,
            "split": split_name,
        }
        row.update(metrics)
        rows.append(row)

adv_metrics_df = pd.DataFrame(rows)

print("\n[INFO] Сводная таблица метрик ансамблей:")
display(adv_metrics_df)

adv_metrics_csv_path = os.path.join(REPORTS_DIR, "metrics_ensembles.csv")
adv_metrics_df.to_csv(adv_metrics_csv_path, index=False, encoding="utf-8-sig")

adv_metrics_json_path = os.path.join(REPORTS_DIR, "metrics_ensembles.json")
with open(adv_metrics_json_path, "w", encoding="utf-8") as f:
    json.dump(advanced_results, f, indent=2, ensure_ascii=False)

print("\n[INFO] Метрики ансамблей сохранены в:")
print("   CSV :", adv_metrics_csv_path)
print("   JSON:", adv_metrics_json_path)

def get_feature_names_from_preprocess(fitted_preprocess, numeric_features, categorical_features):
    """
    Возвращает список имён признаков после ColumnTransformer.
    """
    try:
        # sklearn >= 1.0
        return fitted_preprocess.get_feature_names_out().tolist()
    except AttributeError:
        # ручной fallback
        num_names = numeric_features
        cat_ohe = []
        if "cat" in fitted_preprocess.named_transformers_:
            ohe = fitted_preprocess.named_transformers_["cat"]["onehot"]
            cat_ohe = ohe.get_feature_names_out(categorical_features).tolist()
        return num_names + cat_ohe

rf_pre = rf_best_pipeline.named_steps["preprocess"]
rf_clf = rf_best_pipeline.named_steps["clf"]

rf_feature_names = get_feature_names_from_preprocess(
    rf_pre, NUMERIC_FEATURES, CATEGORICAL_FEATURES
)

rf_importances = rf_clf.feature_importances_
rf_imp_df = pd.DataFrame(
    {"feature": rf_feature_names, "importance": rf_importances}
).sort_values("importance", ascending=False)

rf_imp_csv_path = os.path.join(REPORTS_DIR, "feature_importances_randomforest.csv")
rf_imp_df.to_csv(rf_imp_csv_path, index=False, encoding="utf-8-sig")
print("\n[INFO] Важности признаков RandomForest сохранены в:", rf_imp_csv_path)
display(rf_imp_df.head(20))

# График топ-20 признаков
top_k = 20
rf_top = rf_imp_df.head(top_k).iloc[::-1]  # для красивой ориентации по вертикали

fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(rf_top["feature"], rf_top["importance"])
ax.set_title("RandomForest: топ-20 признаков по важности")
ax.set_xlabel("Важность (feature_importances_)")
plt.tight_layout()

rf_imp_plot_path = os.path.join(PLOTS_DIR, "rf_feature_importances_top20.png")
plt.savefig(rf_imp_plot_path, dpi=150)
plt.show()
plt.close()

print("[INFO] График важностей RandomForest сохранён в:", rf_imp_plot_path)

xgb_pre = xgb_best_pipeline.named_steps["preprocess"]
xgb_clf = xgb_best_pipeline.named_steps["clf"]

xgb_feature_names = get_feature_names_from_preprocess(
    xgb_pre, NUMERIC_FEATURES, CATEGORICAL_FEATURES
)

xgb_importances = xgb_clf.feature_importances_
xgb_imp_df = pd.DataFrame(
    {"feature": xgb_feature_names, "importance": xgb_importances}
).sort_values("importance", ascending=False)

xgb_imp_csv_path = os.path.join(REPORTS_DIR, "feature_importances_xgboost.csv")
xgb_imp_df.to_csv(xgb_imp_csv_path, index=False, encoding="utf-8-sig")
print("\n[INFO] Важности признаков XGBoost сохранены в:", xgb_imp_csv_path)
display(xgb_imp_df.head(20))

xgb_top = xgb_imp_df.head(top_k).iloc[::-1]

fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(xgb_top["feature"], xgb_top["importance"])
ax.set_title("XGBoost: топ-20 признаков по важности")
ax.set_xlabel("Важность (feature_importances_)")
plt.tight_layout()

xgb_imp_plot_path = os.path.join(PLOTS_DIR, "xgb_feature_importances_top20.png")
plt.savefig(xgb_imp_plot_path, dpi=150)
plt.show()
plt.close()

print("[INFO] График важностей XGBoost сохранён в:", xgb_imp_plot_path)

# Будем смотреть зависимости для нескольких числовых признаков,
# которые точно есть в исходном датасете
pdp_features_candidates = [
    "AdSpend",
    "ConversionRate",
    "WebsiteVisits",
    "PreviousPurchases",
]

pdp_features = [f for f in pdp_features_candidates if f in X_train.columns]
print("\n[INFO] Признаки для Partial Dependence:", pdp_features)

os.makedirs(PLOTS_DIR, exist_ok=True)

for feat in pdp_features:
    fig, ax = plt.subplots(figsize=(6, 4))
    PartialDependenceDisplay.from_estimator(
        rf_best_pipeline,
        X_train,
        features=[feat],
        kind="average",
        ax=ax,
    )
    ax.set_title(f"Partial Dependence: {feat} (RandomForest)")
    plt.tight_layout()

    pdp_path = os.path.join(PLOTS_DIR, f"pdp_randomforest_{feat}.png")
    plt.savefig(pdp_path, dpi=150)
    plt.show()
    plt.close()

    print(f"[INFO] Partial dependence для признака {feat} сохранён в:", pdp_path)

import os
import numpy as np
import pandas as pd

from sklearn.base import clone

# При необходимости можно доустановить TensorFlow (в Colab обычно уже есть)
try:
    import tensorflow as tf
except ImportError:
    !pip install -q tensorflow
    import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

print("[INFO] Версия TensorFlow:", tf.__version__)

# Клоним препроцессор, чтобы отдельно фитить его под DL,
# не ломая уже обученные пайплайны sklearn
preprocess_dl = clone(preprocess)

# Фитим на обучающей выборке и трансформируем все сплиты
X_train_dl = preprocess_dl.fit_transform(X_train)
X_valid_dl = preprocess_dl.transform(X_valid)
X_test_dl = preprocess_dl.transform(X_test)

y_train_dl = y_train.values.astype("float32")
y_valid_dl = y_valid.values.astype("float32")
y_test_dl = y_test.values.astype("float32")

print("\n[SHAPE] Формы массивов для DL:")
print("   X_train_dl:", X_train_dl.shape, " | y_train_dl:", y_train_dl.shape)
print("   X_valid_dl:", X_valid_dl.shape, " | y_valid_dl:", y_valid_dl.shape)
print("   X_test_dl :", X_test_dl.shape,  " | y_test_dl :", y_test_dl.shape)

input_dim = X_train_dl.shape[1]
print("\n[INFO] Размер входного слоя MLP:", input_dim)

# Сохраним описание feature-space именно для DL-ветки
from sklearn.preprocessing import OneHotEncoder, RobustScaler

def get_feature_names_from_preprocess(fitted_preprocess, numeric_features, categorical_features):
    try:
        return fitted_preprocess.get_feature_names_out().tolist()
    except Exception:
        num_names = numeric_features
        cat_names = []
        if "cat" in fitted_preprocess.named_transformers_:
            ohe = fitted_preprocess.named_transformers_["cat"]["onehot"]
            cat_names = ohe.get_feature_names_out(categorical_features).tolist()
        return num_names + cat_names

dl_feature_names = get_feature_names_from_preprocess(
    preprocess_dl, NUMERIC_FEATURES, CATEGORICAL_FEATURES
)

dl_feature_space = {
    "input_dim": int(input_dim),
    "feature_names": dl_feature_names,
}

dl_feature_space_path = os.path.join(DATA_DIR, "dl_feature_space.json")
with open(dl_feature_space_path, "w", encoding="utf-8") as f:
    json.dump(dl_feature_space, f, indent=2, ensure_ascii=False)

print("[INFO] DL feature_space сохранён в:", dl_feature_space_path)

def build_mlp_model(
    input_dim: int,
    hidden_units: list[int] = [128, 64, 32],
    dropout_rate: float = 0.3,
) -> keras.Model:
    inputs = keras.Input(shape=(input_dim,), name="features")
    x = inputs

    for i, units in enumerate(hidden_units):
        x = layers.Dense(units, activation=None, name=f"dense_{i+1}")(x)
        x = layers.BatchNormalization(name=f"bn_{i+1}")(x)
        x = layers.Activation("relu", name=f"relu_{i+1}")(x)
        x = layers.Dropout(dropout_rate, name=f"dropout_{i+1}")(x)

    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="mlp_conversion")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


mlp_model = build_mlp_model(input_dim=input_dim)

mlp_model.summary()

BATCH_SIZE = 256
EPOCHS = 100

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_auc",
        patience=10,
        mode="max",
        restore_best_weights=True,
        verbose=1,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_auc",
        factor=0.5,
        patience=5,
        mode="max",
        verbose=1,
        min_lr=1e-5,
    ),
]

history = mlp_model.fit(
    X_train_dl,
    y_train_dl,
    validation_data=(X_valid_dl, y_valid_dl),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=2,
    callbacks=callbacks,
)

# --- графики истории обучения ---
import matplotlib.pyplot as plt

history_dict = history.history

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Потери
axes[0].plot(history_dict["loss"], label="train")
axes[0].plot(history_dict["val_loss"], label="valid")
axes[0].set_title("MLP: динамика функции потерь")
axes[0].set_xlabel("Эпоха")
axes[0].set_ylabel("Loss")
axes[0].legend()

# AUC
axes[1].plot(history_dict["auc"], label="train")
axes[1].plot(history_dict["val_auc"], label="valid")
axes[1].set_title("MLP: динамика AUC")
axes[1].set_xlabel("Эпоха")
axes[1].set_ylabel("AUC")
axes[1].legend()

plt.tight_layout()
mlp_history_plot_path = os.path.join(PLOTS_DIR, "mlp_training_history.png")
plt.savefig(mlp_history_plot_path, dpi=150)
plt.show()
plt.close()

print("[INFO] График истории обучения MLP сохранён в:", mlp_history_plot_path)

def evaluate_mlp_model(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    # Прогнозы вероятностей
    proba_train = model.predict(X_train).ravel()
    proba_valid = model.predict(X_valid).ravel()
    proba_test = model.predict(X_test).ravel()

    metrics_train = _compute_basic_metrics(y_train, proba_train, threshold)
    metrics_valid = _compute_basic_metrics(y_valid, proba_valid, threshold)
    metrics_test = _compute_basic_metrics(y_test, proba_test, threshold)

    print("\n[METRICS] MLP — validation:")
    for k, v in metrics_valid.items():
        if k in ["tn", "fp", "fn", "tp", "support_neg", "support_pos"]:
            continue
        print(f"   {k:10s}: {v:.4f}" if isinstance(v, float) else f"   {k:10s}: {v}")

    # ROC/PR-кривые на тесте
    plot_paths = plot_roc_pr_curves(
        y_true=y_test,
        y_proba=proba_test,
        model_name="MLP",
        split_name="test",
        plots_dir=PLOTS_DIR,
    )

    return {
        "train": metrics_train,
        "valid": metrics_valid,
        "test": metrics_test,
        "plots": plot_paths,
    }


dl_results = {}
dl_results["MLP"] = evaluate_mlp_model(
    model=mlp_model,
    X_train=X_train_dl,
    y_train=y_train_dl,
    X_valid=X_valid_dl,
    y_valid=y_valid_dl,
    X_test=X_test_dl,
    y_test=y_test_dl,
    threshold=0.5,
)

# --- сводная таблица и сохранение ---
rows = []
for model_name, res in dl_results.items():
    for split_name in ["train", "valid", "test"]:
        metrics = res[split_name]
        row = {
            "model": model_name,
            "split": split_name,
        }
        row.update(metrics)
        rows.append(row)

dl_metrics_df = pd.DataFrame(rows)
print("\n[INFO] Метрики MLP:")
display(dl_metrics_df)

dl_metrics_csv_path = os.path.join(REPORTS_DIR, "metrics_dl_mlp.csv")
dl_metrics_df.to_csv(dl_metrics_csv_path, index=False, encoding="utf-8-sig")

dl_metrics_json_path = os.path.join(REPORTS_DIR, "metrics_dl_mlp.json")
with open(dl_metrics_json_path, "w", encoding="utf-8") as f:
    json.dump(dl_results, f, indent=2, ensure_ascii=False)

print("\n[INFO] Метрики MLP сохранены в:")
print("   CSV :", dl_metrics_csv_path)
print("   JSON:", dl_metrics_json_path)

from sklearn.metrics import average_precision_score

def permutation_importance_mlp(
    model: keras.Model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_repeats: int = 5,
) -> pd.DataFrame:
    """
    Простая пермутационная важность: считаем падение PR-AUC при перемешивании
    каждого признака.
    """
    rng = np.random.RandomState(42)

    # базовый PR-AUC
    base_proba = model.predict(X).ravel()
    base_score = average_precision_score(y, base_proba)

    importances = []
    n_features = X.shape[1]

    for j in range(n_features):
        scores = []
        for r in range(n_repeats):
            X_permuted = X.copy()
            perm = rng.permutation(X_permuted.shape[0])
            X_permuted[:, j] = X_permuted[perm, j]
            proba_perm = model.predict(X_permuted, verbose=0).ravel()
            score_perm = average_precision_score(y, proba_perm)
            scores.append(score_perm)

        mean_score_perm = float(np.mean(scores))
        importance = base_score - mean_score_perm
        importances.append(importance)

        if (j + 1) % 10 == 0 or j == n_features - 1:
            print(f"[PI] Обработано признаков: {j+1}/{n_features}")

    imp_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)

    return imp_df


print("\n[INFO] Расчёт пермутационной важности для MLP (по валидационной выборке)...")
mlp_pi_df = permutation_importance_mlp(
    model=mlp_model,
    X=X_valid_dl,
    y=y_valid_dl,
    feature_names=dl_feature_names,
    n_repeats=5,
)

mlp_pi_csv_path = os.path.join(REPORTS_DIR, "feature_permutation_importance_mlp.csv")
mlp_pi_df.to_csv(mlp_pi_csv_path, index=False, encoding="utf-8-sig")
print("[INFO] Пермутационная важность MLP сохранена в:", mlp_pi_csv_path)

display(mlp_pi_df.head(20))

# График топ-20 признаков
top_k = 20
mlp_top = mlp_pi_df.head(top_k).iloc[::-1]

fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(mlp_top["feature"], mlp_top["importance"])
ax.set_title("MLP: топ-20 признаков по пермутационной важности (PR-AUC)")
ax.set_xlabel("Падение PR-AUC при перемешивании признака")
plt.tight_layout()

mlp_pi_plot_path = os.path.join(PLOTS_DIR, "mlp_permutation_importance_top20.png")
plt.savefig(mlp_pi_plot_path, dpi=150)
plt.show()
plt.close()

print("[INFO] График пермутационной важности MLP сохранён в:", mlp_pi_plot_path)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os

plt.style.use("default")

# Чемпионская модель — RandomForest (можно потом поменять вручную)
champion_name = "RandomForest"
champion_pipeline = rf_best_pipeline  # из этапа 4

print(f"[INFO] Чемпионская модель для анализа порогов: {champion_name}")

# Вероятности класса 1 на валид и тест
proba_valid_rf = champion_pipeline.predict_proba(X_valid)[:, 1]
proba_test_rf = champion_pipeline.predict_proba(X_test)[:, 1]

y_valid_np = y_valid.values
y_test_np = y_test.values

print("[INFO] Примеры вероятностей (valid):", proba_valid_rf[:5])
print("[INFO] Примеры вероятностей (test) :", proba_test_rf[:5])

def compute_threshold_curve(y_true: np.ndarray, y_proba: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    rows = []
    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        pos_rate = y_pred.mean()
        rows.append(
            {
                "threshold": float(thr),
                "accuracy": acc,
                "f1": f1,
                "precision": prec,
                "recall": rec,
                "positive_rate": pos_rate,
            }
        )
    return pd.DataFrame(rows)


thresholds = np.linspace(0.1, 0.9, 81)  # 0.10, 0.11, ..., 0.90

curve_valid = compute_threshold_curve(y_valid_np, proba_valid_rf, thresholds)
curve_test = compute_threshold_curve(y_test_np, proba_test_rf, thresholds)

print("\n[INFO] Пример таблицы по порогам (validation):")
display(curve_valid.head())

# --- графики для валидации ---
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

ax[0].plot(curve_valid["threshold"], curve_valid["f1"], label="F1")
ax[0].plot(curve_valid["threshold"], curve_valid["precision"], label="Precision")
ax[0].plot(curve_valid["threshold"], curve_valid["recall"], label="Recall")
ax[0].set_xlabel("Порог отсечения")
ax[0].set_ylabel("Значение метрики")
ax[0].set_title("RandomForest (validation): метрики vs порог")
ax[0].legend()

ax[1].plot(curve_valid["threshold"], curve_valid["positive_rate"])
ax[1].set_xlabel("Порог отсечения")
ax[1].set_ylabel("Доля выбранных клиентов")
ax[1].set_title("RandomForest (validation): positive_rate vs порог")

plt.tight_layout()
thr_curve_plot_path = os.path.join(PLOTS_DIR, "rf_threshold_metrics_valid.png")
plt.savefig(thr_curve_plot_path, dpi=150)
plt.show()
plt.close()

print("[INFO] График метрик по порогу сохранён в:", thr_curve_plot_path)

# --- выбор «оптимальных» порогов по validation ---
# 1) Максимальный F1
idx_best_f1 = curve_valid["f1"].idxmax()
thr_best_f1 = float(curve_valid.loc[idx_best_f1, "threshold"])

# 2) Высокий recall (>= 0.98), максимальный precision
high_recall_mask = curve_valid["recall"] >= 0.98
if high_recall_mask.any():
    high_recall_df = curve_valid[high_recall_mask]
    idx_high_rec = high_recall_df["precision"].idxmax()
    thr_high_recall = float(curve_valid.loc[idx_high_rec, "threshold"])
else:
    thr_high_recall = float(thr_best_f1)

# 3) Высокий precision (>= 0.97), максимальный recall
high_precision_mask = curve_valid["precision"] >= 0.97
if high_precision_mask.any():
    high_precision_df = curve_valid[high_precision_mask]
    idx_high_prec = high_precision_df["recall"].idxmax()
    thr_high_precision = float(curve_valid.loc[idx_high_prec, "threshold"])
else:
    thr_high_precision = float(thr_best_f1)

threshold_scenarios = {
    "best_f1": thr_best_f1,
    "high_recall": thr_high_recall,
    "high_precision": thr_high_precision,
    "default_0_5": 0.5,
}

print("\n[INFO] Выбранные сценарные пороги (по validation):")
for name, thr in threshold_scenarios.items():
    print(f"   {name:>14s}: {thr:.3f}")

# Сохраним кривые и пороги
curve_valid_path = os.path.join(REPORTS_DIR, "rf_threshold_curve_valid.csv")
curve_test_path = os.path.join(REPORTS_DIR, "rf_threshold_curve_test.csv")
curve_valid.to_csv(curve_valid_path, index=False, encoding="utf-8-sig")
curve_test.to_csv(curve_test_path, index=False, encoding="utf-8-sig")

thr_scenarios_path = os.path.join(REPORTS_DIR, "rf_threshold_scenarios.json")
with open(thr_scenarios_path, "w", encoding="utf-8") as f:
    json.dump(threshold_scenarios, f, indent=2, ensure_ascii=False)

print("\n[INFO] Кривая метрик и сценарные пороги сохранены в:")
print("   curve_valid:", curve_valid_path)
print("   curve_test :", curve_test_path)
print("   scenarios  :", thr_scenarios_path)

def compute_lift_table(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Считает лифт и кумулятивные гейны по бинам (децили по умолчанию)."""
    df_tmp = pd.DataFrame({"y_true": y_true, "score": y_proba})
    df_tmp = df_tmp.sort_values("score", ascending=False).reset_index(drop=True)

    df_tmp["bin"] = pd.qcut(df_tmp.index, q=n_bins, labels=False)  # 0..n_bins-1

    total_positives = df_tmp["y_true"].sum()
    total_count = len(df_tmp)
    base_rate = total_positives / total_count

    groups = []
    cum_pos = 0
    cum_cnt = 0

    for b in range(n_bins):
        g = df_tmp[df_tmp["bin"] == b]
        cnt = len(g)
        pos = g["y_true"].sum()
        rate = pos / cnt if cnt > 0 else 0.0
        lift = rate / base_rate if base_rate > 0 else np.nan

        cum_cnt += cnt
        cum_pos += pos

        cum_rate = cum_pos / cum_cnt if cum_cnt > 0 else 0.0
        cum_gain = cum_pos / total_positives if total_positives > 0 else 0.0
        pop_share = cum_cnt / total_count

        groups.append(
            {
                "bin": b + 1,  # 1..n_bins
                "count": cnt,
                "positives": pos,
                "rate": rate,
                "lift": lift,
                "cum_count": cum_cnt,
                "cum_positives": cum_pos,
                "cum_rate": cum_rate,
                "cum_gain": cum_gain,
                "population_share": pop_share,
            }
        )

    return pd.DataFrame(groups), base_rate


lift_df, base_rate_test = compute_lift_table(y_test_np, proba_test_rf, n_bins=10)

print(f"\n[INFO] Базовая доля положительного класса на тесте: {base_rate_test:.4f}")
print("[INFO] Таблица лифта (первые строки):")
display(lift_df.head())

# --- график лифта по децилам ---
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(lift_df["bin"], lift_df["lift"])
ax.set_xlabel("Дециль (1 = топ по скору)")
ax.set_ylabel("Lift (rate / base_rate)")
ax.set_title("RandomForest: лифт по децилам (test)")
plt.xticks(lift_df["bin"])
plt.tight_layout()

lift_plot_path = os.path.join(PLOTS_DIR, "rf_lift_chart_test.png")
plt.savefig(lift_plot_path, dpi=150)
plt.show()
plt.close()

print("[INFO] График лифта сохранён в:", lift_plot_path)

# --- график кумулятивных гейнов ---
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(lift_df["population_share"], lift_df["cum_gain"], marker="o", label="Модель")
ax.plot([0, 1], [0, 1], linestyle="--", label="Случайный выбор")
ax.set_xlabel("Доля отсортированной популяции")
ax.set_ylabel("Кумулятивная доля конверсий")
ax.set_title("RandomForest: кумулятивные гейны (test)")
ax.legend()
plt.tight_layout()

gains_plot_path = os.path.join(PLOTS_DIR, "rf_gains_chart_test.png")
plt.savefig(gains_plot_path, dpi=150)
plt.show()
plt.close()

print("[INFO] График кумулятивных гейнов сохранён в:", gains_plot_path)

# Сохраняем таблицу лифта
lift_csv_path = os.path.join(REPORTS_DIR, "rf_lift_table_test.csv")
lift_df.to_csv(lift_csv_path, index=False, encoding="utf-8-sig")
print("[INFO] Таблица лифта сохранена в:", lift_csv_path)

# Возьмём исходные признаки для тестовой выборки (чтобы получить AdSpend и др.)
df_test_features = df_fe.loc[X_test.index].copy()

def simulate_roi(
    df_subset: pd.DataFrame,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    cost_col: str = "AdSpend",
    revenue_per_conversion: float = None,
) -> dict:
    """
    Простая модель ROI:
    - таргетируем всех клиентов, у кого score >= threshold;
    - стоимость контакта = значение cost_col;
    - выручка за конверсию = revenue_per_conversion (если None, берём медианное значение cost_col).
    """
    if revenue_per_conversion is None:
        revenue_per_conversion = float(df_subset[cost_col].median())

    mask_target = y_proba >= threshold
    n_target = int(mask_target.sum())
    if n_target == 0:
        return {
            "threshold": threshold,
            "n_targeted": 0,
            "n_conversions": 0,
            "total_cost": 0.0,
            "total_revenue": 0.0,
            "net_gain": 0.0,
            "roi": np.nan,
        }

    costs = df_subset.loc[mask_target, cost_col].values
    total_cost = float(costs.sum())

    n_conv = int(((y_true == 1) & mask_target).sum())
    total_revenue = n_conv * revenue_per_conversion
    net_gain = total_revenue - total_cost
    roi = net_gain / total_cost if total_cost > 0 else np.nan

    return {
        "threshold": threshold,
        "n_targeted": n_target,
        "n_conversions": n_conv,
        "total_cost": total_cost,
        "revenue_per_conversion": revenue_per_conversion,
        "total_revenue": total_revenue,
        "net_gain": net_gain,
        "roi": roi,
    }


# Посчитаем ROI для нескольких сценариев на тестовой выборке
roi_rows = []
for name, thr in threshold_scenarios.items():
    res = simulate_roi(
        df_subset=df_test_features,
        y_true=y_test_np,
        y_proba=proba_test_rf,
        threshold=thr,
        cost_col="AdSpend",
        revenue_per_conversion=None,  # возьмём медианный AdSpend как "выручку"
    )
    res["scenario"] = name
    roi_rows.append(res)

roi_df = pd.DataFrame(roi_rows)[
    [
        "scenario",
        "threshold",
        "n_targeted",
        "n_conversions",
        "total_cost",
        "revenue_per_conversion",
        "total_revenue",
        "net_gain",
        "roi",
    ]
]

print("\n[INFO] Результаты ROI-модели по сценарным порогам (test):")
display(roi_df)

roi_csv_path = os.path.join(REPORTS_DIR, "rf_roi_test_scenarios.csv")
roi_df.to_csv(roi_csv_path, index=False, encoding="utf-8-sig")
print("[INFO] Таблица ROI сохранена в:", roi_csv_path)

import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans

plt.style.use("default")

# Признаковое пространство, с которым обучался RF
feature_cols_for_model = NUMERIC_FEATURES + CATEGORICAL_FEATURES

X_all = df_fe[feature_cols_for_model].copy()
y_all = df_fe[TARGET_COL].astype(int).values

# Скоринг по чемпионской модели
rf_scores_all = rf_best_pipeline.predict_proba(X_all)[:, 1]
df_fe["rf_score"] = rf_scores_all

print("[INFO] Примеры скоринговых баллов RandomForest:")
print(df_fe[["CustomerID", "rf_score", TARGET_COL]].head())

# qcut даёт бины по возрастанию; нам нужен 1-й дециль = самые высокие score.
# Сначала считаем дециль по возрастанию, затем инвертируем.
df_fe["rf_decile_asc"] = pd.qcut(df_fe["rf_score"], q=10, labels=False)
df_fe["rf_decile"] = 10 - df_fe["rf_decile_asc"]  # 1 = самый высокий скор, 10 = самый низкий
df_fe["rf_decile"] = df_fe["rf_decile"].astype(int)

# Укрупнённые сегменты приоритета
def map_priority(decile: int) -> str:
    if decile <= 3:
        return "High"    # топ-30 % по скору
    elif decile <= 7:
        return "Medium"  # середина
    else:
        return "Low"     # нижние 30 %

df_fe["priority_segment"] = df_fe["rf_decile"].apply(map_priority)

# Сводка по децилям
decile_summary = (
    df_fe
    .groupby("rf_decile")
    .agg(
        n_clients=("CustomerID", "size"),
        conversion_rate=(TARGET_COL, "mean"),
        avg_adspend=("AdSpend", "mean"),
        avg_engagement=("EngagementScore", "mean"),
        avg_prev_purchases=("PreviousPurchases", "mean"),
    )
    .sort_index()
)

print("\n[INFO] Сводка по децильной сегментации (1 = самые горячие):")
display(decile_summary)

# Сводка по укрупнённым сегментам
priority_summary = (
    df_fe
    .groupby("priority_segment")
    .agg(
        n_clients=("CustomerID", "size"),
        conversion_rate=(TARGET_COL, "mean"),
        avg_adspend=("AdSpend", "mean"),
        avg_engagement=("EngagementScore", "mean"),
        avg_prev_purchases=("PreviousPurchases", "mean"),
    )
    .sort_values("conversion_rate", ascending=False)
)

print("\n[INFO] Сводка по укрупнённым сегментам приоритета:")
display(priority_summary)

# --- графики конверсии по децилям ---
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(decile_summary.index, decile_summary["conversion_rate"])
ax.set_xlabel("Дециль по скору (1 = самый высокий)")
ax.set_ylabel("Конверсия")
ax.set_title("Конверсия по децилям скоринга RandomForest")
plt.xticks(decile_summary.index)
plt.tight_layout()

decile_plot_path = os.path.join(PLOTS_DIR, "rf_decile_conversion.png")
plt.savefig(decile_plot_path, dpi=150)
plt.show()
plt.close()
print("[INFO] График конверсии по децилям сохранён в:", decile_plot_path)

# --- сохранение таблиц сегментации ---
segmentation_cols = [
    ID_COL,
    TARGET_COL,
    "rf_score",
    "rf_decile",
    "priority_segment",
    "AdSpend",
    "ConversionRate",
    "WebsiteVisits",
    "PreviousPurchases",
    "LoyaltyPoints",
    "EngagementScore",
]

segmentation_path = os.path.join(DATA_DIR, "client_segmentation_deciles.csv")
df_fe[segmentation_cols].to_csv(segmentation_path, index=False, encoding="utf-8-sig")

decile_summary_path = os.path.join(REPORTS_DIR, "rf_decile_summary.csv")
priority_summary_path = os.path.join(REPORTS_DIR, "rf_priority_segment_summary.csv")

decile_summary.to_csv(decile_summary_path, encoding="utf-8-sig")
priority_summary.to_csv(priority_summary_path, encoding="utf-8-sig")

print("[INFO] Децильная сегментация сохранена в:", segmentation_path)
print("[INFO] Сводка по децилям сохранена в:", decile_summary_path)
print("[INFO] Сводка по приоритетным сегментам сохранена в:", priority_summary_path)

# Выбираем несколько ключевых числовых признаков для кластеризации.
cluster_features = [
    "EngagementScore",
    "AdSpend",
    "ConversionRate",
    "WebsiteVisits",
    "PagesPerVisit",
    "TimeOnSite",
    "PreviousPurchases",
    "LoyaltyPoints",
]

missing_features = [f for f in cluster_features if f not in df_fe.columns]
if missing_features:
    print("[WARN] В df_fe отсутствуют признаки для кластеризации:", missing_features)

cluster_features = [f for f in cluster_features if f in df_fe.columns]

X_cluster_raw = df_fe[cluster_features].values

# Масштабируем для корректной работы KMeans
cluster_scaler = RobustScaler()
X_cluster = cluster_scaler.fit_transform(X_cluster_raw)

# Количество кластеров можно варьировать; для практики возьмём 4
N_CLUSTERS = 4

kmeans = KMeans(
    n_clusters=N_CLUSTERS,
    random_state=42,
    n_init=20,
)

cluster_labels = kmeans.fit_predict(X_cluster)
df_fe["cluster_kmeans"] = cluster_labels

print(f"\n[INFO] Кластеризация KMeans завершена, найдено кластеров: {N_CLUSTERS}")
print("Распределение клиентов по кластерам:")
print(df_fe["cluster_kmeans"].value_counts().sort_index())

# Сводка по кластерам
cluster_summary = (
    df_fe
    .groupby("cluster_kmeans")
    .agg(
        n_clients=("CustomerID", "size"),
        conversion_rate=(TARGET_COL, "mean"),
        avg_score=("rf_score", "mean"),
        avg_adspend=("AdSpend", "mean"),
        avg_engagement=("EngagementScore", "mean"),
        avg_prev_purchases=("PreviousPurchases", "mean"),
        avg_loyalty=("LoyaltyPoints", "mean"),
    )
    .sort_values("conversion_rate", ascending=False)
)

print("\n[INFO] Сводка по кластерам KMeans:")
display(cluster_summary)

# График: конверсия по кластерам
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(cluster_summary.index.astype(str), cluster_summary["conversion_rate"])
ax.set_xlabel("Кластер KMeans")
ax.set_ylabel("Конверсия")
ax.set_title("Конверсия по кластерам (KMeans)")
plt.tight_layout()

cluster_plot_path = os.path.join(PLOTS_DIR, "kmeans_cluster_conversion.png")
plt.savefig(cluster_plot_path, dpi=150)
plt.show()
plt.close()
print("[INFO] График конверсии по кластерам сохранён в:", cluster_plot_path)

# --- Сохранение результатов кластеризации ---
clusters_path = os.path.join(DATA_DIR, "client_clusters_kmeans.csv")
df_fe[[ID_COL, "cluster_kmeans"] + cluster_features].to_csv(
    clusters_path, index=False, encoding="utf-8-sig"
)

cluster_summary_path = os.path.join(REPORTS_DIR, "kmeans_cluster_summary.csv")
cluster_summary.to_csv(cluster_summary_path, encoding="utf-8-sig")

print("[INFO] Кластеры клиентов сохранены в:", clusters_path)
print("[INFO] Сводка по кластерам сохранена в:", cluster_summary_path)

# --- Дополнительно: сохраняем конфиг кластеризации для Flask ---
cluster_config = {
    "features": cluster_features,
    "n_clusters": int(N_CLUSTERS),
}

cluster_config_path = os.path.join(DATA_DIR, "kmeans_cluster_config.json")
with open(cluster_config_path, "w", encoding="utf-8") as f:
    json.dump(cluster_config, f, indent=2, ensure_ascii=False)

print("[INFO] Конфигурация кластеризации сохранена в:", cluster_config_path)

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

# Каталоги для сохранения артефактов
MODELS_DIR = os.path.join(BASE_DIR, "models")
CONFIGS_DIR = os.path.join(BASE_DIR, "configs")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CONFIGS_DIR, exist_ok=True)

print("[INFO] Каталоги для моделей и конфигов готовы:")
print("   MODELS_DIR :", MODELS_DIR)
print("   CONFIGS_DIR:", CONFIGS_DIR)

# Пути для моделей
RF_MODEL_PATH = os.path.join(MODELS_DIR, "rf_best_pipeline.joblib")
XGB_MODEL_PATH = os.path.join(MODELS_DIR, "xgb_best_pipeline.joblib")
PREPROCESS_DL_PATH = os.path.join(MODELS_DIR, "preprocess_dl.joblib")
CLUSTER_SCALER_PATH = os.path.join(MODELS_DIR, "cluster_scaler.joblib")
KMEANS_MODEL_PATH = os.path.join(MODELS_DIR, "kmeans_model.joblib")

# Чемпионская модель (RandomForest + preprocess)
joblib.dump(rf_best_pipeline, RF_MODEL_PATH)
print("[INFO] Сохранён пайплайн RandomForest:", RF_MODEL_PATH)

# XGBoost-пайплайн (для возможного сравнения в Flask)
joblib.dump(xgb_best_pipeline, XGB_MODEL_PATH)
print("[INFO] Сохранён пайплайн XGBoost:", XGB_MODEL_PATH)

# Препроцессор для DL (скалирование+one-hot до входа в MLP)
joblib.dump(preprocess_dl, PREPROCESS_DL_PATH)
print("[INFO] Сохранён preprocess_dl для MLP:", PREPROCESS_DL_PATH)

# Масштабирование и модель кластеризации
joblib.dump(cluster_scaler, CLUSTER_SCALER_PATH)
print("[INFO] Сохранён scaler для KMeans:", CLUSTER_SCALER_PATH)

joblib.dump(kmeans, KMEANS_MODEL_PATH)
print("[INFO] Сохранён KMeans-кластеризатор:", KMEANS_MODEL_PATH)

import tensorflow as tf

# сохраняем именно файл с расширением .keras
MLP_MODEL_PATH = os.path.join(MODELS_DIR, "mlp_conversion_model.keras")

mlp_model.save(MLP_MODEL_PATH, include_optimizer=True)
print("[INFO] MLP-модель сохранена в файле:", MLP_MODEL_PATH)

# Пути к уже сохранённым JSON/CSV-файлам из предыдущих этапов
WINSOR_CONFIG_PATH = os.path.join(DATA_DIR, "winsorization_config.json")
FEATURE_SPACE_PATH = os.path.join(DATA_DIR, "feature_space.json")
DL_FEATURE_SPACE_PATH = os.path.join(DATA_DIR, "dl_feature_space.json")
THRESHOLD_SCENARIOS_PATH = os.path.join(REPORTS_DIR, "rf_threshold_scenarios.json")
DECILE_SEGMENTATION_PATH = os.path.join(DATA_DIR, "client_segmentation_deciles.csv")
DECILE_SUMMARY_PATH = os.path.join(REPORTS_DIR, "rf_decile_summary.csv")
PRIORITY_SUMMARY_PATH = os.path.join(REPORTS_DIR, "rf_priority_segment_summary.csv")
CLUSTER_CONFIG_PATH = os.path.join(DATA_DIR, "kmeans_cluster_config.json")
CLUSTER_SUMMARY_PATH = os.path.join(REPORTS_DIR, "kmeans_cluster_summary.csv")
LIFT_TABLE_PATH = os.path.join(REPORTS_DIR, "rf_lift_table_test.csv")
ROI_TABLE_PATH = os.path.join(REPORTS_DIR, "rf_roi_test_scenarios.csv")

# Загружаем сценарные пороги (для выбора default_threshold)
with open(THRESHOLD_SCENARIOS_PATH, "r", encoding="utf-8") as f:
    threshold_scenarios = json.load(f)

default_threshold = float(threshold_scenarios.get("best_f1", 0.5))

# Краткое summary по метрикам чемпионской модели
metrics_ensembles_path = os.path.join(REPORTS_DIR, "metrics_ensembles.csv")
metrics_df = pd.read_csv(metrics_ensembles_path)

rf_test_metrics = (
    metrics_df[(metrics_df["model"] == "RandomForest") & (metrics_df["split"] == "test")]
    .iloc[0]
    .to_dict()
)

flask_serving_config = {
    "created_at": datetime.now().isoformat(),
    "project_name": "digital_marketing_conversion",
    "target_column": TARGET_COL,
    "id_column": ID_COL,
    "numeric_features": NUMERIC_FEATURES,
    "categorical_features": CATEGORICAL_FEATURES,
    "derived_features": DERIVED_FEATURES,
    "paths": {
        "models": {
            "random_forest_pipeline": RF_MODEL_PATH,
            "xgboost_pipeline": XGB_MODEL_PATH,
            "mlp_keras_model": MLP_MODEL_PATH,   # <-- обновили
            "preprocess_dl": PREPROCESS_DL_PATH,
            "cluster_scaler": CLUSTER_SCALER_PATH,
            "kmeans_model": KMEANS_MODEL_PATH,
        },
        "configs": {
            "winsorization": WINSOR_CONFIG_PATH,
            "feature_space_tabular": FEATURE_SPACE_PATH,
            "feature_space_dl": DL_FEATURE_SPACE_PATH,
            "threshold_scenarios": THRESHOLD_SCENARIOS_PATH,
            "kmeans_cluster": CLUSTER_CONFIG_PATH,
        },
        "reports": {
            "metrics_ensembles": metrics_ensembles_path,
            "rf_decile_summary": DECILE_SUMMARY_PATH,
            "rf_priority_summary": PRIORITY_SUMMARY_PATH,
            "kmeans_cluster_summary": CLUSTER_SUMMARY_PATH,
            "lift_table_test": LIFT_TABLE_PATH,
            "roi_scenarios_test": ROI_TABLE_PATH,
        },
        "data": {
            "client_segmentation_deciles": DECILE_SEGMENTATION_PATH,
            "client_clusters_kmeans": os.path.join(DATA_DIR, "client_clusters_kmeans.csv"),
        },
    },
    "serving": {
        "champion_model": "random_forest_pipeline",
        "default_threshold": default_threshold,
        "threshold_scenarios": threshold_scenarios,
        "priority_segment_column": "priority_segment",
        "priority_levels_order": ["High", "Medium", "Low"],
    },
    "champion_test_metrics": rf_test_metrics,
}

FLASK_CONFIG_PATH = os.path.join(CONFIGS_DIR, "flask_serving_config.json")
with open(FLASK_CONFIG_PATH, "w", encoding="utf-8") as f:
    json.dump(flask_serving_config, f, indent=2, ensure_ascii=False)

print("\n[INFO] Единый конфиг для Flask-приложения сохранён в:", FLASK_CONFIG_PATH)

# Короткий README о том, что где лежит
readme_lines = [
    "Артефакты эксперимента для Flask-приложения",
    "=========================================",
    "",
    f"Дата создания: {flask_serving_config['created_at']}",
    "",
    "Основное:",
    f"- Чемпионская модель: RandomForest (путь: {RF_MODEL_PATH})",
    f"- Default-порог классификации: {default_threshold:.3f}",
    f"- Конфиг сервинга: {FLASK_CONFIG_PATH}",
    "",
    "Дополнительно см. flask_serving_config.json для всех путей и настроек.",
]

README_PATH = os.path.join(BASE_DIR, "README_artifacts.txt")
with open(README_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(readme_lines))

print("[INFO] README по артефактам сохранён в:", README_PATH)

import shutil
from google.colab import files

# Создаём zip-архив всей папки проекта BASE_DIR
ARCHIVE_BASE = os.path.join(BASE_DIR, "digital_marketing_artifacts")
ARCHIVE_PATH = shutil.make_archive(
    base_name=ARCHIVE_BASE,
    format="zip",
    root_dir=BASE_DIR,
)

print("[INFO] Архив с артефактами создан:", ARCHIVE_PATH)

# В Colab появится кнопка скачивания
files.download(ARCHIVE_PATH)
