# ml_service.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import json
import joblib
import numpy as np
import pandas as pd


@dataclass
class MLConfig:
    """Простая обёртка вокруг конфигурации сервиса."""
    numeric_features: List[str]
    categorical_features: List[str]
    derived_features: List[str]
    cluster_features: List[str]
    default_threshold: float


class MLService:
    """
    Сервис для работы с обученными моделями:
    - загрузка артефактов (модель, scaler, kmeans);
    - расчёт вероятности конверсии;
    - присвоение дециля и приоритетного сегмента;
    - отнесение к кластеру KMeans.
    """

    # Значения по умолчанию (взяты из ноутбука эксперимента)
    DEFAULT_NUMERIC_FEATURES = [
        "Age",
        "Income",
        "AdSpend",
        "ClickThroughRate",
        "ConversionRate",
        "WebsiteVisits",
        "PagesPerVisit",
        "TimeOnSite",
        "SocialShares",
        "EmailOpens",
        "EmailClicks",
        "PreviousPurchases",
        "LoyaltyPoints",
        "EmailEngagementRate",
        "SocialShareRate",
        "CostPerVisit",
        "CostPerEmailClick",
        "LoyaltyPerPurchase",
        "EngagementScore",
    ]

    DEFAULT_CATEGORICAL_FEATURES = [
        "Gender",
        "CampaignChannel",
        "CampaignType",
    ]

    DEFAULT_DERIVED_FEATURES = [
        "EmailEngagementRate",
        "SocialShareRate",
        "CostPerVisit",
        "CostPerEmailClick",
        "LoyaltyPerPurchase",
        "EngagementScore",
    ]

    DEFAULT_CLUSTER_FEATURES = [
        "EngagementScore",
        "AdSpend",
        "ConversionRate",
        "WebsiteVisits",
        "PagesPerVisit",
        "TimeOnSite",
        "PreviousPurchases",
        "LoyaltyPoints",
    ]

    DEFAULT_THRESHOLD = 0.44  # порог из анализа по F1

    # Базовые колонки исходного датасета
    BASE_COLUMNS = [
        "CustomerID",
        "Age",
        "Gender",
        "Income",
        "CampaignChannel",
        "CampaignType",
        "AdSpend",
        "ClickThroughRate",
        "ConversionRate",
        "WebsiteVisits",
        "PagesPerVisit",
        "TimeOnSite",
        "SocialShares",
        "EmailOpens",
        "EmailClicks",
        "PreviousPurchases",
        "LoyaltyPoints",
    ]

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        config_path: Optional[Path] = None,
    ) -> None:
        # Корневая папка проекта (digital_marketing_app/)
        self.base_dir = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parent

        self.models_dir = self.base_dir / "models"
        self.configs_dir = self.base_dir / "configs"
        self.data_dir = self.base_dir / "data"

        if config_path is None:
            config_path = self.configs_dir / "flask_serving_config.json"
        self.config_path = Path(config_path)

        # Конфиг (фичи, порог и т.п.)
        self.config = self._load_or_build_config()

        # Пути к моделям и вспомогательным файлам (дефолт — локальные в ./models)
        self.rf_model_path = self.models_dir / "rf_best_pipeline.joblib"
        self.xgb_model_path = self.models_dir / "xgb_best_pipeline.joblib"
        self.mlp_model_path = self.models_dir / "mlp_conversion_model.keras"
        self.cluster_scaler_path = self.models_dir / "cluster_scaler.joblib"
        self.kmeans_model_path = self.models_dir / "kmeans_model.joblib"

        # Если в конфиге есть явные пути — используем их как "желательные"
        if self.config_path.exists():
            try:
                paths_cfg = self._raw_config.get("paths", {})
                models_cfg = paths_cfg.get("models", {})
                if "random_forest_pipeline" in models_cfg:
                    self.rf_model_path = Path(models_cfg["random_forest_pipeline"])
                if "xgboost_pipeline" in models_cfg:
                    self.xgb_model_path = Path(models_cfg["xgboost_pipeline"])
                if "mlp_keras_model" in models_cfg:
                    self.mlp_model_path = Path(models_cfg["mlp_keras_model"])
                if "cluster_scaler" in models_cfg:
                    self.cluster_scaler_path = Path(models_cfg["cluster_scaler"])
                if "kmeans_model" in models_cfg:
                    self.kmeans_model_path = Path(models_cfg["kmeans_model"])
            except Exception:
                # Если структура конфига неожиданная — тихо используем дефолты
                pass

        # Сегментация по децилям
        self.decile_thresholds: Dict[float, float] = {}
        self._load_decile_thresholds()

        # Модели
        self.rf_pipeline = None
        self.cluster_scaler = None
        self.kmeans_model = None

        self._load_models()

        # Текстовые описания для кластеров (можно менять под отчёт)
        self.cluster_descriptions = {
            0: "Кластер 0: умеренно вовлечённые клиенты со средним уровнем покупательской активности.",
            1: "Кластер 1: клиенты с пониженной вовлечённостью и средним уровнем лояльности.",
            2: "Кластер 2: высоко вовлечённые и лояльные клиенты (основной источник выручки).",
            3: "Кластер 3: активные клиенты со стабильной историей покупок.",
        }

    # ------------------------------------------------------------------ #
    # Загрузка и конфигурация
    # ------------------------------------------------------------------ #

    def _load_or_build_config(self) -> MLConfig:
        """Читает конфиг, если он есть, либо создаёт дефолтный."""
        numeric = list(self.DEFAULT_NUMERIC_FEATURES)
        categorical = list(self.DEFAULT_CATEGORICAL_FEATURES)
        derived = list(self.DEFAULT_DERIVED_FEATURES)
        cluster_features = list(self.DEFAULT_CLUSTER_FEATURES)
        threshold = float(self.DEFAULT_THRESHOLD)

        self._raw_config: Dict[str, Any] = {}

        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    self._raw_config = json.load(f)

                numeric = self._raw_config.get("numeric_features", numeric)
                categorical = self._raw_config.get("categorical_features", categorical)
                derived = self._raw_config.get("derived_features", derived)

                serving_cfg = self._raw_config.get("serving", {})
                threshold = float(serving_cfg.get("default_threshold", threshold))

                # Конфиг для кластеризации (список признаков)
                paths_cfg = self._raw_config.get("paths", {})
                configs_cfg = paths_cfg.get("configs", {})
                kmeans_cfg_path = configs_cfg.get("kmeans_cluster")
                if kmeans_cfg_path:
                    kk_path = Path(kmeans_cfg_path)
                    if not kk_path.is_absolute():
                        kk_path = self.base_dir / kk_path
                    if kk_path.exists():
                        with open(kk_path, "r", encoding="utf-8") as f:
                            kk = json.load(f)
                        cluster_features = kk.get("features", cluster_features)
            except Exception:
                # при любой ошибке остаёмся на дефолтных значениях
                pass

        return MLConfig(
            numeric_features=numeric,
            categorical_features=categorical,
            derived_features=derived,
            cluster_features=cluster_features,
            default_threshold=threshold,
        )

    # --- новый помощник для путей моделей --------------------------------

    def _resolve_model_path(self, configured_path: Path, default_name: str) -> Path:
        """
        Приводит путь к модели к реальному существующему файлу.
        Логика:
        1) Если путь относительный — считаем его относительно base_dir.
        2) Если путь абсолютный (например, /content/drive/...) и файл НЕ существует,
           пробуем взять ./models/<default_name>.
        3) Возвращаем путь (существующий или нет — дальше проверяем).
        """
        p = Path(configured_path)

        # Относительные пути → относительно проекта
        if not p.is_absolute():
            p = self.base_dir / p

        # Если файл не найден — пробуем локальный models/<default_name>
        if not p.exists():
            local = self.models_dir / default_name
            if local.exists():
                return local

        return p

    def _load_models(self) -> None:
        """Загружает пайплайн RandomForest, scaler и модель KMeans."""
        # Приводим пути к реальным моделям (или к тому, что есть)
        self.rf_model_path = self._resolve_model_path(self.rf_model_path, "rf_best_pipeline.joblib")
        self.cluster_scaler_path = self._resolve_model_path(self.cluster_scaler_path, "cluster_scaler.joblib")
        self.kmeans_model_path = self._resolve_model_path(self.kmeans_model_path, "kmeans_model.joblib")

        # RandomForest — обязательная модель
        if not self.rf_model_path.exists():
            raise FileNotFoundError(
                "Не найден файл модели RandomForest.\n"
                f"Ожидался по пути: {self.rf_model_path}\n"
                "Убедитесь, что из Colab/Google Drive скопирована папка models/ "
                "в каталог проекта digital_marketing_app/"
            )

        self.rf_pipeline = joblib.load(self.rf_model_path)

        # scaler и kmeans — опционально
        if self.cluster_scaler_path.exists():
            self.cluster_scaler = joblib.load(self.cluster_scaler_path)

        if self.kmeans_model_path.exists():
            self.kmeans_model = joblib.load(self.kmeans_model_path)

    def _load_decile_thresholds(self) -> None:
        """
        Загружает пороги децилей по rf_score.
        Сначала пробует путь из конфига, затем локальный ./data/client_segmentation_deciles.csv.
        Если ничего не найдено — использует равномерные пороги.
        """
        # Дефолтный локальный путь
        default_segmentation_path = self.data_dir / "client_segmentation_deciles.csv"
        segmentation_path = default_segmentation_path

        if self.config_path.exists():
            try:
                paths_cfg = self._raw_config.get("paths", {})
                data_cfg = paths_cfg.get("data", {})
                seg_path_cfg = data_cfg.get("client_segmentation_deciles")
                if seg_path_cfg:
                    p = Path(seg_path_cfg)
                    if not p.is_absolute():
                        p = self.base_dir / p
                    segmentation_path = p
            except Exception:
                pass

        # Если путь из конфига не существует, пробуем локальный
        if not segmentation_path.exists() and default_segmentation_path.exists():
            segmentation_path = default_segmentation_path

        if segmentation_path.exists():
            df_seg = pd.read_csv(segmentation_path)
            if "rf_score" in df_seg.columns:
                qs = df_seg["rf_score"].quantile(
                    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                )
                self.decile_thresholds = qs.to_dict()
                return

        # Если ничего не нашли — равномерные пороги 0.1, 0.2, ..., 0.9
        self.decile_thresholds = {
            q: q for q in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        }

    # ------------------------------------------------------------------ #
    # Подготовка признаков
    # ------------------------------------------------------------------ #

    def _ensure_base_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Гарантирует наличие всех базовых колонок.
        Числовые — заполняем нулями, категориальные — пустой строкой.
        """
        df = df.copy()

        for col in self.BASE_COLUMNS:
            if col not in df.columns:
                if col in ["Gender", "CampaignChannel", "CampaignType"]:
                    df[col] = ""
                else:
                    df[col] = 0

        return df

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Добавляет производные признаки. Формулы совпадают по смыслу с ноутбуком.
        """
        df = df.copy()

        # Защита от деления на ноль
        def safe_div(num, den):
            return np.where(den > 0, num / den, 0.0)

        df["EmailEngagementRate"] = safe_div(df["EmailClicks"], df["EmailOpens"])
        df["SocialShareRate"] = safe_div(df["SocialShares"], df["WebsiteVisits"])
        df["CostPerVisit"] = safe_div(df["AdSpend"], df["WebsiteVisits"])
        df["CostPerEmailClick"] = safe_div(df["AdSpend"], df["EmailClicks"])
        df["LoyaltyPerPurchase"] = safe_div(df["LoyaltyPoints"], df["PreviousPurchases"])

        # Простейшая агрегированная метрика вовлечённости
        engagement_raw = (
            df["ClickThroughRate"].fillna(0)
            + df["EmailEngagementRate"].fillna(0)
            + (df["TimeOnSite"].fillna(0) / (df["TimeOnSite"].max() or 1))
            + (df["PagesPerVisit"].fillna(0) / (df["PagesPerVisit"].max() or 1))
        )
        max_val = engagement_raw.max()
        if max_val > 0:
            df["EngagementScore"] = engagement_raw / max_val
        else:
            df["EngagementScore"] = 0.0

        return df

    def _prepare_df(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Полная подготовка DataFrame:
        - добиваем базовые колонки;
        - добавляем производные признаки.
        """
        df = self._ensure_base_columns(df_raw)
        df = self._add_derived_features(df)
        return df

    # ------------------------------------------------------------------ #
    # Вспомогательные функции для сегментации
    # ------------------------------------------------------------------ #

    def _assign_deciles(self, scores: pd.Series) -> pd.Series:
        """
        Присваивает каждому скору дециль 1..10.
        1 — самая высокая вероятность, 10 — самая низкая.
        """
        qs = self.decile_thresholds
        result = []

        for s in scores:
            # если пороги не заданы, просто ставим 1
            if not qs:
                result.append(1)
                continue

            if s >= qs.get(0.9, 0.9):
                dec = 1
            elif s >= qs.get(0.8, 0.8):
                dec = 2
            elif s >= qs.get(0.7, 0.7):
                dec = 3
            elif s >= qs.get(0.6, 0.6):
                dec = 4
            elif s >= qs.get(0.5, 0.5):
                dec = 5
            elif s >= qs.get(0.4, 0.4):
                dec = 6
            elif s >= qs.get(0.3, 0.3):
                dec = 7
            elif s >= qs.get(0.2, 0.2):
                dec = 8
            elif s >= qs.get(0.1, 0.1):
                dec = 9
            else:
                dec = 10
            result.append(dec)

        return pd.Series(result, index=scores.index, dtype=int)

    @staticmethod
    def _map_priority_segment(decile: int) -> str:
        """High / Medium / Low в зависимости от дециля."""
        if decile <= 3:
            return "High"
        elif decile <= 7:
            return "Medium"
        else:
            return "Low"

    # ------------------------------------------------------------------ #
    # Кластеризация
    # ------------------------------------------------------------------ #

    def _predict_clusters(self, df_prepared: pd.DataFrame) -> np.ndarray:
        """
        Возвращает номера кластеров KMeans для каждой строки df_prepared.
        Если моделей нет, возвращает нули.
        """
        if self.cluster_scaler is None or self.kmeans_model is None:
            return np.zeros(len(df_prepared), dtype=int)

        # гарантируем наличие всех признаков для кластеризации
        for col in self.config.cluster_features:
            if col not in df_prepared.columns:
                df_prepared[col] = 0.0

        X_clust = df_prepared[self.config.cluster_features].values
        X_scaled = self.cluster_scaler.transform(X_clust)
        labels = self.kmeans_model.predict(X_scaled)
        return labels.astype(int)

    # ------------------------------------------------------------------ #
    # Публичные методы сервиса
    # ------------------------------------------------------------------ #

    @property
    def default_threshold(self) -> float:
        return float(self.config.default_threshold)

    def predict_one(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Принимает словарь с признаками одного клиента в терминах исходного датасета.
        Возвращает словарь с прогнозом, сегментами и текстовыми пояснениями.
        """
        df_raw = pd.DataFrame([features])
        df_prepared = self._prepare_df(df_raw)

        proba = float(self.rf_pipeline.predict_proba(df_prepared)[0, 1])
        predicted_label = int(proba >= self.default_threshold)

        decile_series = self._assign_deciles(pd.Series([proba]))
        decile = int(decile_series.iloc[0])
        priority = self._map_priority_segment(decile)

        cluster_labels = self._predict_clusters(df_prepared)
        cluster_id = int(cluster_labels[0])
        cluster_text = self.cluster_descriptions.get(
            cluster_id, f"Кластер {cluster_id}: поведенческий сегмент клиента."
        )

        result: Dict[str, Any] = dict(features)  # копия исходных признаков
        result.update(
            {
                "rf_score": proba,
                "predicted_conversion": predicted_label,
                "rf_decile": decile,
                "priority_segment": priority,
                "cluster_kmeans": cluster_id,
                # Текстовые пояснения для интерфейса
                "probability_text": f"Оценочная вероятность конверсии: {proba * 100:.1f}%.",
                "priority_text": self._priority_explanation(priority, decile),
                "cluster_text": cluster_text,
            }
        )
        return result

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Пакетный расчёт для DataFrame с лидами.
        На вход подаются сырые признаки (как в исходном датасете),
        на выходе — те же признаки плюс:
        rf_score, predicted_conversion, rf_decile, priority_segment, cluster_kmeans.
        """
        if df.empty:
            raise ValueError("Передан пустой DataFrame для predict_batch().")

        df_raw = df.copy()
        df_prepared = self._prepare_df(df_raw)

        proba = self.rf_pipeline.predict_proba(df_prepared)[:, 1]
        proba_series = pd.Series(proba, index=df_raw.index)

        deciles = self._assign_deciles(proba_series)
        priorities = deciles.apply(self._map_priority_segment)
        clusters = self._predict_clusters(df_prepared)

        df_out = df_raw.copy()
        df_out["rf_score"] = proba_series
        df_out["predicted_conversion"] = (proba_series >= self.default_threshold).astype(int)
        df_out["rf_decile"] = deciles
        df_out["priority_segment"] = priorities
        df_out["cluster_kmeans"] = clusters

        return df_out

    @staticmethod
    def _priority_explanation(priority: str, decile: int) -> str:
        """Человеко-понятное объяснение сегмента приоритета."""
        if priority == "High":
            return (
                "Высокий приоритет: клиент входит в верхние дециля по прогнозируемой "
                "вероятности конверсии (приблизительно топ-30% выборки)."
            )
        if priority == "Medium":
            return (
                "Средний приоритет: вероятность конверсии заметно выше среднего, "
                "но клиент не относится к самой верхней группе."
            )
        return (
            "Низкий приоритет: вероятность конверсии ниже, чем у основной части клиентов. "
            "Имеет смысл применять более бюджетные или долгосрочные стратегии."
        )
