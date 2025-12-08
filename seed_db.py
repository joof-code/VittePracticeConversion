# seed_db.py
from __future__ import annotations

from pathlib import Path
import sys
import argparse

import pandas as pd

# импортируем готовое приложение и объекты из app.py
from app import app, db, BatchRun, Lead, ml_service


def seed_database(limit: int | None = None) -> None:
    """
    Заполняет БД начальными данными из учебного датасета Kaggle.
    - Загружает CSV с клиентами.
    - Прогоняет через ml_service.predict_batch().
    - Создаёт один BatchRun и много Lead.
    """
    base_dir = Path(__file__).resolve().parent
    dataset_path = base_dir / "data" / "digital_marketing_campaign_dataset.csv"

    if not dataset_path.exists():
        print(f"[ERROR] Файл датасета не найден: {dataset_path}", file=sys.stderr)
        print("Скопируй CSV из Google Drive/Colab в папку data/ рядом с app.py.")
        sys.exit(1)

    with app.app_context():
        # Проверяем, не заполнена ли уже БД
        leads_count = Lead.query.count()
        runs_count = BatchRun.query.count()
        if leads_count > 0 or runs_count > 0:
            print("[INFO] В базе уже есть данные — сидирование пропускается.")
            print(f"    Leads: {leads_count}")
            print(f"    BatchRuns: {runs_count}")
            return

        print(f"[INFO] Читаю датасет: {dataset_path}")
        df = pd.read_csv(dataset_path)

        if limit is not None and limit > 0:
            df = df.head(limit)
            print(f"[INFO] Использую только первые {len(df)} строк для сидирования.")
        else:
            print(f"[INFO] Всего строк в датасете: {len(df)}")

        # Прогоняем через ML-сервис (RandomForest + кластеры + дециль)
        print("[INFO] Запуск predict_batch()...")
        df_pred = ml_service.predict_batch(df)
        print("[INFO] Прогнозы посчитаны.")

        # Создаём запись о запуске
        batch = BatchRun(
            name=f"Начальный импорт из Kaggle ({len(df_pred)} клиентов)",
            threshold_used=ml_service.default_threshold,
            n_clients=len(df_pred),
            avg_score=float(df_pred["rf_score"].mean()),
            avg_predicted_conversion=float(df_pred["predicted_conversion"].mean()),
            comment="Автоматическое сидирование БД из учебного датасета Kaggle.",
        )
        db.session.add(batch)
        db.session.flush()  # получаем batch.id

        # Формируем объекты Lead (как в upload_batch)
        leads_to_add: list[Lead] = []
        for _, row in df_pred.iterrows():
            lead = Lead(
                batch_id=batch.id,
                customer_id=str(row.get("CustomerID", "")),
                age=int(row.get("Age", 0)) if not pd.isna(row.get("Age")) else None,
                gender=row.get("Gender") if not pd.isna(row.get("Gender")) else None,
                income=float(row.get("Income", 0))
                if not pd.isna(row.get("Income"))
                else None,
                campaign_channel=row.get("CampaignChannel")
                if not pd.isna(row.get("CampaignChannel"))
                else None,
                campaign_type=row.get("CampaignType")
                if not pd.isna(row.get("CampaignType"))
                else None,
                ad_spend=float(row.get("AdSpend", 0))
                if not pd.isna(row.get("AdSpend"))
                else None,
                click_through_rate=float(row.get("ClickThroughRate", 0))
                if not pd.isna(row.get("ClickThroughRate"))
                else None,
                conversion_rate=float(row.get("ConversionRate", 0))
                if not pd.isna(row.get("ConversionRate"))
                else None,
                website_visits=int(row.get("WebsiteVisits", 0))
                if not pd.isna(row.get("WebsiteVisits"))
                else None,
                pages_per_visit=float(row.get("PagesPerVisit", 0))
                if not pd.isna(row.get("PagesPerVisit"))
                else None,
                time_on_site=float(row.get("TimeOnSite", 0))
                if not pd.isna(row.get("TimeOnSite"))
                else None,
                social_shares=int(row.get("SocialShares", 0))
                if not pd.isna(row.get("SocialShares"))
                else None,
                email_opens=int(row.get("EmailOpens", 0))
                if not pd.isna(row.get("EmailOpens"))
                else None,
                email_clicks=int(row.get("EmailClicks", 0))
                if not pd.isna(row.get("EmailClicks"))
                else None,
                previous_purchases=int(row.get("PreviousPurchases", 0))
                if not pd.isna(row.get("PreviousPurchases"))
                else None,
                loyalty_points=int(row.get("LoyaltyPoints", 0))
                if not pd.isna(row.get("LoyaltyPoints"))
                else None,
                rf_score=float(row.get("rf_score", 0))
                if not pd.isna(row.get("rf_score"))
                else None,
                predicted_conversion=int(row.get("predicted_conversion", 0))
                if not pd.isna(row.get("predicted_conversion"))
                else None,
                rf_decile=int(row.get("rf_decile", 0))
                if not pd.isna(row.get("rf_decile"))
                else None,
                priority_segment=row.get("priority_segment")
                if not pd.isna(row.get("priority_segment"))
                else None,
                cluster_kmeans=int(row.get("cluster_kmeans", 0))
                if not pd.isna(row.get("cluster_kmeans"))
                else None,
            )
            leads_to_add.append(lead)

        db.session.add_all(leads_to_add)
        db.session.commit()

        print("[OK] Сидирование завершено.")
        print(f"     BatchRun #{batch.id}, клиентов записано: {len(leads_to_add)}")
        print(f"     Средний rf_score: {batch.avg_score:.4f}")
        print(
            f"     Доля предсказанных конверсий: {batch.avg_predicted_conversion * 100:.2f}%"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Сидирование БД digital_marketing.db из учебного датасета Kaggle."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Ограничить число строк для сидирования (по умолчанию — весь датасет). "
            "Например: --limit 2000"
        ),
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    seed_database(limit=args.limit)
