# app.py
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import io

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    send_file,
)
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func
import pandas as pd

from ml_service import MLService

# ----------------------------------------------------------------------
# Инициализация приложения и БД
# ----------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)
app.config["SECRET_KEY"] = "change-me-in-production"

# SQLite в подкаталоге instance/
instance_dir = BASE_DIR / "instance"
instance_dir.mkdir(exist_ok=True)
db_path = instance_dir / "digital_marketing.db"

app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{db_path}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# Инициализируем ML-сервис
ml_service = MLService(base_dir=BASE_DIR)

@app.context_processor
def inject_ml_service():
    """Глобальные объекты, доступные во всех шаблонах Jinja."""
    return {
        "ml_service": ml_service
    }

# Справочники для форм (значение в БД / отображаемое имя)
GENDER_OPTIONS = [
    ("Male", "Мужской"),
    ("Female", "Женский"),
]

CAMPAIGN_CHANNEL_OPTIONS = [
    ("Social Media", "Социальные сети"),
    ("Email", "Email-рассылки"),
    ("PPC", "Контекстная реклама (PPC)"),
    ("SEO", "SEO-трафик (поисковые системы)"),
    ("Referral", "Рекомендации / партнёрские программы"),
]

CAMPAIGN_TYPE_OPTIONS = [
    ("Awareness", "Узнаваемость бренда"),
    ("Consideration", "Формирование интереса"),
    ("Conversion", "Конверсионная кампания (продажи)"),
    ("Retention", "Удержание существующих клиентов"),
]


# ----------------------------------------------------------------------
# Модели БД (SQLAlchemy)
# ----------------------------------------------------------------------


class BatchRun(db.Model):
    __tablename__ = "batch_runs"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    threshold_used = db.Column(db.Float)

    n_clients = db.Column(db.Integer)
    avg_score = db.Column(db.Float)
    avg_predicted_conversion = db.Column(db.Float)

    comment = db.Column(db.Text)

    leads = db.relationship("Lead", backref="batch", lazy=True)


class Lead(db.Model):
    __tablename__ = "leads"

    id = db.Column(db.Integer, primary_key=True)
    batch_id = db.Column(db.Integer, db.ForeignKey("batch_runs.id"), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Исходные признаки
    customer_id = db.Column(db.String(64))
    age = db.Column(db.Integer)
    gender = db.Column(db.String(16))
    income = db.Column(db.Float)
    campaign_channel = db.Column(db.String(32))
    campaign_type = db.Column(db.String(32))
    ad_spend = db.Column(db.Float)
    click_through_rate = db.Column(db.Float)
    conversion_rate = db.Column(db.Float)
    website_visits = db.Column(db.Integer)
    pages_per_visit = db.Column(db.Float)
    time_on_site = db.Column(db.Float)
    social_shares = db.Column(db.Integer)
    email_opens = db.Column(db.Integer)
    email_clicks = db.Column(db.Integer)
    previous_purchases = db.Column(db.Integer)
    loyalty_points = db.Column(db.Integer)

    # Результаты модели
    rf_score = db.Column(db.Float)
    predicted_conversion = db.Column(db.Integer)
    rf_decile = db.Column(db.Integer)
    priority_segment = db.Column(db.String(16))
    cluster_kmeans = db.Column(db.Integer)


# Создаём таблицы, если их ещё нет
with app.app_context():
    db.create_all()


# ----------------------------------------------------------------------
# Вспомогательные функции
# ----------------------------------------------------------------------


def _lead_to_dict(lead: Lead) -> dict:
    """Удобное представление лида для выгрузки CSV."""
    return {
        "CustomerID": lead.customer_id,
        "Age": lead.age,
        "Gender": lead.gender,
        "Income": lead.income,
        "CampaignChannel": lead.campaign_channel,
        "CampaignType": lead.campaign_type,
        "AdSpend": lead.ad_spend,
        "ClickThroughRate": lead.click_through_rate,
        "ConversionRate": lead.conversion_rate,
        "WebsiteVisits": lead.website_visits,
        "PagesPerVisit": lead.pages_per_visit,
        "TimeOnSite": lead.time_on_site,
        "SocialShares": lead.social_shares,
        "EmailOpens": lead.email_opens,
        "EmailClicks": lead.email_clicks,
        "PreviousPurchases": lead.previous_purchases,
        "LoyaltyPoints": lead.loyalty_points,
        "rf_score": lead.rf_score,
        "predicted_conversion": lead.predicted_conversion,
        "rf_decile": lead.rf_decile,
        "priority_segment": lead.priority_segment,
        "cluster_kmeans": lead.cluster_kmeans,
    }


# ----------------------------------------------------------------------
# Маршруты
# ----------------------------------------------------------------------


@app.route("/")
def index():
    """Главная страница — дашборд по уже сохранённым данным."""
    total_leads = db.session.query(func.count(Lead.id)).scalar()
    total_batches = db.session.query(func.count(BatchRun.id)).scalar()

    last_batch = BatchRun.query.order_by(BatchRun.created_at.desc()).first()

    # Распределение по приоритетным сегментам
    seg_rows = (
        db.session.query(
            Lead.priority_segment,
            func.count(Lead.id),
            func.avg(Lead.rf_score),
        )
        .group_by(Lead.priority_segment)
        .all()
    )

    segment_stats = [
        {
            "segment": seg or "не задан",
            "n": int(n),
            "avg_score": float(avg or 0.0),
        }
        for seg, n, avg in seg_rows
    ]

    return render_template(
        "index.html",
        total_leads=total_leads,
        total_batches=total_batches,
        last_batch=last_batch,
        segment_stats=segment_stats,
    )


@app.route("/predict", methods=["GET", "POST"])
def predict_single():
    """
    Оценка одного клиента через веб-форму.
    При GET показываем пустую форму, при POST — считаем прогноз и сохраняем в БД.
    """
    result = None
    lead_id = None

    if request.method == "POST":
        try:
            form = request.form

            features = {
                "CustomerID": form.get("CustomerID", "").strip(),
                "Age": int(form.get("Age", "0") or 0),
                "Gender": form.get("Gender", ""),
                "Income": float(form.get("Income", "0") or 0),
                "CampaignChannel": form.get("CampaignChannel", ""),
                "CampaignType": form.get("CampaignType", ""),
                "AdSpend": float(form.get("AdSpend", "0") or 0),
                "ClickThroughRate": float(form.get("ClickThroughRate", "0") or 0),
                "ConversionRate": float(form.get("ConversionRate", "0") or 0),
                "WebsiteVisits": int(form.get("WebsiteVisits", "0") or 0),
                "PagesPerVisit": float(form.get("PagesPerVisit", "0") or 0),
                "TimeOnSite": float(form.get("TimeOnSite", "0") or 0),
                "SocialShares": int(form.get("SocialShares", "0") or 0),
                "EmailOpens": int(form.get("EmailOpens", "0") or 0),
                "EmailClicks": int(form.get("EmailClicks", "0") or 0),
                "PreviousPurchases": int(form.get("PreviousPurchases", "0") or 0),
                "LoyaltyPoints": int(form.get("LoyaltyPoints", "0") or 0),
            }
        except ValueError:
            flash(
                "Ошибка преобразования данных. "
                "Проверьте, что числовые поля заполнены корректными числами.",
                "danger",
            )
            return redirect(request.url)

        # Расчёт прогноза
        try:
            result = ml_service.predict_one(features)
        except Exception as e:
            flash(f"Ошибка при расчёте прогноза: {e}", "danger")
            return redirect(request.url)

        # Сохраняем в БД
        lead = Lead(
            customer_id=result.get("CustomerID"),
            age=result.get("Age"),
            gender=result.get("Gender"),
            income=result.get("Income"),
            campaign_channel=result.get("CampaignChannel"),
            campaign_type=result.get("CampaignType"),
            ad_spend=result.get("AdSpend"),
            click_through_rate=result.get("ClickThroughRate"),
            conversion_rate=result.get("ConversionRate"),
            website_visits=result.get("WebsiteVisits"),
            pages_per_visit=result.get("PagesPerVisit"),
            time_on_site=result.get("TimeOnSite"),
            social_shares=result.get("SocialShares"),
            email_opens=result.get("EmailOpens"),
            email_clicks=result.get("EmailClicks"),
            previous_purchases=result.get("PreviousPurchases"),
            loyalty_points=result.get("LoyaltyPoints"),
            rf_score=result.get("rf_score"),
            predicted_conversion=result.get("predicted_conversion"),
            rf_decile=result.get("rf_decile"),
            priority_segment=result.get("priority_segment"),
            cluster_kmeans=result.get("cluster_kmeans"),
        )
        db.session.add(lead)
        db.session.commit()
        lead_id = lead.id

        flash("Оценка клиента успешно выполнена и сохранена в базе данных.", "success")

    return render_template(
        "predict_single.html",
        result=result,
        lead_id=lead_id,
        gender_options=GENDER_OPTIONS,
        channel_options=CAMPAIGN_CHANNEL_OPTIONS,
        campaign_type_options=CAMPAIGN_TYPE_OPTIONS,
    )


@app.route("/upload", methods=["GET", "POST"])
def upload_batch():
    """
    Пакетная загрузка CSV с лидами и расчёт прогнозов.
    """
    batch = None
    preview_html = None

    if request.method == "POST":
        file = request.files.get("file")
        batch_name = request.form.get("name") or f"Запуск от {datetime.now():%Y-%m-%d %H:%M}"

        if file is None or file.filename == "":
            flash("Файл не выбран. Пожалуйста, выберите CSV-файл.", "danger")
            return redirect(request.url)

        try:
            df = pd.read_csv(file)
        except Exception as e:
            flash(f"Не удалось прочитать CSV-файл: {e}", "danger")
            return redirect(request.url)

        try:
            df_pred = ml_service.predict_batch(df)
        except Exception as e:
            flash(f"Ошибка при расчёте прогноза для набора клиентов: {e}", "danger")
            return redirect(request.url)

        # Создаём запись о запуске
        batch = BatchRun(
            name=batch_name,
            created_at=datetime.utcnow(),
            threshold_used=ml_service.default_threshold,
            n_clients=len(df_pred),
            avg_score=float(df_pred["rf_score"].mean()),
            avg_predicted_conversion=float(df_pred["predicted_conversion"].mean()),
        )
        db.session.add(batch)
        db.session.flush()  # получаем batch.id

        # Сохраняем лидов
        leads_to_add = []
        for _, row in df_pred.iterrows():
            lead = Lead(
                batch_id=batch.id,
                customer_id=str(row.get("CustomerID", "")),
                age=int(row.get("Age", 0)) if not pd.isna(row.get("Age")) else None,
                gender=row.get("Gender"),
                income=float(row.get("Income", 0)) if not pd.isna(row.get("Income")) else None,
                campaign_channel=row.get("CampaignChannel"),
                campaign_type=row.get("CampaignType"),
                ad_spend=float(row.get("AdSpend", 0)) if not pd.isna(row.get("AdSpend")) else None,
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
                rf_score=float(row.get("rf_score", 0)),
                predicted_conversion=int(row.get("predicted_conversion", 0)),
                rf_decile=int(row.get("rf_decile", 0)),
                priority_segment=row.get("priority_segment"),
                cluster_kmeans=int(row.get("cluster_kmeans", 0)),
            )
            leads_to_add.append(lead)

        db.session.add_all(leads_to_add)
        db.session.commit()

        flash(f"Загрузка завершена: обработано {len(df_pred)} клиентов.", "success")

        preview_html = df_pred.head(20).to_html(
            classes="table table-striped table-sm", index=False
        )

    return render_template(
        "upload_batch.html",
        batch=batch,
        preview_html=preview_html,
    )


@app.route("/runs")
def runs_list():
    """Список всех запусков пакетной обработки."""
    runs = BatchRun.query.order_by(BatchRun.created_at.desc()).all()
    return render_template("runs_list.html", runs=runs)


@app.route("/runs/<int:run_id>")
def run_detail(run_id: int):
    """Подробная информация по одному запуску."""
    batch = BatchRun.query.get_or_404(run_id)
    leads = (
        Lead.query.filter_by(batch_id=run_id)
        .order_by(Lead.rf_score.desc())
        .limit(500)
        .all()
    )
    return render_template("run_detail.html", batch=batch, leads=leads)


@app.route("/runs/<int:run_id>/download")
def download_run(run_id: int):
    """Выгрузка всех лидов конкретного запуска в CSV."""
    batch = BatchRun.query.get_or_404(run_id)
    leads = Lead.query.filter_by(batch_id=run_id).all()

    rows = [_lead_to_dict(lead) for lead in leads]
    df = pd.DataFrame(rows)
    csv_data = df.to_csv(index=False, encoding="utf-8-sig")

    filename = f"batch_{batch.id}_leads.csv"
    return send_file(
        io.BytesIO(csv_data.encode("utf-8-sig")),
        mimetype="text/csv",
        as_attachment=True,
        download_name=filename,
    )


@app.route("/clients/<int:lead_id>")
def client_detail(lead_id: int):
    """Карточка отдельного клиента."""
    lead = Lead.query.get_or_404(lead_id)
    return render_template("client_detail.html", lead=lead)


# ----------------------------------------------------------------------
# Точка входа
# ----------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
