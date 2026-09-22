import os
import json
import uuid
import pickle
import logging
import secrets
from pathlib import Path
from functools import wraps

import numpy as np
import pandas as pd
from flask import (
    Flask, request, render_template, session, abort, jsonify,
    g, has_request_context,
)
from catboost import CatBoostClassifier

class RequestIdFilter(logging.Filter):
    def filter(self, record):
        record.request_id = (
            getattr(g, 'request_id', '-') if has_request_context() else '-'
        )
        return True


LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s [%(levelname)s] [rid=%(request_id)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler()],
)
for _h in logging.root.handlers:
    _h.addFilter(RequestIdFilter())
logger = logging.getLogger('scoring')


BASE_DIR = Path(__file__).parent
ARTIFACTS_DIR = Path(os.environ.get('ARTIFACTS_DIR', BASE_DIR / 'artifacts'))
MODELS_DIR = Path(os.environ.get('MODELS_DIR', BASE_DIR / 'models'))

MODEL_PATH = MODELS_DIR / 'catboost_final.cbm'
FALLBACK_MODEL = ARTIFACTS_DIR / 'catboost_trained.cbm'
THRESHOLD_PATH = MODELS_DIR / 'opt_threshold.json'
CALIBRATOR_PATH = MODELS_DIR / 'calibrator.pkl'
CAT_FEATURES_PATH = ARTIFACTS_DIR / 'cat_features.json'
FEATURE_NAMES_PATH = ARTIFACTS_DIR / 'feature_names.json'
TRAIN_PATH = ARTIFACTS_DIR / 'train_processed.parquet'

LGD = float(os.environ.get('LGD', 0.60))
EAD_MEAN = float(os.environ.get('EAD_MEAN', 200_000))
MARGIN_GOOD = float(os.environ.get('MARGIN_GOOD', 20_000))

MIN_AGE = int(os.environ.get('MIN_AGE', 18))
MAX_AGE = int(os.environ.get('MAX_AGE', 100))
MIN_LIMIT = float(os.environ.get('MIN_LIMIT', 10_000))
MAX_LIMIT = float(os.environ.get('MAX_LIMIT', 1_000_000))

# --- Policy для thin-file ---
THIN_FILE_MIN_AGE = int(os.environ.get('THIN_FILE_MIN_AGE', 21))
THIN_FILE_MAX_AGE = int(os.environ.get('THIN_FILE_MAX_AGE', 65))
THIN_FILE_MAX_LIMIT = float(os.environ.get('THIN_FILE_MAX_LIMIT', 100_000))

POLICY_VERSION = os.environ.get('POLICY_VERSION', 'thin_file_v1.0')

# --- Скрывать ли BILL_AMT в форме ---
HIDE_BILL_AMT = os.environ.get('HIDE_BILL_AMT', '0') == '1'

PAY_STATUS_COLS = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
BILL_COLS = [f'BILL_AMT{i}' for i in range(1, 7)]
PAY_AMT_COLS = [f'PAY_AMT{i}' for i in range(1, 7)]
HISTORY_COLS = PAY_STATUS_COLS + BILL_COLS + PAY_AMT_COLS
USER_FIELDS = ['AGE', 'SEX', 'EDUCATION', 'MARRIAGE', 'LIMIT_BAL']

MONTH_LABELS = [
    'Текущий месяц', '1 мес. назад', '2 мес. назад',
    '3 мес. назад', '4 мес. назад', '5 мес. назад',
]

LABELS = {
    'AGE': 'Возраст (лет)',
    'SEX': 'Пол',
    'EDUCATION': 'Образование',
    'MARRIAGE': 'Семейное положение',
    'LIMIT_BAL': 'Кредитный лимит (руб.)',
}

HELP_TEXTS = {
    'AGE': 'Ваш возраст, от 18 до 100 лет',
    'SEX': 'Мужской / Женский',
    'EDUCATION': 'Высшее / магистратура / среднее / другое',
    'MARRIAGE': 'Женат / не замужем / другое',
    'LIMIT_BAL': 'Максимальная сумма, которую банк готов вам дать. '
                 'Для действующей карты — её текущий лимит.',
}

COLUMN_LABELS = {
    'month':      'Месяц',
    'pay_status': 'Статус платежа',
    'bill_amt':   'Остаток долга',
    'pay_amt':    'Сколько заплатили',
}

PLACEHOLDERS = {
    'bill_amt': 'например, 45 000',
    'pay_amt':  'например, 5 000',
}

TOOLTIPS = {
    'bill_amt': (
        'Остаток долга по всем кредитам и картам на конец месяца. '
        'Сюда входят покупки, проценты, комиссии и штрафы. '
        'Если вы не пользовались кредитом — впишите 0. '
        'Если на конец месяца у вас переплата (отрицательный баланс) — '
        'впишите отрицательное число.'
    ),
    'pay_amt': (
        'Сумма, фактически внесённая в этом месяце: '
        'ежемесячный платёж, досрочное погашение, любые переводы в счёт долга. '
        'Если не платили — впишите 0. '
        'Отрицательных значений быть не может.'
    ),
    'pay_status': (
        'Нет потребления — вы не пользовались кредитом в этом месяце. '
        'Оплачено вовремя — платёж внесён в срок. '
        'Револьверный кредит — платите только минимальный платёж. '
        'Просрочка N мес. — задержка платежа на N месяцев на конец месяца.'
    ),
    'month': (
        'Месяцы идут от текущего к самому старому. '
        '«Текущий месяц» — последний закрытый отчётный период. '
        'Если данные за какой-то месяц неизвестны, оставьте поля пустыми.'
    ),
}

SEX_OPTIONS = {1: 'Мужской', 2: 'Женский'}
EDUCATION_OPTIONS = {
    1: 'Аспирантура / магистратура',
    2: 'Бакалавриат / университет',
    3: 'Среднее образование',
    4: 'Другое',
}
MARRIAGE_OPTIONS = {
    1: 'Женат / замужем',
    2: 'Холост / не замужем',
    3: 'Другое',
}
PAY_STATUS_OPTIONS = {
    -2: 'Нет потребления по кредиту',
    -1: 'Оплачено вовремя',
    0: 'Использование револьверного кредита',
    1: 'Просрочка 1 мес.',
    2: 'Просрочка 2 мес.',
    3: 'Просрочка 3 мес.',
    4: 'Просрочка 4 мес.',
    5: 'Просрочка 5 мес.',
    6: 'Просрочка 6 мес.',
    7: 'Просрочка 7 мес.',
    8: 'Просрочка 8+ мес.',
}
PAY_STATUS_ALLOWED = set(PAY_STATUS_OPTIONS.keys())

MSG_APPROVED = '🎉 Поздравляем! Ваша заявка на кредит одобрена.'
MSG_REJECTED = ('К сожалению, мы не можем одобрить кредит в текущих условиях. '
                'Вы можете подать заявку на кредитную карту с меньшим лимитом.')
MSG_REVIEW = '🕓 Заявка передана андеррайтеру на ручную проверку.'

MSG_THIN_FILE_APPROVED = (
    '✅ Предварительно одобрено. Для первого кредита лимит ограничен '
    f'{THIN_FILE_MAX_LIMIT:,.0f} руб., ставка выше стандартной.'
)
MSG_THIN_FILE_REJECTED = (
    '⚠️ По правилам для клиентов без кредитной истории '
    'мы не можем одобрить заявку. Рекомендуем начать с кредитной карты.'
)
MSG_THIN_FILE_REVIEW = (
    '🕓 Заявка передана андеррайтеру — для первого кредита требуется '
    'ручная проверка документов.'
)

def compute_features(row: dict) -> dict:
    eps = 1.0
    bill_vals = np.array([row.get(c, 0) or 0 for c in BILL_COLS], dtype=float)
    pay_vals = np.array([row.get(c, 0) or 0 for c in PAY_AMT_COLS], dtype=float)
    status_vals = np.array([row.get(c, 0) or 0 for c in PAY_STATUS_COLS], dtype=float)

    row['avg_bill'] = float(bill_vals.mean())
    row['avg_payment'] = float(pay_vals.mean())
    row['std_bill'] = float(bill_vals.std(ddof=1)) if len(bill_vals) > 1 else 0.0
    row['std_payment'] = float(pay_vals.std(ddof=1)) if len(pay_vals) > 1 else 0.0

    row['total_bill'] = float(bill_vals.sum())
    row['total_payment'] = float(pay_vals.sum())
    row['total_debt'] = row['total_bill'] - row['total_payment']

    limit = float(row.get('LIMIT_BAL', 0) or 0)
    age = float(row.get('AGE', 0) or 0)
    bill1 = float(row.get('BILL_AMT1', 0) or 0)
    pay1 = float(row.get('PAY_AMT1', 0) or 0)
    pay0 = float(row.get('PAY_0', 0) or 0)

    row['utilization'] = bill1 / (limit + eps)
    row['avg_utilization'] = row['avg_bill'] / (limit + eps)
    row['payment_ratio'] = row['avg_payment'] / (row['avg_bill'] + eps)
    row['payment_to_limit'] = row['avg_payment'] / (limit + eps)
    row['limit_per_age'] = limit / (age + eps)
    row['payment_ratio_1'] = pay1 / (bill1 + eps)

    row['bill_trend'] = bill1 - float(row.get('BILL_AMT6', 0) or 0)
    row['payment_trend'] = pay1 - float(row.get('PAY_AMT6', 0) or 0)
    row['delay_trend'] = pay0 - float(row.get('PAY_6', 0) or 0)

    row['max_delay'] = float(status_vals.max())
    row['min_delay'] = float(status_vals.min())
    row['mean_delay'] = float(status_vals.mean())
    row['months_with_debt'] = int((status_vals > 0).sum())
    row['months_fully_paid'] = int((status_vals == -1).sum())
    row['sum_positive_delay'] = float(np.clip(status_vals, 0, None).sum())

    row['has_negative_bill'] = int((bill_vals < 0).any())
    row['bill_x_age'] = bill1 * age
    row['pay_x_pay0'] = pay1 * max(pay0, 0)
    row['utilization_change'] = row['utilization'] - row['avg_utilization']

    row['log_bill1'] = float(np.log1p(abs(bill1)))
    row['log_pay1'] = float(np.log1p(abs(pay1)))

    for c in ['utilization', 'avg_utilization', 'payment_ratio',
              'payment_to_limit', 'limit_per_age', 'payment_ratio_1',
              'utilization_change']:
        v = row.get(c, 0.0)
        if not np.isfinite(v):
            row[c] = 0.0
    return row


def build_dataframe(form_data: dict) -> pd.DataFrame:
    if not MEDIANS:
        raise RuntimeError('Медианы не загружены (нет train_processed.parquet). '
                           'Инференс невозможен.')

    missing = [f for f in MODEL_FEATURES if f not in MEDIANS]
    if missing:
        raise RuntimeError(
            f"Признаки модели отсутствуют в MEDIANS: {missing[:10]}... "
            f"Проверьте, что model и train_processed.parquet из одного запуска."
        )

    row = {feat: MEDIANS[feat] for feat in MODEL_FEATURES}

    for feat, val in form_data.items():
        if val is None or val == '':
            continue
        if feat in row:
            row[feat] = val

    for c in CAT_FEATURES:
        if c in row:
            row[c] = int(row[c])

    row = compute_features(row)
    df = pd.DataFrame([row])
    df = df.reindex(columns=MODEL_FEATURES)
    for c in CAT_FEATURES:
        if c in df.columns:
            df[c] = df[c].astype(int)
    return df


def calibrate(raw_pd: float) -> float:
    """Применяет калибратор (Platt / Isotonic) к сырой вероятности."""
    if CALIBRATOR is None:
        return float(raw_pd)
    try:
        eps = 1e-7
        raw_pd = float(np.clip(raw_pd, eps, 1 - eps))
        logit = np.log(raw_pd / (1 - raw_pd)).reshape(-1, 1)

        if CALIB_TYPE == 'platt':
            cal_pd = CALIBRATOR.predict_proba(logit)[0, 1]
        elif CALIB_TYPE == 'isotonic':
            cal_pd = float(CALIBRATOR.predict(np.array([raw_pd]))[0])
        else:
            logger.warning(f"Неизвестный тип калибратора '{CALIB_TYPE}', "
                           f"используется сырая PD")
            return float(raw_pd)

        return float(np.clip(cal_pd, 0.0, 1.0))
    except Exception as e:
        logger.error(f"Ошибка калибратора: {e}")
        return float(raw_pd)

def policy_for_thin_file(age: float, limit_bal: float,
                         education: int, marriage: int) -> dict:
    if age < THIN_FILE_MIN_AGE or age > THIN_FILE_MAX_AGE:
        return {
            'decision': 'rejected',
            'message': (f'Для первого кредита возраст должен быть '
                        f'{THIN_FILE_MIN_AGE}–{THIN_FILE_MAX_AGE} лет.'),
            'policy': 'thin_file_age',
            'score': None,
        }
    if limit_bal > THIN_FILE_MAX_LIMIT:
        return {
            'decision': 'rejected',
            'message': (f'Максимальный лимит для первого кредита — '
                        f'{THIN_FILE_MAX_LIMIT:,.0f} руб.'),
            'policy': 'thin_file_limit',
            'score': None,
        }

    score = 0
    if 25 <= age <= 45:
        score += 2
    if education in (1, 2):
        score += 1
    if marriage in (1, 2):
        score += 1

    if score >= 3:
        return {'decision': 'approved',
                'message': MSG_THIN_FILE_APPROVED,
                'policy': 'thin_file_rules', 'score': score}
    if score == 2:
        return {'decision': 'review',
                'message': MSG_THIN_FILE_REVIEW,
                'policy': 'thin_file_review', 'score': score}
    return {'decision': 'rejected',
            'message': MSG_THIN_FILE_REJECTED,
            'policy': 'thin_file_rules', 'score': score}


def audit_log(event: str, payload: dict):
    """Единый audit-log: пишет одну JSON-строку в stdout."""
    record = {
        'ts': pd.Timestamp.utcnow().isoformat(),
        'request_id': getattr(g, 'request_id', None) if has_request_context() else None,
        'event': event,
        'model_version': MODEL_VERSION,
        'policy_version': POLICY_VERSION,
        **payload,
    }
    logger.info(f"AUDIT {json.dumps(record, ensure_ascii=False, default=str)}")

def load_artifacts():
    model_path = MODEL_PATH if MODEL_PATH.exists() else FALLBACK_MODEL
    if not model_path.exists():
        raise FileNotFoundError(
            f"Модель не найдена: {MODEL_PATH} или {FALLBACK_MODEL}. "
            f"Сначала запустите 2.ipynb."
        )
    logger.info(f"Загрузка модели из {model_path}")
    model = CatBoostClassifier()
    model.load_model(str(model_path))
    model_version = f"{model_path.name}:{int(model_path.stat().st_mtime)}"

    # threshold
    opt_threshold = 0.5
    if THRESHOLD_PATH.exists():
        with open(THRESHOLD_PATH, encoding='utf-8') as f:
            opt_threshold = float(json.load(f).get('opt_threshold', 0.5))
    else:
        logger.warning("opt_threshold.json не найден — используется 0.5")
    logger.info(f"Порог одобрения: {opt_threshold:.4f}")

    # calibrator
    calibrator, calib_type = None, 'none'
    if CALIBRATOR_PATH.exists():
        try:
            with open(CALIBRATOR_PATH, 'rb') as f:
                calib_data = pickle.load(f)
            if isinstance(calib_data, dict):
                calibrator = calib_data.get('calibrator')
                calib_type = calib_data.get('type', 'unknown')
            else:
                calibrator, calib_type = calib_data, 'unknown'
            if calibrator is None:
                logger.warning("Калибратор пустой — используется сырая PD")
                calib_type = 'none'
        except Exception as e:
            logger.warning(f"Не удалось загрузить калибратор: {e}")
            calibrator, calib_type = None, 'none'
    else:
        logger.warning("Файл калибратора не найден — используется сырая PD")

    # cat features
    if not CAT_FEATURES_PATH.exists():
        raise FileNotFoundError(f"Файл {CAT_FEATURES_PATH} не найден")
    with open(CAT_FEATURES_PATH, encoding='utf-8') as f:
        cat_features = json.load(f)['cat_features']
    logger.info(f"Категориальные признаки: {cat_features}")

    # feature names — берём из feature_names.json, иначе из модели
    if FEATURE_NAMES_PATH.exists():
        with open(FEATURE_NAMES_PATH, encoding='utf-8') as f:
            payload = json.load(f)
            feature_names = payload.get('feature_names', payload)
        logger.info(f"Feature names из {FEATURE_NAMES_PATH.name}: {len(feature_names)}")
    else:
        feature_names = list(model.feature_names_)
        logger.info(f"Feature names из .cbm: {len(feature_names)}")

    # медианы
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Файл {TRAIN_PATH} не найден — без него инференс невозможен "
            f"(нет дефолтов для пустых полей)."
        )
    train_df = pd.read_parquet(TRAIN_PATH)
    medians = {}
    numeric = train_df.drop(columns='default').median(numeric_only=True)
    for k, v in numeric.items():
        if not np.isfinite(v):
            continue
        medians[k] = float(v)
    for c in cat_features:
        if c in train_df.columns:
            medians[c] = int(train_df[c].mode().iloc[0])
    logger.info(f"Медианы загружены для {len(medians)} признаков")

    return (model, model_version, opt_threshold, calibrator, calib_type,
            cat_features, feature_names, medians)


(MODEL, MODEL_VERSION, OPT_THRESHOLD, CALIBRATOR, CALIB_TYPE,
 CAT_FEATURES, MODEL_FEATURES, MEDIANS) = load_artifacts()

logger.info(
    f"Модель: {MODEL_VERSION}; признаков: {len(MODEL_FEATURES)}; "
    f"калибратор: {CALIB_TYPE}; порог: {OPT_THRESHOLD:.4f}; "
    f"policy: {POLICY_VERSION}; hide_bill_amt={HIDE_BILL_AMT}"
)

app = Flask(__name__)

_secret = os.environ.get('SECRET_KEY')
if not _secret:
    _secret = secrets.token_hex(32)
    logger.warning(
        "SECRET_KEY не задан в env — сгенерирован случайный. "
        "Под gunicorn -w N>1 CSRF-токены будут ломаться. "
        "Задайте SECRET_KEY в проде."
    )
app.secret_key = _secret


@app.before_request
def _attach_request_id():
    g.request_id = request.headers.get('X-Request-Id') or uuid.uuid4().hex[:12]


def generate_csrf_token():
    if '_csrf_token' not in session:
        session['_csrf_token'] = secrets.token_hex(32)
    return session['_csrf_token']


app.jinja_env.globals['csrf_token'] = generate_csrf_token


def csrf_protect(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if request.method == 'POST':
            token = session.get('_csrf_token')
            if not token or token != request.form.get('_csrf_token'):
                logger.warning("CSRF token mismatch")
                abort(403)
        return f(*args, **kwargs)
    return wrapper

def base_context():
    return dict(
        user_fields=USER_FIELDS,
        pay_status_cols=PAY_STATUS_COLS,
        bill_cols=BILL_COLS,
        pay_amt_cols=PAY_AMT_COLS,
        month_labels=MONTH_LABELS,
        labels=LABELS,
        help_texts=HELP_TEXTS,
        column_labels=COLUMN_LABELS,
        placeholders=PLACEHOLDERS,
        tooltips=TOOLTIPS,
        sex_options=SEX_OPTIONS,
        education_options=EDUCATION_OPTIONS,
        marriage_options=MARRIAGE_OPTIONS,
        pay_status_options=PAY_STATUS_OPTIONS,
        opt_threshold=OPT_THRESHOLD,
        thin_file_max_limit=THIN_FILE_MAX_LIMIT,
        hide_bill_amt=HIDE_BILL_AMT,
    )


def empty_context(error_message=None):
    ctx = base_context()
    ctx.update(
        result=None,
        error_message=error_message,
        form_data={f: '' for f in USER_FIELDS + HISTORY_COLS},
        no_history=False,
    )
    return ctx



@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': MODEL is not None,
        'model_version': MODEL_VERSION,
        'policy_version': POLICY_VERSION,
        'calibrator': CALIB_TYPE,
        'threshold': OPT_THRESHOLD,
        'n_features': len(MODEL_FEATURES),
        'hide_bill_amt': HIDE_BILL_AMT,
    }), 200


@app.route('/model_info', methods=['GET'])
def model_info():
    return jsonify({
        'model_version': MODEL_VERSION,
        'policy_version': POLICY_VERSION,
        'calibrator': CALIB_TYPE,
        'threshold': OPT_THRESHOLD,
        'features': MODEL_FEATURES,
        'cat_features': CAT_FEATURES,
        'lgd': LGD,
        'ead_mean': EAD_MEAN,
        'margin_good': MARGIN_GOOD,
        'thin_file': {
            'min_age': THIN_FILE_MIN_AGE,
            'max_age': THIN_FILE_MAX_AGE,
            'max_limit': THIN_FILE_MAX_LIMIT,
        },
    }), 200


@app.route('/', methods=['GET', 'POST'])
@csrf_protect
def index():
    result = None
    error_message = None
    no_history = False
    form_data = {f: '' for f in USER_FIELDS + HISTORY_COLS}
    form_data['SEX'] = 1
    form_data['EDUCATION'] = 2
    form_data['MARRIAGE'] = 2
    for c in PAY_STATUS_COLS:
        form_data[c] = -1

    if request.method == 'POST':
        no_history = request.form.get('no_history') == '1'

        # --- USER_FIELDS ---
        for f in USER_FIELDS:
            raw = request.form.get(f, '').strip()
            if raw == '':
                form_data[f] = None
                continue
            try:
                form_data[f] = int(raw) if f in CAT_FEATURES else float(raw)
            except ValueError:
                form_data[f] = None

        # --- HISTORY ---
        if no_history:
            for c in PAY_STATUS_COLS:
                form_data[c] = -2
            for c in BILL_COLS:
                form_data[c] = 0.0
            for c in PAY_AMT_COLS:
                form_data[c] = 0.0
        else:
            for f in HISTORY_COLS:
                if HIDE_BILL_AMT and f in BILL_COLS:
                    form_data[f] = None  # подставится медиана
                    continue
                raw = request.form.get(f, '').strip()
                if raw == '':
                    form_data[f] = None
                    continue
                try:
                    form_data[f] = int(raw) if f in PAY_STATUS_COLS else float(raw)
                except ValueError:
                    form_data[f] = None

        # --- Валидация общих полей ---
        age = form_data.get('AGE')
        limit = form_data.get('LIMIT_BAL')

        if age is None:
            error_message = 'Пожалуйста, укажите возраст.'
        elif not (MIN_AGE <= age <= MAX_AGE):
            error_message = f'⚠️ Возраст должен быть от {MIN_AGE} до {MAX_AGE} лет.'
        elif limit is None:
            error_message = 'Пожалуйста, укажите кредитный лимит.'
        elif not (MIN_LIMIT <= limit <= MAX_LIMIT):
            error_message = (f'⚠️ Кредитный лимит должен быть от '
                             f'{MIN_LIMIT:,.0f} до {MAX_LIMIT:,.0f} руб.')

        # --- Валидация истории ---
        if not error_message and not no_history:
            # статусы платежа
            for f in PAY_STATUS_COLS:
                v = form_data.get(f)
                if v is None:
                    continue
                if int(v) not in PAY_STATUS_ALLOWED:
                    error_message = (f'Недопустимое значение статуса платежа '
                                     f'{f}={v}.')
                    break
            # суммы
            if not error_message:
                for f in (BILL_COLS + PAY_AMT_COLS):
                    if HIDE_BILL_AMT and f in BILL_COLS:
                        continue
                    v = form_data.get(f)
                    if v is not None and not np.isfinite(v):
                        error_message = f'Некорректное значение для {f}'
                        break
            if not error_message:
                for f in PAY_AMT_COLS:
                    v = form_data.get(f)
                    if v is not None and v < 0:
                        error_message = (f'Сумма платежа не может быть '
                                         f'отрицательной: {f}')
                        break

        # --- Расчёт ---
        if not error_message:
            try:
                if no_history:
                    education = int(form_data.get('EDUCATION') or 4)
                    marriage = int(form_data.get('MARRIAGE') or 3)
                    policy_result = policy_for_thin_file(
                        age=float(age),
                        limit_bal=float(limit),
                        education=education,
                        marriage=marriage,
                    )
                    result = {
                        **policy_result,
                        'pd_raw': None,
                        'pd_cal': None,
                        'threshold': None,
                        'expected_loss': None,
                        'is_thin_file': True,
                    }
                    audit_log('thin_file_decision', {
                        'age': age, 'sex': form_data.get('SEX'),
                        'education': education, 'marriage': marriage,
                        'limit_bal': limit,
                        'decision': result['decision'],
                        'policy': result['policy'],
                        'score': result['score'],
                    })
                else:
                    X = build_dataframe(form_data)
                    proba = MODEL.predict_proba(X)[0]
                    raw_pd = float(proba[1])
                    calibrated_pd = calibrate(raw_pd)

                    decision = ('approved'
                                if calibrated_pd < OPT_THRESHOLD
                                else 'rejected')
                    message = MSG_APPROVED if decision == 'approved' else MSG_REJECTED
                    expected_loss = calibrated_pd * LGD * EAD_MEAN

                    result = {
                        'decision': decision,
                        'message': message,
                        'pd_raw': round(raw_pd, 4),
                        'pd_cal': round(calibrated_pd, 4),
                        'threshold': round(OPT_THRESHOLD, 4),
                        'expected_loss': round(expected_loss, 0),
                        'is_thin_file': False,
                        'policy': 'ml_model',
                        'score': None,
                    }
                    audit_log('ml_decision', {
                        'decision': decision,
                        'pd_raw': round(raw_pd, 6),
                        'pd_cal': round(calibrated_pd, 6),
                        'threshold': OPT_THRESHOLD,
                        'expected_loss': round(expected_loss, 2),
                        'hide_bill_amt': HIDE_BILL_AMT,
                        'features': {k: form_data.get(k) for k in
                                     USER_FIELDS + HISTORY_COLS},
                    })
            except Exception as e:
                logger.exception("Ошибка при расчёте")
                error_message = f'Ошибка при расчёте: {e}'

        # None → '' для шаблона
        for k in list(form_data.keys()):
            if form_data[k] is None:
                form_data[k] = ''

    return render_template(
        'index.html',
        result=result,
        error_message=error_message,
        form_data=form_data,
        no_history=no_history,
        **base_context(),
    )


@app.errorhandler(403)
def forbidden(e):
    return render_template(
        'index.html',
        **empty_context(
            'Ошибка безопасности: недействительный CSRF-токен. '
            'Обновите страницу.'
        ),
    ), 403


@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def internal_error(e):
    logger.exception("Internal server error")
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)