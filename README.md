# Credit Scoring

Pet-проект по кредитному скорингу: от EDA до Flask-сервиса и Docker-контейнера.
Модель предсказывает вероятность дефолта клиента по истории платежей за 6 месяцев
и используется для автоматического решения «одобрить / отклонить» с учётом
экономики банка.

---

## Бизнес-задача

Для каждой заявки на кредит нужно принять решение:

- **APPROVE** — если ожидаемая прибыль положительна;
- **REJECT** — если ожидаемые потери превышают прибыль.

Модель выдаёт калиброванную вероятность дефолта `PD`, а порог одобрения подобран
не по метрике, а по **максимуму прибыли (P&L)** на OOF-предсказаниях.

Экономика решения:

| Параметр | Значение | Смысл |
|---|---|---|
| `LGD` | 0.60 | доля потерь при дефолте |
| `EAD` | 200 000 ₽ | средняя сумма под риском |
| `margin_good` | 20 000 ₽ | прибыль с одного хорошего клиента |

Следствие: пропущенный дефолт (`FN`) стоит **120 000 ₽**, а ошибочно отклонённый
хороший клиент (`FP`) теряет **20 000 ₽**. То есть `FN` в 6 раз дороже `FP` —
именно поэтому порог смещён в сторону осторожности.

---

## Данные

Датасет **Default of Credit Card Clients** (UCI):
30 000 клиентов, 25 колонок, целевая — `default payment next month`.

- 23 признака: лимит, пол, образование, семейное положение, возраст,
  история платежей `PAY_0..PAY_6`, суммы счетов `BILL_AMT1..6`,
  суммы платежей `PAY_AMT1..6`;
- дисбаланс классов ≈ **78 % / 22 %**;
- пропусков нет, удалено 35 полных дубликатов;
- обнаружены и исправлены недокументированные коды `EDUCATION` (0, 5, 6) и
  `MARRIAGE` (0);
- отрицательные `BILL_AMT` оставлены — это сигнал переплаты, а не ошибка.

Данные **не входят в репозиторий**. Положи `data.xls` в корень проекта перед
запуском `1.ipynb`.

---

## Этапы проекта

```
1.ipynb  →  EDA + feature engineering + train/test split
2.ipynb  →  CatBoost + Optuna + калибровка + подбор порога
3.ipynb  →  SHAP-интерпретация + анализ ошибок + P&L
app.py   →  Flask API для inference
Docker   →  контейнеризация сервиса
```

### 1. `1.ipynb` — EDA и подготовка данных

- очистка данных, rule-based правки категорий;
- **28 новых признаков**: агрегаты за 6 месяцев (`avg_bill`, `std_bill`,
  `total_payment`), отношения (`utilization`, `payment_ratio`, `limit_per_age`),
  тренды (`bill_trend`, `delay_trend`), статистики просрочек (`max_delay`,
  `months_with_debt`, `sum_positive_delay`), взаимодействия
  (`bill_x_age`, `pay_x_pay0`), лог-признаки;
- **adversarial validation** `AUC = 0.51` → train и test однородны, обычный
  стратифицированный split валиден;
- **Mutual Information**: engineered-признаки (`sum_positive_delay`,
  `months_with_debt`, `mean_delay`, `max_delay`) обошли сырые `PAY_2..PAY_6`,
  что подтверждает ценность FE;
- артефакты: `train_processed.parquet`, `test_processed.parquet`,
  `cat_features.json`, `preprocessing_artifacts.pkl`, `summary.json`.

### 2. `2.ipynb` — обучение моделей

- бейзлайны: `Dummy`, `LogisticRegression`, `XGBoost`, `LightGBM`;
- тюнинг `CatBoost` через **Optuna** (TPE + MedianPruner), objective =
  `0.5 · ROC-AUC + 0.5 · PR-AUC`;
- 5-fold **OOF**-предсказания;
- калибровка **Platt scaling** (логистическая регрессия на logit-вероятностях)
  поверх OOF;
- подбор порога по максимуму P&L на OOF;
- бизнес-метрики: approval rate, bad rate, Expected Loss, P&L, Bootstrap-CI.

### 3. `3.ipynb` — интерпретация

- **TreeSHAP** на 3 000 объектах;
- глобальная важность `mean(|SHAP|)` и сравнение с CatBoost FI;
- beeswarm, bar, dependence plots;
- **waterfall** для трёх клиентов: BAD / BORDERLINE / GOOD;
- перевод топ-15 фич на бизнес-язык;
- анализ ошибок `FN` / `FP` с денежной оценкой.

---

## Результаты

### Качество на тесте

| Метрика | Значение |
|---|---|
| ROC-AUC | **0.7819** |
| PR-AUC | **0.5637** |
| Brier | 0.1344 |
| ECE | **0.0130** |
| KS | 0.42 |
| Gini | 0.56 |

### Порог и экономика

| Параметр | Значение |
|---|---|
| Оптимальный порог | **0.1288** |
| TN (хорошие одобрены) | 2 555 |
| FP (хорошие отклонены) | 2 112 |
| FN (плохие одобрены) | 243 |
| TP (плохие отклонены) | 1 083 |
| **P&L** | **≈ 22.2 млн ₽** |

### Топ-5 драйверов дефолта (SHAP)

| # | Признак | mean\|SHAP\| | Бизнес-смысл |
|---|---|---|---|
| 1 | `PAY_0` | 0.251 | свежесть последнего платежа |
| 2 | `max_delay` | 0.116 | максимальная задержка за 6 мес. |
| 3 | `months_with_debt` | 0.092 | число месяцев с задолженностью |
| 4 | `sum_positive_delay` | 0.091 | сумма положительных задержек |
| 5 | `utilization` | 0.068 | утилизация лимита |

**Ключевой инсайт:** дефолт определяет не размер долга, а **поведение по его
обслуживанию**. Признак `total_debt` одинаков у дефолтных и недефолтных
(≈ 238k ₽), а `payment_ratio` у дефолтных в 5 раз ниже.

---

## Структура проекта

```
Credit_Scoring/
├── 1.ipynb                       # EDA + FE
├── 2.ipynb                       # модели + калибровка + порог
├── 3.ipynb                       # SHAP + анализ ошибок
├── app.py                        # Flask API
├── templates/                    # HTML формы для app.py
├── Dockerfile
├── requirements.txt
├── .gitignore
├── .dockerignore
├── README.md
├── data.xls                      # ← не коммитится, положить вручную
├── artifacts/                    # ← генерируется 1.ipynb
└── models/                       # ← генерируется 2.ipynb
```

---

## Установка и запуск

### 1. Клонировать репозиторий

```bash
git clone https://github.com/ТВОЙ_ЛОГИН/credit-scoring.git
cd credit-scoring
```

### 2. Создать окружение

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
```

### 3. Положить данные

Скачай `data.xls` (UCI Default of Credit Card Clients) и положи в корень проекта.

### 4. Прогнать ноутбуки по порядку

```
1.ipynb → 2.ipynb → 3.ipynb
```

После `1.ipynb` появится папка `artifacts/`,
после `2.ipynb` — папка `models/` и файлы `catboost_final.cbm`,
`calibrator.pkl`, `opt_threshold.json`, `business_metrics.json`.

### 5. Запустить Flask-сервис

```bash
python app.py
```

Открой http://localhost:5000 — форма скоринга.
Проверка: http://localhost:5000/health.

---

## Запуск через Docker

### Сборка

```bash
docker build -t credit-scoring .
```

### Запуск

```bash
docker run --rm -p 5000:5000 credit-scoring
```

Если модели и артефакты лежат локально, а не внутри образа — примонтируй их:

```bash
docker run --rm -p 5000:5000 ^
  -v "%cd%/models:/app/models" ^
  -v "%cd%/artifacts:/app/artifacts" ^
  credit-scoring
```

Для Linux / macOS замени `%cd%` на `$(pwd)`.


## Flask API

| Метод | Endpoint | Назначение |
|---|---|---|
| `GET` | `/health` | статус сервиса, версия модели, порог |
| `GET` | `/model_info` | список фич, cat_features, бизнес-параметры |
| `GET` | `/` | HTML-форма скоринга |
| `POST` | `/` | расчёт PD и решение |

Сервис поддерживает **thin-file сценарий**: если у клиента нет кредитной
истории, вместо ML-модели применяется policy на основе возраста, лимита и
образования.

Каждый расчёт пишется в audit-log одной JSON-строкой — с `request_id`,
версией модели и версией policy.

---

## Технологии

**Данные и ML:** `pandas`, `numpy`, `scikit-learn`, `catboost`, `xgboost`,
`lightgbm`, `optuna`, `shap`.

**Визуализация:** `matplotlib`, `seaborn`.

**Сервис:** `Flask`, `gunicorn`.

**Инфраструктура:** `Docker`, `Git`.

