# Lab 5 — агрегированный отчет по экспериментам

## 1) Цель эксперимента

Сравнить три нейросетевые архитектуры для классификации тональности отзывов и выбрать лучшую модель по `macro-F1` на validation с дополнительной проверкой на test.

## 2) Конфигурация запуска

- `SEED`: `42`
- Число классов: `NUM_CLASSES = len(LABEL_ORDER)`
- Предобработка:
  - `PREPROCESS_MODE = "soft"`
  - `PREPROCESS_VERSION = "v4_rating_train_bigrams"`
  - `USE_RATING_TOKENS = True`
  - `USE_FREQUENT_BIGRAMS = True`
- Длины последовательностей:
  - `CNN_MAX_LEN = 768`
  - `RNN_MAX_LEN = 640`
  - `COMBINED_MAX_LEN = 768`
- Размеры batch:
  - `CNN_BATCH_SIZE = 512`
  - `RNN_BATCH_SIZE = 256`
  - `COMBINED_BATCH_SIZE = 256`
- Обучение:
  - `MAX_EPOCHS = 25`
  - `PATIENCE = 5`
  - `BALANCE_STRATEGY = "sampler"` (`WeightedRandomSampler`)
  - `criterion = CrossEntropyLoss(weight=None, label_smoothing=0.02)`
  - `scheduler = ReduceLROnPlateau` по validation `macro-F1`
- `RUN_SEED_ENSEMBLE = False` (ensemble отключен)

## 3) Сравниваемые модели

1. `TrainableCNN`  
   Embedding + multi-kernel Conv1D + max/mean pooling
2. `TrainablePackedBiGRU`  
   Embedding + packed bidirectional GRU + final/mean/max/attention pooling
3. `TrainableCNNBiGRU`  
   Параллельные ветки Conv1D и packed BiGRU с attention pooling

## 4) Логика оценки

- Первичный критерий выбора: `calibrated_val_macro_f1`.
- На test считаются raw и calibrated метрики.
- Для лучшей модели дополнительно строятся:
  - `classification_report`
  - `confusion_matrix`
- Calibration делается через подбор `logit bias` на validation.

## 5) Сводка результатов

Ниже агрегированы фактические значения из output ячеек `7.9`, `7.12` и `7.13` ноутбука.

### 5.1 Validation (после calibration)

| Модель | best_epoch | raw_val_macro_f1 | calibrated_val_macro_f1 | raw_val_accuracy | calibrated_val_accuracy | neutral_f1 | train_time | params |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TrainableCNN | 3 | 0.707477 | 0.712266 | 0.782039 | 0.785710 | 0.514011 | 178.530544 | 22575427 |
| TrainableCNNBiGRU | 12 | 0.708073 | 0.711226 | 0.779191 | 0.789634 | 0.507767 | 1331.143338 | 22663812 |
| TrainablePackedBiGRU | 9 | 0.695935 | 0.696633 | 0.785773 | 0.784507 | 0.483127 | 1062.504230 | 21758724 |

### 5.2 Test (лучшие чекпоинты, raw vs calibrated)

| Модель | raw_macro_f1 | calibrated_macro_f1 | raw_accuracy | calibrated_accuracy | weighted_f1 | neutral_f1 | train_time | params |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TrainableCNN | 0.702034 | 0.700158 | 0.779866 | 0.780132 | 0.784099 | 0.495572 | 178.530544 | 22575427 |
| TrainableCNNBiGRU | 0.692969 | 0.692828 | 0.768816 | 0.777474 | 0.779901 | 0.475029 | 1331.143338 | 22663812 |
| TrainablePackedBiGRU | 0.681901 | 0.685846 | 0.776942 | 0.777094 | 0.775915 | 0.455980 | 1062.504230 | 21758724 |

## 6) Итоговый выбор модели

- Лучшая модель по validation `macro-F1`: `TrainableCNN`
- Ее `macro-F1`:
  - validation (calibrated): `0.712266`
  - test (calibrated): `0.700158`

## 7) Анализ ошибок

- По `classification_report` лучшей модели (`TrainableCNN`) на test:
  - `negative`: precision `0.74`, recall `0.71`, f1 `0.73` (support `3965`)
  - `neutral`: precision `0.47`, recall `0.53`, f1 `0.50` (support `4941`)
  - `positive`: precision `0.89`, recall `0.87`, f1 `0.88` (support `17428`)
  - `accuracy`: `0.78`
  - `macro avg`: precision `0.70`, recall `0.70`, f1 `0.70`
  - `weighted avg`: precision `0.79`, recall `0.78`, f1 `0.78`
- Наиболее сложный класс остается `neutral`: его F1 ниже `negative` и `positive`, в текущем запуске он равен `0.50` на test.

## 8) Краткие выводы

1. Для несбалансированных классов основная метрика — `macro-F1`, а не accuracy.
2. Calibration следует оценивать отдельно от архитектуры: на validation она повышает `macro-F1` у всех трех моделей, но на test улучшение сохраняется только у `TrainablePackedBiGRU`; у `TrainableCNN` и `TrainableCNNBiGRU` calibrated `macro-F1` немного ниже raw.
3. По validation-критерию оптимальным выбором стала `TrainableCNN`: лучший `calibrated_val_macro_f1` и минимальное время обучения среди трех архитектур. На test у нее также лучший raw и calibrated `macro-F1`, но отрыв от `TrainableCNNBiGRU` стал меньше, чем в предыдущем запуске.

---

### Источник данных

Числа перенесены из output ячеек ноутбука:
- `validation_summary_df` (раздел `7.9`)
- `test_summary_df` и `Best by calibrated validation macro-F1` (раздел `7.12`)
- `classification_report` (раздел `7.13`)
