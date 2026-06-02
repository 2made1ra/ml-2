# Lab 5 — агрегированный отчет по экспериментам

## 1) Цель эксперимента

Сравнить три нейросетевые архитектуры для классификации тональности отзывов и выбрать лучшую модель по `macro-F1` на validation с дополнительной проверкой на test.

## 2) Конфигурация запуска

- `SEED`: `42`
- Число классов: `NUM_CLASSES = len(LABEL_ORDER)`
- Предобработка:
  - `PREPROCESS_MODE = "soft"`
  - `PREPROCESS_VERSION = "v3_rating_train_bigrams"`
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
| TrainableCNN | 3 | 0.715784 | 0.723081 | 0.781976 | 0.801975 | 0.531014 | 167.939237 | 22575427 |
| TrainableCNNBiGRU | 6 | 0.711117 | 0.716087 | 0.774002 | 0.789064 | 0.521109 | 890.697260 | 22663812 |
| TrainablePackedBiGRU | 5 | 0.702943 | 0.704528 | 0.781849 | 0.786279 | 0.500966 | 763.228147 | 21758724 |

### 5.2 Test (лучшие чекпоинты, raw vs calibrated)

| Модель | raw_macro_f1 | calibrated_macro_f1 | raw_accuracy | calibrated_accuracy | weighted_f1 | neutral_f1 | train_time | params |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TrainableCNN | 0.712333 | 0.714502 | 0.780170 | 0.796803 | 0.796555 | 0.514570 | 167.939237 | 22575427 |
| TrainableCNNBiGRU | 0.699888 | 0.701957 | 0.768588 | 0.781955 | 0.785730 | 0.491127 | 890.697260 | 22663812 |
| TrainablePackedBiGRU | 0.699950 | 0.700158 | 0.780360 | 0.784195 | 0.786116 | 0.489800 | 763.228147 | 21758724 |

## 6) Итоговый выбор модели

- Лучшая модель по validation `macro-F1`: `TrainableCNN`
- Ее `macro-F1`:
  - validation (calibrated): `0.723081`
  - test (calibrated): `0.714502`

## 7) Анализ ошибок

- По `classification_report` лучшей модели (`TrainableCNN`) на test:
  - `negative`: precision `0.80`, recall `0.69`, f1 `0.74` (support `3965`)
  - `neutral`: precision `0.51`, recall `0.52`, f1 `0.51` (support `4941`)
  - `positive`: precision `0.88`, recall `0.90`, f1 `0.89` (support `17428`)
  - `accuracy`: `0.80`
  - `macro avg`: precision `0.73`, recall `0.70`, f1 `0.71`
  - `weighted avg`: precision `0.80`, recall `0.80`, f1 `0.80`
- Наиболее сложный класс остается `neutral`: его F1 ниже `negative` и `positive`, но после текущего обучения он вырос до `0.51` на test.

## 8) Краткие выводы

1. Для несбалансированных классов основная метрика — `macro-F1`, а не accuracy.
2. Calibration следует оценивать отдельно от архитектуры, так как он корректирует смещение logits и в текущем запуске повышает `macro-F1` у всех трех моделей.
3. По совокупности метрик оптимальным выбором стала `TrainableCNN`: лучший `calibrated_val_macro_f1` и лучший `calibrated_macro_f1` на test при наименьшем времени обучения.

---

### Источник данных

Числа перенесены из output ячеек ноутбука:
- `validation_summary_df` (раздел `7.9`)
- `test_summary_df` и `Best by calibrated validation macro-F1` (раздел `7.12`)
- `classification_report` (раздел `7.13`)
