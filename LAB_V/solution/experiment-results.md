# Lab 5 — агрегированный отчет по экспериментам

## 1) Цель эксперимента

Сравнить три нейросетевые архитектуры для классификации тональности отзывов и выбрать лучшую модель по `macro-F1` на validation с дополнительной проверкой на test.

## 2) Конфигурация запуска

- `SEED`: `42`
- Число классов: `NUM_CLASSES = len(LABEL_ORDER)`
- Длины последовательностей:
  - `CNN_MAX_LEN = 512`
  - `RNN_MAX_LEN = 400`
  - `COMBINED_MAX_LEN = 512`
- Размеры batch:
  - `CNN_BATCH_SIZE = 512`
  - `RNN_BATCH_SIZE = 256`
  - `COMBINED_BATCH_SIZE = 256`
- Обучение:
  - `MAX_EPOCHS = 15`
  - `PATIENCE = 4`
  - `criterion = CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)`
  - `scheduler = ReduceLROnPlateau` по validation `macro-F1`
- `RUN_SEED_ENSEMBLE = False` (ensemble отключен)

## 3) Сравниваемые модели

1. `TrainableCNN`  
   Embedding + multi-kernel Conv1D + max/mean pooling
2. `TrainablePackedBiGRU`  
   Embedding + packed bidirectional GRU + final/mean/max pooling
3. `TrainableCNNBiGRU`  
   Параллельные ветки Conv1D и packed BiGRU

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
| TrainableCNN | 9 | 0.630229 | 0.632007 | 0.728372 | 0.729701 | 0.409657 | 184.745183 | 14575427 |
| TrainablePackedBiGRU | 7 | 0.634017 | 0.640036 | 0.731220 | 0.740649 | 0.397672 | 526.170627 | 13495043 |
| TrainableCNNBiGRU | 9 | 0.636386 | 0.637774 | 0.728435 | 0.731283 | 0.406504 | 713.572685 | 14466051 |

### 5.2 Test (лучшие чекпоинты, raw vs calibrated)

| Модель | raw_macro_f1 | calibrated_macro_f1 | raw_accuracy | calibrated_accuracy | weighted_f1 | neutral_f1 | train_time | params |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TrainableCNN | 0.629158 | 0.627437 | 0.727956 | 0.727539 | 0.733011 | 0.398916 | 184.745183 | 14575427 |
| TrainablePackedBiGRU | 0.635414 | 0.638981 | 0.733083 | 0.740943 | 0.743359 | 0.399259 | 526.170627 | 13495043 |
| TrainableCNNBiGRU | 0.636427 | 0.635552 | 0.729475 | 0.731868 | 0.739556 | 0.404472 | 713.572685 | 14466051 |

## 6) Итоговый выбор модели

- Лучшая модель по validation `macro-F1`: `TrainablePackedBiGRU`
- Ее `macro-F1`:
  - validation (calibrated): `0.640036`
  - test (calibrated): `0.638981`

## 7) Анализ ошибок

- По `classification_report` лучшей модели (`TrainablePackedBiGRU`) на test:
  - `negative`: precision `0.68`, recall `0.64`, f1 `0.66` (support `3965`)
  - `neutral`: precision `0.39`, recall `0.41`, f1 `0.40` (support `4941`)
  - `positive`: precision `0.86`, recall `0.86`, f1 `0.86` (support `17428`)
  - `accuracy`: `0.74`
  - `macro avg`: precision `0.64`, recall `0.64`, f1 `0.64`
  - `weighted avg`: precision `0.75`, recall `0.74`, f1 `0.74`
- Наиболее сложный класс остается `neutral`: его F1 заметно ниже `negative` и `positive`, что согласуется с ожидаемыми смешанными формулировками в нейтральных отзывах.

## 8) Краткие выводы

1. Для несбалансированных классов основная метрика — `macro-F1`, а не accuracy.
2. Calibration следует оценивать отдельно от архитектуры, так как он корректирует смещение logits.
3. По совокупности метрик оптимальным выбором стала `TrainablePackedBiGRU`: лучший `calibrated_val_macro_f1` и лучший `calibrated_macro_f1` на test.

---

### Источник данных

Числа перенесены из output ячеек ноутбука:
- `validation_summary_df` (раздел `7.9`)
- `test_summary_df` и `Best by calibrated validation macro-F1` (раздел `7.12`)
- `classification_report` (раздел `7.13`)
