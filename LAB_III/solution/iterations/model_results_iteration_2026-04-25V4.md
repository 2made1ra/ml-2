# Итерация V4: переход на `google/vit-base-patch16-224`

Документ фиксирует внесённые изменения в `LAB_III/solution/lab_3.ipynb`. Метрики нужно заполнить после полного запуска ноутбука, чтобы не смешивать результаты старого `EfficientNet-B0`/`EfficientNet-V2-S` прогона с новой ViT-архитектурой.

---

## Цель

Улучшить transfer-learning baseline для мультиклассовой классификации жанров WikiArt (`target_column = genre`) за счёт универсального ImageNet-pretrained backbone:

- новая предобученная модель: `google/vit-base-patch16-224`;
- прежний baseline для сравнения из V3: `EfficientNet-B0`, `accuracy = 0.5107`, `macro F1 = 0.4529`;
- своя CNN остаётся в ноутбуке как обязательный baseline лабораторной.

---

## Изменения в архитектуре и обучении

- Добавлена зависимость `transformers`.
- Transfer-learning ветка заменена на ViT-B/16 через `AutoModelForImageClassification`.
- ImageNet-классификатор заменяется на голову с `NUM_CLASSES` выходами через `num_labels`, `id2label`, `label2id`, `ignore_mismatched_sizes=True`.
- На первой фазе заморожен `model.vit`, обучается только новая голова.
- На второй фазе размораживаются последние `CFG.vit_unfreeze_layers = 2` encoder layers.
- Добавлен early stopping по `val_acc` с `CFG.early_stopping_patience = 3`.
- Для ViT preprocessing берётся из `AutoImageProcessor`: вход `224x224`, mean/std модели.

---

## Ключевые параметры

| Параметр | Значение |
| --- | --- |
| `pretrained_model_name` | `google/vit-base-patch16-224` |
| `target_column` | `genre` |
| `max_samples` | `40_000` |
| `vit_image_size` | `224` |
| `vit_batch_size` | `96` |
| `vit_head_epochs` | `4` |
| `vit_finetune_epochs` | `6` |
| `vit_lr_head` | `1e-3` |
| `vit_lr_finetune` | `2e-5` |
| `vit_weight_decay` | `0.05` |
| `balanced_train_sampler` | `True` |
| `use_class_weights` | `False` |

---

## Метрики после запуска

Заполнить после полного `Run All`.

| Модель | `test_loss` | accuracy | macro precision | macro recall | macro F1 | Время |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Custom WikiArtLiteCNN` | TBD | TBD | TBD | TBD | TBD | TBD |
| `ViT-B/16 ImageNet transfer/fine-tune` | TBD | TBD | TBD | TBD | TBD | TBD |

Критерий успеха относительно V3:

- `accuracy > 0.5107`;
- `macro F1 > 0.4529`;
- желательно `val_acc >= 0.58` при умеренном времени обучения.

---

## Примечания

- Оценка около `85%` по стилям не переносится напрямую на текущую задачу, потому что ноутбук классифицирует `genre`, а не `style`.
- Если будет OOM на RTX 4090, первым шагом уменьшить `CFG.vit_batch_size` с `96` до `64`.
- После запуска нужно обновить confusion matrix, top-10 пар ошибок, число ошибочных примеров и выводы в конце ноутбука.
