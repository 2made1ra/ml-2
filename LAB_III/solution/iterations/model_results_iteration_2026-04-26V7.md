# Итерация V7: partial fine-tuning и поддержка малых классов

Документ фиксирует плановые изменения V7 в `LAB_III/solution/lab_3.ipynb`. Метрики V7 нужно заполнить после полного `Run All`.

---

## База V6

Последний зафиксированный прогон V6:

| Модель | accuracy | macro F1 | Время |
| --- | ---: | ---: | ---: |
| `Custom WikiArtTinyCNN` | `0.3560` | `0.33` | `8.0 min` |
| `ViT-B/16 ImageNet transfer/fine-tune` | `0.6675` | `0.65` | `31.4 min` |

V3 baseline для исторического сравнения:

- `EfficientNet-B0`: `accuracy = 0.5107`, `macro F1 = 0.4529`.

---

## Требование лабораторной

Полный fine-tuning всей предобученной сети не используется. V7 соблюдает partial fine-tuning:

- нижние ViT encoder layers остаются замороженными;
- обучаются classifier head и только верхние encoder layers;
- модель `google/vit-base-patch16-224` не заменяется.

В V7 размораживаются последние `6` из `12` encoder layers, то есть нижняя половина ViT остаётся frozen.

---

## Изменения `Config`

| Параметр | V6 | V7 |
| --- | ---: | ---: |
| `vit_unfreeze_layers` | `7` | `6` |
| `vit_finetune_epochs` | `16` | `20` |
| `early_stopping_patience` | `5` | `6` |
| `vit_lr_head` | `6e-4` | `5e-4` |
| `vit_lr_finetune` | `8e-6` | `6e-6` |
| `vit_classifier_lr_finetune` | n/a | `5e-5` |
| `vit_weight_decay` | `0.05` | `0.08` |
| `max_grad_norm` | n/a | `1.0` |
| `custom_epochs` | `8` | `5` |
| `lr_custom` | `8e-4` | `1e-3` |
| `sampler_alpha` | n/a | `0.75` |
| `rare_class_threshold` | n/a | `1500` |

Без изменений:

- `pretrained_model_name = "google/vit-base-patch16-224"`;
- `target_column = "genre"`;
- `max_samples = 40_000`;
- `vit_batch_size = 128`;
- `num_workers = 0`;
- `balanced_train_sampler = True`;
- `use_class_weights = False`;
- `Unknown Genre` остаётся в 11-классовой задаче.

---

## Обучение и данные

ViT fine-tuning:

- head training: `3` эпохи, backbone полностью frozen;
- partial fine-tuning: последние `6` encoder layers + classifier head;
- две LR-группы:
  - верхние ViT layers: `6e-6`;
  - classifier head: `5e-5`;
- gradient clipping: `max_grad_norm = 1.0`;
- checkpoint выбирается по лучшему `val_acc`.

Малые классы:

- `Unknown Genre` не удаляется из train/val/test;
- oversampling сглажен: `weight = 1 / count[label] ** 0.75`;
- train-классы с количеством меньше `1500` получают умеренно усиленный train transform:
  - `RandomResizedCrop(scale=(0.55, 1.0))`;
  - немного более сильный `ColorJitter`.

Custom baseline:

- остаётся `WikiArtTinyCNN` из V6;
- расписание возвращено к более быстрому V5-режиму: `custom_epochs = 5`, `lr_custom = 1e-3`.

---

## Метрики после запуска

Заполнено по текущему сохранённому output из `lab_3.ipynb` (последний прогон V7).

| Модель | `test_loss` | accuracy | macro precision | macro recall | macro F1 | Время |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Custom WikiArtTinyCNN` | n/a в output | 0.3742 | 0.32 | 0.41 | 0.33 | 5.4 min |
| `ViT-B/16 ImageNet partial fine-tune` | n/a в output | 0.6632 | 0.62 | 0.71 | 0.65 | 40.5 min (head 5.0 + partial finetune 35.5) |

Критерий успеха:

- ViT accuracy выше `0.6675` или macro F1 выше `0.65`;
- precision/recall малых классов улучшаются без удаления `Unknown Genre`;
- реализация остаётся partial fine-tuning, не full fine-tuning.

Факт по текущему прогону:

- ViT partial fine-tuning сохранён (требование лабораторной соблюдено).
- По quality-порогам относительно V6: `accuracy 0.6632` ниже `0.6675`, `macro F1 0.65` на уровне порога, но не выше.
- Custom baseline остаётся заметно ниже V5/V6 по качеству (`accuracy 0.3742`, `macro F1 0.33`).

---

## Fallback

- Если будет OOM: снизить `vit_batch_size` с `128` до `96`.
- Если fine-tuning переобучается: снизить `vit_finetune_epochs` до `14` или увеличить `vit_weight_decay` только в следующей итерации.
- Если редкие классы переобучаются: снизить `sampler_alpha` до `0.6`.
