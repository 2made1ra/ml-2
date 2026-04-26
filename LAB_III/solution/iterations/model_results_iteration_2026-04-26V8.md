# Итерация V8: LiteRes custom CNN и ограничение ViT fine-tuning

Документ фиксирует изменения V8 в `LAB_III/solution/lab_3.ipynb`.

---

## База сравнения

Лучший custom baseline из прошлых итераций:

| Модель | accuracy | macro F1 | Итерация |
| --- | ---: | ---: | --- |
| `Custom WikiArtTinyCNN` | `0.4187` | `0.38` | V5 |

Лучший общий результат предобученной модели:

| Модель | accuracy | macro F1 | Итерация |
| --- | ---: | ---: | --- |
| `ViT-B/16 ImageNet transfer/fine-tune` | `0.6675` | `0.65` | V6 |

V7 показала, что ViT validation accuracy почти выходит на плато после 13-14 эпох partial fine-tuning: лучший `val_acc=0.6710` был на 17-й эпохе, но прирост после 14-й эпохи минимальный на фоне роста train accuracy.

---

## Изменения V8

### Config

| Параметр | V7 | V8 |
| --- | ---: | ---: |
| `vit_finetune_epochs` | `20` | `14` |
| `early_stopping_patience` | `6` | `3` |
| `custom_epochs` | `5` | `5` |
| `lr_custom` | `1e-3` | `1e-3` |
| `weight_decay` | `1e-4` | `1e-4` |

Без изменений:

- `pretrained_model_name = "google/vit-base-patch16-224"`;
- `vit_head_epochs = 3`;
- `vit_unfreeze_layers = 6`;
- `vit_lr_finetune = 6e-6`;
- `vit_classifier_lr_finetune = 5e-5`;
- `sampler_alpha = 0.75`;
- `Unknown Genre` остается в 11-классовой задаче.

### Custom model

`WikiArtTinyCNN` заменена на `WikiArtLiteResCNN`.

Архитектура:

- stem: `ConvNormAct(3, 32, stride=2)`;
- stage 1: `DepthwiseSeparableBlock(32, 64, stride=2)`, затем residual `64 -> 64`;
- stage 2: `DepthwiseSeparableBlock(64, 128, stride=2)`, затем residual `128 -> 128`;
- stage 3: `DepthwiseSeparableBlock(128, 160, stride=2)`, затем residual `160 -> 160`;
- head: `1x1 ConvNormAct(160, 192)`, `AdaptiveAvgPool2d(1)`, `Dropout(0.25)`, `Linear(192, NUM_CLASSES)`.

Residual skip применяется только при `stride=1` и одинаковом числе каналов.

### ViT

- full fine-tuning не используется;
- нижние 6 из 12 encoder layers остаются frozen;
- обучаются classifier head и верхние 6 encoder layers;
- fine-tuning ограничен 14 эпохами;
- checkpoint по-прежнему выбирается по лучшему `val_acc`.

---

## Метрики после запуска

Заполнено по текущему сохранённому output из `lab_3.ipynb`.

| Модель | `test_loss` | accuracy | macro precision | macro recall | macro F1 | Время |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Custom WikiArtLiteResCNN` | n/a в output | 0.3742 | 0.32 | 0.41 | 0.33 | n/a в output |
| `ViT-B/16 ImageNet partial fine-tune` | n/a в output | 0.6632 | 0.62 | 0.71 | 0.65 | n/a в output |

Критерии проверки:

- custom-модель должна приблизиться к V5 или превысить `accuracy=0.4187`, `macro F1=0.38`;
- ViT должен сохранить уровень около `accuracy >= 0.66`, `macro F1 ~= 0.65`;
- время ViT fine-tuning должно снизиться относительно V7 за счет лимита 14 эпох и `patience=3`.

Факт по текущему сохранению:

- custom-модель пока ниже целевого уровня V5 (`accuracy 0.3742`, `macro F1 0.33`);
- ViT сохраняет целевой уровень (`accuracy 0.6632`, `macro F1 0.65`);
- проверить снижение времени по текущему `ipynb` нельзя: строки `finished in ... min` в сохранённом output отсутствуют.

---

## Fallback

- Если custom CNN переобучается: увеличить `Dropout(0.25)` до `0.30`.
- Если custom CNN не добирает качество V5: поднять `custom_epochs` до `6-8`, не меняя архитектуру.
- Если ViT заметно теряет качество: вернуть `early_stopping_patience=4`, но оставить `vit_finetune_epochs=14`.
