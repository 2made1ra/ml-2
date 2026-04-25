# Итерация V5: быстрый custom baseline и более тяжёлый ViT fine-tuning

Документ фиксирует плановые изменения V5 в `LAB_III/solution/lab_3.ipynb`. Метрики нужно заполнить после полного `Run All`, чтобы не смешивать результаты старых прогонов с новой конфигурацией.

---

## Цель

Сместить основной ресурс обучения на предобученную модель `google/vit-base-patch16-224`, а собственную CNN сделать более простой и быстрой:

- custom-модель: быстрый baseline `WikiArtTinyCNN`;
- pretrained-модель: тот же ViT-B/16 из V4, но с большим batch size, более глубокой разморозкой и более длинным fine-tuning;
- задача не меняется: `target_column = genre`;
- V3 baseline для сравнения: `EfficientNet-B0`, `accuracy = 0.5107`, `macro F1 = 0.4529`.

---

## Изменения `Config`

| Параметр | V5 |
| --- | --- |
| `custom_image_size` | `160` |
| `batch_size` | `512` |
| `custom_epochs` | `5` |
| `lr_custom` | `1e-3` |
| `weight_decay` | `1e-4` |
| `vit_batch_size` | `192` |
| `num_workers` | `2` |
| `vit_head_epochs` | `3` |
| `vit_finetune_epochs` | `10` |
| `vit_unfreeze_layers` | `4` |
| `early_stopping_patience` | `4` |
| `vit_lr_head` | `8e-4` |
| `vit_lr_finetune` | `1.5e-5` |
| `ram_budget_gb` | `12.0` |

Без изменений:

- `pretrained_model_name = "google/vit-base-patch16-224"`;
- `target_column = "genre"`;
- `max_samples = 40_000`;
- `vit_image_size = 224`;
- `balanced_train_sampler = True`;
- `use_class_weights = False`.

---

## Архитектура и обучение

Custom baseline заменён на `WikiArtTinyCNN`:

- `ConvNormAct(3, 32, stride=2)`;
- `ConvNormAct(32, 64, stride=2)`;
- `ConvNormAct(64, 128, stride=2)`;
- `ConvNormAct(128, 160, stride=2)`;
- `AdaptiveAvgPool2d(1)`;
- `Flatten`;
- `Dropout(0.20)`;
- `Linear(160, NUM_CLASSES)`.

Удалена custom-only сложность прошлой версии:

- `SqueezeExcite`;
- `StochasticDepth`;
- `LiteResidualSEBlock`;
- deep residual feature stack.

Custom transforms упрощены: остаются `RandomResizedCrop`, `RandomHorizontalFlip`, лёгкий `ColorJitter`, `ToTensor`, ImageNet normalize; `RandomGrayscale` и `RandomErasing` удалены.

ViT-модель не заменяется. Усилено только обучение:

- head training сокращён до `3` эпох;
- fine-tuning увеличен до `10` эпох;
- размораживаются последние `4` encoder layers;
- batch size увеличен до `192`;
- `num_workers=2` для роста throughput.

---

## GPU Utilization

На скриншоте RTX 4090 показывает около `19%` utilization и примерно `4.5 / 24.0 GB` dedicated GPU memory. В V5 это трактуется как практический сигнал увеличить `vit_batch_size`, число размороженных ViT-слоёв и загрузку данных.

Для точной диагностики CUDA-нагрузки желательно дополнительно смотреть `nvidia-smi`, потому что Windows Task Manager в режиме графика `3D` не всегда точно показывает compute-загрузку CUDA.

Fallback:

- если будет OOM, снизить `vit_batch_size` с `192` до `128`;
- если Windows/Jupyter DataLoader зависает при `num_workers=2`, вернуть `num_workers=0`.

---

## Метрики после запуска

Обновлено по текущему сохранённому output из `lab_3.ipynb` (последний запуск ячеек сравнения и отчётов).

| Модель | `test_loss` | accuracy | macro precision | macro recall | macro F1 | Время |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Custom WikiArtTinyCNN` | n/a в output | 0.4187 | 0.41 | 0.49 | 0.38 | n/a в output |
| `ViT-B/16 ImageNet transfer/fine-tune` | n/a в output | 0.6340 | 0.58 | 0.70 | 0.61 | 18.1 min (head 7.4 + finetune 10.7) |

Критерий успеха относительно V3:

- `accuracy > 0.5107`;
- `macro F1 > 0.4529`;
- рост качества должен идти от ViT, а не от custom baseline.

Факт по текущим результатам:

- `Custom WikiArtTinyCNN`: ниже V3 по `accuracy` и `macro F1`;
- `ViT-B/16`: выше V3 по `accuracy` и `macro F1`, целевой критерий выполнен.

---

## Примечания

- V5 не меняет pretrained architecture и не меняет целевую колонку.
- Если V5 окажется медленнее без прироста качества, следующая итерация должна сравнить `vit_unfreeze_layers=2/4/6` и `vit_batch_size=128/192` на одинаковом split.
