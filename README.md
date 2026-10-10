# CXR Conformal XAI (`cxr-conformal-xai`)

Фреймворк для оценки неопределенности (Uncertainty Quantification) и объяснимого искусственного интеллекта (Explainable AI / Grad-CAM) в задачах анализа рентгенограмм органов грудной клетки (Chest X-Ray, CXR).

---

## 📌 Архитектура и структура проекта

```text
cxr-conformal-xai/
├── data/                             # Датасеты (добавлены в .gitignore, тяжелые файлы не пушатся)
├── notebooks/                        # Ноутбуки для Colab / Jupyter
│   ├── 01_baseline_inference.ipynb   # Зона Напарника: ResNet + вычисление logits/вероятностей
│   └── 02_conformal_calibration.ipynb # Зона Тимлида: Conformal Prediction (MAPIE) + XAI (Grad-CAM)
├── src/                              # Переиспользуемый код и модули
│   ├── __init__.py
│   ├── conformal.py                  # Калибровка non-conformity scores и построение множеств C(x)
│   └── metrics.py                    # Coverage, средний размер множеств, связь с Grad-CAM
├── outputs/                          # Артефакты: predictions.csv, графики, примеры изображений
├── .gitignore                        # Исключения для Git
├── requirements.txt                  # Зафиксированные версии библиотек
└── README.md                         # Инструкция по установке и воспроизведению
```

---

## 👥 Разделение зон ответственности

| Участник | Ноутбук / Модуль | Основная задача |
|---|---|---|
| **Напарник** | [`notebooks/01_baseline_inference.ipynb`](file:///notebooks/01_baseline_inference.ipynb) | Загрузка модели (ResNet / TorchXRayVision), предобработка CXR, прогон инференса, экспорт сырых логитов и вероятностей в `outputs/predictions.csv`. |
| **Тимлид** | [`notebooks/02_conformal_calibration.ipynb`](file:///notebooks/02_conformal_calibration.ipynb) & [`src/`](file:///src/) | Конформная калибровка (MAPIE / LAC / APS), расчет гарантированного покрытия ($1 - \alpha$), оценка эффективности $|C(x)|$, анализ согласования тепловых карт Grad-CAM. |

---

## 📂 Data Contract (Формат передачи данных)

Инференс модели сохраняется в `outputs/predictions.csv` со следующими столбцами:
- `image_id`: имя файла рентгенограммы;
- `split`: выборка (`calib` или `test`);
- `true_label`: метка класса (0 — норма, 1 — патология);
- `prob_normal`: прогнозируемая вероятность нормы;
- `prob_pneumonia`: прогнозируемая вероятность патологии.

---

## 🚀 Быстрый старт и установка

### 1. Клонирование репозитория

```bash
git clone https://github.com/ecxqua/cxr-conformal-xai.git
cd cxr-conformal-xai
```

### 2. Создание и активация виртуального окружения

```bash
# Linux / macOS
python -m venv venv
source venv/bin/activate

# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Установка зависимостей

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Примечание по CUDA**: Для установки PyTorch с поддержкой вашей версии CUDA используйте инструкцию с [официального сайта PyTorch](https://pytorch.org/get-started/locally/).

---

## 🔬 Пайплайн воспроизведения экспериментов

### Шаг 1: Инференс базовой модели (Напарник)
Откройте ноутбук [`notebooks/01_baseline_inference.ipynb`](file:///notebooks/01_baseline_inference.ipynb):
1. Загрузите рентгеновские снимки в папку `data/`.
2. Запустите инференс предобученной модели ResNet / TorchXRayVision.
3. Убедитесь, что файл с предсказаниями сохранен в `outputs/predictions.csv`.

### Шаг 2: Конформная калибровка и XAI (Тимлид)
Откройте ноутбук [`notebooks/02_conformal_calibration.ipynb`](file:///notebooks/02_conformal_calibration.ipynb):
1. Загрузите полученные вероятности из `outputs/predictions.csv`.
2. Разделите данные на калибровочную ($D_{cal}$) и тестовую ($D_{test}$) выборки.
3. Рассчитайте калибровочный квантиль $\hat{q}$ для целевого уровня надежности $1 - \alpha$ (например, 90% при $\alpha = 0.1$).
4. Постройте множества предсказаний $C(x)$.
5. Оцените метрики:
   - **Empirical Coverage**: фактическая доля покрытых истинных меток ($\ge 1 - \alpha$).
   - **Average Set Size**: средний размер предсказательного множества $|C(x)|$.
   - **Grad-CAM Alignment Score**: степень пространственного наложения карт внимания диагнозов, входящих в неоднозначные множества ($|C(x)| > 1$).

---

## 📊 Описание модулей в `src/`

- **`src.conformal`**:
  - `compute_nonconformity_scores(probs, labels, method='lac')`: вычисление оценок несоответствия.
  - `calibrate_threshold(scores, alpha=0.1)`: нахождение квантиля $\hat{q}$ с поправкой на конечную выборку.
  - `predict_conformal_sets(probs, q_hat, method='lac')`: генерация предиктивных множеств $C(x)$.
  - `evaluate_conformal_coverage(prediction_sets, true_labels)`: быстрая оценка покрытия и размера множеств.

- **`src.metrics`**:
  - `calculate_coverage(...)`: расчет покрытия выборки.
  - `calculate_average_set_size(...)`: расчет средней кардинальности множеств.
  - `calculate_set_size_distribution(...)`: частотное распределение размеров множеств.
  - `conformal_classification_metrics(...)`: сводный отчет по всем ключевым метрикам калибровки.
  - `cam_conformal_alignment_score(...)`: метрика пространственного согласия / расхождения между Grad-CAM картами классов внутри предиктивного множества.

