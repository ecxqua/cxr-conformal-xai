# CXR Conformal XAI (`cxr-conformal-xai`)

Фреймворк для оценки неопределенности (Uncertainty Quantification) и объяснимого искусственного интеллекта (Explainable AI / Grad-CAM) в задачах анализа рентгенограмм органов грудной клетки (Chest X-Ray, CXR)


## 📌 Архитектура и структура проекта

```text
cxr-conformal-xai/
├── data/                             # Датасеты (добавлены в .gitignore, тяжелые файлы не пушатся)
├── notebooks/                        # Ноутбуки для Colab / Jupyter
│   ├── 01_baseline_inference.ipynb   # Зона Напарника: ResNet + вычисление logits/вероятностей
│   ├── 02_conformal_calibration.ipynb # Зона Тимлида: Conformal Prediction (MAPIE) + XAI (Grad-CAM)
│   ├── 011_grad_cam.ipynb # Формирование карт внимания Grad-CAM
│   ├── 012_inference_table.ipynb # Инференс модели для формирования predictions.csv
│   └── 03_statistical_analysis_week3.ipynb # Зона Тимлида: статистический анализ результатов
├── outputs/                          # Артефакты: predictions.csv, графики, примеры изображений
├── .gitignore                        # Исключения для Git
├── requirements.txt                  # Зафиксированные версии библиотек
├── article_base.pdf                  # Черновик статьи
└── README.md                         # Инструкция по установке и воспроизведению
```

---

## 👥 Разделение зон ответственности

| Участник | Ноутбук / Модуль | Основная задача |
|---|---|---|
| **Напарник** | [`notebooks/01_baseline_inference.ipynb`](file:///notebooks/01_baseline_inference.ipynb) | Загрузка модели (ResNet / TorchXRayVision), предобработка CXR, прогон инференса, экспорт сырых логитов и вероятностей в `outputs/predictions.csv`. |
| **Тимлид** | [`notebooks/02_conformal_calibration.ipynb`](file:///notebooks/02_conformal_calibration.ipynb) & [`src/`](file:///src/) | Конформная калибровка (MAPIE / LAC / APS), расчет гарантированного покрытия ($1 - \alpha$), оценка эффективности $\|C(x)\|$, анализ согласования тепловых карт Grad-CAM. |

---

## 📂 Data Contract (Формат передачи данных)

Инференс модели сохраняется в `outputs/predictions.csv` со следующими столбцами:
- `image_id`: имя файла рентгенограммы (например, `IM-0140-0001.jpeg`, `person1_bacteria_1.jpeg`);
- `split`: выборка (`calib` — для калибровки квантиля $\hat{q}$, `test` — для расчета предиктивных множеств $C(x)$ и построения карт Grad-CAM);
- `true_label`: целочисленная метка истинного класса:
  - `0`: норма (`NORMAL`)
  - `1`: бактериальная пневмония (`BACTERIAL`)
  - `2`: вирусная пневмония (`VIRAL`)
- `prob_normal`: прогнозируемая вероятность нормы ($P(\text{NORMAL}) \in [0.0, 1.0]$);
- `prob_bacterial`: прогнозируемая вероятность бактериальной пневмонии ($P(\text{BACTERIAL}) \in [0.0, 1.0]$);
- `prob_viral`: прогнозируемая вероятность вирусной пневмонии ($P(\text{VIRAL}) \in [0.0, 1.0]$).

> **Валидация распределения:** для каждой записи сумма вероятностей должна удовлетворять условию:  
> `prob_normal + prob_bacterial + prob_viral == 1.0` (выход слоя Softmax).

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
