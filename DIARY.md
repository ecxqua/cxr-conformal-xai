# Дневник исследований
## Недели 1-2: 🔬 Пайплайн воспроизведения экспериментов

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

## 🔬 Неделя 3: Статистический анализ связки конформных множеств и Grad-CAM

### Артефакты Недели 3:

ДОРАБОТАТЬ! НЕЙРОСЛОП! ВЫВОДЫ НЕВЕРНЫЕ! ОПИРАЙТЕСЬ НА PDF!

- Ноутбук анализа: [`notebooks/03_statistical_analysis_week3.ipynb`](notebooks/03_statistical_analysis_week3.ipynb)
- Модуль статистической оценки: [`src/stats_evaluation.py`](src/stats_evaluation.py)
- Датасет метрик: [`outputs/test_conformal_xai_metrics.csv`](outputs/test_conformal_xai_metrics.csv)
- Сводный отчет в JSON: [`outputs/statistical_summary_week3.json`](outputs/statistical_summary_week3.json)
- Графики публикации:
  - [`outputs/figure1_attention_area_comparison.png`](outputs/figure1_attention_area_comparison.png) — боксплот площади внимания
  - [`outputs/figure2_spatial_entropy_comparison.png`](outputs/figure2_spatial_entropy_comparison.png) — скрипичная диаграмма энтропии
  - [`outputs/figure3_multiclass_trends.png`](outputs/figure3_multiclass_trends.png) — динамика по размерам множеств $|C(x)| \in \{1, 2, 3\}$
  - [`outputs/figure4_clinical_cases_panel.png`](outputs/figure4_clinical_cases_panel.png) — клиническая панель кейсов