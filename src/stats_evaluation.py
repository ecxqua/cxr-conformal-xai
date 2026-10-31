"""
Модуль статистического анализа связи между размером конформного множества |C(x)|
и метриками внимания Grad-CAM (площадь внимания S_CAM и пространственная энтропия H_norm).
"""
import numpy as np
import pandas as pd
import scipy.stats as stats


def evaluate_conformal_xai_link(df: pd.DataFrame):
    """
    Вычисляет описательные статистики, абсолютную и относительную разницу,
    t-критерий Уэлча, критерий Манна-Уитни и ранговую корреляцию Спирмена.
    """
    g1 = df[df['set_size'] == 1]
    g2 = df[df['set_size'] >= 2]

    # Attention Area (tau = 0.2)
    a1 = g1['attention_area_02']
    a2 = g2['attention_area_02']
    delta_a = a2.mean() - a1.mean()
    t_a, p_t_a = stats.ttest_ind(a2, a1, equal_var=False)
    u_a, p_u_a = stats.mannwhitneyu(a2, a1, alternative='two-sided')
    r_a, p_r_a = stats.spearmanr(df['set_size'], df['attention_area_02'])

    # Spatial Entropy
    e1 = g1['spatial_entropy']
    e2 = g2['spatial_entropy']
    delta_e = e2.mean() - e1.mean()
    u_e, p_u_e = stats.mannwhitneyu(e2, e1, alternative='two-sided')
    r_e, p_r_e = stats.spearmanr(df['set_size'], df['spatial_entropy'])

    return {
        'n_total': len(df),
        'n_group_1': len(g1),
        'n_group_2_plus': len(g2),
        'area_mean_g1': float(a1.mean()),
        'area_mean_g2': float(a2.mean()),
        'area_delta': float(delta_a),
        'area_rel_pct': float((delta_a / a1.mean()) * 100),
        'area_welch_p': float(p_t_a),
        'area_mannwhitney_p': float(p_u_a),
        'area_spearman_rho': float(r_a),
        'entropy_mean_g1': float(e1.mean()),
        'entropy_mean_g2': float(e2.mean()),
        'entropy_delta': float(delta_e),
        'entropy_mannwhitney_p': float(p_u_e),
        'entropy_spearman_rho': float(r_e)
    }
