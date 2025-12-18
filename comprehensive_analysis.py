"""
综合分析模块 - 完整的慢性乙型肝炎治疗效果分析

分析内容：
1. 基线可比性检验 - 验证两组基线是否相似
2. 治疗后肝功能差异 - 比较变化值差异
3. 非劣效性检验 - 验证实验组减少肝损伤风险
4. 箱式图 - 比较肝功能指标分布
5. 柱状图 - 治疗前后各指标比较
6. 肝硬度值分析 - 组内组间差异
7. 治愈速度差异分析

临床参考：
- 正常范围：白蛋白[40,55] ALT[7-40] AST[13-35] ALP[35-100] GGT[7-45]
- 重点关注：ALT、AST

图表风格：柳叶刀(Lancet)论文风格
"""
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 导入柳叶刀风格
from lancet_style import (
    setup_lancet_style, get_lancet_colors, format_pvalue,
    TREATMENT_COLOR, CONTROL_COLOR, LANCET_PALETTE,
    create_lancet_boxplot, save_lancet_figure
)

# 设置柳叶刀风格
setup_lancet_style()

# 设置中文字体（在柳叶刀风格基础上添加中文支持）
import platform
if platform.system() == 'Windows':
    plt.rcParams['font.sans-serif'] = ['Arial', 'Microsoft YaHei', 'SimHei']
else:
    plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

import os
import warnings
warnings.filterwarnings('ignore')

from data_preprocessing import (
    load_and_preprocess_data, ensure_result_dir,
    COVARIATES, OUTCOMES_PAIRS, PRIMARY_OUTCOMES_PAIRS, SECONDARY_OUTCOMES_PAIRS,
    NORMAL_RANGES
)

# 结果目录
RESULT_DIR = 'result/Comprehensive'
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)


# ==================== PSM匹配函数 ====================
def perform_psm_matching(df, covariates, treatment_col='treatment', caliper=0.2):
    """
    执行倾向性评分匹配
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import NearestNeighbors
    
    print("正在执行PSM匹配...")
    df = df.copy()
    
    # 准备协变量矩阵
    X = df[covariates].values
    y = df[treatment_col].values
    
    # 训练逻辑回归模型
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X, y)
    
    # 计算倾向性评分
    df['propensity_score'] = model.predict_proba(X)[:, 1]
    
    # 最近邻匹配
    treated = df[df[treatment_col] == 1]
    control = df[df[treatment_col] == 0]
    
    treated_ps = treated['propensity_score'].values.reshape(-1, 1)
    control_ps = control['propensity_score'].values.reshape(-1, 1)
    
    nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
    nn.fit(control_ps)
    
    distances, indices = nn.kneighbors(treated_ps)
    
    # 应用caliper
    ps_std = df['propensity_score'].std()
    caliper_dist = caliper * ps_std
    
    matched_treated_idx = []
    matched_control_idx = []
    
    for i, (dist, idx) in enumerate(zip(distances.flatten(), indices.flatten())):
        if dist <= caliper_dist:
            matched_treated_idx.append(treated.index[i])
            matched_control_idx.append(control.index[idx])
    
    # 合并匹配的数据
    matched_treated = df.loc[matched_treated_idx]
    matched_control = df.loc[matched_control_idx]
    df_matched = pd.concat([matched_treated, matched_control])
    
    print(f"PSM匹配: 治疗组{len(matched_treated)}人, 对照组{len(matched_control)}人")
    
    return df_matched


# ==================== CEM匹配函数 ====================
def perform_cem_matching(df, covariates, treatment_col='treatment', n_bins=4):
    """
    执行粗化精确匹配
    """
    print("正在执行CEM匹配...")
    df = df.copy()
    
    # 选择关键协变量进行匹配（避免维度灾难）
    key_covariates = ['性别', '年龄', '是否合并脂肪肝', '基线ALT', '基线AST']
    key_covariates = [c for c in key_covariates if c in covariates]
    
    # 粗化每个协变量
    coarsened_cols = []
    for cov in key_covariates:
        coarsened_col = f'{cov}_coarsened'
        series = df[cov]
        if series.nunique() <= n_bins:
            df[coarsened_col] = series.astype(str)
        else:
            try:
                df[coarsened_col] = pd.qcut(series, q=n_bins, labels=False, duplicates='drop').astype(str)
            except:
                df[coarsened_col] = pd.qcut(series, q=3, labels=False, duplicates='drop').astype(str)
        coarsened_cols.append(coarsened_col)
    
    # 创建层标识
    df['stratum'] = df[coarsened_cols].apply(lambda x: '_'.join(x.values), axis=1)
    
    # 找到同时包含治疗组和对照组的层
    stratum_counts = df.groupby(['stratum', treatment_col]).size().unstack(fill_value=0)
    valid_strata = stratum_counts[(stratum_counts[0] > 0) & (stratum_counts[1] > 0)].index
    
    # 筛选匹配的样本
    df_matched = df[df['stratum'].isin(valid_strata)].copy()
    
    # 删除临时列
    df_matched = df_matched.drop(columns=coarsened_cols + ['stratum'], errors='ignore')
    
    n_treated = (df_matched[treatment_col] == 1).sum()
    n_control = (df_matched[treatment_col] == 0).sum()
    print(f"CEM匹配: 治疗组{n_treated}人, 对照组{n_control}人")
    
    return df_matched


# ==================== 1. 基线可比性检验 ====================
def baseline_comparability_test(df, treatment_col='treatment', result_dir=None):
    """
    验证两组基线是否可比
    使用t检验（连续变量）和卡方检验（分类变量）
    """
    if result_dir is None:
        result_dir = RESULT_DIR
    print("\n" + "="*60)
    print("1. 基线可比性检验")
    print("="*60)
    
    results = []
    treated = df[df[treatment_col] == 1]
    control = df[df[treatment_col] == 0]
    
    # 分类变量
    categorical_vars = ['性别', '饮酒史', '吸烟史', '是否合并脂肪肝']
    # 连续变量
    continuous_vars = ['年龄', '基线白蛋白', '基线胆红素', '基线ALT', '基线AST', 
                       '基线GGT', '基线ALP', '肝硬度值基线', '血小板基线', 'HBsAg基线']
    
    # 连续变量 - t检验和Mann-Whitney U检验
    for var in continuous_vars:
        if var in df.columns:
            t_data = treated[var].dropna()
            c_data = control[var].dropna()
            
            if len(t_data) > 1 and len(c_data) > 1:
                # t检验
                t_stat, t_pval = stats.ttest_ind(t_data, c_data, equal_var=False)
                # Mann-Whitney U检验（非参数）
                u_stat, u_pval = stats.mannwhitneyu(t_data, c_data, alternative='two-sided')
                # 正态性检验
                _, norm_p_t = stats.shapiro(t_data[:50]) if len(t_data) >= 3 else (0, 1)
                _, norm_p_c = stats.shapiro(c_data[:50]) if len(c_data) >= 3 else (0, 1)
                
                results.append({
                    '变量': var,
                    '变量类型': '连续',
                    '治疗组均值': t_data.mean(),
                    '治疗组SD': t_data.std(),
                    '对照组均值': c_data.mean(),
                    '对照组SD': c_data.std(),
                    't检验P值': t_pval,
                    'Mann-Whitney P值': u_pval,
                    '推荐P值': u_pval if (norm_p_t < 0.05 or norm_p_c < 0.05) else t_pval,
                    '基线可比': '是' if (u_pval if (norm_p_t < 0.05 or norm_p_c < 0.05) else t_pval) > 0.05 else '否'
                })
    
    # 分类变量 - 卡方检验
    for var in categorical_vars:
        if var in df.columns:
            contingency = pd.crosstab(df[var], df[treatment_col])
            if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
                chi2, chi_pval, dof, expected = stats.chi2_contingency(contingency)
                
                # Fisher精确检验（当期望频数<5时）
                if (expected < 5).any():
                    try:
                        _, fisher_pval = stats.fisher_exact(contingency)
                        final_pval = fisher_pval
                    except:
                        final_pval = chi_pval
                else:
                    final_pval = chi_pval
                
                results.append({
                    '变量': var,
                    '变量类型': '分类',
                    '治疗组均值': treated[var].mean(),
                    '治疗组SD': np.nan,
                    '对照组均值': control[var].mean(),
                    '对照组SD': np.nan,
                    't检验P值': np.nan,
                    'Mann-Whitney P值': np.nan,
                    '推荐P值': final_pval,
                    '基线可比': '是' if final_pval > 0.05 else '否'
                })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'{result_dir}/baseline_comparability.csv', index=False, encoding='utf-8-sig')
    
    # 汇总
    comparable_count = (results_df['基线可比'] == '是').sum()
    total_count = len(results_df)
    print(f"\n基线可比的变量: {comparable_count}/{total_count}")
    print(results_df[['变量', '治疗组均值', '对照组均值', '推荐P值', '基线可比']].to_string(index=False))
    
    return results_df


# ==================== 2. 治疗后肝功能差异分析 ====================
def liver_function_difference_analysis(df, treatment_col='treatment', result_dir=None):
    """
    比较治疗后肝功能指标的变化值差异
    """
    if result_dir is None:
        result_dir = RESULT_DIR
    print("\n" + "="*60)
    print("2. 治疗后肝功能差异分析")
    print("="*60)
    
    # 肝功能相关指标
    liver_outcomes = [
        ('基线ALT', 'ALT12个月', 'ALT'),
        ('基线AST', 'AST12个月', 'AST'),
        ('基线GGT', 'GGT12个月', 'GGT'),
        ('基线ALP', 'ALP12个月', 'ALP'),
        ('基线白蛋白', '白蛋白12个月', '白蛋白'),
        ('基线胆红素', '总胆红素12个月', '胆红素'),
    ]
    
    results = []
    
    for base_col, end_col, name in liver_outcomes:
        if base_col in df.columns and end_col in df.columns:
            treated = df[df[treatment_col] == 1]
            control = df[df[treatment_col] == 0]
            
            # 计算变化值
            t_valid = treated[[base_col, end_col]].dropna()
            c_valid = control[[base_col, end_col]].dropna()
            
            t_change = t_valid[end_col] - t_valid[base_col]
            c_change = c_valid[end_col] - c_valid[base_col]
            
            if len(t_change) > 1 and len(c_change) > 1:
                # 独立样本t检验
                t_stat, t_pval = stats.ttest_ind(t_change, c_change, equal_var=False)
                # Mann-Whitney U检验
                u_stat, u_pval = stats.mannwhitneyu(t_change, c_change, alternative='two-sided')
                # 效应量Cohen's d
                pooled_std = np.sqrt((t_change.var() + c_change.var()) / 2)
                cohens_d = (t_change.mean() - c_change.mean()) / pooled_std if pooled_std > 0 else 0
                
                # 判断方向
                is_primary = name in ['ALT', 'AST']
                
                results.append({
                    '指标': name,
                    '指标类型': '主要' if is_primary else '次要',
                    '治疗组例数': len(t_change),
                    '治疗组变化均值': t_change.mean(),
                    '治疗组变化SD': t_change.std(),
                    '对照组例数': len(c_change),
                    '对照组变化均值': c_change.mean(),
                    '对照组变化SD': c_change.std(),
                    '组间差异': t_change.mean() - c_change.mean(),
                    't检验P值': t_pval,
                    'Mann-Whitney P值': u_pval,
                    "Cohen's d": cohens_d,
                    '统计学差异': '是' if t_pval < 0.05 else '否',
                    '临床意义': '治疗组更优' if (name != '白蛋白' and t_change.mean() < c_change.mean()) or 
                               (name == '白蛋白' and t_change.mean() > c_change.mean()) else '对照组更优或无差异'
                })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'{result_dir}/liver_function_difference.csv', index=False, encoding='utf-8-sig')
    
    print("\n肝功能变化值组间比较:")
    print(results_df[['指标', '治疗组变化均值', '对照组变化均值', '组间差异', 't检验P值', '统计学差异']].to_string(index=False))
    
    return results_df


# ==================== 3. 非劣效性检验 ====================
def non_inferiority_test(df, treatment_col='treatment', margin=10, result_dir=None):
    """
    非劣效性检验 - 验证实验组是否能减少肝损伤风险
    
    H0: 治疗组变化 - 对照组变化 >= margin (劣效)
    H1: 治疗组变化 - 对照组变化 < margin (非劣效)
    
    对于ALT/AST，下降越多越好，因此治疗组变化应该<=对照组变化
    margin: 非劣效界值（U/L），临床上通常取10 U/L或正常上限的25%
    """
    if result_dir is None:
        result_dir = RESULT_DIR
    print("\n" + "="*60)
    print("3. 非劣效性检验")
    print("="*60)
    print(f"非劣效界值(margin): {margin} U/L")
    
    # 主要针对ALT和AST
    ni_outcomes = [
        ('基线ALT', 'ALT12个月', 'ALT', 10),  # margin = 10 U/L
        ('基线AST', 'AST12个月', 'AST', 8.75),  # margin = 35*0.25 = 8.75 U/L
    ]
    
    results = []
    
    for base_col, end_col, name, ni_margin in ni_outcomes:
        if base_col in df.columns and end_col in df.columns:
            treated = df[df[treatment_col] == 1]
            control = df[df[treatment_col] == 0]
            
            t_valid = treated[[base_col, end_col]].dropna()
            c_valid = control[[base_col, end_col]].dropna()
            
            t_change = t_valid[end_col] - t_valid[base_col]
            c_change = c_valid[end_col] - c_valid[base_col]
            
            if len(t_change) > 1 and len(c_change) > 1:
                # 计算差异及其置信区间
                diff = t_change.mean() - c_change.mean()
                se_diff = np.sqrt(t_change.var()/len(t_change) + c_change.var()/len(c_change))
                
                # 95%置信区间
                ci_lower = diff - 1.96 * se_diff
                ci_upper = diff + 1.96 * se_diff
                
                # 非劣效性判断：CI上限 < margin (对于下降指标)
                # 由于ALT/AST下降更好，如果治疗组变化更负（下降更多），差异为负
                # 非劣效条件：差异的上限 < margin
                is_non_inferior = ci_upper < ni_margin
                
                # 优效性判断：CI上限 < 0
                is_superior = ci_upper < 0
                
                results.append({
                    '指标': name,
                    '非劣效界值': ni_margin,
                    '治疗组变化均值': t_change.mean(),
                    '对照组变化均值': c_change.mean(),
                    '差异(治疗-对照)': diff,
                    '差异SE': se_diff,
                    '95%CI下限': ci_lower,
                    '95%CI上限': ci_upper,
                    '非劣效结论': '非劣效' if is_non_inferior else '未证明非劣效',
                    '优效结论': '优效' if is_superior else '未证明优效',
                    '综合结论': '优效' if is_superior else ('非劣效' if is_non_inferior else '结果不确定')
                })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'{result_dir}/non_inferiority_test.csv', index=False, encoding='utf-8-sig')
    
    print("\n非劣效性检验结果:")
    print(results_df[['指标', '差异(治疗-对照)', '95%CI下限', '95%CI上限', '非劣效界值', '综合结论']].to_string(index=False))
    
    # 绘制非劣效性森林图 - 柳叶刀风格
    setup_lancet_style()
    fig, ax = plt.subplots(figsize=(10, 3.5), facecolor='white')
    ax.set_facecolor('white')
    
    y_pos = np.arange(len(results_df))
    
    for i, row in results_df.iterrows():
        # 柳叶刀配色：优效绿色，非劣效蓝色，未证明红色
        if row['综合结论'] == '优效':
            color = '#42B540'  # 绿色
        elif row['综合结论'] == '非劣效':
            color = '#00468B'  # 深蓝
        else:
            color = '#ED0000'  # 红色
        
        # 置信区间线
        ax.plot([row['95%CI下限'], row['95%CI上限']], [i, i], 
               color=color, linewidth=2.5, solid_capstyle='round')
        # 端点
        ax.plot([row['95%CI下限'], row['95%CI上限']], [i, i], '|', 
               color=color, markersize=10, markeredgewidth=2)
        # 点估计 - 菱形
        ax.plot(row['差异(治疗-对照)'], i, 'D', color=color, markersize=10, 
               markeredgecolor='white', markeredgewidth=1)
    
    # 参考线
    ax.axvline(x=0, color='#333333', linestyle='-', linewidth=1.2, zorder=1)
    ax.axvline(x=results_df['非劣效界值'].max(), color='#AD002A', linestyle='--', 
              linewidth=1.5, label=f'非劣效界值 ({results_df["非劣效界值"].max():.1f})')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(results_df['指标'], fontsize=16)
    ax.set_xlabel('变化值差异 (治疗组 - 对照组)', fontsize=16, fontweight='bold')
    ax.set_title('非劣效性检验森林图', fontsize=20, fontweight='bold', pad=15)
    
    # 添加图例说明
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#42B540', label='优效'),
        Patch(facecolor='#00468B', label='非劣效'),
        Patch(facecolor='#ED0000', label='未证明'),
        plt.Line2D([0], [0], color='#AD002A', linestyle='--', label='非劣效界值')
    ]
    ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9, fontsize=14)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(ax.get_xlim()[0] - 2, results_df['非劣效界值'].max() + 3)
    
    plt.tight_layout()
    save_lancet_figure(fig, f'{result_dir}/non_inferiority_forest.png')
    
    return results_df


# ==================== 4. 箱式图比较 ====================
def boxplot_comparison(df, treatment_col='treatment', result_dir=None):
    """
    使用箱式图比较实验组对照组肝功能指标分布 - 柳叶刀风格
    """
    if result_dir is None:
        result_dir = RESULT_DIR
    print("\n" + "="*60)
    print("4. 箱式图比较")
    print("="*60)
    
    # 应用Lancet风格确保中文正确显示
    setup_lancet_style()
    
    # 肝功能指标
    indicators = [
        ('基线ALT', 'ALT12个月', 'ALT'),
        ('基线AST', 'AST12个月', 'AST'),
        ('基线GGT', 'GGT12个月', 'GGT'),
        ('基线ALP', 'ALP12个月', 'ALP'),
        ('基线白蛋白', '白蛋白12个月', '白蛋白'),
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), facecolor='white')
    axes = axes.flatten()
    
    for idx, (base_col, end_col, name) in enumerate(indicators):
        if base_col in df.columns and end_col in df.columns:
            ax = axes[idx]
            
            # 准备数据
            treated = df[df[treatment_col] == 1]
            control = df[df[treatment_col] == 0]
            
            data_to_plot = [
                treated[base_col].dropna(),
                treated[end_col].dropna(),
                control[base_col].dropna(),
                control[end_col].dropna()
            ]
            
            # 柳叶刀风格箱式图
            bp = ax.boxplot(data_to_plot, patch_artist=True,
                          labels=['治疗组\n基线', '治疗组\n12月', '对照组\n基线', '对照组\n12月'],
                          widths=0.6,
                          medianprops=dict(color='white', linewidth=2),
                          whiskerprops=dict(color='#333333', linewidth=1),
                          capprops=dict(color='#333333', linewidth=1),
                          flierprops=dict(marker='o', markerfacecolor='#666666', 
                                         markersize=4, alpha=0.5, markeredgecolor='none'))
            
            # 柳叶刀配色
            colors = ['#ED0000', '#AD002A', '#00468B', '#003366']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.75)
                patch.set_edgecolor('#333333')
                patch.set_linewidth(1)
            
            # 添加正常范围线
            if name in NORMAL_RANGES:
                lower, upper, unit = NORMAL_RANGES[name]
                ax.axhline(y=upper, color='#42B540', linestyle='--', linewidth=1.5, 
                          alpha=0.8, label=f'ULN ({upper})')
            
            ax.set_title(f'{name}', fontsize=18, fontweight='bold')
            ax.set_ylabel(f'{name} (U/L)' if name != '白蛋白' else f'{name} (g/L)', fontsize=15, fontweight='bold')
            ax.tick_params(axis='both', labelsize=14)
            if name in NORMAL_RANGES:
                ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
    
    # 隐藏多余的子图
    if len(indicators) < 6:
        axes[-1].set_visible(False)
    
    plt.suptitle('肝功能指标分布比较', fontsize=21, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_lancet_figure(fig, f'{result_dir}/boxplot_comparison.png')
    
    print("箱式图已保存: boxplot_comparison.png")
    
    # 变化值箱式图 - 柳叶刀风格
    fig, axes = plt.subplots(1, 5, figsize=(15, 4.5))
    
    for idx, (base_col, end_col, name) in enumerate(indicators):
        if base_col in df.columns and end_col in df.columns:
            ax = axes[idx]
            
            treated = df[df[treatment_col] == 1]
            control = df[df[treatment_col] == 0]
            
            t_change = (treated[end_col] - treated[base_col]).dropna()
            c_change = (control[end_col] - control[base_col]).dropna()
            
            # 柳叶刀风格箱式图
            bp = ax.boxplot([t_change, c_change], patch_artist=True,
                          labels=['治疗组', '对照组'],
                          widths=0.5,
                          medianprops=dict(color='white', linewidth=2),
                          whiskerprops=dict(color='#333333', linewidth=1),
                          capprops=dict(color='#333333', linewidth=1),
                          flierprops=dict(marker='o', markerfacecolor='#666666', 
                                         markersize=4, alpha=0.5, markeredgecolor='none'))
            
            colors = [TREATMENT_COLOR, CONTROL_COLOR]
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.75)
                patch.set_edgecolor('#333333')
                patch.set_linewidth(1)
            
            # 添加散点
            for i, (data, color) in enumerate(zip([t_change, c_change], colors)):
                x = np.random.normal(i + 1, 0.04, len(data))
                ax.scatter(x, data, alpha=0.3, s=12, color=color, edgecolors='none', zorder=3)
            
            ax.axhline(y=0, color='#333333', linestyle='-', linewidth=1, alpha=0.5)
            ax.set_title(f'{name}', fontsize=17, fontweight='bold')
            ax.set_ylabel('变化值', fontsize=15, fontweight='bold')
            ax.tick_params(axis='both', labelsize=14)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # 添加统计检验P值
            _, pval = stats.mannwhitneyu(t_change, c_change, alternative='two-sided')
            p_text = format_pvalue(pval)
            ax.text(0.5, 0.02, p_text, transform=ax.transAxes, 
                   ha='center', fontsize=14, style='italic', color='#333333')
    
    plt.suptitle('肝功能指标变化值比较 (12个月 - 基线)', fontsize=20, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_lancet_figure(fig, f'{result_dir}/boxplot_change_comparison.png')
    
    print("变化值箱式图已保存: boxplot_change_comparison.png")


# ==================== 5. 柱状图比较 ====================
def barplot_comparison(df, treatment_col='treatment', result_dir=None):
    """
    用柱状图比较实验组和对照组治疗前后各个指标 - Lancet风格
    """
    if result_dir is None:
        result_dir = RESULT_DIR
    print("\n" + "="*60)
    print("5. 柱状图比较（治疗前后）")
    print("="*60)
    
    # 应用Lancet风格
    setup_lancet_style()
    
    indicators = [
        ('基线ALT', 'ALT12个月', 'ALT', 'U/L'),
        ('基线AST', 'AST12个月', 'AST', 'U/L'),
        ('基线GGT', 'GGT12个月', 'GGT', 'U/L'),
        ('基线ALP', 'ALP12个月', 'ALP', 'U/L'),
        ('基线白蛋白', '白蛋白12个月', '白蛋白', 'g/L'),
        ('基线胆红素', '总胆红素12个月', '胆红素', 'μmol/L'),
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 10), facecolor='white')
    axes = axes.flatten()
    
    for idx, (base_col, end_col, name, unit) in enumerate(indicators):
        if base_col in df.columns and end_col in df.columns:
            ax = axes[idx]
            ax.set_facecolor('white')
            
            treated = df[df[treatment_col] == 1]
            control = df[df[treatment_col] == 0]
            
            # 计算均值和标准误
            t_base_mean = treated[base_col].mean()
            t_base_se = treated[base_col].std() / np.sqrt(len(treated[base_col].dropna()))
            t_end_mean = treated[end_col].mean()
            t_end_se = treated[end_col].std() / np.sqrt(len(treated[end_col].dropna()))
            
            c_base_mean = control[base_col].mean()
            c_base_se = control[base_col].std() / np.sqrt(len(control[base_col].dropna()))
            c_end_mean = control[end_col].mean()
            c_end_se = control[end_col].std() / np.sqrt(len(control[end_col].dropna()))
            
            x = np.arange(2)
            width = 0.35
            
            # Lancet风格柱状图
            bars1 = ax.bar(x - width/2, [t_base_mean, t_end_mean], width, 
                          yerr=[t_base_se, t_end_se], capsize=4,
                          label='治疗组', color=TREATMENT_COLOR, 
                          edgecolor='white', linewidth=1.2,
                          error_kw={'elinewidth': 1.5, 'capthick': 1.5, 'ecolor': '#333333'})
            bars2 = ax.bar(x + width/2, [c_base_mean, c_end_mean], width,
                          yerr=[c_base_se, c_end_se], capsize=4,
                          label='对照组', color=CONTROL_COLOR,
                          edgecolor='white', linewidth=1.2,
                          error_kw={'elinewidth': 1.5, 'capthick': 1.5, 'ecolor': '#333333'})
            
            ax.set_xticks(x)
            ax.set_xticklabels(['基线', '12个月'], fontsize=15)
            ax.set_ylabel(f'{name} ({unit})', fontsize=15, fontweight='bold')
            ax.set_title(f'{name}', fontsize=18, fontweight='bold', pad=10)
            ax.tick_params(axis='y', labelsize=14)
            
            # 调整y轴范围避免图例重叠
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymin * 0.95 if ymin > 0 else ymin, ymax * 1.25)
            ax.legend(loc='upper right', fontsize=12, frameon=False)
            
            # Lancet风格：只保留左边和底部的边框
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(1.2)
            ax.spines['bottom'].set_linewidth(1.2)
            
            # 添加正常范围参考线（使用绿色）
            if name in NORMAL_RANGES:
                lower, upper, _ = NORMAL_RANGES[name]
                ax.axhline(y=upper, color='#42B540', linestyle='--', alpha=0.8, linewidth=1.5,
                          label=f'正常上限={upper}')
            
            # 在柱子上标注数值
            for bar, val in zip(bars1, [t_base_mean, t_end_mean]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + t_base_se + 1, 
                       f'{val:.1f}', ha='center', va='bottom', fontsize=14, fontweight='bold')
            for bar, val in zip(bars2, [c_base_mean, c_end_mean]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + c_base_se + 1,
                       f'{val:.1f}', ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    plt.suptitle('肝功能指标治疗前后柱状图比较\n(误差棒表示标准误)', 
                 fontsize=21, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_lancet_figure(fig, f'{result_dir}/barplot_comparison.png')
    plt.close('all')
    
    print("柱状图已保存: barplot_comparison.png")
    
    # 变化值柱状图 - Lancet风格
    setup_lancet_style()
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
    ax.set_facecolor('white')
    
    names = []
    t_changes = []
    c_changes = []
    t_ses = []
    c_ses = []
    
    for base_col, end_col, name, unit in indicators:
        if base_col in df.columns and end_col in df.columns:
            treated = df[df[treatment_col] == 1]
            control = df[df[treatment_col] == 0]
            
            t_change = (treated[end_col] - treated[base_col]).dropna()
            c_change = (control[end_col] - control[base_col]).dropna()
            
            names.append(name)
            t_changes.append(t_change.mean())
            c_changes.append(c_change.mean())
            t_ses.append(t_change.std() / np.sqrt(len(t_change)))
            c_ses.append(c_change.std() / np.sqrt(len(c_change)))
    
    x = np.arange(len(names))
    width = 0.35
    
    # Lancet风格柱状图
    bars1 = ax.bar(x - width/2, t_changes, width, yerr=t_ses, capsize=4,
                  label='治疗组', color=TREATMENT_COLOR,
                  edgecolor='white', linewidth=1.2,
                  error_kw={'elinewidth': 1.5, 'capthick': 1.5, 'ecolor': '#333333'})
    bars2 = ax.bar(x + width/2, c_changes, width, yerr=c_ses, capsize=4,
                  label='对照组', color=CONTROL_COLOR,
                  edgecolor='white', linewidth=1.2,
                  error_kw={'elinewidth': 1.5, 'capthick': 1.5, 'ecolor': '#333333'})
    
    ax.axhline(y=0, color='#333333', linestyle='-', linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=17, fontweight='bold')
    ax.set_ylabel('变化值', fontsize=18, fontweight='bold')
    ax.tick_params(axis='y', labelsize=14)
    ax.set_title('各指标变化值比较（12个月 - 基线）\n负值表示下降', 
                 fontsize=20, fontweight='bold', pad=15)
    ax.legend(frameon=False, fontsize=17)
    
    # Lancet风格边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    
    plt.tight_layout()
    save_lancet_figure(fig, f'{result_dir}/barplot_change_comparison.png')
    plt.close()
    
    print("变化值柱状图已保存: barplot_change_comparison.png")


# ==================== 6. 肝硬度值分析 ====================
def liver_stiffness_analysis(df, treatment_col='treatment', result_dir=None):
    """
    分析肝硬度值治疗前后的组内和组间差异
    """
    if result_dir is None:
        result_dir = RESULT_DIR
    print("\n" + "="*60)
    print("6. 肝硬度值分析")
    print("="*60)
    
    base_col = '肝硬度值基线'
    end_col = '肝硬度值12个月'
    
    if base_col not in df.columns or end_col not in df.columns:
        print("警告: 未找到肝硬度值列")
        return None
    
    treated = df[df[treatment_col] == 1]
    control = df[df[treatment_col] == 0]
    
    t_valid = treated[[base_col, end_col]].dropna()
    c_valid = control[[base_col, end_col]].dropna()
    
    results = {}
    
    # 组内比较（配对t检验）
    print("\n--- 组内比较（治疗前后配对t检验）---")
    
    # 治疗组
    t_stat_t, p_t = stats.ttest_rel(t_valid[end_col], t_valid[base_col])
    _, wilcox_p_t = stats.wilcoxon(t_valid[end_col], t_valid[base_col])
    t_change = t_valid[end_col] - t_valid[base_col]
    
    print(f"治疗组: 基线={t_valid[base_col].mean():.2f}±{t_valid[base_col].std():.2f}, "
          f"12月={t_valid[end_col].mean():.2f}±{t_valid[end_col].std():.2f}")
    print(f"  变化: {t_change.mean():.2f}±{t_change.std():.2f}, P={p_t:.4f} (配对t), P={wilcox_p_t:.4f} (Wilcoxon)")
    
    # 对照组
    c_stat_t, p_c = stats.ttest_rel(c_valid[end_col], c_valid[base_col])
    _, wilcox_p_c = stats.wilcoxon(c_valid[end_col], c_valid[base_col])
    c_change = c_valid[end_col] - c_valid[base_col]
    
    print(f"对照组: 基线={c_valid[base_col].mean():.2f}±{c_valid[base_col].std():.2f}, "
          f"12月={c_valid[end_col].mean():.2f}±{c_valid[end_col].std():.2f}")
    print(f"  变化: {c_change.mean():.2f}±{c_change.std():.2f}, P={p_c:.4f} (配对t), P={wilcox_p_c:.4f} (Wilcoxon)")
    
    # 组间比较
    print("\n--- 组间比较（变化值独立样本检验）---")
    t_stat_between, p_between = stats.ttest_ind(t_change, c_change, equal_var=False)
    _, mw_p_between = stats.mannwhitneyu(t_change, c_change, alternative='two-sided')
    
    print(f"治疗组变化 vs 对照组变化:")
    print(f"  差异: {t_change.mean() - c_change.mean():.2f}")
    print(f"  P={p_between:.4f} (独立t), P={mw_p_between:.4f} (Mann-Whitney)")
    
    # 保存结果
    results_df = pd.DataFrame([{
        '分析类型': '治疗组组内',
        '基线均值': t_valid[base_col].mean(),
        '基线SD': t_valid[base_col].std(),
        '终点均值': t_valid[end_col].mean(),
        '终点SD': t_valid[end_col].std(),
        '变化均值': t_change.mean(),
        '变化SD': t_change.std(),
        '配对t检验P值': p_t,
        'Wilcoxon P值': wilcox_p_t,
        '例数': len(t_valid)
    }, {
        '分析类型': '对照组组内',
        '基线均值': c_valid[base_col].mean(),
        '基线SD': c_valid[base_col].std(),
        '终点均值': c_valid[end_col].mean(),
        '终点SD': c_valid[end_col].std(),
        '变化均值': c_change.mean(),
        '变化SD': c_change.std(),
        '配对t检验P值': p_c,
        'Wilcoxon P值': wilcox_p_c,
        '例数': len(c_valid)
    }, {
        '分析类型': '组间比较',
        '基线均值': np.nan,
        '基线SD': np.nan,
        '终点均值': np.nan,
        '终点SD': np.nan,
        '变化均值': t_change.mean() - c_change.mean(),
        '变化SD': np.nan,
        '配对t检验P值': p_between,
        'Wilcoxon P值': mw_p_between,
        '例数': len(t_valid) + len(c_valid)
    }])
    
    results_df.to_csv(f'{result_dir}/liver_stiffness_analysis.csv', index=False, encoding='utf-8-sig')
    
    # 绘图 - Lancet风格
    setup_lancet_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor='white')
    
    # 左图：治疗前后对比
    ax1 = axes[0]
    ax1.set_facecolor('white')
    x = [0, 1]
    
    # 治疗组 - 红色
    ax1.plot(x, [t_valid[base_col].mean(), t_valid[end_col].mean()], 'o-', 
            color=TREATMENT_COLOR, linewidth=2.5, markersize=12, label='治疗组',
            markeredgecolor='white', markeredgewidth=1.5)
    ax1.fill_between(x, 
                     [t_valid[base_col].mean() - t_valid[base_col].std(), 
                      t_valid[end_col].mean() - t_valid[end_col].std()],
                     [t_valid[base_col].mean() + t_valid[base_col].std(),
                      t_valid[end_col].mean() + t_valid[end_col].std()],
                     color=TREATMENT_COLOR, alpha=0.15)
    
    # 对照组 - 蓝色
    ax1.plot(x, [c_valid[base_col].mean(), c_valid[end_col].mean()], 's-',
            color=CONTROL_COLOR, linewidth=2.5, markersize=12, label='对照组',
            markeredgecolor='white', markeredgewidth=1.5)
    ax1.fill_between(x,
                     [c_valid[base_col].mean() - c_valid[base_col].std(),
                      c_valid[end_col].mean() - c_valid[end_col].std()],
                     [c_valid[base_col].mean() + c_valid[base_col].std(),
                      c_valid[end_col].mean() + c_valid[end_col].std()],
                     color=CONTROL_COLOR, alpha=0.15)
    
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['基线', '12个月'], fontsize=17, fontweight='bold')
    ax1.set_ylabel('肝硬度值 (kPa)', fontsize=17, fontweight='bold')
    ax1.tick_params(axis='y', labelsize=14)
    ax1.set_title('肝硬度值治疗前后变化\n(阴影区域表示±1SD)', 
                  fontsize=18, fontweight='bold', pad=10)
    ax1.legend(frameon=False, fontsize=15)
    
    # Lancet风格边框
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_linewidth(1.2)
    ax1.spines['bottom'].set_linewidth(1.2)
    
    # 添加P值标注 - Lancet风格
    ax1.annotate(f'治疗组: {format_pvalue(p_t)}', xy=(0.5, t_valid[base_col].mean()), 
                 fontsize=15, color=TREATMENT_COLOR, fontweight='bold')
    ax1.annotate(f'对照组: {format_pvalue(p_c)}', xy=(0.5, c_valid[base_col].mean()-2), 
                 fontsize=15, color=CONTROL_COLOR, fontweight='bold')
    
    # 右图：变化值箱式图 - Lancet风格
    ax2 = axes[1]
    ax2.set_facecolor('white')
    
    bp = ax2.boxplot([t_change, c_change], patch_artist=True, labels=['治疗组', '对照组'],
                     widths=0.6)
    colors = [TREATMENT_COLOR, CONTROL_COLOR]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('white')
        patch.set_linewidth(1.5)
    for median in bp['medians']:
        median.set_color('white')
        median.set_linewidth(2)
    for whisker in bp['whiskers']:
        whisker.set_color('#333333')
        whisker.set_linewidth(1.5)
    for cap in bp['caps']:
        cap.set_color('#333333')
        cap.set_linewidth(1.5)
    for flier in bp['fliers']:
        flier.set(marker='o', markerfacecolor='#333333', markeredgecolor='white', 
                  markersize=6, alpha=0.7)
    
    # 添加散点
    for i, (data, color) in enumerate(zip([t_change, c_change], colors)):
        x_jitter = np.random.normal(i+1, 0.08, len(data))
        ax2.scatter(x_jitter, data, alpha=0.4, s=25, color=color, edgecolor='white', linewidth=0.5)
    
    ax2.axhline(y=0, color='#333333', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.set_ylabel('肝硬度值变化 (kPa)', fontsize=17, fontweight='bold')
    ax2.tick_params(axis='both', labelsize=14)
    ax2.set_title(f'肝硬度值变化值比较\n组间{format_pvalue(p_between)}', 
                  fontsize=18, fontweight='bold', pad=10)
    
    # Lancet风格边框
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_linewidth(1.2)
    ax2.spines['bottom'].set_linewidth(1.2)
    
    plt.tight_layout()
    save_lancet_figure(fig, f'{result_dir}/liver_stiffness_analysis.png')
    plt.close()
    
    print("\n肝硬度值分析图已保存: liver_stiffness_analysis.png")
    
    return results_df


# ==================== 7. 治愈速度差异分析 ====================
def cure_speed_analysis(df, treatment_col='treatment', result_dir=None):
    """
    分析实验组和对照组的治愈速度差异
    使用复常率和下降速度作为评估指标
    """
    if result_dir is None:
        result_dir = RESULT_DIR
    print("\n" + "="*60)
    print("7. 治愈速度/改善速度差异分析")
    print("="*60)
    
    indicators = [
        ('基线ALT', 'ALT12个月', 'ALT', 40),
        ('基线AST', 'AST12个月', 'AST', 35),
        ('基线GGT', 'GGT12个月', 'GGT', 45),
    ]
    
    results = []
    
    for base_col, end_col, name, upper_limit in indicators:
        if base_col in df.columns and end_col in df.columns:
            treated = df[df[treatment_col] == 1]
            control = df[df[treatment_col] == 0]
            
            for group, group_name in [(treated, '治疗组'), (control, '对照组')]:
                valid = group[[base_col, end_col]].dropna()
                
                # 基线异常的患者（考虑容忍度+2）
                abnormal = valid[valid[base_col] > upper_limit + 2]
                
                if len(abnormal) > 0:
                    # 复常率
                    normalized = abnormal[abnormal[end_col] <= upper_limit + 2]
                    norm_rate = len(normalized) / len(abnormal) * 100
                    
                    # 下降速度（月均下降值）
                    change = abnormal[end_col] - abnormal[base_col]
                    monthly_decrease = -change.mean() / 12  # 负号使下降为正值
                    
                    # 相对下降率
                    relative_decrease = -change.mean() / abnormal[base_col].mean() * 100
                    
                    results.append({
                        '指标': name,
                        '分组': group_name,
                        '基线异常例数': len(abnormal),
                        '复常例数': len(normalized),
                        '复常率(%)': norm_rate,
                        '月均下降值': monthly_decrease,
                        '相对下降率(%)': relative_decrease,
                        '基线均值': abnormal[base_col].mean(),
                        '终点均值': abnormal[end_col].mean()
                    })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'{result_dir}/cure_speed_analysis.csv', index=False, encoding='utf-8-sig')
    
    print("\n各指标改善速度分析:")
    print(results_df.to_string(index=False))
    
    # 复常率组间比较（Fisher精确检验）
    print("\n--- 复常率组间比较 ---")
    comparison_results = []
    
    for name in ['ALT', 'AST', 'GGT']:
        t_data = results_df[(results_df['指标'] == name) & (results_df['分组'] == '治疗组')]
        c_data = results_df[(results_df['指标'] == name) & (results_df['分组'] == '对照组')]
        
        if len(t_data) > 0 and len(c_data) > 0:
            t_row = t_data.iloc[0]
            c_row = c_data.iloc[0]
            
            # Fisher精确检验
            table = [[int(t_row['复常例数']), int(t_row['基线异常例数'] - t_row['复常例数'])],
                     [int(c_row['复常例数']), int(c_row['基线异常例数'] - c_row['复常例数'])]]
            
            try:
                odds_ratio, fisher_p = stats.fisher_exact(table)
            except:
                odds_ratio, fisher_p = np.nan, np.nan
            
            comparison_results.append({
                '指标': name,
                '治疗组复常率': t_row['复常率(%)'],
                '对照组复常率': c_row['复常率(%)'],
                '复常率差异': t_row['复常率(%)'] - c_row['复常率(%)'],
                'OR值': odds_ratio,
                'Fisher P值': fisher_p
            })
            
            print(f"{name}: 治疗组{t_row['复常率(%)']:.1f}% vs 对照组{c_row['复常率(%)']:.1f}%, P={fisher_p:.4f}")
    
    comparison_df = pd.DataFrame(comparison_results)
    comparison_df.to_csv(f'{result_dir}/cure_rate_comparison.csv', index=False, encoding='utf-8-sig')
    
    # 绘图 - Lancet风格
    setup_lancet_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor='white')
    
    # 左图：复常率比较
    ax1 = axes[0]
    ax1.set_facecolor('white')
    indicators_list = results_df['指标'].unique()
    x = np.arange(len(indicators_list))
    width = 0.35
    
    t_rates = [results_df[(results_df['指标']==ind) & (results_df['分组']=='治疗组')]['复常率(%)'].values[0] 
               if len(results_df[(results_df['指标']==ind) & (results_df['分组']=='治疗组')]) > 0 else 0 
               for ind in indicators_list]
    c_rates = [results_df[(results_df['指标']==ind) & (results_df['分组']=='对照组')]['复常率(%)'].values[0]
               if len(results_df[(results_df['指标']==ind) & (results_df['分组']=='对照组')]) > 0 else 0
               for ind in indicators_list]
    
    bars1 = ax1.bar(x - width/2, t_rates, width, label='治疗组', color=TREATMENT_COLOR,
                    edgecolor='white', linewidth=1.2)
    bars2 = ax1.bar(x + width/2, c_rates, width, label='对照组', color=CONTROL_COLOR,
                    edgecolor='white', linewidth=1.2)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(indicators_list, fontsize=17, fontweight='bold')
    ax1.set_ylabel('复常率 (%)', fontsize=17, fontweight='bold')
    ax1.tick_params(axis='y', labelsize=14)
    ax1.set_title('各指标复常率比较', fontsize=18, fontweight='bold', pad=10)
    
    # 调整y轴范围避免图例重叠
    max_rate = max(max(t_rates), max(c_rates))
    ax1.set_ylim(0, max_rate * 1.35)
    ax1.legend(loc='upper right', frameon=False, fontsize=13)
    
    # Lancet风格边框
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_linewidth(1.2)
    ax1.spines['bottom'].set_linewidth(1.2)
    
    # 标注数值
    for bar, val in zip(bars1, t_rates):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{val:.1f}%', ha='center', va='bottom', fontsize=13, fontweight='bold')
    for bar, val in zip(bars2, c_rates):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=13, fontweight='bold')
    
    # 右图：月均下降速度比较
    ax2 = axes[1]
    ax2.set_facecolor('white')
    t_speeds = [results_df[(results_df['指标']==ind) & (results_df['分组']=='治疗组')]['月均下降值'].values[0]
               if len(results_df[(results_df['指标']==ind) & (results_df['分组']=='治疗组')]) > 0 else 0
               for ind in indicators_list]
    c_speeds = [results_df[(results_df['指标']==ind) & (results_df['分组']=='对照组')]['月均下降值'].values[0]
               if len(results_df[(results_df['指标']==ind) & (results_df['分组']=='对照组')]) > 0 else 0
               for ind in indicators_list]
    
    bars1 = ax2.bar(x - width/2, t_speeds, width, label='治疗组', color=TREATMENT_COLOR,
                    edgecolor='white', linewidth=1.2)
    bars2 = ax2.bar(x + width/2, c_speeds, width, label='对照组', color=CONTROL_COLOR,
                    edgecolor='white', linewidth=1.2)
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(indicators_list, fontsize=17, fontweight='bold')
    ax2.set_ylabel('月均下降值 (U/L/月)', fontsize=17, fontweight='bold')
    ax2.tick_params(axis='y', labelsize=14)
    ax2.set_title('各指标月均改善速度比较', fontsize=18, fontweight='bold', pad=10)
    
    # 调整y轴范围避免图例重叠
    max_speed = max(max(t_speeds), max(c_speeds))
    ax2.set_ylim(0, max_speed * 1.35)
    ax2.legend(loc='upper right', frameon=False, fontsize=13)
    
    # Lancet风格边框
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_linewidth(1.2)
    ax2.spines['bottom'].set_linewidth(1.2)
    
    plt.tight_layout()
    save_lancet_figure(fig, f'{result_dir}/cure_speed_comparison.png')
    plt.close()
    
    print("\n治愈速度分析图已保存: cure_speed_comparison.png")
    
    return results_df, comparison_df


# ==================== 主函数 ====================
def run_comprehensive_analysis():
    """运行完整的综合分析 - 三种数据来源"""
    print("="*70)
    print("       综合分析 - 慢性乙型肝炎治疗效果评估")
    print("       (原始数据 / PSM匹配 / CEM匹配)")
    print("="*70)
    print("\n临床参考范围:")
    print("  ALT: 7-40 U/L | AST: 13-35 U/L | GGT: 7-45 U/L")
    print("  ALP: 35-100 U/L | 白蛋白: 40-55 g/L | 胆红素: 3.4-17.1 μmol/L")
    
    # 保存原始结果目录
    base_result_dir = RESULT_DIR
    
    # 加载原始数据
    df_raw = load_and_preprocess_data(fill_missing=True)
    print(f"\n原始数据样本量: {len(df_raw)}")
    print(f"治疗组: {(df_raw['treatment']==1).sum()}, 对照组: {(df_raw['treatment']==0).sum()}")
    
    # PSM匹配
    df_psm = perform_psm_matching(df_raw, COVARIATES)
    
    # CEM匹配
    df_cem = perform_cem_matching(df_raw, COVARIATES)
    
    # 三种数据来源
    data_sources = {
        'Raw': ('原始数据', df_raw),
        'PSM': ('PSM匹配', df_psm),
        'CEM': ('CEM匹配', df_cem)
    }
    
    all_results = {}
    
    for source_key, (source_name, df) in data_sources.items():
        print("\n" + "="*70)
        print(f"       {source_name}分析")
        print("="*70)
        
        # 创建子目录
        source_dir = f'{base_result_dir}/{source_key}'
        if not os.path.exists(source_dir):
            os.makedirs(source_dir)
        
        results = {}
        
        try:
            # 1. 基线可比性检验
            results['baseline'] = baseline_comparability_test(df, result_dir=source_dir)
            
            # 2. 治疗后肝功能差异
            results['liver_function'] = liver_function_difference_analysis(df, result_dir=source_dir)
            
            # 3. 非劣效性检验
            results['non_inferiority'] = non_inferiority_test(df, result_dir=source_dir)
            
            # 4. 箱式图比较
            boxplot_comparison(df, result_dir=source_dir)
            
            # 5. 柱状图比较
            barplot_comparison(df, result_dir=source_dir)
            
            # 6. 肝硬度值分析
            results['liver_stiffness'] = liver_stiffness_analysis(df, result_dir=source_dir)
            
            # 7. 治愈速度分析
            results['cure_speed'], results['cure_comparison'] = cure_speed_analysis(df, result_dir=source_dir)
            
        except Exception as e:
            print(f"警告: {source_name}分析部分失败: {e}")
            import traceback
            traceback.print_exc()
        
        all_results[source_key] = results
    
    print("\n" + "="*70)
    print("综合分析完成！")
    print("="*70)
    print(f"\n所有结果已保存至: {base_result_dir}/")
    print("\n结果目录结构:")
    print("  - Raw/    (原始数据分析)")
    print("  - PSM/    (PSM匹配后分析)")
    print("  - CEM/    (CEM匹配后分析)")
    
    # 生成汇总对比表
    generate_comparison_summary(all_results, data_sources, base_result_dir)
    
    return all_results


def generate_comparison_summary(all_results, data_sources, result_dir):
    """生成三种方法的对比汇总"""
    print("\n正在生成对比汇总...")
    
    # 汇总非劣效性检验结果
    ni_summary = []
    for source_key, (source_name, _) in data_sources.items():
        if source_key in all_results and 'non_inferiority' in all_results[source_key]:
            ni_df = all_results[source_key]['non_inferiority']
            if ni_df is not None:
                for _, row in ni_df.iterrows():
                    ni_summary.append({
                        '数据来源': source_name,
                        '指标': row['指标'],
                        '差异值': row.get('差异(治疗-对照)', np.nan),
                        '95%CI下限': row.get('95%CI下限', np.nan),
                        '95%CI上限': row.get('95%CI上限', np.nan),
                        '结论': row.get('综合结论', '')
                    })
    
    if ni_summary:
        ni_summary_df = pd.DataFrame(ni_summary)
        ni_summary_df.to_csv(f'{result_dir}/comparison_non_inferiority.csv', 
                             index=False, encoding='utf-8-sig')
    
    # 汇总治愈速度结果
    cure_summary = []
    for source_key, (source_name, _) in data_sources.items():
        if source_key in all_results and 'cure_comparison' in all_results[source_key]:
            cure_df = all_results[source_key]['cure_comparison']
            if cure_df is not None and len(cure_df) > 0:
                for _, row in cure_df.iterrows():
                    cure_summary.append({
                        '数据来源': source_name,
                        '指标': row.get('指标', ''),
                        '治疗组复常率': row.get('治疗组复常率', np.nan),
                        '对照组复常率': row.get('对照组复常率', np.nan),
                        '复常率差异': row.get('复常率差异', np.nan),
                        'P值': row.get('Fisher P值', np.nan)
                    })
    
    if cure_summary:
        cure_summary_df = pd.DataFrame(cure_summary)
        cure_summary_df.to_csv(f'{result_dir}/comparison_cure_rate.csv', 
                               index=False, encoding='utf-8-sig')
    
    print("对比汇总已保存")


if __name__ == '__main__':
    run_comprehensive_analysis()
