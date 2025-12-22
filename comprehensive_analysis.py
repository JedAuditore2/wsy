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


# ==================== 分层分析功能 ====================
# 扩展的正常范围（考虑临床意义）
EXTENDED_NORMAL_RANGES = {
    '白蛋白': {'lower': 40, 'upper': 55, 'unit': 'g/L', 'direction': 'higher_better'},
    'ALT': {'lower': 7, 'upper': 40, 'unit': 'U/L', 'direction': 'lower_better'},
    'AST': {'lower': 13, 'upper': 35, 'unit': 'U/L', 'direction': 'lower_better'},
    'ALP': {'lower': 35, 'upper': 100, 'unit': 'U/L', 'direction': 'lower_better'},
    'GGT': {'lower': 7, 'upper': 45, 'unit': 'U/L', 'direction': 'lower_better'},
    '胆红素': {'lower': 3.4, 'upper': 17.1, 'unit': 'μmol/L', 'direction': 'lower_better'},
    '肝硬度值': {'lower': 2.5, 'upper': 7.0, 'unit': 'kPa', 'direction': 'lower_better'},
    '血小板': {'lower': 125, 'upper': 350, 'unit': '×10^9/L', 'direction': 'higher_better'},
}


def get_indicator_range(indicator_name):
    """获取指标的正常范围信息"""
    clean_name = indicator_name.replace('基线', '').replace('12个月', '').strip()
    for key, value in EXTENDED_NORMAL_RANGES.items():
        if key in clean_name:
            return value
    return None


def classify_by_normal_range(baseline_val, endpoint_val, indicator_name):
    """
    根据正常范围对患者进行分类
    
    返回:
        'both_normal': 基线和终点均在正常范围内 → 变化可能是正常生理波动
        'both_abnormal': 基线和终点均异常 → 可能需更长治疗时间
        'improved': 基线异常→终点正常 → 真正的治疗改善
        'worsened': 基线正常→终点异常 → 恶化
        'normal_improved': 基线正常且有改善趋势 → 在正常范围内优化
    """
    range_info = get_indicator_range(indicator_name)
    if range_info is None:
        return 'unknown'
    
    lower, upper = range_info['lower'], range_info['upper']
    direction = range_info.get('direction', 'lower_better')
    
    baseline_normal = lower <= baseline_val <= upper
    endpoint_normal = lower <= endpoint_val <= upper
    
    if baseline_normal and endpoint_normal:
        return 'both_normal'
    elif not baseline_normal and not endpoint_normal:
        return 'both_abnormal'
    elif not baseline_normal and endpoint_normal:
        return 'improved'
    else:  # baseline_normal and not endpoint_normal
        return 'worsened'


def analyze_normal_range_fluctuation(df, treatment_col='treatment', result_dir=None):
    """
    分析"正常范围内波动"情况
    
    临床意义：
    - 基线和终点均在正常范围内的患者，其变化值可能只是正常生理波动
    - 这类患者不应作为治疗效果评估的主要依据
    - 真正的治疗效果应来自基线异常→终点正常的"复常"患者
    """
    if result_dir is None:
        result_dir = RESULT_DIR
        
    print("\n" + "="*60)
    print("额外分析: 正常范围内波动分析")
    print("="*60)
    print("说明: 区分正常波动与真正的治疗效果")
    
    indicators = [
        ('基线ALT', 'ALT12个月', 'ALT'),
        ('基线AST', 'AST12个月', 'AST'),
        ('基线GGT', 'GGT12个月', 'GGT'),
        ('基线ALP', 'ALP12个月', 'ALP'),
        ('基线白蛋白', '白蛋白12个月', '白蛋白'),
        ('基线胆红素', '总胆红素12个月', '胆红素'),
    ]
    
    results = []
    
    for base_col, end_col, name in indicators:
        if base_col not in df.columns or end_col not in df.columns:
            continue
            
        for group_name, group_val in [('治疗组', 1), ('对照组', 0)]:
            group = df[df[treatment_col] == group_val].copy()
            valid = group[[base_col, end_col]].dropna()
            
            if len(valid) == 0:
                continue
            
            # 分类每个患者
            categories = valid.apply(
                lambda row: classify_by_normal_range(row[base_col], row[end_col], name), 
                axis=1
            )
            
            total = len(categories)
            both_normal = (categories == 'both_normal').sum()
            both_abnormal = (categories == 'both_abnormal').sum()
            improved = (categories == 'improved').sum()
            worsened = (categories == 'worsened').sum()
            
            # 计算各类别的变化值
            change = valid[end_col] - valid[base_col]
            
            results.append({
                '指标': name,
                '分组': group_name,
                '总例数': total,
                '双正常(可能为波动)': both_normal,
                '双正常比例': f'{both_normal/total*100:.1f}%',
                '基线异常→正常(复常)': improved,
                '复常率': f'{improved/(both_abnormal+improved)*100:.1f}%' if (both_abnormal+improved) > 0 else 'N/A',
                '基线正常→异常(恶化)': worsened,
                '持续异常': both_abnormal,
                '全组变化均值': change.mean(),
                '双正常组变化均值': change[categories == 'both_normal'].mean() if both_normal > 0 else np.nan,
                '注释': '双正常组变化可能是正常生理波动，不代表治疗效果'
            })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'{result_dir}/normal_range_fluctuation.csv', index=False, encoding='utf-8-sig')
    
    print("\n正常范围内波动分析:")
    print(results_df[['指标', '分组', '双正常(可能为波动)', '双正常比例', '基线异常→正常(复常)']].to_string(index=False))
    print("\n⚠️ 注意: '双正常'患者的变化值可能只是正常生理波动，需谨慎解读")
    
    return results_df


def stratified_analysis(df, treatment_col='treatment', result_dir=None):
    """
    临床分层分析 - 区分基线正常和异常患者的治疗效果
    
    解决问题：
    1. 天花板效应：基线正常者改善空间有限
    2. 地板效应：基线已经处于最佳状态
    3. 效应修饰：治疗效果可能因基线状态而异
    """
    if result_dir is None:
        result_dir = RESULT_DIR
        
    print("\n" + "="*60)
    print("8. 临床分层分析")
    print("="*60)
    print("说明: 按基线状态(正常/异常)分层，消除天花板/地板效应")
    
    # 分析的指标对
    indicator_pairs = [
        ('基线ALT', 'ALT12个月', 'ALT'),
        ('基线AST', 'AST12个月', 'AST'),
        ('基线GGT', 'GGT12个月', 'GGT'),
        ('基线胆红素', '胆红素12个月', '胆红素'),
        ('肝硬度值基线', '肝硬度值12个月', '肝硬度值'),
    ]
    
    all_results = []
    tolerance = 2  # 容忍度
    
    for base_col, end_col, name in indicator_pairs:
        if base_col not in df.columns or end_col not in df.columns:
            continue
            
        range_info = get_indicator_range(name)
        if range_info is None:
            continue
            
        upper = range_info['upper']
        lower = range_info['lower']
        direction = range_info['direction']
        
        # 有效数据
        df_valid = df[[base_col, end_col, treatment_col]].dropna().copy()
        if len(df_valid) < 10:
            continue
        
        # 分类基线状态
        def classify(x):
            if x > upper + tolerance:
                return '异常偏高'
            elif x < lower - tolerance:
                return '异常偏低'
            else:
                return '正常'
        
        df_valid['baseline_status'] = df_valid[base_col].apply(classify)
        df_valid['change'] = df_valid[end_col] - df_valid[base_col]
        
        # 分层分析
        for status in ['全人群', '基线异常偏高', '基线正常']:
            if status == '全人群':
                subgroup = df_valid
            elif status == '基线异常偏高':
                subgroup = df_valid[df_valid['baseline_status'] == '异常偏高']
            else:
                subgroup = df_valid[df_valid['baseline_status'] == '正常']
            
            if len(subgroup) < 5:
                continue
            
            treated = subgroup[subgroup[treatment_col] == 1]
            control = subgroup[subgroup[treatment_col] == 0]
            
            if len(treated) < 2 or len(control) < 2:
                continue
            
            # t检验
            try:
                _, p_val = stats.ttest_ind(treated['change'], control['change'], equal_var=False)
            except:
                p_val = np.nan
            
            result = {
                '指标': name,
                '分层': status,
                'N': len(subgroup),
                '治疗组N': len(treated),
                '对照组N': len(control),
                '治疗组变化': treated['change'].mean(),
                '对照组变化': control['change'].mean(),
                '组间差异': treated['change'].mean() - control['change'].mean(),
                'P值': p_val,
                '显著': '是' if p_val < 0.05 else '否'
            }
            all_results.append(result)
        
        # 计算复常率（基线异常→终点正常）
        abnormal_treated = df_valid[(df_valid['baseline_status'] == '异常偏高') & (df_valid[treatment_col] == 1)]
        abnormal_control = df_valid[(df_valid['baseline_status'] == '异常偏高') & (df_valid[treatment_col] == 0)]
        
        if len(abnormal_treated) > 0:
            normalized_t = len(abnormal_treated[abnormal_treated[end_col] <= upper + tolerance])
            norm_rate_t = normalized_t / len(abnormal_treated) * 100
        else:
            norm_rate_t = np.nan
            
        if len(abnormal_control) > 0:
            normalized_c = len(abnormal_control[abnormal_control[end_col] <= upper + tolerance])
            norm_rate_c = normalized_c / len(abnormal_control) * 100
        else:
            norm_rate_c = np.nan
        
        # 计算维持率（基线正常→终点仍正常）
        normal_treated = df_valid[(df_valid['baseline_status'] == '正常') & (df_valid[treatment_col] == 1)]
        normal_control = df_valid[(df_valid['baseline_status'] == '正常') & (df_valid[treatment_col] == 0)]
        
        if len(normal_treated) > 0:
            maintained_t = len(normal_treated[(normal_treated[end_col] >= lower - tolerance) & 
                                               (normal_treated[end_col] <= upper + tolerance)])
            maint_rate_t = maintained_t / len(normal_treated) * 100
        else:
            maint_rate_t = np.nan
            
        if len(normal_control) > 0:
            maintained_c = len(normal_control[(normal_control[end_col] >= lower - tolerance) & 
                                               (normal_control[end_col] <= upper + tolerance)])
            maint_rate_c = maintained_c / len(normal_control) * 100
        else:
            maint_rate_c = np.nan
        
        # 添加复常率和维持率结果
        all_results.append({
            '指标': name,
            '分层': '复常率',
            'N': len(abnormal_treated) + len(abnormal_control),
            '治疗组N': len(abnormal_treated),
            '对照组N': len(abnormal_control),
            '治疗组变化': norm_rate_t,
            '对照组变化': norm_rate_c,
            '组间差异': norm_rate_t - norm_rate_c if not np.isnan(norm_rate_t) and not np.isnan(norm_rate_c) else np.nan,
            'P值': np.nan,
            '显著': ''
        })
        
        all_results.append({
            '指标': name,
            '分层': '维持率',
            'N': len(normal_treated) + len(normal_control),
            '治疗组N': len(normal_treated),
            '对照组N': len(normal_control),
            '治疗组变化': maint_rate_t,
            '对照组变化': maint_rate_c,
            '组间差异': maint_rate_t - maint_rate_c if not np.isnan(maint_rate_t) and not np.isnan(maint_rate_c) else np.nan,
            'P值': np.nan,
            '显著': ''
        })
    
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(f'{result_dir}/stratified_analysis.csv', index=False, encoding='utf-8-sig')
    
    # 打印关键结果
    print("\n分层分析结果摘要:")
    print("-" * 80)
    
    for indicator in results_df['指标'].unique():
        ind_df = results_df[results_df['指标'] == indicator]
        print(f"\n【{indicator}】")
        
        # 全人群
        full = ind_df[ind_df['分层'] == '全人群']
        if len(full) > 0:
            row = full.iloc[0]
            p_str = f"P={row['P值']:.4f}" if not np.isnan(row['P值']) else ""
            sig = " *" if row['显著'] == '是' else ""
            print(f"  全人群(N={row['N']}): 治疗组{row['治疗组变化']:.2f} vs 对照组{row['对照组变化']:.2f} {p_str}{sig}")
        
        # 基线异常
        abnormal = ind_df[ind_df['分层'] == '基线异常偏高']
        if len(abnormal) > 0:
            row = abnormal.iloc[0]
            p_str = f"P={row['P值']:.4f}" if not np.isnan(row['P值']) else ""
            sig = " *" if row['显著'] == '是' else ""
            print(f"  基线异常(N={row['N']}): 治疗组{row['治疗组变化']:.2f} vs 对照组{row['对照组变化']:.2f} {p_str}{sig}")
        
        # 基线正常
        normal = ind_df[ind_df['分层'] == '基线正常']
        if len(normal) > 0:
            row = normal.iloc[0]
            p_str = f"P={row['P值']:.4f}" if not np.isnan(row['P值']) else ""
            sig = " *" if row['显著'] == '是' else ""
            print(f"  基线正常(N={row['N']}): 治疗组{row['治疗组变化']:.2f} vs 对照组{row['对照组变化']:.2f} {p_str}{sig}")
        
        # 复常率和维持率
        norm = ind_df[ind_df['分层'] == '复常率']
        maint = ind_df[ind_df['分层'] == '维持率']
        if len(norm) > 0:
            row = norm.iloc[0]
            if not np.isnan(row['治疗组变化']) and not np.isnan(row['对照组变化']):
                print(f"  复常率: 治疗组{row['治疗组变化']:.1f}% vs 对照组{row['对照组变化']:.1f}%")
        if len(maint) > 0:
            row = maint.iloc[0]
            if not np.isnan(row['治疗组变化']) and not np.isnan(row['对照组变化']):
                print(f"  维持率: 治疗组{row['治疗组变化']:.1f}% vs 对照组{row['对照组变化']:.1f}%")
    
    # 生成分层分析可视化
    create_stratified_visualization(results_df, result_dir)
    
    return results_df


def create_stratified_visualization(results_df, result_dir):
    """创建分层分析可视化图"""
    setup_lancet_style()
    
    # 筛选主要分层
    main_strata = ['全人群', '基线异常偏高', '基线正常']
    
    # 动态获取实际存在的指标（优先选择主要指标）
    preferred_indicators = ['ALT', 'AST', '肝硬度值', '胆红素', 'GGT']
    available_indicators = results_df[results_df['分层'].isin(main_strata)]['指标'].unique().tolist()
    
    # 按优先级排序，选择前4个存在的指标
    main_indicators = []
    for ind in preferred_indicators:
        if ind in available_indicators:
            main_indicators.append(ind)
        if len(main_indicators) == 4:
            break
    
    # 如果不足4个，补充其他可用指标
    if len(main_indicators) < 4:
        for ind in available_indicators:
            if ind not in main_indicators:
                main_indicators.append(ind)
            if len(main_indicators) == 4:
                break
    
    plot_df = results_df[
        (results_df['指标'].isin(main_indicators)) & 
        (results_df['分层'].isin(main_strata))
    ].copy()
    
    if len(plot_df) == 0 or len(main_indicators) == 0:
        return
    
    # 根据实际指标数量动态调整布局
    n_indicators = len(main_indicators)
    if n_indicators == 1:
        fig, axes = plt.subplots(1, 1, figsize=(7, 5), facecolor='white')
        axes = [axes]
    elif n_indicators == 2:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor='white')
        axes = axes.flatten()
    elif n_indicators == 3:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor='white')
        axes = axes.flatten()
    else:  # 4个或更多
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor='white')
        axes = axes.flatten()
    
    for idx, indicator in enumerate(main_indicators):
        if idx >= len(axes):
            break
        ax = axes[idx]
        ax.set_facecolor('white')
        
        ind_df = plot_df[plot_df['指标'] == indicator]
        
        if len(ind_df) == 0:
            continue
        
        # 只保留实际存在的分层
        actual_strata = []
        t_vals = []
        c_vals = []
        p_vals = []
        
        for stratum in main_strata:
            row = ind_df[ind_df['分层'] == stratum]
            if len(row) > 0:
                actual_strata.append(stratum)
                t_vals.append(row.iloc[0]['治疗组变化'])
                c_vals.append(row.iloc[0]['对照组变化'])
                p_vals.append(row.iloc[0]['P值'])
        
        if len(actual_strata) == 0:
            continue
        
        x = np.arange(len(actual_strata))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, t_vals, width, label='治疗组', color=TREATMENT_COLOR,
                       edgecolor='white', linewidth=1)
        bars2 = ax.bar(x + width/2, c_vals, width, label='对照组', color=CONTROL_COLOR,
                       edgecolor='white', linewidth=1)
        
        ax.set_xticks(x)
        ax.set_xticklabels(actual_strata, fontsize=13, fontweight='bold')
        ax.set_ylabel('变化值', fontsize=14, fontweight='bold')
        ax.set_title(f'{indicator}分层分析', fontsize=16, fontweight='bold', pad=10)
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        if idx == 0:
            ax.legend(loc='upper right', frameon=False, fontsize=11)
        
        # 标注显著性
        for i, p in enumerate(p_vals):
            if not np.isnan(p) and p < 0.05:
                # 获取当前柱状图的最大/最小值
                t_val = t_vals[i]
                c_val = c_vals[i]
                
                # 根据数据正负确定星号位置
                if t_val >= 0 or c_val >= 0:
                    # 至少有一个正值，星号放在正值上方
                    max_val = max(t_val, c_val)
                    y_pos = max_val + abs(max_val) * 0.15 + 0.5
                    va = 'bottom'
                else:
                    # 都是负值，星号放在负值下方
                    min_val = min(t_val, c_val)
                    y_pos = min_val - abs(min_val) * 0.15 - 0.5
                    va = 'top'
                
                ax.text(i, y_pos, '*', ha='center', va=va, fontsize=16, fontweight='bold', color='red')
    
    plt.tight_layout()
    save_lancet_figure(fig, f'{result_dir}/stratified_analysis.png')
    print("\n分层分析图已保存: stratified_analysis.png")


# ==================== 相关性分析 ====================
def correlation_analysis(df, treatment_col='treatment', result_dir=None):
    """
    指标间相关性分析
    根据正态性选择Pearson（正态）或Spearman（非正态）相关系数
    """
    if result_dir is None:
        result_dir = RESULT_DIR
        
    print("\n" + "="*60)
    print("9. 指标间相关性分析")
    print("="*60)
    print("方法选择: 正态分布→Pearson相关 | 非正态→Spearman相关")
    
    # 选择基线指标进行相关性分析
    baseline_vars = ['基线ALT', '基线AST', '基线GGT', '基线ALP', '基线白蛋白', 
                     '基线胆红素', '肝硬度值基线', '血小板基线']
    baseline_vars = [v for v in baseline_vars if v in df.columns]
    
    # 变化值指标
    change_pairs = [
        ('基线ALT', 'ALT12个月', 'ALT变化'),
        ('基线AST', 'AST12个月', 'AST变化'),
        ('肝硬度值基线', '肝硬度值12个月', '肝硬度变化'),
    ]
    
    # 计算变化值
    df_corr = df.copy()
    for base, end, change in change_pairs:
        if base in df.columns and end in df.columns:
            df_corr[change] = df_corr[end] - df_corr[base]
    
    all_vars = baseline_vars + [c[2] for c in change_pairs if c[2] in df_corr.columns]
    
    results = []
    
    # 计算相关系数矩阵
    for i, var1 in enumerate(all_vars):
        for j, var2 in enumerate(all_vars):
            if i < j and var1 in df_corr.columns and var2 in df_corr.columns:
                data1 = df_corr[var1].dropna()
                data2 = df_corr[var2].dropna()
                
                # 取两个变量的共同索引
                common_idx = data1.index.intersection(data2.index)
                if len(common_idx) < 10:
                    continue
                    
                x = df_corr.loc[common_idx, var1]
                y = df_corr.loc[common_idx, var2]
                
                # 正态性检验
                _, norm_p1 = stats.shapiro(x[:50]) if len(x) >= 3 else (0, 1)
                _, norm_p2 = stats.shapiro(y[:50]) if len(y) >= 3 else (0, 1)
                
                is_normal = (norm_p1 >= 0.05) and (norm_p2 >= 0.05)
                
                if is_normal:
                    # Pearson相关
                    corr, pval = stats.pearsonr(x, y)
                    method = 'Pearson'
                    reason = '两变量均正态分布'
                else:
                    # Spearman相关
                    corr, pval = stats.spearmanr(x, y)
                    method = 'Spearman'
                    reason = '存在非正态分布变量'
                
                # 相关强度判断
                abs_corr = abs(corr)
                if abs_corr >= 0.7:
                    strength = '强相关'
                elif abs_corr >= 0.4:
                    strength = '中等相关'
                elif abs_corr >= 0.2:
                    strength = '弱相关'
                else:
                    strength = '极弱/无相关'
                
                results.append({
                    '变量1': var1,
                    '变量2': var2,
                    'N': len(common_idx),
                    '相关系数': corr,
                    'P值': pval,
                    '检验方法': method,
                    '选择依据': reason,
                    '相关强度': strength,
                    '统计显著': '是' if pval < 0.05 else '否'
                })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'{result_dir}/correlation_analysis.csv', index=False, encoding='utf-8-sig')
    
    # 打印显著相关
    print("\n显著相关（P<0.05）的变量对:")
    print("-" * 70)
    sig_results = results_df[results_df['统计显著'] == '是'].sort_values('相关系数', key=abs, ascending=False)
    
    for _, row in sig_results.head(10).iterrows():
        print(f"  {row['变量1']} ↔ {row['变量2']}: r={row['相关系数']:.3f}, P={row['P值']:.4f} ({row['检验方法']}, {row['相关强度']})")
    
    # 创建相关系数热力图
    create_correlation_heatmap(df_corr, baseline_vars, result_dir)
    
    return results_df


def create_correlation_heatmap(df, variables, result_dir):
    """创建相关系数热力图"""
    setup_lancet_style()
    
    # 计算相关矩阵
    corr_matrix = df[variables].corr(method='spearman')
    
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
    ax.set_facecolor('white')
    
    # 绘制热力图
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('相关系数', fontsize=14, fontweight='bold')
    
    # 设置刻度
    ax.set_xticks(np.arange(len(variables)))
    ax.set_yticks(np.arange(len(variables)))
    
    # 简化变量名
    short_names = [v.replace('基线', '').replace('12个月', '') for v in variables]
    ax.set_xticklabels(short_names, fontsize=12, fontweight='bold', rotation=45, ha='right')
    ax.set_yticklabels(short_names, fontsize=12, fontweight='bold')
    
    # 添加数值标注
    for i in range(len(variables)):
        for j in range(len(variables)):
            val = corr_matrix.iloc[i, j]
            color = 'white' if abs(val) > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                   fontsize=10, fontweight='bold', color=color)
    
    ax.set_title('基线指标相关性热力图（Spearman）', fontsize=16, fontweight='bold', pad=15)
    
    plt.tight_layout()
    save_lancet_figure(fig, f'{result_dir}/correlation_heatmap.png')
    print("\n相关性热力图已保存: correlation_heatmap.png")


# ==================== 多因素回归分析 ====================
def multivariate_regression_analysis(df, treatment_col='treatment', result_dir=None):
    """
    多因素回归分析 - 控制混杂因素后评估治疗效果
    使用线性回归分析各结局指标
    """
    if result_dir is None:
        result_dir = RESULT_DIR
        
    print("\n" + "="*60)
    print("10. 多因素回归分析")
    print("="*60)
    print("目的: 控制混杂因素后评估治疗的独立效应")
    
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    
    # 结局变量（变化值）
    outcomes = [
        ('基线ALT', 'ALT12个月', 'ALT变化'),
        ('基线AST', 'AST12个月', 'AST变化'),
        ('肝硬度值基线', '肝硬度值12个月', '肝硬度变化'),
        ('基线胆红素', '胆红素12个月', '胆红素变化'),
    ]
    
    # 协变量（混杂因素）
    confounders = ['年龄', '性别', '是否合并脂肪肝']
    confounders = [c for c in confounders if c in df.columns]
    
    all_results = []
    
    for base_col, end_col, outcome_name in outcomes:
        if base_col not in df.columns or end_col not in df.columns:
            continue
        
        # 创建分析数据
        df_analysis = df.copy()
        df_analysis['outcome'] = df_analysis[end_col] - df_analysis[base_col]
        df_analysis['baseline'] = df_analysis[base_col]  # 基线值作为协变量
        
        # 准备特征
        features = [treatment_col, 'baseline'] + confounders
        
        # 删除缺失值
        df_valid = df_analysis[['outcome'] + features].dropna()
        
        if len(df_valid) < 30:
            continue
        
        X = df_valid[features]
        y = df_valid['outcome']
        
        # 拟合模型
        model = LinearRegression()
        model.fit(X, y)
        
        # 获取系数
        coef_treatment = model.coef_[0]
        
        # 计算标准误和P值（使用statsmodels获得更详细的统计信息）
        try:
            import statsmodels.api as sm
            X_with_const = sm.add_constant(X)
            ols_model = sm.OLS(y, X_with_const).fit()
            
            # treatment系数在第2个位置（第1个是常数项）
            treatment_idx = 1
            coef = ols_model.params.iloc[treatment_idx]
            se = ols_model.bse.iloc[treatment_idx]
            pval = ols_model.pvalues.iloc[treatment_idx]
            ci_low = ols_model.conf_int().iloc[treatment_idx, 0]
            ci_high = ols_model.conf_int().iloc[treatment_idx, 1]
            r_squared = ols_model.rsquared
            
            # 判断显著性
            significant = pval < 0.05
            
            result = {
                '结局变量': outcome_name,
                'N': len(df_valid),
                '治疗效应系数': coef,
                '标准误': se,
                '95%CI下限': ci_low,
                '95%CI上限': ci_high,
                'P值': pval,
                'R²': r_squared,
                '显著': '是' if significant else '否',
                '控制变量': '基线值+' + '+'.join(confounders),
                '解释': f"控制混杂因素后，治疗组{outcome_name}{'显著' if significant else '无显著'}{'减少' if coef < 0 else '增加'}{abs(coef):.2f}单位"
            }
            
            all_results.append(result)
            
            print(f"\n【{outcome_name}】")
            print(f"  样本量: {len(df_valid)}")
            print(f"  治疗效应: β={coef:.3f}, SE={se:.3f}, P={pval:.4f}")
            print(f"  95%CI: [{ci_low:.3f}, {ci_high:.3f}]")
            print(f"  模型R²: {r_squared:.3f}")
            print(f"  结论: {result['解释']}")
            
        except Exception as e:
            print(f"  警告: {outcome_name}回归分析失败: {e}")
    
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(f'{result_dir}/multivariate_regression.csv', index=False, encoding='utf-8-sig')
    
    # 绘制森林图
    if len(all_results) > 0:
        create_regression_forest_plot(results_df, result_dir)
    
    return results_df


def create_regression_forest_plot(results_df, result_dir):
    """创建多因素回归森林图"""
    setup_lancet_style()
    
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    ax.set_facecolor('white')
    
    outcomes = results_df['结局变量'].tolist()
    y_pos = np.arange(len(outcomes))
    
    coefs = results_df['治疗效应系数'].values
    ci_lows = results_df['95%CI下限'].values
    ci_highs = results_df['95%CI上限'].values
    pvals = results_df['P值'].values
    
    # 绘制误差线和点
    colors = [TREATMENT_COLOR if p < 0.05 else '#888888' for p in pvals]
    
    for i, (y, coef, ci_l, ci_h, color) in enumerate(zip(y_pos, coefs, ci_lows, ci_highs, colors)):
        ax.plot([ci_l, ci_h], [y, y], color=color, linewidth=2, solid_capstyle='round')
        ax.scatter(coef, y, color=color, s=100, zorder=5, edgecolors='white', linewidth=1)
    
    # 添加零参考线
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.7)
    
    # 设置轴
    ax.set_yticks(y_pos)
    ax.set_yticklabels(outcomes, fontsize=14, fontweight='bold')
    ax.set_xlabel('治疗效应 (β系数)', fontsize=14, fontweight='bold')
    ax.set_title('多因素回归分析：治疗对各指标变化的独立效应', fontsize=16, fontweight='bold', pad=15)
    
    # 添加P值标注
    for i, (coef, p) in enumerate(zip(coefs, pvals)):
        sig_mark = '*' if p < 0.05 else ''
        ax.text(coef, i + 0.25, f'P={p:.3f}{sig_mark}', ha='center', fontsize=11, fontweight='bold')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 添加图例说明
    ax.text(0.02, 0.98, '红色: P<0.05 (显著)\n灰色: P≥0.05 (不显著)', 
            transform=ax.transAxes, fontsize=11, va='top',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.8))
    
    plt.tight_layout()
    save_lancet_figure(fig, f'{result_dir}/regression_forest_plot.png')
    print("\n多因素回归森林图已保存: regression_forest_plot.png")


# ==================== 1. 基线可比性检验 ====================
def baseline_comparability_test(df, treatment_col='treatment', result_dir=None):
    """
    验证两组基线是否可比
    检验选择依据：
    - 连续变量：先Shapiro-Wilk正态性检验
      - 两组均正态 → 独立样本t检验（Welch）
      - 任一组非正态 → Mann-Whitney U检验
    - 分类变量：
      - 期望频数均≥5 → 卡方检验
      - 存在期望频数<5 → Fisher精确检验
    """
    if result_dir is None:
        result_dir = RESULT_DIR
    print("\n" + "="*60)
    print("1. 基线可比性检验")
    print("="*60)
    print("检验选择策略:")
    print("  - 连续变量: Shapiro-Wilk正态性检验 → t检验/Mann-Whitney U")
    print("  - 分类变量: 期望频数检验 → 卡方检验/Fisher精确检验")
    
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
                # t检验（Welch，不假设方差齐性）
                t_stat, t_pval = stats.ttest_ind(t_data, c_data, equal_var=False)
                # Mann-Whitney U检验（非参数）
                u_stat, u_pval = stats.mannwhitneyu(t_data, c_data, alternative='two-sided')
                # 正态性检验 (Shapiro-Wilk)
                _, norm_p_t = stats.shapiro(t_data[:50]) if len(t_data) >= 3 else (0, 1)
                _, norm_p_c = stats.shapiro(c_data[:50]) if len(c_data) >= 3 else (0, 1)
                
                # 选择依据
                is_normal = (norm_p_t >= 0.05) and (norm_p_c >= 0.05)
                recommended_pval = t_pval if is_normal else u_pval
                test_used = 't检验(Welch)' if is_normal else 'Mann-Whitney U'
                selection_reason = '两组均正态分布' if is_normal else f'存在非正态(治疗组P={norm_p_t:.3f},对照组P={norm_p_c:.3f})'
                
                results.append({
                    '变量': var,
                    '变量类型': '连续',
                    '治疗组均值': t_data.mean(),
                    '治疗组SD': t_data.std(),
                    '对照组均值': c_data.mean(),
                    '对照组SD': c_data.std(),
                    't检验P值': t_pval,
                    'Mann-Whitney P值': u_pval,
                    '推荐P值': recommended_pval,
                    '采用检验': test_used,
                    '选择依据': selection_reason,
                    '基线可比': '是' if recommended_pval > 0.05 else '否'
                })
    
    # 分类变量 - 卡方检验
    for var in categorical_vars:
        if var in df.columns:
            contingency = pd.crosstab(df[var], df[treatment_col])
            if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
                chi2, chi_pval, dof, expected = stats.chi2_contingency(contingency)
                
                # Fisher精确检验（当期望频数<5时）
                min_expected = expected.min()
                if (expected < 5).any():
                    try:
                        _, fisher_pval = stats.fisher_exact(contingency)
                        final_pval = fisher_pval
                        test_used = 'Fisher精确检验'
                        selection_reason = f'存在期望频数<5(最小={min_expected:.1f})'
                    except:
                        final_pval = chi_pval
                        test_used = '卡方检验'
                        selection_reason = 'Fisher检验失败，使用卡方'
                else:
                    final_pval = chi_pval
                    test_used = '卡方检验'
                    selection_reason = f'期望频数均≥5(最小={min_expected:.1f})'
                
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
                    '采用检验': test_used,
                    '选择依据': selection_reason,
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
    
    分层分析策略：
    1. 全人群分析
    2. 基线异常患者分析 - 消除天花板效应
    3. 基线正常患者分析 - 评估维持稳定能力
    4. 排除"双正常"敏感性分析
    """
    if result_dir is None:
        result_dir = RESULT_DIR
    print("\n" + "="*60)
    print("2. 治疗后肝功能差异分析（含分层）")
    print("="*60)
    print("分层策略: 全人群 / 基线异常 / 基线正常 / 排除双正常")
    
    # 肝功能相关指标及其正常上限
    liver_outcomes = [
        ('基线ALT', 'ALT12个月', 'ALT', 40),
        ('基线AST', 'AST12个月', 'AST', 35),
        ('基线GGT', 'GGT12个月', 'GGT', 45),
        ('基线ALP', 'ALP12个月', 'ALP', 100),
        ('基线白蛋白', '白蛋白12个月', '白蛋白', 55),  # 白蛋白是下限40
        ('基线胆红素', '总胆红素12个月', '胆红素', 17.1),
    ]
    
    results = []
    stratified_results = []
    tolerance = 2  # 容忍度
    
    for base_col, end_col, name, upper_limit in liver_outcomes:
        if base_col in df.columns and end_col in df.columns:
            # 特殊处理白蛋白（下限异常）
            if name == '白蛋白':
                lower_limit = 40
                is_abnormal_func = lambda x: x < lower_limit - tolerance
            else:
                is_abnormal_func = lambda x: x > upper_limit + tolerance
            
            treated = df[df[treatment_col] == 1]
            control = df[df[treatment_col] == 0]
            
            # 计算变化值（全人群）
            t_valid = treated[[base_col, end_col]].dropna()
            c_valid = control[[base_col, end_col]].dropna()
            
            t_change = t_valid[end_col] - t_valid[base_col]
            c_change = c_valid[end_col] - c_valid[base_col]
            
            # 分类：识别"双正常"患者
            t_categories = t_valid.apply(
                lambda row: classify_by_normal_range(row[base_col], row[end_col], name), 
                axis=1
            )
            c_categories = c_valid.apply(
                lambda row: classify_by_normal_range(row[base_col], row[end_col], name), 
                axis=1
            )
            
            # === 分层分析 ===
            # 基线异常患者
            t_abnormal_idx = t_valid[base_col].apply(is_abnormal_func)
            c_abnormal_idx = c_valid[base_col].apply(is_abnormal_func)
            t_change_abnormal = t_change[t_abnormal_idx]
            c_change_abnormal = c_change[c_abnormal_idx]
            
            # 基线正常患者
            t_change_normal = t_change[~t_abnormal_idx]
            c_change_normal = c_change[~c_abnormal_idx]
            
            # 排除双正常后的变化值
            t_change_excl = t_change[t_categories != 'both_normal']
            c_change_excl = c_change[c_categories != 'both_normal']
            
            if len(t_change) > 1 and len(c_change) > 1:
                # 全人群分析
                t_stat, t_pval = stats.ttest_ind(t_change, c_change, equal_var=False)
                u_stat, u_pval = stats.mannwhitneyu(t_change, c_change, alternative='two-sided')
                pooled_std = np.sqrt((t_change.var() + c_change.var()) / 2)
                cohens_d = (t_change.mean() - c_change.mean()) / pooled_std if pooled_std > 0 else 0
                
                # 基线异常亚组分析
                abnormal_pval = np.nan
                abnormal_diff = np.nan
                if len(t_change_abnormal) > 1 and len(c_change_abnormal) > 1:
                    _, abnormal_pval = stats.mannwhitneyu(t_change_abnormal, c_change_abnormal, alternative='two-sided')
                    abnormal_diff = t_change_abnormal.mean() - c_change_abnormal.mean()
                
                # 基线正常亚组分析
                normal_pval = np.nan
                normal_diff = np.nan
                if len(t_change_normal) > 1 and len(c_change_normal) > 1:
                    _, normal_pval = stats.mannwhitneyu(t_change_normal, c_change_normal, alternative='two-sided')
                    normal_diff = t_change_normal.mean() - c_change_normal.mean()
                
                # 排除双正常后的敏感性分析
                excl_pval = np.nan
                if len(t_change_excl) > 1 and len(c_change_excl) > 1:
                    _, excl_pval = stats.mannwhitneyu(t_change_excl, c_change_excl, alternative='two-sided')
                
                # 判断方向
                is_primary = name in ['ALT', 'AST']
                
                results.append({
                    '指标': name,
                    '指标类型': '主要' if is_primary else '次要',
                    '分层': '全人群',
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
                    '统计学差异': '是' if u_pval < 0.05 else '否',
                    '临床意义': '治疗组更优' if (name != '白蛋白' and t_change.mean() < c_change.mean()) or 
                               (name == '白蛋白' and t_change.mean() > c_change.mean()) else '对照组更优或无差异',
                })
                
                # 添加分层结果
                # 基线异常
                if len(t_change_abnormal) > 0 or len(c_change_abnormal) > 0:
                    stratified_results.append({
                        '指标': name,
                        '分层': '基线异常',
                        '治疗组N': len(t_change_abnormal),
                        '治疗组变化': t_change_abnormal.mean() if len(t_change_abnormal) > 0 else np.nan,
                        '对照组N': len(c_change_abnormal),
                        '对照组变化': c_change_abnormal.mean() if len(c_change_abnormal) > 0 else np.nan,
                        '组间差异': abnormal_diff,
                        'P值': abnormal_pval,
                        '显著': '是' if abnormal_pval < 0.05 else '否' if not np.isnan(abnormal_pval) else 'N/A',
                        '注释': '消除天花板效应后的真实治疗效果'
                    })
                
                # 基线正常
                if len(t_change_normal) > 0 or len(c_change_normal) > 0:
                    stratified_results.append({
                        '指标': name,
                        '分层': '基线正常',
                        '治疗组N': len(t_change_normal),
                        '治疗组变化': t_change_normal.mean() if len(t_change_normal) > 0 else np.nan,
                        '对照组N': len(c_change_normal),
                        '对照组变化': c_change_normal.mean() if len(c_change_normal) > 0 else np.nan,
                        '组间差异': normal_diff,
                        'P值': normal_pval,
                        '显著': '是' if normal_pval < 0.05 else '否' if not np.isnan(normal_pval) else 'N/A',
                        '注释': '评估在正常范围内的稳定维持能力'
                    })
                
                # 排除双正常
                stratified_results.append({
                    '指标': name,
                    '分层': '排除双正常',
                    '治疗组N': len(t_change_excl),
                    '治疗组变化': t_change_excl.mean() if len(t_change_excl) > 0 else np.nan,
                    '对照组N': len(c_change_excl),
                    '对照组变化': c_change_excl.mean() if len(c_change_excl) > 0 else np.nan,
                    '组间差异': t_change_excl.mean() - c_change_excl.mean() if len(t_change_excl) > 0 and len(c_change_excl) > 0 else np.nan,
                    'P值': excl_pval,
                    '显著': '是' if excl_pval < 0.05 else '否' if not np.isnan(excl_pval) else 'N/A',
                    '注释': f'排除{(t_categories=="both_normal").sum()}+{(c_categories=="both_normal").sum()}例双正常患者'
                })
    
    results_df = pd.DataFrame(results)
    stratified_df = pd.DataFrame(stratified_results)
    
    results_df.to_csv(f'{result_dir}/liver_function_difference.csv', index=False, encoding='utf-8-sig')
    stratified_df.to_csv(f'{result_dir}/liver_function_stratified.csv', index=False, encoding='utf-8-sig')
    
    print("\n全人群肝功能变化值组间比较:")
    print(results_df[['指标', '治疗组变化均值', '对照组变化均值', '组间差异', 'Mann-Whitney P值', '统计学差异']].to_string(index=False))
    
    print("\n分层分析结果（基线异常 vs 基线正常）:")
    print("-" * 80)
    for indicator in stratified_df['指标'].unique():
        ind_df = stratified_df[stratified_df['指标'] == indicator]
        print(f"\n【{indicator}】")
        for _, row in ind_df.iterrows():
            p_str = f"P={row['P值']:.4f}" if not np.isnan(row['P值']) else "N/A"
            sig = " *" if row['显著'] == '是' else ""
            if not np.isnan(row['治疗组变化']) and not np.isnan(row['对照组变化']):
                print(f"  {row['分层']}(治疗组N={row['治疗组N']}, 对照组N={row['对照组N']}): "
                      f"治疗组{row['治疗组变化']:.2f} vs 对照组{row['对照组变化']:.2f} {p_str}{sig}")
    
    return results_df, stratified_df


# ==================== 3. 非劣效性检验 ====================
def non_inferiority_test(df, treatment_col='treatment', margin=10, result_dir=None):
    """
    非劣效性检验 - 验证实验组是否能减少肝损伤风险
    
    H0: 治疗组变化 - 对照组变化 >= margin (劣效)
    H1: 治疗组变化 - 对照组变化 < margin (非劣效)
    
    对于ALT/AST，下降越多越好，因此治疗组变化应该<=对照组变化
    margin: 非劣效界值（U/L），临床上通常取10 U/L或正常上限的25%
    
    分层分析：
    1. 全人群分析
    2. 基线异常患者分析 - 更能反映真实治疗效果
    """
    if result_dir is None:
        result_dir = RESULT_DIR
    print("\n" + "="*60)
    print("3. 非劣效性检验（含分层）")
    print("="*60)
    print(f"非劣效界值(margin): ALT={margin} U/L, AST=8.75 U/L")
    print("分层策略: 全人群 / 基线异常患者")
    
    # 主要针对ALT和AST
    ni_outcomes = [
        ('基线ALT', 'ALT12个月', 'ALT', 10, 40),  # margin = 10 U/L, upper_limit = 40
        ('基线AST', 'AST12个月', 'AST', 8.75, 35),  # margin = 35*0.25 = 8.75 U/L, upper_limit = 35
    ]
    
    results = []
    tolerance = 2  # 容忍度
    
    for base_col, end_col, name, ni_margin, upper_limit in ni_outcomes:
        if base_col in df.columns and end_col in df.columns:
            treated = df[df[treatment_col] == 1]
            control = df[df[treatment_col] == 0]
            
            t_valid = treated[[base_col, end_col]].dropna()
            c_valid = control[[base_col, end_col]].dropna()
            
            # === 分层分析函数 ===
            def analyze_subgroup(t_data, c_data, subgroup_name):
                t_change = t_data[end_col] - t_data[base_col]
                c_change = c_data[end_col] - c_data[base_col]
                
                if len(t_change) < 2 or len(c_change) < 2:
                    return None
                
                # 计算差异及其置信区间
                diff = t_change.mean() - c_change.mean()
                se_diff = np.sqrt(t_change.var()/len(t_change) + c_change.var()/len(c_change))
                
                # 95%置信区间
                ci_lower = diff - 1.96 * se_diff
                ci_upper = diff + 1.96 * se_diff
                
                # 非劣效性判断
                is_non_inferior = ci_upper < ni_margin
                is_superior = ci_upper < 0
                
                return {
                    '指标': name,
                    '分层': subgroup_name,
                    'N': len(t_change) + len(c_change),
                    '治疗组N': len(t_change),
                    '对照组N': len(c_change),
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
                }
            
            # 全人群分析
            full_result = analyze_subgroup(t_valid, c_valid, '全人群')
            if full_result:
                results.append(full_result)
            
            # 基线异常患者分析
            t_abnormal = t_valid[t_valid[base_col] > upper_limit + tolerance]
            c_abnormal = c_valid[c_valid[base_col] > upper_limit + tolerance]
            abnormal_result = analyze_subgroup(t_abnormal, c_abnormal, '基线异常')
            if abnormal_result:
                results.append(abnormal_result)
            
            # 基线正常患者分析
            t_normal = t_valid[t_valid[base_col] <= upper_limit + tolerance]
            c_normal = c_valid[c_valid[base_col] <= upper_limit + tolerance]
            normal_result = analyze_subgroup(t_normal, c_normal, '基线正常')
            if normal_result:
                results.append(normal_result)
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'{result_dir}/non_inferiority_test.csv', index=False, encoding='utf-8-sig')
    
    print("\n非劣效性检验结果（全人群 + 分层）:")
    print("-" * 90)
    for indicator in ['ALT', 'AST']:
        ind_df = results_df[results_df['指标'] == indicator]
        print(f"\n【{indicator}】")
        for _, row in ind_df.iterrows():
            print(f"  {row['分层']}(N={row['N']}): 差异={row['差异(治疗-对照)']:.2f}, "
                  f"95%CI=[{row['95%CI下限']:.2f}, {row['95%CI上限']:.2f}], "
                  f"界值={row['非劣效界值']:.1f} → {row['综合结论']}")
    
    # 绘制非劣效性森林图（只显示全人群结果）- 柳叶刀风格
    full_results = results_df[results_df['分层'] == '全人群']
    
    setup_lancet_style()
    fig, ax = plt.subplots(figsize=(10, 3.5), facecolor='white')
    ax.set_facecolor('white')
    
    y_pos = np.arange(len(full_results))
    
    for i, row in full_results.iterrows():
        y_idx = list(full_results.index).index(i)
        # 柳叶刀配色：优效绿色，非劣效蓝色，未证明红色
        if row['综合结论'] == '优效':
            color = '#42B540'  # 绿色
        elif row['综合结论'] == '非劣效':
            color = '#00468B'  # 深蓝
        else:
            color = '#ED0000'  # 红色
        
        # 置信区间线
        ax.plot([row['95%CI下限'], row['95%CI上限']], [y_idx, y_idx], 
               color=color, linewidth=2.5, solid_capstyle='round')
        # 端点
        ax.plot([row['95%CI下限'], row['95%CI上限']], [y_idx, y_idx], '|', 
               color=color, markersize=10, markeredgewidth=2)
        # 点估计 - 菱形
        ax.plot(row['差异(治疗-对照)'], y_idx, 'D', color=color, markersize=10, 
               markeredgecolor='white', markeredgewidth=1)
    
    # 参考线
    ax.axvline(x=0, color='#333333', linestyle='-', linewidth=1.2, zorder=1)
    ax.axvline(x=full_results['非劣效界值'].max(), color='#AD002A', linestyle='--', 
              linewidth=1.5, label=f'非劣效界值 ({full_results["非劣效界值"].max():.1f})')
    
    ax.set_yticks(np.arange(len(full_results)))
    ax.set_yticklabels(full_results['指标'], fontsize=16)
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
    
    # === 新增：分层变化值箱式图（基线异常 vs 基线正常）===
    create_stratified_change_boxplot(df, treatment_col, result_dir)


def create_stratified_change_boxplot(df, treatment_col='treatment', result_dir=None):
    """
    创建分层变化值箱式图 - 展示基线异常和基线正常患者的变化值差异
    
    目的：消除天花板效应，展示不同基线状态下的治疗效果
    """
    if result_dir is None:
        result_dir = RESULT_DIR
        
    print("\n--- 分层变化值箱式图（基线异常 vs 基线正常）---")
    
    setup_lancet_style()
    
    # 需要分层分析的关键指标
    indicators = [
        ('基线ALT', 'ALT12个月', 'ALT', 40),
        ('基线AST', 'AST12个月', 'AST', 35),
        ('基线胆红素', '总胆红素12个月', '胆红素', 17.1),
        ('肝硬度值基线', '肝硬度值12个月', '肝硬度', 7.0),
    ]
    
    tolerance = 2  # 容忍度
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor='white')
    axes = axes.flatten()
    
    for idx, (base_col, end_col, name, upper_limit) in enumerate(indicators):
        if base_col not in df.columns or end_col not in df.columns:
            continue
            
        ax = axes[idx]
        ax.set_facecolor('white')
        
        # 肝硬度容忍度为0
        tol = 0 if name == '肝硬度' else tolerance
        
        treated = df[df[treatment_col] == 1]
        control = df[df[treatment_col] == 0]
        
        # 计算变化值
        t_valid = treated[[base_col, end_col]].dropna()
        c_valid = control[[base_col, end_col]].dropna()
        
        t_change = t_valid[end_col] - t_valid[base_col]
        c_change = c_valid[end_col] - c_valid[base_col]
        
        # 分层：基线异常 vs 基线正常
        t_abnormal_idx = t_valid[base_col] > upper_limit + tol
        c_abnormal_idx = c_valid[base_col] > upper_limit + tol
        
        # 准备四组数据
        data_groups = []
        labels = []
        colors_list = []
        
        # 基线异常-治疗组
        t_change_abnormal = t_change[t_abnormal_idx]
        if len(t_change_abnormal) >= 2:
            data_groups.append(t_change_abnormal)
            labels.append(f'基线异常\n治疗组(n={len(t_change_abnormal)})')
            colors_list.append(TREATMENT_COLOR)
        
        # 基线异常-对照组
        c_change_abnormal = c_change[c_abnormal_idx]
        if len(c_change_abnormal) >= 2:
            data_groups.append(c_change_abnormal)
            labels.append(f'基线异常\n对照组(n={len(c_change_abnormal)})')
            colors_list.append(CONTROL_COLOR)
        
        # 基线正常-治疗组
        t_change_normal = t_change[~t_abnormal_idx]
        if len(t_change_normal) >= 2:
            data_groups.append(t_change_normal)
            labels.append(f'基线正常\n治疗组(n={len(t_change_normal)})')
            colors_list.append('#FF6B6B')  # 浅红
        
        # 基线正常-对照组
        c_change_normal = c_change[~c_abnormal_idx]
        if len(c_change_normal) >= 2:
            data_groups.append(c_change_normal)
            labels.append(f'基线正常\n对照组(n={len(c_change_normal)})')
            colors_list.append('#6B9FFF')  # 浅蓝
        
        if len(data_groups) < 2:
            ax.text(0.5, 0.5, '数据不足', ha='center', va='center', fontsize=14)
            ax.set_title(f'{name}', fontsize=16, fontweight='bold')
            continue
        
        # 绘制箱式图
        bp = ax.boxplot(data_groups, patch_artist=True, labels=labels, widths=0.6,
                       medianprops=dict(color='white', linewidth=2),
                       whiskerprops=dict(color='#333333', linewidth=1),
                       capprops=dict(color='#333333', linewidth=1),
                       flierprops=dict(marker='o', markerfacecolor='#666666', 
                                      markersize=4, alpha=0.5, markeredgecolor='none'))
        
        for patch, color in zip(bp['boxes'], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
            patch.set_edgecolor('#333333')
            patch.set_linewidth(1)
        
        # 添加散点
        for i, (data, color) in enumerate(zip(data_groups, colors_list)):
            x_jitter = np.random.normal(i + 1, 0.05, len(data))
            ax.scatter(x_jitter, data, alpha=0.3, s=15, color=color, edgecolors='none', zorder=3)
        
        ax.axhline(y=0, color='#333333', linestyle='--', linewidth=1, alpha=0.7)
        ax.set_title(f'{name}变化值分层比较', fontsize=16, fontweight='bold')
        ax.set_ylabel('变化值', fontsize=14, fontweight='bold')
        ax.tick_params(axis='both', labelsize=11)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # 添加P值标注（基线异常亚组）
        if len(t_change_abnormal) >= 2 and len(c_change_abnormal) >= 2:
            _, p_abnormal = stats.mannwhitneyu(t_change_abnormal, c_change_abnormal, alternative='two-sided')
            sig = '*' if p_abnormal < 0.05 else ''
            ax.text(0.5, 0.95, f'基线异常组间: P={p_abnormal:.3f}{sig}', 
                   transform=ax.transAxes, ha='center', fontsize=12, 
                   fontweight='bold' if p_abnormal < 0.05 else 'normal',
                   color=TREATMENT_COLOR if p_abnormal < 0.05 else '#333333')
    
    plt.suptitle('分层变化值比较（消除天花板效应）\n深色=基线异常，浅色=基线正常', 
                fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_lancet_figure(fig, f'{result_dir}/boxplot_stratified_change.png')
    plt.close()
    
    print("分层变化值箱式图已保存: boxplot_stratified_change.png")


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
            
            # 调整y轴范围 - 设置非零起始值以突显差异
            all_values = [t_base_mean, t_end_mean, c_base_mean, c_end_mean]
            all_se = [t_base_se, t_end_se, c_base_se, c_end_se]
            
            # 过滤掉NaN值
            valid_values = [v for v in all_values if not np.isnan(v)]
            valid_se = [s for s in all_se if not np.isnan(s)]
            
            if valid_values and valid_se:
                min_val = min(valid_values) - max(valid_se) * 2
                max_val = max(valid_values) + max(valid_se) * 3  # 留空间给数值标注
                
                # Y轴起始值设为最小值的60-80%，但不能为负（除非数据本身为负）
                if min_val > 0:
                    y_start = min_val * 0.7  # 从最小值的开始
                else:
                    y_start = min_val * 1.1
                
                ax.set_ylim(y_start, max_val * 1.15)
            
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
    
    # === 新增：分层变化值柱状图（基线异常 vs 基线正常）===
    create_stratified_change_barplot(df, treatment_col, result_dir)


def create_stratified_change_barplot(df, treatment_col='treatment', result_dir=None):
    """
    创建分层变化值柱状图 - 展示基线异常患者的变化值比较
    
    目的：消除天花板效应，聚焦于基线异常患者的治疗效果
    """
    if result_dir is None:
        result_dir = RESULT_DIR
        
    print("\n--- 分层变化值柱状图（基线异常亚组）---")
    
    setup_lancet_style()
    
    indicators = [
        ('基线ALT', 'ALT12个月', 'ALT', 40, 2),
        ('基线AST', 'AST12个月', 'AST', 35, 2),
        ('基线GGT', 'GGT12个月', 'GGT', 45, 2),
        ('基线ALP', 'ALP12个月', 'ALP', 100, 2),
        ('基线白蛋白', '白蛋白12个月', '白蛋白', 38, 2),  # 低于正常视为异常
        ('基线胆红素', '总胆红素12个月', '胆红素', 17.1, 2),
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), facecolor='white')
    axes = axes.flatten()
    
    for idx, (base_col, end_col, name, threshold, tolerance) in enumerate(indicators):
        if base_col not in df.columns or end_col not in df.columns:
            continue
            
        ax = axes[idx]
        ax.set_facecolor('white')
        
        treated = df[df[treatment_col] == 1]
        control = df[df[treatment_col] == 0]
        
        # 分层：基线异常 vs 基线正常
        # 白蛋白特殊：低于正常视为异常
        if name == '白蛋白':
            t_abnormal = treated[treated[base_col] < threshold - tolerance]
            c_abnormal = control[control[base_col] < threshold - tolerance]
            t_normal = treated[treated[base_col] >= threshold - tolerance]
            c_normal = control[control[base_col] >= threshold - tolerance]
        else:
            t_abnormal = treated[treated[base_col] > threshold + tolerance]
            c_abnormal = control[control[base_col] > threshold + tolerance]
            t_normal = treated[treated[base_col] <= threshold + tolerance]
            c_normal = control[control[base_col] <= threshold + tolerance]
        
        # 计算变化值
        t_change_abn = (t_abnormal[end_col] - t_abnormal[base_col]).dropna()
        c_change_abn = (c_abnormal[end_col] - c_abnormal[base_col]).dropna()
        t_change_nor = (t_normal[end_col] - t_normal[base_col]).dropna()
        c_change_nor = (c_normal[end_col] - c_normal[base_col]).dropna()
        
        # 准备数据 - 只包含有效数据
        all_data = [
            (t_change_abn, '基线异常\n治疗组', TREATMENT_COLOR),
            (c_change_abn, '基线异常\n对照组', CONTROL_COLOR),
            (t_change_nor, '基线正常\n治疗组', '#FF6B6B'),
            (c_change_nor, '基线正常\n对照组', '#6B9FFF'),
        ]
        
        # 过滤有效数据
        valid_data = [(d, l, c) for d, l, c in all_data if len(d) >= 2]
        
        if len(valid_data) < 2:
            ax.text(0.5, 0.5, '数据不足', ha='center', va='center', fontsize=14)
            ax.set_title(f'{name}', fontsize=15, fontweight='bold')
            continue
        
        means = [d.mean() for d, _, _ in valid_data]
        ses = [d.std() / np.sqrt(len(d)) for d, _, _ in valid_data]
        ns = [len(d) for d, _, _ in valid_data]
        labels = [l for _, l, _ in valid_data]
        colors = [c for _, _, c in valid_data]
        
        x = np.arange(len(valid_data))
        bars = ax.bar(x, means, yerr=ses, capsize=4, color=colors,
                     edgecolor='white', linewidth=1.2,
                     error_kw={'elinewidth': 1.5, 'capthick': 1.5, 'ecolor': '#333333'})
        
        # 添加样本量标注
        for i, (bar, n) in enumerate(zip(bars, ns)):
            ax.text(bar.get_x() + bar.get_width()/2, 0.02, f'n={n}', 
                   ha='center', va='bottom', fontsize=10, transform=ax.get_xaxis_transform())
        
        ax.axhline(y=0, color='#333333', linestyle='-', linewidth=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_title(f'{name}变化值分层比较', fontsize=15, fontweight='bold')
        ax.set_ylabel('变化值', fontsize=13, fontweight='bold')
        ax.tick_params(axis='y', labelsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # 添加组间P值（基线异常亚组）
        if len(t_change_abn) >= 2 and len(c_change_abn) >= 2:
            _, p_val = stats.mannwhitneyu(t_change_abn, c_change_abn, alternative='two-sided')
            sig = '*' if p_val < 0.05 else ''
            ax.text(0.5, 0.97, f'基线异常组间P={p_val:.3f}{sig}', 
                   transform=ax.transAxes, ha='center', fontsize=11,
                   fontweight='bold' if p_val < 0.05 else 'normal',
                   color=TREATMENT_COLOR if p_val < 0.05 else '#333333')
    
    plt.suptitle('分层变化值柱状图比较（消除天花板效应）\n深色=基线异常，浅色=基线正常', 
                fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_lancet_figure(fig, f'{result_dir}/barplot_stratified_change.png')
    plt.close()
    
    print("分层变化值柱状图已保存: barplot_stratified_change.png")


# ==================== 6. 肝硬度值分析 ====================
def liver_stiffness_analysis(df, treatment_col='treatment', result_dir=None):
    """
    分析肝硬度值治疗前后的组内和组间差异
    
    分层分析策略：
    1. 全人群分析
    2. 基线异常患者（肝硬度>7.0 kPa）分析 - 消除天花板效应
    3. 基线正常患者（肝硬度≤7.0 kPa）分析 - 评估维持稳定能力
    """
    if result_dir is None:
        result_dir = RESULT_DIR
    print("\n" + "="*60)
    print("6. 肝硬度值分析（含分层）")
    print("="*60)
    print("分层策略: 全人群 / 基线异常(>7.0kPa) / 基线正常(≤7.0kPa)")
    
    base_col = '肝硬度值基线'
    end_col = '肝硬度值12个月'
    upper_limit = 7.0  # 肝硬度正常上限
    tolerance = 0  # 肝硬度容忍度
    
    if base_col not in df.columns or end_col not in df.columns:
        print("警告: 未找到肝硬度值列")
        return None
    
    treated = df[df[treatment_col] == 1]
    control = df[df[treatment_col] == 0]
    
    t_valid = treated[[base_col, end_col]].dropna()
    c_valid = control[[base_col, end_col]].dropna()
    
    all_results = []
    stratified_results = []
    
    # === 全人群分析 ===
    print("\n--- 全人群分析 ---")
    
    # 组内比较（配对t检验）
    t_stat_t, p_t = stats.ttest_rel(t_valid[end_col], t_valid[base_col])
    _, wilcox_p_t = stats.wilcoxon(t_valid[end_col], t_valid[base_col])
    t_change = t_valid[end_col] - t_valid[base_col]
    
    print(f"治疗组: 基线={t_valid[base_col].mean():.2f}±{t_valid[base_col].std():.2f}, "
          f"12月={t_valid[end_col].mean():.2f}±{t_valid[end_col].std():.2f}")
    print(f"  变化: {t_change.mean():.2f}±{t_change.std():.2f}, P={p_t:.4f} (配对t)")
    
    c_stat_t, p_c = stats.ttest_rel(c_valid[end_col], c_valid[base_col])
    _, wilcox_p_c = stats.wilcoxon(c_valid[end_col], c_valid[base_col])
    c_change = c_valid[end_col] - c_valid[base_col]
    
    print(f"对照组: 基线={c_valid[base_col].mean():.2f}±{c_valid[base_col].std():.2f}, "
          f"12月={c_valid[end_col].mean():.2f}±{c_valid[end_col].std():.2f}")
    print(f"  变化: {c_change.mean():.2f}±{c_change.std():.2f}, P={p_c:.4f} (配对t)")
    
    # 组间比较
    t_stat_between, p_between = stats.ttest_ind(t_change, c_change, equal_var=False)
    _, mw_p_between = stats.mannwhitneyu(t_change, c_change, alternative='two-sided')
    
    print(f"组间差异: {t_change.mean() - c_change.mean():.2f}, P={p_between:.4f} (t), P={mw_p_between:.4f} (MW)")
    
    # 保存全人群结果
    all_results.extend([{
        '分析类型': '治疗组组内',
        '分层': '全人群',
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
        '分层': '全人群',
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
        '分层': '全人群',
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
    
    stratified_results.append({
        '分层': '全人群',
        '治疗组N': len(t_valid),
        '对照组N': len(c_valid),
        '治疗组变化': t_change.mean(),
        '治疗组SD': t_change.std(),
        '对照组变化': c_change.mean(),
        '对照组SD': c_change.std(),
        '组间差异': t_change.mean() - c_change.mean(),
        't检验P值': p_between,
        'Mann-Whitney P值': mw_p_between,
        '显著': '是' if mw_p_between < 0.05 else '否'
    })
    
    # === 基线异常患者分析 ===
    print("\n--- 基线异常患者(>7.0kPa)分析 ---")
    
    t_abnormal = t_valid[t_valid[base_col] > upper_limit + tolerance]
    c_abnormal = c_valid[c_valid[base_col] > upper_limit + tolerance]
    
    if len(t_abnormal) >= 2 and len(c_abnormal) >= 2:
        t_change_abnormal = t_abnormal[end_col] - t_abnormal[base_col]
        c_change_abnormal = c_abnormal[end_col] - c_abnormal[base_col]
        
        _, p_abnormal = stats.ttest_ind(t_change_abnormal, c_change_abnormal, equal_var=False)
        _, mw_p_abnormal = stats.mannwhitneyu(t_change_abnormal, c_change_abnormal, alternative='two-sided')
        
        print(f"治疗组(N={len(t_abnormal)}): 变化={t_change_abnormal.mean():.2f}±{t_change_abnormal.std():.2f}")
        print(f"对照组(N={len(c_abnormal)}): 变化={c_change_abnormal.mean():.2f}±{c_change_abnormal.std():.2f}")
        print(f"组间差异: {t_change_abnormal.mean() - c_change_abnormal.mean():.2f}, P={p_abnormal:.4f} (t), P={mw_p_abnormal:.4f} (MW)")
        
        stratified_results.append({
            '分层': '基线异常(>7.0kPa)',
            '治疗组N': len(t_abnormal),
            '对照组N': len(c_abnormal),
            '治疗组变化': t_change_abnormal.mean(),
            '治疗组SD': t_change_abnormal.std(),
            '对照组变化': c_change_abnormal.mean(),
            '对照组SD': c_change_abnormal.std(),
            '组间差异': t_change_abnormal.mean() - c_change_abnormal.mean(),
            't检验P值': p_abnormal,
            'Mann-Whitney P值': mw_p_abnormal,
            '显著': '是' if mw_p_abnormal < 0.05 else '否'
        })
    else:
        print(f"基线异常患者数量不足(治疗组{len(t_abnormal)}, 对照组{len(c_abnormal)})")
    
    # === 基线正常患者分析 ===
    print("\n--- 基线正常患者(≤7.0kPa)分析 ---")
    
    t_normal = t_valid[t_valid[base_col] <= upper_limit + tolerance]
    c_normal = c_valid[c_valid[base_col] <= upper_limit + tolerance]
    
    if len(t_normal) >= 2 and len(c_normal) >= 2:
        t_change_normal = t_normal[end_col] - t_normal[base_col]
        c_change_normal = c_normal[end_col] - c_normal[base_col]
        
        _, p_normal = stats.ttest_ind(t_change_normal, c_change_normal, equal_var=False)
        _, mw_p_normal = stats.mannwhitneyu(t_change_normal, c_change_normal, alternative='two-sided')
        
        print(f"治疗组(N={len(t_normal)}): 变化={t_change_normal.mean():.2f}±{t_change_normal.std():.2f}")
        print(f"对照组(N={len(c_normal)}): 变化={c_change_normal.mean():.2f}±{c_change_normal.std():.2f}")
        print(f"组间差异: {t_change_normal.mean() - c_change_normal.mean():.2f}, P={p_normal:.4f} (t), P={mw_p_normal:.4f} (MW)")
        
        stratified_results.append({
            '分层': '基线正常(≤7.0kPa)',
            '治疗组N': len(t_normal),
            '对照组N': len(c_normal),
            '治疗组变化': t_change_normal.mean(),
            '治疗组SD': t_change_normal.std(),
            '对照组变化': c_change_normal.mean(),
            '对照组SD': c_change_normal.std(),
            '组间差异': t_change_normal.mean() - c_change_normal.mean(),
            't检验P值': p_normal,
            'Mann-Whitney P值': mw_p_normal,
            '显著': '是' if mw_p_normal < 0.05 else '否'
        })
    else:
        print(f"基线正常患者数量不足(治疗组{len(t_normal)}, 对照组{len(c_normal)})")
    
    # 保存结果
    results_df = pd.DataFrame(all_results)
    stratified_df = pd.DataFrame(stratified_results)
    
    results_df.to_csv(f'{result_dir}/liver_stiffness_analysis.csv', index=False, encoding='utf-8-sig')
    stratified_df.to_csv(f'{result_dir}/liver_stiffness_stratified.csv', index=False, encoding='utf-8-sig')
    
    # 打印分层汇总
    print("\n=== 肝硬度值分层分析汇总 ===")
    for _, row in stratified_df.iterrows():
        sig = " *" if row['显著'] == '是' else ""
        print(f"  {row['分层']}: 治疗组{row['治疗组变化']:.2f} vs 对照组{row['对照组变化']:.2f}, "
              f"差异={row['组间差异']:.2f}, P={row['Mann-Whitney P值']:.4f}{sig}")
    
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
            
            # 2. 治疗后肝功能差异（含分层分析）
            results['liver_function'], results['liver_function_stratified'] = liver_function_difference_analysis(df, result_dir=source_dir)
            
            # 2.5 正常范围内波动分析（区分正常生理波动与真正治疗效果）
            results['normal_fluctuation'] = analyze_normal_range_fluctuation(df, result_dir=source_dir)
            
            # 3. 非劣效性检验（含分层）
            results['non_inferiority'] = non_inferiority_test(df, result_dir=source_dir)
            
            # 4. 箱式图比较
            boxplot_comparison(df, result_dir=source_dir)
            
            # 5. 柱状图比较
            barplot_comparison(df, result_dir=source_dir)
            
            # 6. 肝硬度值分析（含分层）
            results['liver_stiffness'] = liver_stiffness_analysis(df, result_dir=source_dir)
            
            # 7. 治愈速度分析
            results['cure_speed'], results['cure_comparison'] = cure_speed_analysis(df, result_dir=source_dir)
            
            # 8. 临床分层分析（消除天花板/地板效应）
            results['stratified'] = stratified_analysis(df, result_dir=source_dir)
            
            # 9. 相关性分析（Pearson/Spearman）
            results['correlation'] = correlation_analysis(df, result_dir=source_dir)
            
            # 10. 多因素回归分析（控制混杂因素）
            results['regression'] = multivariate_regression_analysis(df, result_dir=source_dir)
            
        except Exception as e:
            print(f"警告: {source_name}分析部分失败: {e}")
            import traceback
            traceback.print_exc()
        
        all_results[source_key] = results
    
    print("\n" + "="*70)
    print("综合分析完成！（共10项分析）")
    print("="*70)
    print(f"\n所有结果已保存至: {base_result_dir}/")
    print("\n结果目录结构:")
    print("  - Raw/    (原始数据分析)")
    print("  - PSM/    (PSM匹配后分析)")
    print("  - CEM/    (CEM匹配后分析)")
    print("\n分析项目:")
    print("  1. 基线可比性检验     6. 肝硬度值分析")
    print("  2. 肝功能差异分析     7. 治愈速度分析")
    print("  3. 非劣效性检验       8. 临床分层分析")
    print("  4. 箱式图比较         9. 相关性分析")
    print("  5. 柱状图比较        10. 多因素回归")
    
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
    
    # 汇总分层分析结果
    stratified_summary = []
    for source_key, (source_name, _) in data_sources.items():
        if source_key in all_results and 'stratified' in all_results[source_key]:
            strat_df = all_results[source_key]['stratified']
            if strat_df is not None and len(strat_df) > 0:
                # 只提取关键分层结果
                key_strata = strat_df[strat_df['分层'].isin(['全人群', '基线异常偏高', '基线正常'])]
                for _, row in key_strata.iterrows():
                    stratified_summary.append({
                        '数据来源': source_name,
                        '指标': row.get('指标', ''),
                        '分层': row.get('分层', ''),
                        'N': row.get('N', 0),
                        '治疗组变化': row.get('治疗组变化', np.nan),
                        '对照组变化': row.get('对照组变化', np.nan),
                        '组间差异': row.get('组间差异', np.nan),
                        'P值': row.get('P值', np.nan),
                        '显著': row.get('显著', '')
                    })
    
    if stratified_summary:
        stratified_summary_df = pd.DataFrame(stratified_summary)
        stratified_summary_df.to_csv(f'{result_dir}/comparison_stratified.csv', 
                                      index=False, encoding='utf-8-sig')
    
    print("对比汇总已保存")


if __name__ == '__main__':
    run_comprehensive_analysis()
