"""
多变量分析模块
考虑肝功能指标间的相关性，进行综合分析

临床背景：
- 各个表征指标间非独立（如ALT/AST/GGT/ALP等肝酶指标相关性高）
- 需要使用多变量方法进行综合评估
"""
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from data_preprocessing import (
    load_and_preprocess_data, ensure_result_dir,
    COVARIATES, OUTCOMES_PAIRS, PRIMARY_OUTCOMES_PAIRS, SECONDARY_OUTCOMES_PAIRS,
    NORMAL_RANGES
)

def calculate_correlation_matrix(df, outcome_pairs):
    """
    计算结局变量变化值之间的相关性矩阵
    
    Parameters:
    -----------
    df : DataFrame
    outcome_pairs : list of tuples
    
    Returns:
    --------
    corr_matrix : DataFrame
        相关性矩阵
    """
    changes = {}
    
    for base_col, end_col in outcome_pairs:
        if base_col in df.columns and end_col in df.columns:
            indicator = base_col.replace('基线', '').strip()
            change = df[end_col] - df[base_col]
            changes[indicator] = change
    
    change_df = pd.DataFrame(changes)
    corr_matrix = change_df.corr(method='spearman')  # 使用Spearman相关，对非正态分布更稳健
    
    return corr_matrix

def multivariate_treatment_effect(df, treatment_col='treatment'):
    """
    多变量治疗效果分析（使用Hotelling's T²检验）
    考虑多个结局变量之间的相关性
    
    Parameters:
    -----------
    df : DataFrame
    treatment_col : str
    
    Returns:
    --------
    dict: 多变量分析结果
    """
    # 提取主要结局变量的变化
    changes = []
    indicator_names = []
    
    for base_col, end_col in PRIMARY_OUTCOMES_PAIRS:
        if base_col in df.columns and end_col in df.columns:
            indicator = base_col.replace('基线', '').strip()
            indicator_names.append(indicator)
            changes.append(df[end_col] - df[base_col])
    
    if len(changes) < 2:
        return None
    
    # 构建变化矩阵
    change_matrix = pd.concat(changes, axis=1)
    change_matrix.columns = indicator_names
    
    # 分组
    treated = change_matrix[df[treatment_col] == 1].dropna()
    control = change_matrix[df[treatment_col] == 0].dropna()
    
    n1, p = treated.shape
    n2 = len(control)
    
    if n1 < p + 2 or n2 < p + 2:
        return {'error': '样本量不足以进行多变量分析'}
    
    # 计算均值向量
    mean1 = treated.mean().values
    mean2 = control.mean().values
    diff = mean1 - mean2
    
    # 计算池化协方差矩阵
    cov1 = treated.cov().values
    cov2 = control.cov().values
    pooled_cov = ((n1 - 1) * cov1 + (n2 - 1) * cov2) / (n1 + n2 - 2)
    
    # Hotelling's T²
    try:
        pooled_cov_inv = np.linalg.inv(pooled_cov)
        t2 = (n1 * n2) / (n1 + n2) * diff @ pooled_cov_inv @ diff
        
        # 转换为F统计量
        f_stat = t2 * (n1 + n2 - p - 1) / (p * (n1 + n2 - 2))
        df1 = p
        df2 = n1 + n2 - p - 1
        p_value = 1 - stats.f.cdf(f_stat, df1, df2)
        
        return {
            '分析类型': '多变量Hotelling T²检验',
            '变量': indicator_names,
            '治疗组样本量': n1,
            '对照组样本量': n2,
            '治疗组变化均值': dict(zip(indicator_names, mean1)),
            '对照组变化均值': dict(zip(indicator_names, mean2)),
            '组间差异': dict(zip(indicator_names, diff)),
            'T²统计量': t2,
            'F统计量': f_stat,
            '自由度': (df1, df2),
            'P值': p_value,
            '结论': '治疗组显著优于对照组' if p_value < 0.05 and all(d < 0 for d in diff) else 
                   '治疗组显著差于对照组' if p_value < 0.05 and all(d > 0 for d in diff) else
                   '组间无显著差异' if p_value >= 0.05 else '组间存在显著差异'
        }
    except np.linalg.LinAlgError:
        return {'error': '协方差矩阵奇异，无法进行多变量分析'}

def liver_enzyme_composite_score(df, treatment_col='treatment'):
    """
    构建肝酶综合评分（基于临床正常范围标准化）
    
    将ALT和AST标准化后取平均，作为综合肝酶指标
    """
    results = []
    
    # ALT和AST的正常上限
    alt_upper = NORMAL_RANGES.get('ALT', (7, 40, 'U/L'))[1]
    ast_upper = NORMAL_RANGES.get('AST', (13, 35, 'U/L'))[1]
    
    for group in [1, 0]:
        group_data = df[df[treatment_col] == group].copy()
        group_name = '治疗组' if group == 1 else '对照组'
        
        # 计算基线和终点的标准化值（相对于正常上限）
        if '基线ALT' in df.columns and 'ALT12个月' in df.columns:
            alt_baseline_ratio = group_data['基线ALT'] / alt_upper
            alt_endpoint_ratio = group_data['ALT12个月'] / alt_upper
        else:
            alt_baseline_ratio = pd.Series([np.nan])
            alt_endpoint_ratio = pd.Series([np.nan])
        
        if '基线AST' in df.columns and 'AST12个月' in df.columns:
            ast_baseline_ratio = group_data['基线AST'] / ast_upper
            ast_endpoint_ratio = group_data['AST12个月'] / ast_upper
        else:
            ast_baseline_ratio = pd.Series([np.nan])
            ast_endpoint_ratio = pd.Series([np.nan])
        
        # 综合评分（ALT和AST相对于正常上限的平均倍数）
        baseline_score = (alt_baseline_ratio + ast_baseline_ratio) / 2
        endpoint_score = (alt_endpoint_ratio + ast_endpoint_ratio) / 2
        score_change = endpoint_score - baseline_score
        
        results.append({
            '分组': group_name,
            '例数': len(group_data),
            '基线综合评分': baseline_score.mean(),
            '终点综合评分': endpoint_score.mean(),
            '评分变化': score_change.mean(),
            '评分变化SD': score_change.std(),
            '基线ALT/ULN': alt_baseline_ratio.mean(),
            '终点ALT/ULN': alt_endpoint_ratio.mean(),
            '基线AST/ULN': ast_baseline_ratio.mean(),
            '终点AST/ULN': ast_endpoint_ratio.mean(),
        })
    
    results_df = pd.DataFrame(results)
    
    # 组间比较
    treated_change = (df[df[treatment_col] == 1]['ALT12个月'] / alt_upper + 
                      df[df[treatment_col] == 1]['AST12个月'] / ast_upper) / 2 - \
                     (df[df[treatment_col] == 1]['基线ALT'] / alt_upper + 
                      df[df[treatment_col] == 1]['基线AST'] / ast_upper) / 2
    
    control_change = (df[df[treatment_col] == 0]['ALT12个月'] / alt_upper + 
                      df[df[treatment_col] == 0]['AST12个月'] / ast_upper) / 2 - \
                     (df[df[treatment_col] == 0]['基线ALT'] / alt_upper + 
                      df[df[treatment_col] == 0]['基线AST'] / ast_upper) / 2
    
    treated_change = treated_change.dropna()
    control_change = control_change.dropna()
    
    if len(treated_change) > 1 and len(control_change) > 1:
        _, p_val = stats.ttest_ind(treated_change, control_change, equal_var=False)
        _, mw_p_val = stats.mannwhitneyu(treated_change, control_change, alternative='two-sided')
    else:
        p_val = np.nan
        mw_p_val = np.nan
    
    return {
        'summary': results_df,
        't_test_p': p_val,
        'mannwhitney_p': mw_p_val,
        'interpretation': '肝酶综合评分反映ALT和AST相对于正常上限(ULN)的平均倍数，评分下降表示改善'
    }

def bonferroni_correction(p_values, alpha=0.05):
    """
    Bonferroni校正多重比较
    """
    n = len(p_values)
    adjusted_alpha = alpha / n
    
    results = []
    for name, p in p_values.items():
        results.append({
            '指标': name,
            '原始P值': p,
            '校正后显著性水平': adjusted_alpha,
            '是否显著': '是' if p < adjusted_alpha else '否'
        })
    
    return pd.DataFrame(results)

def fdr_correction(p_values, alpha=0.05):
    """
    Benjamini-Hochberg FDR校正
    """
    names = list(p_values.keys())
    pvals = np.array([p_values[n] for n in names])
    
    n = len(pvals)
    sorted_idx = np.argsort(pvals)
    sorted_pvals = pvals[sorted_idx]
    sorted_names = [names[i] for i in sorted_idx]
    
    # 计算调整后的p值
    adjusted_pvals = np.zeros(n)
    for i in range(n):
        adjusted_pvals[i] = sorted_pvals[i] * n / (i + 1)
    
    # 确保单调性
    for i in range(n - 2, -1, -1):
        adjusted_pvals[i] = min(adjusted_pvals[i], adjusted_pvals[i + 1])
    adjusted_pvals = np.minimum(adjusted_pvals, 1)
    
    results = []
    for i in range(n):
        results.append({
            '指标': sorted_names[i],
            '原始P值': sorted_pvals[i],
            '调整后P值(FDR)': adjusted_pvals[i],
            '是否显著(FDR<0.05)': '是' if adjusted_pvals[i] < alpha else '否'
        })
    
    return pd.DataFrame(results)

def run_multivariate_analysis(df=None):
    """
    运行完整的多变量分析
    """
    print("=" * 60)
    print("多变量分析（考虑指标间相关性）")
    print("=" * 60)
    
    if df is None:
        df = load_and_preprocess_data(fill_missing=True)
    
    result_dir = ensure_result_dir('Multivariate')
    
    # 1. 相关性分析
    print("\n1. 计算结局变量间的相关性矩阵...")
    corr_matrix = calculate_correlation_matrix(df, OUTCOMES_PAIRS)
    corr_matrix.to_csv(f'{result_dir}/correlation_matrix.csv', encoding='utf-8-sig')
    print("主要指标(ALT, AST)相关系数:", 
          corr_matrix.loc['ALT', 'AST'] if 'ALT' in corr_matrix.index and 'AST' in corr_matrix.columns else 'N/A')
    
    # 2. 多变量检验
    print("\n2. 多变量Hotelling T²检验...")
    mv_result = multivariate_treatment_effect(df)
    if mv_result and 'error' not in mv_result:
        print(f"  T²统计量: {mv_result['T²统计量']:.4f}")
        print(f"  P值: {mv_result['P值']:.4f}")
        print(f"  结论: {mv_result['结论']}")
        pd.DataFrame([mv_result]).to_csv(f'{result_dir}/hotelling_test.csv', encoding='utf-8-sig', index=False)
    else:
        print(f"  {mv_result.get('error', '分析失败')}")
    
    # 3. 综合评分分析
    print("\n3. 肝酶综合评分分析...")
    composite_result = liver_enzyme_composite_score(df)
    print(composite_result['summary'].to_string(index=False))
    print(f"  组间比较P值(t检验): {composite_result['t_test_p']:.4f}")
    print(f"  组间比较P值(Mann-Whitney): {composite_result['mannwhitney_p']:.4f}")
    composite_result['summary'].to_csv(f'{result_dir}/composite_score.csv', encoding='utf-8-sig', index=False)
    
    # 4. 多重比较校正
    print("\n4. 多重比较校正...")
    # 收集所有结局变量的p值
    from data_preprocessing import analyze_outcomes
    outcome_df = analyze_outcomes(df)
    
    if len(outcome_df) > 0:
        p_values = {}
        for _, row in outcome_df.iterrows():
            if pd.notna(row['组间变化差异P值']):
                p_values[row['指标']] = row['组间变化差异P值']
        
        if p_values:
            bonf_df = bonferroni_correction(p_values)
            fdr_df = fdr_correction(p_values)
            
            print("\nBonferroni校正结果:")
            print(bonf_df.to_string(index=False))
            
            print("\nFDR校正结果:")
            print(fdr_df.to_string(index=False))
            
            bonf_df.to_csv(f'{result_dir}/bonferroni_correction.csv', encoding='utf-8-sig', index=False)
            fdr_df.to_csv(f'{result_dir}/fdr_correction.csv', encoding='utf-8-sig', index=False)
    
    print("\n" + "=" * 60)
    print("多变量分析完成！")
    print(f"结果已保存至: {result_dir}")
    print("=" * 60)
    
    return {
        'correlation_matrix': corr_matrix,
        'multivariate_test': mv_result,
        'composite_score': composite_result
    }

if __name__ == '__main__':
    run_multivariate_analysis()
