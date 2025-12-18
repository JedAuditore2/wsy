"""
粗化精确匹配 (Coarsened Exact Matching, CEM) 分析程序
将连续变量粗化后进行精确匹配
"""
import pandas as pd
import numpy as np
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

# 导入共享模块
from data_preprocessing import (
    load_and_preprocess_data, ensure_result_dir, calculate_smd,
    analyze_outcomes, COVARIATES, OUTCOMES_PAIRS
)

def coarsen_variable(series, n_bins=5, method='quantile'):
    """
    粗化连续变量
    
    Parameters:
    -----------
    series : pd.Series
        要粗化的变量
    n_bins : int
        分箱数量
    method : str
        'quantile' 或 'uniform'
    
    Returns:
    --------
    coarsened : pd.Series
        粗化后的变量
    """
    if series.nunique() <= n_bins:
        # 已经是离散变量，直接返回
        return series.astype(str)
    
    try:
        if method == 'quantile':
            coarsened = pd.qcut(series, q=n_bins, labels=False, duplicates='drop')
        else:
            coarsened = pd.cut(series, bins=n_bins, labels=False)
    except ValueError:
        # 如果分箱失败，使用较少的箱数
        coarsened = pd.qcut(series, q=3, labels=False, duplicates='drop')
    
    return coarsened.astype(str)

def cem_matching(df, covariates, treatment_col='treatment', n_bins=5, use_subset=True):
    """
    执行粗化精确匹配
    
    Parameters:
    -----------
    df : DataFrame
    covariates : list
    treatment_col : str
    n_bins : int
    use_subset : bool
        如果全变量匹配效果不好，是否使用子集变量
    
    Returns:
    --------
    df_matched : DataFrame
        匹配后的数据
    strata_info : DataFrame
        层信息
    """
    df = df.copy()
    
    # 如果协变量太多，只使用最重要的几个（减少维度灾难）
    if use_subset and len(covariates) > 5:
        # 选择关键协变量进行匹配
        key_covariates = ['性别', '年龄', '是否合并脂肪肝', '基线ALT', '基线AST']
        key_covariates = [c for c in key_covariates if c in covariates]
        print(f"使用关键协变量进行匹配: {key_covariates}")
    else:
        key_covariates = covariates
    
    # 粗化每个协变量
    coarsened_cols = []
    for cov in key_covariates:
        coarsened_col = f'{cov}_coarsened'
        df[coarsened_col] = coarsen_variable(df[cov], n_bins=n_bins)
        coarsened_cols.append(coarsened_col)
    
    # 创建层标识
    df['stratum'] = df[coarsened_cols].apply(lambda x: '_'.join(x.values), axis=1)
    
    # 找到同时包含治疗组和对照组的层
    stratum_counts = df.groupby(['stratum', treatment_col]).size().unstack(fill_value=0)
    
    # 只保留两组都有样本的层
    valid_strata = stratum_counts[(stratum_counts[0] > 0) & (stratum_counts[1] > 0)].index
    
    print(f"总层数: {len(df['stratum'].unique())}")
    print(f"有效匹配层数: {len(valid_strata)}")
    
    # 筛选匹配的样本
    df_matched = df[df['stratum'].isin(valid_strata)].copy()
    
    # 计算每层的权重（可选：使用CEM权重）
    # 权重 = 层内对照组/治疗组比例的调整
    weights = []
    for stratum in df_matched['stratum'].unique():
        stratum_data = df_matched[df_matched['stratum'] == stratum]
        n_treated = (stratum_data[treatment_col] == 1).sum()
        n_control = (stratum_data[treatment_col] == 0).sum()
        
        for idx, row in stratum_data.iterrows():
            if row[treatment_col] == 1:
                weights.append(1.0)
            else:
                # 对照组权重调整
                weights.append(n_treated / n_control if n_control > 0 else 1.0)
    
    df_matched['cem_weight'] = weights
    
    # 层信息统计
    strata_info = []
    for stratum in valid_strata:
        stratum_data = df_matched[df_matched['stratum'] == stratum]
        strata_info.append({
            '层标识': stratum[:50] + '...' if len(stratum) > 50 else stratum,
            '治疗组人数': (stratum_data[treatment_col] == 1).sum(),
            '对照组人数': (stratum_data[treatment_col] == 0).sum()
        })
    
    strata_df = pd.DataFrame(strata_info)
    
    # 删除临时列
    cols_to_drop = coarsened_cols + ['stratum']
    df_matched = df_matched.drop(columns=cols_to_drop, errors='ignore')
    
    return df_matched, strata_df

def main():
    print("=" * 60)
    print("粗化精确匹配 (Coarsened Exact Matching, CEM) 分析")
    print("=" * 60)
    
    # 加载数据
    df = load_and_preprocess_data(fill_missing=True)
    result_dir = ensure_result_dir('CEM')
    
    print(f"原始样本量: {len(df)}")
    print(f"治疗组: {(df['treatment']==1).sum()}, 对照组: {(df['treatment']==0).sum()}")
    
    # 执行CEM匹配
    print("\n正在执行粗化精确匹配...")
    
    # 可以调整分箱数量
    n_bins = 4  # 较少的箱数可以提高匹配率
    
    df_matched, strata_info = cem_matching(df, COVARIATES, n_bins=n_bins)
    
    n_treated_matched = (df_matched['treatment'] == 1).sum()
    n_control_matched = (df_matched['treatment'] == 0).sum()
    
    print(f"\n匹配后样本量: {len(df_matched)}")
    print(f"匹配后治疗组: {n_treated_matched}, 对照组: {n_control_matched}")
    
    # 保存结果
    df_matched.to_csv(os.path.join(result_dir, 'matched_data.csv'), index=False, encoding='utf-8-sig')
    strata_info.to_csv(os.path.join(result_dir, 'strata_info.csv'), index=False, encoding='utf-8-sig')
    
    # 平衡性检验
    print("\n正在进行平衡性检验...")
    smd_before = calculate_smd(df, COVARIATES)
    smd_after = calculate_smd(df_matched, COVARIATES)
    
    balance_df = pd.merge(
        smd_before[['协变量', 'SMD']], 
        smd_after[['协变量', 'SMD']], 
        on='协变量', 
        suffixes=('_before', '_after')
    )
    balance_df = balance_df.rename(columns={'SMD_before': '匹配前SMD', 'SMD_after': '匹配后SMD'})
    balance_df.to_csv(os.path.join(result_dir, 'balance_check.csv'), index=False, encoding='utf-8-sig')
    
    print("\n平衡性检验结果:")
    print(balance_df.to_string(index=False))
    
    # 结果分析
    print("\n正在进行结果分析...")
    outcome_df = analyze_outcomes(df_matched)
    outcome_df.to_csv(os.path.join(result_dir, 'outcome_analysis.csv'), index=False, encoding='utf-8-sig')
    
    # 保存汇总
    summary = {
        '方法': '粗化精确匹配 (CEM)',
        '原始样本量': len(df),
        '原始治疗组人数': (df['treatment']==1).sum(),
        '原始对照组人数': (df['treatment']==0).sum(),
        '匹配后样本量': len(df_matched),
        '匹配后治疗组人数': n_treated_matched,
        '匹配后对照组人数': n_control_matched,
        '有效匹配层数': len(strata_info),
        '分箱数量': n_bins,
        '协变量数量': len(COVARIATES),
        '平衡协变量数(|SMD|<0.1)': sum(abs(balance_df['匹配后SMD']) < 0.1)
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(os.path.join(result_dir, 'summary.csv'), index=False, encoding='utf-8-sig')
    
    print("\n" + "=" * 60)
    print("CEM分析完成！")
    print(f"结果已保存至: {result_dir}")
    print("=" * 60)

if __name__ == '__main__':
    main()
