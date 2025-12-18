"""
双重差分法 (Difference-in-Differences, DID) 分析程序
比较两组患者在治疗前后的变化差异
"""
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import os
import warnings
warnings.filterwarnings('ignore')

# 导入共享模块
from data_preprocessing import (
    load_and_preprocess_data, ensure_result_dir, calculate_smd,
    COVARIATES, OUTCOMES_PAIRS
)

def reshape_to_long_format(df, outcome_pairs):
    """
    将数据从宽格式转换为长格式，用于DID分析
    
    Returns:
    --------
    df_long : DataFrame
        长格式数据，包含 id, treatment, time, outcome 等列
    """
    long_data = []
    
    for idx, row in df.iterrows():
        patient_id = row.get('病历号', idx)
        treatment = row['treatment']
        
        for base_col, end_col in outcome_pairs:
            if base_col in df.columns and end_col in df.columns:
                outcome_name = base_col.replace('基线', '').replace('基线', '')
                
                # 基线时间点
                if pd.notna(row[base_col]):
                    long_data.append({
                        'patient_id': patient_id,
                        'treatment': treatment,
                        'time': 0,  # 基线
                        'post': 0,
                        'outcome': outcome_name,
                        'value': row[base_col]
                    })
                
                # 终点时间点
                if pd.notna(row[end_col]):
                    long_data.append({
                        'patient_id': patient_id,
                        'treatment': treatment,
                        'time': 1,  # 12个月
                        'post': 1,
                        'outcome': outcome_name,
                        'value': row[end_col]
                    })
    
    return pd.DataFrame(long_data)

def run_did_analysis(df_long, outcome_name, covariates_df=None):
    """
    运行DID回归分析
    
    Y = β0 + β1*Treatment + β2*Post + β3*(Treatment×Post) + ε
    
    β3 是DID估计量，表示治疗组相对于对照组的额外变化
    
    Returns:
    --------
    dict with DID results
    """
    outcome_data = df_long[df_long['outcome'] == outcome_name].copy()
    
    if len(outcome_data) < 20:
        return None
    
    # 创建交互项
    outcome_data['treat_post'] = outcome_data['treatment'] * outcome_data['post']
    
    try:
        # 简单DID模型
        model = smf.ols('value ~ treatment + post + treat_post', data=outcome_data).fit()
        
        # 提取结果
        did_estimate = model.params.get('treat_post', np.nan)
        did_se = model.bse.get('treat_post', np.nan)
        did_pvalue = model.pvalues.get('treat_post', np.nan)
        ci_lower = model.conf_int().loc['treat_post', 0] if 'treat_post' in model.conf_int().index else np.nan
        ci_upper = model.conf_int().loc['treat_post', 1] if 'treat_post' in model.conf_int().index else np.nan
        
        # 计算各组均值
        treated_pre = outcome_data[(outcome_data['treatment']==1) & (outcome_data['post']==0)]['value'].mean()
        treated_post = outcome_data[(outcome_data['treatment']==1) & (outcome_data['post']==1)]['value'].mean()
        control_pre = outcome_data[(outcome_data['treatment']==0) & (outcome_data['post']==0)]['value'].mean()
        control_post = outcome_data[(outcome_data['treatment']==0) & (outcome_data['post']==1)]['value'].mean()
        
        return {
            '指标': outcome_name,
            '治疗组_基线': treated_pre,
            '治疗组_终点': treated_post,
            '治疗组_变化': treated_post - treated_pre,
            '对照组_基线': control_pre,
            '对照组_终点': control_post,
            '对照组_变化': control_post - control_pre,
            'DID估计量': did_estimate,
            'DID标准误': did_se,
            'DID_P值': did_pvalue,
            'DID_95%CI下限': ci_lower,
            'DID_95%CI上限': ci_upper,
            'R²': model.rsquared,
            '样本量': len(outcome_data)
        }
    except Exception as e:
        print(f"  {outcome_name}: 分析失败 - {e}")
        return None

def parallel_trends_test(df_long, outcome_name):
    """
    平行趋势检验（简化版）
    由于本数据只有两个时间点，无法进行完整的平行趋势检验
    这里只检验基线时两组是否可比
    """
    baseline_data = df_long[(df_long['outcome'] == outcome_name) & (df_long['post'] == 0)]
    
    if len(baseline_data) < 10:
        return None
    
    treated_baseline = baseline_data[baseline_data['treatment'] == 1]['value']
    control_baseline = baseline_data[baseline_data['treatment'] == 0]['value']
    
    try:
        stat, pvalue = stats.ttest_ind(treated_baseline, control_baseline, equal_var=False)
        return {
            '指标': outcome_name,
            '治疗组基线均值': treated_baseline.mean(),
            '对照组基线均值': control_baseline.mean(),
            '差异': treated_baseline.mean() - control_baseline.mean(),
            'T统计量': stat,
            'P值': pvalue,
            '基线可比': '是' if pvalue > 0.05 else '否'
        }
    except:
        return None

def main():
    print("=" * 60)
    print("双重差分法 (Difference-in-Differences, DID) 分析")
    print("=" * 60)
    
    # 加载数据
    df = load_and_preprocess_data(fill_missing=True)
    result_dir = ensure_result_dir('DID')
    
    print(f"样本量: {len(df)}")
    print(f"治疗组: {(df['treatment']==1).sum()}, 对照组: {(df['treatment']==0).sum()}")
    
    # 转换为长格式
    print("\n正在转换数据格式...")
    df_long = reshape_to_long_format(df, OUTCOMES_PAIRS)
    df_long.to_csv(os.path.join(result_dir, 'long_format_data.csv'), index=False, encoding='utf-8-sig')
    
    print(f"长格式数据行数: {len(df_long)}")
    
    # 获取所有结果变量
    outcomes = df_long['outcome'].unique()
    print(f"分析的结果变量: {outcomes}")
    
    # 运行DID分析
    print("\n正在运行DID分析...")
    did_results = []
    
    for outcome in outcomes:
        result = run_did_analysis(df_long, outcome)
        if result:
            did_results.append(result)
            print(f"  {outcome}: DID={result['DID估计量']:.4f}, P={result['DID_P值']:.4f}")
    
    did_df = pd.DataFrame(did_results)
    did_df.to_csv(os.path.join(result_dir, 'did_analysis.csv'), index=False, encoding='utf-8-sig')
    
    # 平行趋势检验（基线可比性）
    print("\n正在进行基线可比性检验...")
    parallel_results = []
    
    for outcome in outcomes:
        result = parallel_trends_test(df_long, outcome)
        if result:
            parallel_results.append(result)
    
    parallel_df = pd.DataFrame(parallel_results)
    parallel_df.to_csv(os.path.join(result_dir, 'baseline_comparability.csv'), index=False, encoding='utf-8-sig')
    
    print("\n基线可比性检验结果:")
    print(parallel_df[['指标', '治疗组基线均值', '对照组基线均值', 'P值', '基线可比']].to_string(index=False))
    
    # 保存原始数据（用于可视化）
    df.to_csv(os.path.join(result_dir, 'original_data.csv'), index=False, encoding='utf-8-sig')
    
    # 汇总统计
    if len(did_results) > 0:
        significant_count = sum(1 for r in did_results if r['DID_P值'] < 0.05)
        
        summary = {
            '方法': '双重差分法 (DID)',
            '样本量': len(df),
            '治疗组人数': (df['treatment']==1).sum(),
            '对照组人数': (df['treatment']==0).sum(),
            '分析指标数': len(outcomes),
            'DID显著指标数(P<0.05)': significant_count,
            '基线可比指标数(P>0.05)': sum(1 for r in parallel_results if r['P值'] > 0.05)
        }
        
        summary_df = pd.DataFrame([summary])
        summary_df.to_csv(os.path.join(result_dir, 'summary.csv'), index=False, encoding='utf-8-sig')
    
    print("\n" + "=" * 60)
    print("DID分析完成！")
    print(f"结果已保存至: {result_dir}")
    print("=" * 60)
    
    # 打印主要结果
    if len(did_results) > 0:
        print("\n主要DID分析结果:")
        print("=" * 80)
        print(f"{'指标':<15} {'治疗组变化':>12} {'对照组变化':>12} {'DID估计量':>12} {'P值':>10}")
        print("-" * 80)
        for r in did_results:
            sig = '*' if r['DID_P值'] < 0.05 else ''
            print(f"{r['指标']:<15} {r['治疗组_变化']:>12.2f} {r['对照组_变化']:>12.2f} {r['DID估计量']:>12.2f} {r['DID_P值']:>9.4f}{sig}")
        print("=" * 80)
        print("注: * 表示 P < 0.05")

if __name__ == '__main__':
    main()
