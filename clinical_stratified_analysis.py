"""
临床分层分析模块
处理指标正常值范围对统计分析的影响

核心问题：
当基线指标在正常范围内时，即使没有显著变化也不能说明治疗无效（天花板效应/地板效应）
本模块提供分层分析方法，区分：
1. 基线异常者的改善效果
2. 基线正常者的维持效果

统计学原理：
- 天花板效应 (Ceiling Effect): 基线值接近上限时，改善空间有限
- 地板效应 (Floor Effect): 基线值接近下限时，改善空间有限
- 效应修饰 (Effect Modification): 基线状态可能影响治疗效果的大小
"""

import pandas as pd
import numpy as np
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

from data_preprocessing import (
    load_and_preprocess_data, OUTCOMES_PAIRS, COVARIATES,
    TREATMENT_COL, NORMAL_RANGES, ensure_result_dir,
    PRIMARY_OUTCOMES_PAIRS, SECONDARY_OUTCOMES_PAIRS
)

# 扩展的正常范围（包含更多指标）
EXTENDED_NORMAL_RANGES = {
    '白蛋白': {'lower': 40, 'upper': 55, 'unit': 'g/L', 'direction': 'higher_better'},
    'ALT': {'lower': 7, 'upper': 40, 'unit': 'U/L', 'direction': 'lower_better'},
    'AST': {'lower': 13, 'upper': 35, 'unit': 'U/L', 'direction': 'lower_better'},
    'ALP': {'lower': 35, 'upper': 100, 'unit': 'U/L', 'direction': 'lower_better'},
    'GGT': {'lower': 7, 'upper': 45, 'unit': 'U/L', 'direction': 'lower_better'},
    '胆红素': {'lower': 3.4, 'upper': 17.1, 'unit': 'μmol/L', 'direction': 'lower_better'},
    '肝硬度值': {'lower': 2.5, 'upper': 7.0, 'unit': 'kPa', 'direction': 'lower_better'},
    '血小板': {'lower': 125, 'upper': 350, 'unit': '×10^9/L', 'direction': 'higher_better'},
    'HBsAg': {'lower': 0, 'upper': 0.05, 'unit': 'IU/mL', 'direction': 'lower_better'},
}


def get_indicator_range(indicator_name):
    """获取指标的正常范围信息"""
    clean_name = indicator_name.replace('基线', '').replace('12个月', '').strip()
    
    for key, value in EXTENDED_NORMAL_RANGES.items():
        if key in clean_name:
            return value
    return None


def classify_baseline_status(value, indicator_name, tolerance=2):
    """
    分类基线状态
    
    Parameters:
    -----------
    value : float
        基线值
    indicator_name : str
        指标名称
    tolerance : float
        容忍度（考虑检测误差和个体变异）
    
    Returns:
    --------
    str: '正常', '异常偏高', '异常偏低'
    """
    if pd.isna(value):
        return '缺失'
    
    range_info = get_indicator_range(indicator_name)
    if range_info is None:
        return '未知'
    
    lower = range_info['lower']
    upper = range_info['upper']
    
    # 考虑容忍度
    if value < lower - tolerance:
        return '异常偏低'
    elif value > upper + tolerance:
        return '异常偏高'
    else:
        return '正常'


def stratified_outcome_analysis(df, base_col, end_col, treatment_col='treatment'):
    """
    分层结局分析
    
    按基线状态（正常/异常）分层分析治疗效果
    
    Parameters:
    -----------
    df : DataFrame
    base_col : str
        基线列名
    end_col : str
        终点列名
    treatment_col : str
        治疗分组列
    
    Returns:
    --------
    dict: 分层分析结果
    """
    indicator = base_col.replace('基线', '').replace('肝硬度值基线', '肝硬度值').replace('血小板基线', '血小板').replace('HBsAg基线', 'HBsAg').strip()
    range_info = get_indicator_range(indicator)
    
    if range_info is None:
        return None
    
    upper = range_info['upper']
    lower = range_info['lower']
    direction = range_info['direction']
    tolerance = 2
    
    results = {
        'indicator': indicator,
        'normal_range': f"{lower}-{upper}",
        'direction': direction,
        'strata': {}
    }
    
    # 定义基线状态
    df_valid = df[[base_col, end_col, treatment_col]].dropna().copy()
    
    if len(df_valid) < 10:
        return None
    
    # 分类基线状态
    def classify(x):
        if x > upper + tolerance:
            return '异常偏高'
        elif x < lower - tolerance:
            return '异常偏低'
        else:
            return '正常'
    
    df_valid['baseline_status'] = df_valid[base_col].apply(classify)
    
    # 1. 全人群分析（作为参照）
    results['overall'] = analyze_subgroup(df_valid, base_col, end_col, treatment_col, '全人群')
    
    # 2. 按基线状态分层分析
    for status in ['正常', '异常偏高', '异常偏低']:
        subgroup = df_valid[df_valid['baseline_status'] == status]
        if len(subgroup) >= 5:  # 至少5人才分析
            results['strata'][status] = analyze_subgroup(subgroup, base_col, end_col, treatment_col, status)
    
    # 3. 计算复常率（仅针对异常者）
    results['normalization'] = calculate_normalization(df_valid, base_col, end_col, treatment_col, upper, tolerance)
    
    # 4. 计算维持率（仅针对正常者）
    results['maintenance'] = calculate_maintenance(df_valid, base_col, end_col, treatment_col, upper, lower, tolerance)
    
    # 5. 效应修饰检验（交互作用）
    results['effect_modification'] = test_effect_modification(df_valid, base_col, end_col, treatment_col)
    
    return results


def analyze_subgroup(df, base_col, end_col, treatment_col, subgroup_name):
    """分析亚组"""
    result = {
        'name': subgroup_name,
        'n_total': len(df),
        'treatment': {},
        'control': {},
        'comparison': {}
    }
    
    for group, name in [(1, 'treatment'), (0, 'control')]:
        grp = df[df[treatment_col] == group]
        if len(grp) < 2:
            result[name] = {'n': len(grp), 'valid': False}
            continue
        
        diff = grp[end_col] - grp[base_col]
        
        result[name] = {
            'n': len(grp),
            'valid': True,
            'baseline_mean': grp[base_col].mean(),
            'endpoint_mean': grp[end_col].mean(),
            'change_mean': diff.mean(),
            'change_std': diff.std(),
            'change_median': diff.median()
        }
        
        # 配对t检验
        if len(grp) >= 3:
            try:
                _, p_val = stats.ttest_rel(grp[end_col], grp[base_col])
                result[name]['paired_p'] = p_val
            except:
                result[name]['paired_p'] = np.nan
    
    # 组间比较
    t_grp = df[df[treatment_col] == 1]
    c_grp = df[df[treatment_col] == 0]
    
    if len(t_grp) >= 2 and len(c_grp) >= 2:
        t_diff = t_grp[end_col] - t_grp[base_col]
        c_diff = c_grp[end_col] - c_grp[base_col]
        
        try:
            _, p_val = stats.ttest_ind(t_diff, c_diff, equal_var=False)
            result['comparison']['p_value'] = p_val
            result['comparison']['effect_size'] = t_diff.mean() - c_diff.mean()
        except:
            result['comparison']['p_value'] = np.nan
            result['comparison']['effect_size'] = np.nan
    
    return result


def calculate_normalization(df, base_col, end_col, treatment_col, upper, tolerance):
    """计算复常率（基线异常→终点正常）"""
    results = {}
    
    for group, name in [(1, 'treatment'), (0, 'control')]:
        grp = df[df[treatment_col] == group]
        abnormal = grp[grp[base_col] > upper + tolerance]
        
        if len(abnormal) == 0:
            results[name] = {'n_abnormal': 0, 'n_normalized': 0, 'rate': np.nan}
            continue
        
        normalized = abnormal[abnormal[end_col] <= upper + tolerance]
        rate = len(normalized) / len(abnormal) * 100
        
        results[name] = {
            'n_abnormal': len(abnormal),
            'n_normalized': len(normalized),
            'rate': rate
        }
    
    # 比较两组复常率 (Fisher精确检验)
    if results.get('treatment', {}).get('n_abnormal', 0) > 0 and results.get('control', {}).get('n_abnormal', 0) > 0:
        try:
            table = [
                [results['treatment']['n_normalized'], results['treatment']['n_abnormal'] - results['treatment']['n_normalized']],
                [results['control']['n_normalized'], results['control']['n_abnormal'] - results['control']['n_normalized']]
            ]
            _, p_val = stats.fisher_exact(table)
            results['comparison_p'] = p_val
        except:
            results['comparison_p'] = np.nan
    else:
        results['comparison_p'] = np.nan
    
    return results


def calculate_maintenance(df, base_col, end_col, treatment_col, upper, lower, tolerance):
    """计算维持率（基线正常→终点仍正常）"""
    results = {}
    
    for group, name in [(1, 'treatment'), (0, 'control')]:
        grp = df[df[treatment_col] == group]
        # 基线正常
        normal = grp[(grp[base_col] >= lower - tolerance) & (grp[base_col] <= upper + tolerance)]
        
        if len(normal) == 0:
            results[name] = {'n_normal': 0, 'n_maintained': 0, 'rate': np.nan}
            continue
        
        # 终点仍正常
        maintained = normal[(normal[end_col] >= lower - tolerance) & (normal[end_col] <= upper + tolerance)]
        rate = len(maintained) / len(normal) * 100
        
        results[name] = {
            'n_normal': len(normal),
            'n_maintained': len(maintained),
            'rate': rate
        }
    
    # 比较两组维持率
    if results.get('treatment', {}).get('n_normal', 0) > 0 and results.get('control', {}).get('n_normal', 0) > 0:
        try:
            table = [
                [results['treatment']['n_maintained'], results['treatment']['n_normal'] - results['treatment']['n_maintained']],
                [results['control']['n_maintained'], results['control']['n_normal'] - results['control']['n_maintained']]
            ]
            _, p_val = stats.fisher_exact(table)
            results['comparison_p'] = p_val
        except:
            results['comparison_p'] = np.nan
    else:
        results['comparison_p'] = np.nan
    
    return results


def test_effect_modification(df, base_col, end_col, treatment_col):
    """
    效应修饰检验（检验基线状态×治疗的交互作用）
    
    使用线性回归模型：
    Y_change = β0 + β1*Treatment + β2*Baseline_Abnormal + β3*(Treatment×Baseline_Abnormal) + ε
    
    如果β3显著，说明治疗效果在基线异常和正常人群中不同（效应修饰）
    """
    try:
        import statsmodels.api as sm
        
        df_model = df.copy()
        df_model['change'] = df_model[end_col] - df_model[base_col]
        
        # 创建异常指示变量（这里简化为是否超过上限）
        range_info = get_indicator_range(base_col)
        if range_info:
            upper = range_info['upper']
            df_model['abnormal'] = (df_model[base_col] > upper + 2).astype(int)
        else:
            df_model['abnormal'] = (df_model[base_col] > df_model[base_col].median()).astype(int)
        
        # 创建交互项
        df_model['interaction'] = df_model[treatment_col] * df_model['abnormal']
        
        # 拟合模型
        X = df_model[[treatment_col, 'abnormal', 'interaction']]
        X = sm.add_constant(X)
        y = df_model['change']
        
        model = sm.OLS(y, X).fit()
        
        return {
            'interaction_coef': model.params.get('interaction', np.nan),
            'interaction_p': model.pvalues.get('interaction', np.nan),
            'significant': model.pvalues.get('interaction', 1) < 0.05,
            'interpretation': '基线状态影响治疗效果' if model.pvalues.get('interaction', 1) < 0.05 else '基线状态不影响治疗效果'
        }
    except Exception as e:
        return {
            'interaction_coef': np.nan,
            'interaction_p': np.nan,
            'significant': False,
            'interpretation': f'无法计算: {str(e)}'
        }


def run_stratified_analysis(df=None):
    """
    运行完整的分层分析
    """
    if df is None:
        df = load_and_preprocess_data()
    
    print("=" * 60)
    print("临床分层分析 - 考虑正常值范围的影响")
    print("=" * 60)
    
    all_results = []
    detailed_results = {}
    
    for base_col, end_col in OUTCOMES_PAIRS:
        print(f"\n分析指标: {base_col} -> {end_col}")
        
        result = stratified_outcome_analysis(df, base_col, end_col)
        
        if result is None:
            print(f"  跳过（数据不足或无正常范围定义）")
            continue
        
        indicator = result['indicator']
        detailed_results[indicator] = result
        
        # 提取关键信息
        summary = {
            '指标': indicator,
            '正常范围': result['normal_range'],
            '方向': '越低越好' if result['direction'] == 'lower_better' else '越高越好',
        }
        
        # 全人群
        if result.get('overall'):
            ov = result['overall']
            summary['全人群_N'] = ov['n_total']
            if ov.get('comparison'):
                summary['全人群_组间P值'] = ov['comparison'].get('p_value', np.nan)
        
        # 基线异常亚组
        if '异常偏高' in result.get('strata', {}):
            ab = result['strata']['异常偏高']
            summary['基线异常_N'] = ab['n_total']
            if ab.get('comparison'):
                summary['基线异常_组间P值'] = ab['comparison'].get('p_value', np.nan)
        
        # 基线正常亚组
        if '正常' in result.get('strata', {}):
            nm = result['strata']['正常']
            summary['基线正常_N'] = nm['n_total']
            if nm.get('comparison'):
                summary['基线正常_组间P值'] = nm['comparison'].get('p_value', np.nan)
        
        # 复常率
        if result.get('normalization'):
            norm = result['normalization']
            summary['治疗组_复常率(%)'] = norm.get('treatment', {}).get('rate', np.nan)
            summary['对照组_复常率(%)'] = norm.get('control', {}).get('rate', np.nan)
            summary['复常率_P值'] = norm.get('comparison_p', np.nan)
        
        # 维持率
        if result.get('maintenance'):
            maint = result['maintenance']
            summary['治疗组_维持率(%)'] = maint.get('treatment', {}).get('rate', np.nan)
            summary['对照组_维持率(%)'] = maint.get('control', {}).get('rate', np.nan)
            summary['维持率_P值'] = maint.get('comparison_p', np.nan)
        
        # 效应修饰
        if result.get('effect_modification'):
            em = result['effect_modification']
            summary['交互作用P值'] = em.get('interaction_p', np.nan)
            summary['效应修饰'] = '是' if em.get('significant', False) else '否'
        
        all_results.append(summary)
        
        # 打印简要结果
        print(f"  正常范围: {result['normal_range']}")
        print(f"  全人群N={summary.get('全人群_N', 'N/A')}, P={summary.get('全人群_组间P值', 'N/A'):.4f}" if not np.isnan(summary.get('全人群_组间P值', np.nan)) else f"  全人群N={summary.get('全人群_N', 'N/A')}")
        print(f"  基线异常N={summary.get('基线异常_N', 'N/A')}, P={summary.get('基线异常_组间P值', 'N/A'):.4f}" if not np.isnan(summary.get('基线异常_组间P值', np.nan)) else f"  基线异常N={summary.get('基线异常_N', 'N/A')}")
        print(f"  复常率: 治疗组{summary.get('治疗组_复常率(%)', 'N/A'):.1f}% vs 对照组{summary.get('对照组_复常率(%)', 'N/A'):.1f}%" if not np.isnan(summary.get('治疗组_复常率(%)', np.nan)) else "  复常率: N/A")
        print(f"  效应修饰: {summary.get('效应修饰', 'N/A')}")
    
    # 保存结果
    result_dir = ensure_result_dir('Stratified')
    
    df_summary = pd.DataFrame(all_results)
    df_summary.to_csv(os.path.join(result_dir, 'stratified_summary.csv'), index=False, encoding='utf-8-sig')
    
    # 生成详细报告
    generate_stratified_report(detailed_results, result_dir)
    
    print(f"\n结果已保存到 {result_dir}")
    
    return df_summary, detailed_results


def generate_stratified_report(detailed_results, result_dir):
    """生成详细的分层分析报告"""
    
    report_lines = [
        "# 临床分层分析报告",
        "",
        "## 分析说明",
        "",
        "### 为什么需要分层分析？",
        "",
        "在临床研究中，直接比较治疗前后的变化可能产生误导，原因如下：",
        "",
        "1. **天花板效应 (Ceiling Effect)**：如果基线值已经在正常范围内，",
        "   改善空间有限，即使治疗有效也难以观察到统计显著差异。",
        "",
        "2. **地板效应 (Floor Effect)**：对于'越低越好'的指标（如ALT），",
        "   基线正常者已经处于最佳状态。",
        "",
        "3. **效应修饰 (Effect Modification)**：治疗效果可能因基线状态而异，",
        "   基线异常者可能获益更多。",
        "",
        "### 分析策略",
        "",
        "| 分析类型 | 说明 | 临床意义 |",
        "|----------|------|----------|",
        "| 全人群分析 | 所有患者的治疗效果 | 整体疗效评估 |",
        "| 基线异常亚组 | 仅分析基线异常患者 | 真正需要治疗的人群 |",
        "| 基线正常亚组 | 仅分析基线正常患者 | 维护效果评估 |",
        "| 复常率分析 | 异常→正常的比例 | 治愈效果 |",
        "| 维持率分析 | 正常→正常的比例 | 稳定效果 |",
        "| 效应修饰检验 | 交互作用检验 | 疗效是否因人而异 |",
        "",
        "---",
        "",
        "## 各指标分析结果",
        ""
    ]
    
    for indicator, result in detailed_results.items():
        report_lines.extend([
            f"### {indicator}",
            "",
            f"- 正常范围: {result['normal_range']} ({result.get('direction', '')})",
            ""
        ])
        
        # 分层结果表格
        report_lines.append("| 分层 | N | 治疗组变化 | 对照组变化 | 组间P值 |")
        report_lines.append("|------|---|-----------|-----------|---------|")
        
        # 全人群
        if result.get('overall'):
            ov = result['overall']
            t_change = ov.get('treatment', {}).get('change_mean', np.nan)
            c_change = ov.get('control', {}).get('change_mean', np.nan)
            p_val = ov.get('comparison', {}).get('p_value', np.nan)
            report_lines.append(f"| 全人群 | {ov['n_total']} | {t_change:.2f} | {c_change:.2f} | {p_val:.4f} |" 
                              if not np.isnan(p_val) else f"| 全人群 | {ov['n_total']} | - | - | - |")
        
        # 各分层
        for status in ['异常偏高', '正常', '异常偏低']:
            if status in result.get('strata', {}):
                st = result['strata'][status]
                t_change = st.get('treatment', {}).get('change_mean', np.nan)
                c_change = st.get('control', {}).get('change_mean', np.nan)
                p_val = st.get('comparison', {}).get('p_value', np.nan)
                if not np.isnan(t_change):
                    report_lines.append(f"| 基线{status} | {st['n_total']} | {t_change:.2f} | {c_change:.2f} | {p_val:.4f} |" 
                                      if not np.isnan(p_val) else f"| 基线{status} | {st['n_total']} | {t_change:.2f} | {c_change:.2f} | - |")
        
        report_lines.append("")
        
        # 复常率和维持率
        if result.get('normalization'):
            norm = result['normalization']
            t_rate = norm.get('treatment', {}).get('rate', np.nan)
            c_rate = norm.get('control', {}).get('rate', np.nan)
            if not np.isnan(t_rate):
                report_lines.append(f"**复常率**: 治疗组 {t_rate:.1f}% vs 对照组 {c_rate:.1f}% (P={norm.get('comparison_p', np.nan):.4f})")
        
        if result.get('maintenance'):
            maint = result['maintenance']
            t_rate = maint.get('treatment', {}).get('rate', np.nan)
            c_rate = maint.get('control', {}).get('rate', np.nan)
            if not np.isnan(t_rate):
                report_lines.append(f"**维持率**: 治疗组 {t_rate:.1f}% vs 对照组 {c_rate:.1f}% (P={maint.get('comparison_p', np.nan):.4f})")
        
        # 效应修饰
        if result.get('effect_modification'):
            em = result['effect_modification']
            report_lines.append(f"**效应修饰检验**: P={em.get('interaction_p', np.nan):.4f} - {em.get('interpretation', '')}")
        
        report_lines.extend(["", "---", ""])
    
    # 统计学注释
    report_lines.extend([
        "## 统计学方法说明",
        "",
        "### 1. 分层分析原理",
        "",
        "根据基线值是否在正常范围内，将患者分为不同亚组分别分析：",
        "",
        "$$\\text{基线状态} = \\begin{cases} \\text{正常} & \\text{if } L-\\delta \\leq X_0 \\leq U+\\delta \\\\ \\text{异常偏高} & \\text{if } X_0 > U+\\delta \\\\ \\text{异常偏低} & \\text{if } X_0 < L-\\delta \\end{cases}$$",
        "",
        "其中 $L$ 为正常下限，$U$ 为正常上限，$\\delta$ 为容忍度（通常取2）。",
        "",
        "### 2. 复常率计算",
        "",
        "$$\\text{复常率} = \\frac{\\text{基线异常且终点正常的例数}}{\\text{基线异常的例数}} \\times 100\\%$$",
        "",
        "### 3. 维持率计算",
        "",
        "$$\\text{维持率} = \\frac{\\text{基线正常且终点仍正常的例数}}{\\text{基线正常的例数}} \\times 100\\%$$",
        "",
        "### 4. 效应修饰检验",
        "",
        "使用包含交互项的线性回归模型：",
        "",
        "$$\\Delta Y = \\beta_0 + \\beta_1 \\cdot T + \\beta_2 \\cdot A + \\beta_3 \\cdot (T \\times A) + \\varepsilon$$",
        "",
        "其中：",
        "- $\\Delta Y$ = 终点值 - 基线值",
        "- $T$ = 治疗组指示变量 (1=治疗组, 0=对照组)",
        "- $A$ = 基线异常指示变量 (1=异常, 0=正常)",
        "- $\\beta_3$ = 交互作用系数",
        "",
        "如果 $\\beta_3$ 的P值 < 0.05，说明存在效应修饰，即治疗效果在基线异常和正常人群中不同。",
        "",
        "### 5. 结果解读建议",
        "",
        "| 情景 | 全人群P值 | 基线异常亚组P值 | 解读 |",
        "|------|----------|----------------|------|",
        "| A | <0.05 | <0.05 | 治疗有效 |",
        "| B | >0.05 | <0.05 | 治疗对异常者有效，全人群被正常者稀释 |",
        "| C | <0.05 | >0.05 | 需审慎解读，可能有混杂 |",
        "| D | >0.05 | >0.05 | 治疗可能无效 |",
        "",
        "**情景B是本分析着重解决的问题**：当基线正常的患者占比较高时，",
        "全人群分析可能错误地得出'治疗无效'的结论。",
        ""
    ])
    
    # 写入文件
    with open(os.path.join(result_dir, 'stratified_report.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))


def baseline_status_summary(df=None):
    """生成基线状态分布汇总"""
    if df is None:
        df = load_and_preprocess_data()
    
    summary = []
    
    for base_col, end_col in OUTCOMES_PAIRS:
        indicator = base_col.replace('基线', '').replace('肝硬度值基线', '肝硬度值').replace('血小板基线', '血小板').replace('HBsAg基线', 'HBsAg').strip()
        range_info = get_indicator_range(indicator)
        
        if range_info is None:
            continue
        
        valid = df[base_col].dropna()
        n_total = len(valid)
        
        upper = range_info['upper']
        lower = range_info['lower']
        tolerance = 2
        
        n_normal = ((valid >= lower - tolerance) & (valid <= upper + tolerance)).sum()
        n_high = (valid > upper + tolerance).sum()
        n_low = (valid < lower - tolerance).sum()
        
        summary.append({
            '指标': indicator,
            '正常范围': f"{lower}-{upper}",
            '总例数': n_total,
            '正常例数': n_normal,
            '正常比例(%)': n_normal / n_total * 100,
            '异常偏高例数': n_high,
            '异常偏高比例(%)': n_high / n_total * 100,
            '异常偏低例数': n_low,
            '异常偏低比例(%)': n_low / n_total * 100
        })
    
    return pd.DataFrame(summary)


if __name__ == '__main__':
    print("=" * 70)
    print("临床分层分析 - 考虑正常值范围对统计结果的影响")
    print("=" * 70)
    
    # 加载数据
    df = load_and_preprocess_data()
    
    # 1. 基线状态分布
    print("\n### 基线状态分布 ###")
    baseline_summary = baseline_status_summary(df)
    print(baseline_summary.to_string())
    
    # 保存
    result_dir = ensure_result_dir('Stratified')
    baseline_summary.to_csv(os.path.join(result_dir, 'baseline_status_distribution.csv'), 
                           index=False, encoding='utf-8-sig')
    
    # 2. 分层分析
    print("\n### 分层结局分析 ###")
    summary_df, detailed = run_stratified_analysis(df)
    
    print("\n分析完成！")
    print(f"结果保存在: {result_dir}")
