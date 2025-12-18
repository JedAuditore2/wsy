"""
临床参考配置模块
定义肝功能指标的正常范围和临床意义判断标准
"""
import numpy as np
import pandas as pd

# ==================== 临床正常范围定义 ====================
# 格式: {指标名: (下限, 上限, 单位, 是否重点关注)}
NORMAL_RANGES = {
    '白蛋白': (40, 55, 'g/L', False),
    'ALT': (7, 40, 'U/L', True),      # 重点关注
    'AST': (13, 35, 'U/L', True),     # 重点关注
    'ALP': (35, 100, 'U/L', False),
    'GGT': (7, 45, 'U/L', False),
    '胆红素': (3.4, 17.1, 'μmol/L', False),  # 总胆红素参考范围
    '肝硬度值': (2.5, 7.0, 'kPa', False),    # <7.0通常认为无明显纤维化
    '血小板': (125, 350, '×10^9/L', False),
    'HBsAg': (0, 0.05, 'IU/mL', False),      # <0.05为阴性
}

# 主要结局变量（重点关注的指标）
PRIMARY_OUTCOMES = ['ALT', 'AST']
SECONDARY_OUTCOMES = ['GGT', 'ALP', '白蛋白', '胆红素', '肝硬度值', '血小板', 'HBsAg']

# 指标临床方向性（下降为好=True, 上升为好=False）
LOWER_IS_BETTER = {
    'ALT': True,
    'AST': True,
    'GGT': True,
    'ALP': True,
    '胆红素': True,
    '肝硬度值': True,
    'HBsAg': True,
    '白蛋白': False,  # 白蛋白升高为好
    '血小板': False,  # 血小板升高通常为好（肝病患者）
}

# ==================== 临床意义判断函数 ====================

def get_normal_range(indicator_name):
    """
    获取指标的正常范围
    
    Parameters:
    -----------
    indicator_name : str
        指标名称（可以包含"基线"等前缀）
    
    Returns:
    --------
    tuple: (下限, 上限, 单位, 是否重点)
    """
    # 清理指标名
    clean_name = indicator_name.replace('基线', '').replace('12个月', '').strip()
    
    for key in NORMAL_RANGES:
        if key in clean_name:
            return NORMAL_RANGES[key]
    
    return None

def classify_value(value, indicator_name, tolerance_low=0, tolerance_high=2):
    """
    判断单个值的临床分类
    
    Parameters:
    -----------
    value : float
        检测值
    indicator_name : str
        指标名称
    tolerance_low : float
        下限容忍度（略低于下限仍可认为正常）
    tolerance_high : float
        上限容忍度（略高于上限1-2单位仍可认为正常）
    
    Returns:
    --------
    str: '正常', '偏低', '轻度升高', '明显升高', '严重升高'
    """
    if pd.isna(value):
        return '缺失'
    
    range_info = get_normal_range(indicator_name)
    if range_info is None:
        return '未知'
    
    lower, upper, unit, is_key = range_info
    
    # 考虑容忍度（基于正态分布思想）
    adjusted_lower = lower - tolerance_low
    adjusted_upper = upper + tolerance_high
    
    if value < adjusted_lower:
        return '偏低'
    elif value <= adjusted_upper:
        return '正常'
    elif value <= upper * 1.5:  # 上限的1.5倍以内
        return '轻度升高'
    elif value <= upper * 3:   # 上限的3倍以内
        return '明显升高'
    else:
        return '严重升高'

def assess_clinical_significance(baseline, endpoint, indicator_name):
    """
    评估变化的临床意义
    
    Parameters:
    -----------
    baseline : float
        基线值
    endpoint : float
        终点值
    indicator_name : str
        指标名称
    
    Returns:
    --------
    dict: 包含临床评估结果
    """
    if pd.isna(baseline) or pd.isna(endpoint):
        return {'status': '数据缺失', 'significance': '无法评估'}
    
    range_info = get_normal_range(indicator_name)
    if range_info is None:
        return {'status': '未知指标', 'significance': '无法评估'}
    
    lower, upper, unit, is_key = range_info
    change = endpoint - baseline
    change_pct = (change / baseline * 100) if baseline != 0 else 0
    
    # 获取方向性
    clean_name = indicator_name.replace('基线', '').replace('12个月', '').strip()
    lower_is_better = LOWER_IS_BETTER.get(clean_name, None)
    
    # 分类基线和终点状态
    baseline_class = classify_value(baseline, indicator_name)
    endpoint_class = classify_value(endpoint, indicator_name)
    
    # 判断临床改善/恶化
    if lower_is_better is True:  # 如ALT, AST等，下降为好
        if change < 0:
            if endpoint_class == '正常' and baseline_class != '正常':
                clinical_effect = '复常'
            elif abs(change_pct) > 30:
                clinical_effect = '明显改善'
            elif abs(change_pct) > 10:
                clinical_effect = '轻度改善'
            else:
                clinical_effect = '稳定'
        else:
            if abs(change_pct) > 30:
                clinical_effect = '明显恶化'
            elif abs(change_pct) > 10:
                clinical_effect = '轻度恶化'
            else:
                clinical_effect = '稳定'
    elif lower_is_better is False:  # 如白蛋白，上升为好
        if change > 0:
            if endpoint_class == '正常' and baseline_class == '偏低':
                clinical_effect = '复常'
            elif abs(change_pct) > 10:
                clinical_effect = '改善'
            else:
                clinical_effect = '稳定'
        else:
            if abs(change_pct) > 10:
                clinical_effect = '恶化'
            else:
                clinical_effect = '稳定'
    else:
        clinical_effect = '无法判断'
    
    return {
        'baseline': baseline,
        'endpoint': endpoint,
        'change': change,
        'change_pct': change_pct,
        'baseline_class': baseline_class,
        'endpoint_class': endpoint_class,
        'clinical_effect': clinical_effect,
        'is_key_indicator': is_key
    }

def calculate_normalization_rate(df, indicator_name, baseline_col, endpoint_col, treatment_col='treatment'):
    """
    计算复常率（从异常变为正常的比例）
    
    Parameters:
    -----------
    df : DataFrame
    indicator_name : str
    baseline_col : str
        基线列名
    endpoint_col : str
        终点列名
    treatment_col : str
        治疗分组列
    
    Returns:
    --------
    dict: 各组的复常率统计
    """
    range_info = get_normal_range(indicator_name)
    if range_info is None:
        return None
    
    lower, upper, unit, is_key = range_info
    
    results = {}
    
    for group in [0, 1]:
        group_data = df[df[treatment_col] == group].copy()
        valid_data = group_data[[baseline_col, endpoint_col]].dropna()
        
        if len(valid_data) == 0:
            continue
        
        # 基线异常（主要关注升高）的患者
        abnormal_baseline = valid_data[valid_data[baseline_col] > upper + 2]  # 容忍度2
        
        if len(abnormal_baseline) > 0:
            # 终点恢复正常的患者数
            normalized = abnormal_baseline[abnormal_baseline[endpoint_col] <= upper + 2]
            normalization_rate = len(normalized) / len(abnormal_baseline) * 100
        else:
            normalization_rate = np.nan
        
        group_name = '治疗组' if group == 1 else '对照组'
        results[group_name] = {
            '基线异常例数': len(abnormal_baseline),
            '终点复常例数': len(normalized) if len(abnormal_baseline) > 0 else 0,
            '复常率(%)': normalization_rate
        }
    
    return results

def get_correlation_groups():
    """
    返回相关性较高的指标组
    用于多变量分析时的分组处理
    """
    return {
        '肝酶组': ['ALT', 'AST', 'GGT', 'ALP'],  # 肝酶指标相关性高
        '肝功能组': ['白蛋白', '胆红素'],
        '纤维化组': ['肝硬度值', '血小板'],
        '病毒学组': ['HBsAg'],
    }

def summarize_clinical_outcomes(df, treatment_col='treatment'):
    """
    生成临床结果汇总
    
    Parameters:
    -----------
    df : DataFrame
        包含基线和终点数据的DataFrame
    treatment_col : str
        治疗分组列名
    
    Returns:
    --------
    DataFrame: 临床汇总表
    """
    from data_preprocessing import OUTCOMES_PAIRS
    
    results = []
    
    for base_col, end_col in OUTCOMES_PAIRS:
        indicator = base_col.replace('基线', '').strip()
        
        if base_col not in df.columns or end_col not in df.columns:
            continue
        
        is_primary = indicator in PRIMARY_OUTCOMES
        
        for group in [1, 0]:
            group_data = df[df[treatment_col] == group]
            valid_data = group_data[[base_col, end_col]].dropna()
            
            if len(valid_data) == 0:
                continue
            
            # 计算基本统计
            baseline_mean = valid_data[base_col].mean()
            endpoint_mean = valid_data[end_col].mean()
            change_mean = (valid_data[end_col] - valid_data[base_col]).mean()
            
            # 计算复常率
            range_info = get_normal_range(indicator)
            if range_info:
                lower, upper, unit, is_key = range_info
                abnormal_n = (valid_data[base_col] > upper + 2).sum()
                if abnormal_n > 0:
                    normalized_n = ((valid_data[base_col] > upper + 2) & 
                                   (valid_data[end_col] <= upper + 2)).sum()
                    norm_rate = normalized_n / abnormal_n * 100
                else:
                    normalized_n = 0
                    norm_rate = np.nan
            else:
                abnormal_n = np.nan
                normalized_n = np.nan
                norm_rate = np.nan
            
            group_name = '治疗组' if group == 1 else '对照组'
            
            results.append({
                '指标': indicator,
                '指标类型': '主要' if is_primary else '次要',
                '分组': group_name,
                '例数': len(valid_data),
                '基线均值': baseline_mean,
                '终点均值': endpoint_mean,
                '变化均值': change_mean,
                '基线异常例数': abnormal_n,
                '复常例数': normalized_n,
                '复常率(%)': norm_rate
            })
    
    return pd.DataFrame(results)


if __name__ == '__main__':
    # 测试代码
    print("=== 临床配置模块测试 ===")
    print("\n正常范围定义:")
    for k, v in NORMAL_RANGES.items():
        print(f"  {k}: {v[0]}-{v[1]} {v[2]} {'(重点)' if v[3] else ''}")
    
    print("\n测试值分类:")
    test_cases = [
        ('ALT', 25),   # 正常
        ('ALT', 45),   # 轻度升高
        ('ALT', 100),  # 明显升高
        ('ALT', 200),  # 严重升高
        ('白蛋白', 38), # 偏低
        ('白蛋白', 45), # 正常
    ]
    for indicator, value in test_cases:
        result = classify_value(value, indicator)
        print(f"  {indicator}={value}: {result}")
