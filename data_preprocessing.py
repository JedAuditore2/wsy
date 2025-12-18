"""
共享数据预处理模块
用于所有匹配方法的统一数据读取和预处理

临床参考信息：
- 正常指标范围：白蛋白[40,55] ALT[7-40] AST[13-35] ALP[35-100] GGT[7-45]
- 一般偏低情况或者偏高1-2情况考虑正常（以正态分布考虑）
- 偏高过多考虑异常，重点关注指标ALT AST
- 各个表征指标间非独立
"""
import pandas as pd
import numpy as np
import re
import os

DATA_PATH = 'data/data.csv'
RESULT_DIR = 'result'

# 定义变量
TREATMENT_COL = '治疗方案'
COVARIATES = [
    '性别', '年龄', '饮酒史', '吸烟史', '是否合并脂肪肝', 
    '基线白蛋白', '基线胆红素', '基线ALT', '基线AST', '基线GGT', '基线ALP'
]

# 主要结局变量（重点关注）- ALT和AST
PRIMARY_OUTCOMES_PAIRS = [
    ('基线ALT', 'ALT12个月'),
    ('基线AST', 'AST12个月'),
]

# 次要结局变量
SECONDARY_OUTCOMES_PAIRS = [
    ('基线GGT', 'GGT12个月'),
    ('基线ALP', 'ALP12个月'),
    ('基线白蛋白', '白蛋白12个月'),
    ('基线胆红素', '总胆红素12个月'),
    ('肝硬度值基线', '肝硬度值12个月'),
    ('血小板基线', '血小板12个月'),
    ('HBsAg基线', 'HBsAg12个月'),
]

# 全部结果变量 (基线 -> 12个月)
OUTCOMES_PAIRS = PRIMARY_OUTCOMES_PAIRS + SECONDARY_OUTCOMES_PAIRS

# 临床正常范围（下限, 上限, 单位）
NORMAL_RANGES = {
    '白蛋白': (40, 55, 'g/L'),
    'ALT': (7, 40, 'U/L'),
    'AST': (13, 35, 'U/L'),
    'ALP': (35, 100, 'U/L'),
    'GGT': (7, 45, 'U/L'),
    '胆红素': (3.4, 17.1, 'μmol/L'),
}

def clean_numeric(x):
    """清洗数值型数据"""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        if "小于" in x:
            nums = re.findall(r'\d+\.?\d*', x)
            if nums:
                return float(nums[0])
        nums = re.findall(r'\d+\.?\d*', x)
        if nums:
            return float(nums[0])
        return np.nan
    return x

def load_and_preprocess_data(fill_missing=False):
    """
    加载并预处理数据
    
    Parameters:
    -----------
    fill_missing : bool
        是否使用均值填充缺失值
    
    Returns:
    --------
    df : DataFrame
        预处理后的数据
    """
    print("正在读取数据...")
    df = pd.read_csv(DATA_PATH, header=1)
    df.columns = [c.strip() for c in df.columns]
    
    print("原始数据形状:", df.shape)
    
    # 处理治疗变量
    df[TREATMENT_COL] = pd.to_numeric(df[TREATMENT_COL], errors='coerce')
    df = df[df[TREATMENT_COL].isin([1, 2])].copy()
    df['treatment'] = df[TREATMENT_COL].apply(lambda x: 1 if x == 1 else 0)
    
    print(f"过滤治疗方案后样本量: {len(df)}")
    
    # 清洗数值列
    all_cols = COVARIATES + [p[0] for p in OUTCOMES_PAIRS] + [p[1] for p in OUTCOMES_PAIRS]
    all_cols = list(set(all_cols))
    
    for col in all_cols:
        if col in df.columns:
            df[col] = df[col].apply(clean_numeric)
    
    # 处理缺失值
    if fill_missing:
        for col in COVARIATES:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mean())
    
    df_clean = df.dropna(subset=COVARIATES).copy()
    print(f"清洗后样本量: {len(df_clean)}")
    
    return df_clean

def ensure_result_dir(method_name):
    """确保结果目录存在"""
    result_path = os.path.join(RESULT_DIR, method_name)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    return result_path

def calculate_smd(df, covariates, treatment_col='treatment', weights=None):
    """
    计算标准化均值差异 (SMD)
    
    Parameters:
    -----------
    df : DataFrame
    covariates : list
    treatment_col : str
    weights : array-like, optional
        加权权重
    
    Returns:
    --------
    DataFrame with SMD values
    """
    smd_data = []
    treated = df[df[treatment_col] == 1]
    control = df[df[treatment_col] == 0]
    
    if weights is not None:
        weights_t = weights[df[treatment_col] == 1]
        weights_c = weights[df[treatment_col] == 0]
    
    for cov in covariates:
        if weights is None:
            mean_t = treated[cov].mean()
            mean_c = control[cov].mean()
            var_t = treated[cov].var()
            var_c = control[cov].var()
        else:
            # 加权均值和方差
            mean_t = np.average(treated[cov], weights=weights_t)
            mean_c = np.average(control[cov], weights=weights_c)
            var_t = np.average((treated[cov] - mean_t)**2, weights=weights_t)
            var_c = np.average((control[cov] - mean_c)**2, weights=weights_c)
        
        pooled_std = np.sqrt((var_t + var_c) / 2)
        if pooled_std == 0:
            smd = 0
        else:
            smd = (mean_t - mean_c) / pooled_std
            
        smd_data.append({
            '协变量': cov,
            '治疗组均值': mean_t,
            '对照组均值': mean_c,
            'SMD': smd
        })
    return pd.DataFrame(smd_data)

def analyze_outcomes(df, treatment_col='treatment', weights=None):
    """
    分析结果变量（含临床意义评估）
    
    Returns:
    --------
    DataFrame with outcome analysis results
    """
    from scipy import stats
    
    analysis_results = []
    
    for base_col, end_col in OUTCOMES_PAIRS:
        if base_col in df.columns and end_col in df.columns:
            t_group = df[df[treatment_col] == 1]
            c_group = df[df[treatment_col] == 0]
            
            # 获取指标名和类型
            indicator = base_col.replace('基线', '').strip()
            is_primary = (base_col, end_col) in PRIMARY_OUTCOMES_PAIRS
            
            # 获取正常范围
            normal_range = None
            for key in NORMAL_RANGES:
                if key in indicator or key in base_col:
                    normal_range = NORMAL_RANGES[key]
                    break
            
            # 治疗组
            t_valid = t_group[[base_col, end_col]].dropna()
            if len(t_valid) > 1:
                t_diff = t_valid[end_col] - t_valid[base_col]
                t_mean_diff = t_diff.mean()
                try:
                    _, t_p_val = stats.ttest_rel(t_valid[end_col], t_valid[base_col])
                except:
                    t_p_val = np.nan
                
                # 计算复常率（基线异常->终点正常）
                if normal_range:
                    upper = normal_range[1]
                    tolerance = 2  # 容忍度
                    t_abnormal = t_valid[t_valid[base_col] > upper + tolerance]
                    if len(t_abnormal) > 0:
                        t_normalized = t_abnormal[t_abnormal[end_col] <= upper + tolerance]
                        t_norm_rate = len(t_normalized) / len(t_abnormal) * 100
                        t_abnormal_n = len(t_abnormal)
                    else:
                        t_norm_rate = np.nan
                        t_abnormal_n = 0
                else:
                    t_norm_rate = np.nan
                    t_abnormal_n = np.nan
            else:
                t_mean_diff = np.nan
                t_p_val = np.nan
                t_diff = pd.Series([])
                t_norm_rate = np.nan
                t_abnormal_n = np.nan
                
            # 对照组
            c_valid = c_group[[base_col, end_col]].dropna()
            if len(c_valid) > 1:
                c_diff = c_valid[end_col] - c_valid[base_col]
                c_mean_diff = c_diff.mean()
                try:
                    _, c_p_val = stats.ttest_rel(c_valid[end_col], c_valid[base_col])
                except:
                    c_p_val = np.nan
                
                # 计算复常率
                if normal_range:
                    upper = normal_range[1]
                    tolerance = 2
                    c_abnormal = c_valid[c_valid[base_col] > upper + tolerance]
                    if len(c_abnormal) > 0:
                        c_normalized = c_abnormal[c_abnormal[end_col] <= upper + tolerance]
                        c_norm_rate = len(c_normalized) / len(c_abnormal) * 100
                        c_abnormal_n = len(c_abnormal)
                    else:
                        c_norm_rate = np.nan
                        c_abnormal_n = 0
                else:
                    c_norm_rate = np.nan
                    c_abnormal_n = np.nan
            else:
                c_mean_diff = np.nan
                c_p_val = np.nan
                c_diff = pd.Series([])
                c_norm_rate = np.nan
                c_abnormal_n = np.nan

            # 组间比较
            if len(t_diff) > 1 and len(c_diff) > 1:
                try:
                    _, group_p_val = stats.ttest_ind(t_diff, c_diff, equal_var=False)
                except:
                    group_p_val = np.nan
            else:
                group_p_val = np.nan
            
            # 复常率组间比较（Fisher精确检验）
            if normal_range and t_abnormal_n > 0 and c_abnormal_n > 0:
                try:
                    # 构建2x2列联表
                    t_norm_n = int(t_abnormal_n * t_norm_rate / 100) if not np.isnan(t_norm_rate) else 0
                    c_norm_n = int(c_abnormal_n * c_norm_rate / 100) if not np.isnan(c_norm_rate) else 0
                    table = [[t_norm_n, int(t_abnormal_n) - t_norm_n],
                             [c_norm_n, int(c_abnormal_n) - c_norm_n]]
                    _, norm_rate_p = stats.fisher_exact(table)
                except:
                    norm_rate_p = np.nan
            else:
                norm_rate_p = np.nan
                
            analysis_results.append({
                '指标': indicator,
                '指标类型': '主要' if is_primary else '次要',
                '治疗组_例数': len(t_valid),
                '治疗组_基线均值': t_valid[base_col].mean() if len(t_valid) > 0 else np.nan,
                '治疗组_终点均值': t_valid[end_col].mean() if len(t_valid) > 0 else np.nan,
                '治疗组_变化均值': t_mean_diff,
                '治疗组_前后P值': t_p_val,
                '治疗组_基线异常例数': t_abnormal_n,
                '治疗组_复常率(%)': t_norm_rate,
                '对照组_例数': len(c_valid),
                '对照组_基线均值': c_valid[base_col].mean() if len(c_valid) > 0 else np.nan,
                '对照组_终点均值': c_valid[end_col].mean() if len(c_valid) > 0 else np.nan,
                '对照组_变化均值': c_mean_diff,
                '对照组_前后P值': c_p_val,
                '对照组_基线异常例数': c_abnormal_n,
                '对照组_复常率(%)': c_norm_rate,
                '组间变化差异P值': group_p_val,
                '复常率差异P值': norm_rate_p,
                '正常范围上限': normal_range[1] if normal_range else np.nan
            })

    return pd.DataFrame(analysis_results)
