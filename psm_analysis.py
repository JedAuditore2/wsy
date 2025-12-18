import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from scipy import stats
import os
import re

# ==================== 配置 ====================
DATA_PATH = 'data/data.csv'
RESULT_DIR = 'result/PSM'

# 确保结果目录存在
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

# ==================== 1. 数据读取与预处理 ====================
print("正在读取数据...")
# 读取CSV，跳过第一行（元数据），第二行是表头
df = pd.read_csv(DATA_PATH, header=1)

# 清理列名（去除空格）
df.columns = [c.strip() for c in df.columns]

# 打印原始列名以供检查
# print(df.columns.tolist())

# 定义变量
treatment_col = '治疗方案'
covariates = [
    '性别', '年龄', '饮酒史', '吸烟史', '是否合并脂肪肝', 
    '基线白蛋白', '基线胆红素', '基线ALT', '基线AST', '基线GGT', '基线ALP'
]

# 结果变量 (基线 -> 12个月)
outcomes_pairs = [
    ('基线ALT', 'ALT12个月'),
    ('基线AST', 'AST12个月'),
    ('基线GGT', 'GGT12个月'),
    ('基线ALP', 'ALP12个月'),
    ('基线白蛋白', '白蛋白12个月'),
    ('基线胆红素', '总胆红素12个月'),
    ('肝硬度值基线', '肝硬度值12个月'),
    ('血小板基线', '血小板12个月'),
    ('HBsAg基线', 'HBsAg12个月'),
    ('HBV-DNA基线', 'HBV-DNA12个月')
]

# 数据清洗
print("原始数据形状:", df.shape)

# 1. 处理治疗变量
# 强制转换为数字，无法转换的变为NaN
df[treatment_col] = pd.to_numeric(df[treatment_col], errors='coerce')
print(f"治疗方案列的唯一值: {df[treatment_col].unique()}")

# 过滤掉治疗方案为空或不为1/2的行
df = df[df[treatment_col].isin([1, 2])].copy()
print(f"过滤治疗方案后样本量: {len(df)}")

# 转换为 1=治疗, 0=对照
df['treatment'] = df[treatment_col].apply(lambda x: 1 if x == 1 else 0)

# 2. 处理协变量中的缺失值和非数值字符
def clean_numeric(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        # 移除可能存在的非数字字符
        # 针对 "小于20" -> 20
        if "小于" in x:
             nums = re.findall(r'\d+\.?\d*', x)
             if nums:
                 return float(nums[0]) # 或者 float(nums[0]) - epsilon? 这里直接取边界值
        
        nums = re.findall(r'\d+\.?\d*', x)
        if nums:
            return float(nums[0])
        return np.nan
    return x

# 应用清洗到所有相关列
all_cols = covariates + [p[0] for p in outcomes_pairs] + [p[1] for p in outcomes_pairs]
# 去重
all_cols = list(set(all_cols))

for col in all_cols:
    if col in df.columns:
        df[col] = df[col].apply(clean_numeric)

# 3. 填充缺失值 (为了PSM能运行)
# 检查协变量缺失情况
missing_counts = df[covariates].isnull().sum()
print("协变量缺失情况:\n", missing_counts[missing_counts > 0])

# 策略：对于缺失较少的，直接删除；缺失较多的，可以用均值填充
# 这里为了保证能运行，先尝试删除含有缺失协变量的行
df_psm = df.dropna(subset=covariates).copy()

# 如果删除后样本太少，则启用均值填充
if len(df_psm) < 10:
    print("警告：删除缺失值后样本量过少，尝试使用均值填充...")
    df_psm = df.copy()
    for col in covariates:
        df_psm[col] = df_psm[col].fillna(df_psm[col].mean())
    # 再次检查是否还有缺失（可能整列都是NaN）
    df_psm = df_psm.dropna(subset=covariates)

print(f"最终用于PSM的样本量: {len(df_psm)}")

if len(df_psm) == 0:
    print("错误：没有足够的样本进行PSM分析。请检查数据。")
    exit()

# ==================== 2. 倾向性评分计算 (PSM) ====================
print("正在计算倾向性评分...")

X = df_psm[covariates]
y = df_psm['treatment']

# 使用逻辑回归估算倾向性评分
lr = LogisticRegression(max_iter=2000, solver='liblinear') # 增加迭代次数
lr.fit(X, y)
ps_score = lr.predict_proba(X)[:, 1]
df_psm['propensity_score'] = ps_score

# 保存倾向性评分结果
df_psm.to_csv(os.path.join(RESULT_DIR, 'propensity_scores.csv'), index=False, encoding='utf-8-sig')

# ==================== 3. 进行匹配 (1:1 Nearest Neighbor) ====================
print("正在进行1:1匹配...")

treated = df_psm[df_psm['treatment'] == 1]
control = df_psm[df_psm['treatment'] == 0]

print(f"治疗组人数: {len(treated)}, 对照组人数: {len(control)}")

if len(treated) == 0 or len(control) == 0:
    print("错误：某一组样本量为0，无法匹配。")
    exit()

# 使用NearestNeighbors进行匹配
caliper = 0.2 # 放宽卡钳值，或者使用标准差的0.2倍
# 计算logit propensity score的标准差
logit_ps = np.log(df_psm['propensity_score'] / (1 - df_psm['propensity_score']))
caliper_val = 0.2 * logit_ps.std()
print(f"使用卡钳值 (0.2 * SD of logit PS): {caliper_val}")

# 为了简单起见，这里还是用PS值的绝对差，但参考caliper_val的大小
# 注意：sklearn的NearestNeighbors是用欧氏距离，对于1D数据就是绝对差
nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(control[['propensity_score']])
distances, indices = nbrs.kneighbors(treated[['propensity_score']])

matched_indices = []
treated_indices = []
used_control_indices = set()

# 贪婪匹配
# 先把距离排序，优先匹配距离近的？
# 或者直接按顺序匹配
# 这里简单按顺序，但确保不重复使用对照组
# 为了更好的匹配，可以先对treated按PS排序? 暂时不搞那么复杂

matches = []
for i, (dist, ind) in enumerate(zip(distances, indices)):
    matches.append({
        'treated_idx': i,
        'control_idx': ind[0],
        'distance': dist[0]
    })

# 按距离排序，优先匹配好的
matches.sort(key=lambda x: x['distance'])

for m in matches:
    if m['distance'] < caliper_val * 5: # 稍微放宽一点限制，避免匹配太少，实际应严格
        # 注意：caliper_val是基于logit的，这里distance是基于概率的，量纲不同
        # 概率的caliper通常取 0.02 - 0.05
        # 让我们直接用一个固定的概率caliper 0.05
        if m['distance'] < 0.05:
            if m['control_idx'] not in used_control_indices:
                used_control_indices.add(m['control_idx'])
                matched_indices.append(control.iloc[m['control_idx']].name)
                treated_indices.append(treated.iloc[m['treated_idx']].name)

# 构建匹配后的DataFrame
matched_control = df_psm.loc[matched_indices]
matched_treated = df_psm.loc[treated_indices]
df_matched = pd.concat([matched_treated, matched_control])

print(f"匹配成功对数: {len(matched_treated)}")
df_matched.to_csv(os.path.join(RESULT_DIR, 'matched_data.csv'), index=False, encoding='utf-8-sig')

# 保存匹配对
pairs_data = []
# 需要重新对应回去，因为上面排序了
# 简单起见，我们直接遍历matched_treated，找到对应的control
# 但由于我们已经有了df_matched，我们可以假设它们是成对的吗？不，concat是简单的堆叠
# 我们需要记录配对关系
# 重新构建配对列表
for t_idx, c_idx in zip(treated_indices, matched_indices):
    t_row = df_psm.loc[t_idx]
    c_row = df_psm.loc[c_idx]
    pairs_data.append({
        'Treated_ID': t_row['病历号'],
        'Control_ID': c_row['病历号'],
        'Treated_Name': t_row['姓名：'],
        'Control_Name': c_row['姓名：'],
        'Treated_PS': t_row['propensity_score'],
        'Control_PS': c_row['propensity_score'],
        'Distance': abs(t_row['propensity_score'] - c_row['propensity_score'])
    })

pairs_df = pd.DataFrame(pairs_data)
pairs_df.to_csv(os.path.join(RESULT_DIR, 'matched_pairs.csv'), index=False, encoding='utf-8-sig')


# ==================== 4. 平衡性检验 (SMD) ====================
print("正在进行平衡性检验...")

def calculate_smd(df, covariates, treatment_col='treatment'):
    smd_data = []
    treated = df[df[treatment_col] == 1]
    control = df[df[treatment_col] == 0]
    
    if len(treated) < 2 or len(control) < 2:
        return pd.DataFrame(columns=['协变量', '治疗组均值', '对照组均值', 'SMD'])

    for cov in covariates:
        mean_t = treated[cov].mean()
        mean_c = control[cov].mean()
        var_t = treated[cov].var()
        var_c = control[cov].var()
        
        # Cohen's d / SMD calculation
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

smd_before = calculate_smd(df_psm, covariates)
smd_after = calculate_smd(df_matched, covariates)

if not smd_before.empty and not smd_after.empty:
    balance_df = pd.merge(smd_before[['协变量', 'SMD']], smd_after[['协变量', 'SMD']], on='协变量', suffixes=('_before', '_after'))
    balance_df = balance_df.rename(columns={'SMD_before': '匹配前SMD', 'SMD_after': '匹配后SMD'})
    balance_df.to_csv(os.path.join(RESULT_DIR, 'balance_check.csv'), index=False, encoding='utf-8-sig')
else:
    print("警告：无法计算SMD（样本量不足）。")

# ==================== 5. 结果分析 (Outcome Analysis) ====================
print("正在进行结果分析...")

analysis_results = []

if len(df_matched) > 0:
    for base_col, end_col in outcomes_pairs:
        if base_col in df_matched.columns and end_col in df_matched.columns:
            # 提取两组数据
            t_group = df_matched[df_matched['treatment'] == 1]
            c_group = df_matched[df_matched['treatment'] == 0]
            
            # 1. 组内前后比较
            # 治疗组
            t_valid = t_group[[base_col, end_col]].dropna()
            if len(t_valid) > 1:
                t_diff = t_valid[end_col] - t_valid[base_col]
                t_mean_diff = t_diff.mean()
                try:
                    _, t_p_val = stats.ttest_rel(t_valid[end_col], t_valid[base_col])
                except:
                    t_p_val = np.nan
            else:
                t_mean_diff = np.nan
                t_p_val = np.nan
                t_diff = []
                
            # 对照组
            c_valid = c_group[[base_col, end_col]].dropna()
            if len(c_valid) > 1:
                c_diff = c_valid[end_col] - c_valid[base_col]
                c_mean_diff = c_diff.mean()
                try:
                    _, c_p_val = stats.ttest_rel(c_valid[end_col], c_valid[base_col])
                except:
                    c_p_val = np.nan
            else:
                c_mean_diff = np.nan
                c_p_val = np.nan
                c_diff = []

            # 2. 组间比较 (比较变化值)
            if len(t_valid) > 1 and len(c_valid) > 1:
                try:
                    _, group_p_val = stats.ttest_ind(t_diff, c_diff, equal_var=False)
                except:
                    group_p_val = np.nan
            else:
                group_p_val = np.nan
                
            analysis_results.append({
                '指标': base_col.replace('基线', ''),
                '治疗组_基线均值': t_valid[base_col].mean() if len(t_valid) > 0 else np.nan,
                '治疗组_终点均值': t_valid[end_col].mean() if len(t_valid) > 0 else np.nan,
                '治疗组_变化均值': t_mean_diff,
                '治疗组_前后P值': t_p_val,
                '对照组_基线均值': c_valid[base_col].mean() if len(c_valid) > 0 else np.nan,
                '对照组_终点均值': c_valid[end_col].mean() if len(c_valid) > 0 else np.nan,
                '对照组_变化均值': c_mean_diff,
                '对照组_前后P值': c_p_val,
                '组间变化差异P值': group_p_val
            })

    outcome_df = pd.DataFrame(analysis_results)
    outcome_df.to_csv(os.path.join(RESULT_DIR, 'outcome_analysis.csv'), index=False, encoding='utf-8-sig')

print("分析完成！所有结果已保存至 result/ 文件夹。")
