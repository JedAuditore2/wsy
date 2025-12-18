import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import FontProperties
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体 - Windows系统
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'KaiTi', 'FangSong']
plt.rcParams['axes.unicode_minus'] = False

# 尝试找到可用的中文字体
import matplotlib.font_manager as fm
font_path = None
for font in ['msyh.ttc', 'simhei.ttf', 'simsun.ttc']:
    try:
        font_path = fm.findfont(fm.FontProperties(family=['Microsoft YaHei', 'SimHei']))
        if font_path:
            break
    except:
        pass

if font_path:
    plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()

# ==================== 读取已生成的数据 ====================
print("=" * 60)
print("倾向性评分匹配结果可视化报告")
print("=" * 60)

RESULT_DIR = "result/PSM"

df_scores = pd.read_csv(f"{RESULT_DIR}/propensity_scores.csv")
df_matched = pd.read_csv(f"{RESULT_DIR}/matched_data.csv")
df_balance = pd.read_csv(f"{RESULT_DIR}/balance_check.csv")

# ==================== 创建可视化报告 ====================
fig = plt.figure(figsize=(16, 14))
fig.suptitle('倾向性评分匹配分析报告\nPropensity Score Matching Analysis Report', 
             fontsize=16, fontweight='bold', y=0.98)

# ----- 图1: 匹配前后倾向性评分分布对比 -----
ax1 = fig.add_subplot(2, 2, 1)
treated_before = df_scores[df_scores['treatment']==1]['propensity_score']
control_before = df_scores[df_scores['treatment']==0]['propensity_score']

ax1.hist(treated_before, bins=20, alpha=0.6, label=f'治疗组 (n={len(treated_before)})', color='#e74c3c', density=True)
ax1.hist(control_before, bins=20, alpha=0.6, label=f'对照组 (n={len(control_before)})', color='#3498db', density=True)
ax1.set_xlabel('倾向性评分 (Propensity Score)', fontsize=11)
ax1.set_ylabel('密度 (Density)', fontsize=11)
ax1.set_title('匹配前: 倾向性评分分布', fontsize=12, fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# ----- 图2: 匹配后倾向性评分分布 -----
ax2 = fig.add_subplot(2, 2, 2)
treated_after = df_matched[df_matched['treatment']==1]['propensity_score']
control_after = df_matched[df_matched['treatment']==0]['propensity_score']

ax2.hist(treated_after, bins=20, alpha=0.6, label=f'治疗组 (n={len(treated_after)})', color='#e74c3c', density=True)
ax2.hist(control_after, bins=20, alpha=0.6, label=f'对照组 (n={len(control_after)})', color='#3498db', density=True)
ax2.set_xlabel('倾向性评分 (Propensity Score)', fontsize=11)
ax2.set_ylabel('密度 (Density)', fontsize=11)
ax2.set_title('匹配后: 倾向性评分分布', fontsize=12, fontweight='bold')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

# ----- 图3: 协变量平衡性对比 (Love Plot) -----
ax3 = fig.add_subplot(2, 2, 3)
covariates = df_balance['协变量'].tolist()
smd_before = df_balance['匹配前SMD'].abs().tolist()
smd_after = df_balance['匹配后SMD'].abs().tolist()

y_pos = np.arange(len(covariates))
height = 0.35

bars1 = ax3.barh(y_pos - height/2, smd_before, height, label='匹配前', color='#e74c3c', alpha=0.7)
bars2 = ax3.barh(y_pos + height/2, smd_after, height, label='匹配后', color='#27ae60', alpha=0.7)

ax3.axvline(x=0.1, color='#2c3e50', linestyle='--', linewidth=2, label='平衡阈值 (0.1)')
ax3.set_yticks(y_pos)
ax3.set_yticklabels(covariates, fontsize=10)
ax3.set_xlabel('|标准化均值差异| (|SMD|)', fontsize=11)
ax3.set_title('协变量平衡性对比 (Love Plot)', fontsize=12, fontweight='bold')
ax3.legend(loc='upper right')
ax3.grid(True, alpha=0.3, axis='x')
ax3.set_xlim(0, max(max(smd_before), max(smd_after)) * 1.2)

# ----- 图4: 匹配结果统计摘要 -----
ax4 = fig.add_subplot(2, 2, 4)
ax4.axis('off')

# 计算统计信息
n_total = len(df_scores)
n_treated_before = len(df_scores[df_scores['treatment']==1])
n_control_before = len(df_scores[df_scores['treatment']==0])
n_matched_pairs = len(df_matched) // 2
n_balanced = sum(1 for x in smd_after if x < 0.1)

summary_text = f"""
============================================
       倾向性评分匹配分析摘要
============================================

【样本信息】
  原始总样本量:  {n_total:>6} 人
  治疗组样本量:  {n_treated_before:>6} 人
  对照组样本量:  {n_control_before:>6} 人

【匹配结果】
  成功匹配对数:  {n_matched_pairs:>6} 对
  匹配后总样本:  {len(df_matched):>6} 人
  匹配成功率:    {n_matched_pairs/n_treated_before*100:>5.1f}%

【平衡性评估】
  协变量总数:    {len(covariates):>6} 个
  平衡协变量:    {n_balanced:>6} 个 (|SMD|<0.1)
  平衡比例:      {n_balanced/len(covariates)*100:>5.1f}%

【倾向性评分】
  治疗组: 均值={treated_before.mean():.3f}, 标准差={treated_before.std():.3f}
  对照组: 均值={control_before.mean():.3f}, 标准差={control_before.std():.3f}
============================================
"""

ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8))

plt.tight_layout(rect=[0, 0.02, 1, 0.95])
plt.savefig(f'{RESULT_DIR}/PSM_Report.png', dpi=150, bbox_inches='tight', facecolor='white')
print(f"\n✓ 可视化报告已保存: {RESULT_DIR}/PSM_Report.png")

# ==================== 生成详细文本报告 ====================
report = f"""
================================================================================
                    倾向性评分匹配分析报告
                Propensity Score Matching Report
================================================================================

生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

一、研究设计
--------------------------------------------------------------------------------
    本分析采用倾向性评分匹配(PSM)方法，根据患者基线特征计算倾向性评分，
    并使用1:1最近邻匹配法进行治疗组与对照组的匹配，以减少混杂因素影响。

二、分组依据
--------------------------------------------------------------------------------
    • 治疗组(treatment=1): 抗病毒药物类型=1 (如恩替卡韦)
    • 对照组(treatment=0): 抗病毒药物类型=2,3,4 (其他类型)

三、协变量选择
--------------------------------------------------------------------------------
    用于计算倾向性评分的基线协变量包括：
    1. 人口学特征: 性别、年龄
    2. 生活习惯: 饮酒史、吸烟史
    3. 合并症: 是否合并脂肪肝
    4. 肝功能指标: 基线白蛋白、基线胆红素、基线ALT、基线AST、基线GGT、基线ALP

四、样本信息
--------------------------------------------------------------------------------
    匹配前:
      - 总样本量: {n_total} 人
      - 治疗组: {n_treated_before} 人 ({n_treated_before/n_total*100:.1f}%)
      - 对照组: {n_control_before} 人 ({n_control_before/n_total*100:.1f}%)
    
    匹配后:
      - 成功匹配: {n_matched_pairs} 对
      - 匹配后样本: {len(df_matched)} 人
      - 匹配成功率: {n_matched_pairs/n_treated_before*100:.1f}%

五、倾向性评分分布
--------------------------------------------------------------------------------
    匹配前:
      - 治疗组: 均值={treated_before.mean():.4f}, 标准差={treated_before.std():.4f}, 
                范围=[{treated_before.min():.4f}, {treated_before.max():.4f}]
      - 对照组: 均值={control_before.mean():.4f}, 标准差={control_before.std():.4f},
                范围=[{control_before.min():.4f}, {control_before.max():.4f}]
    
    匹配后:
      - 治疗组: 均值={treated_after.mean():.4f}, 标准差={treated_after.std():.4f}
      - 对照组: 均值={control_after.mean():.4f}, 标准差={control_after.std():.4f}

六、协变量平衡性评估 (标准化均值差异 SMD)
--------------------------------------------------------------------------------
    判断标准: |SMD| < 0.1 表示协变量在两组间达到良好平衡
    
    {'协变量':<16} {'匹配前SMD':>12} {'匹配后SMD':>12} {'平衡状态':>10}
    {'-'*60}
"""

for _, row in df_balance.iterrows():
    status = "✓ 平衡" if abs(row['匹配后SMD']) < 0.1 else "✗ 不平衡"
    report += f"    {row['协变量']:<16} {row['匹配前SMD']:>12.4f} {row['匹配后SMD']:>12.4f} {status:>10}\n"

report += f"""    {'-'*60}
    平衡协变量数: {n_balanced}/{len(covariates)} ({n_balanced/len(covariates)*100:.1f}%)

七、结论
--------------------------------------------------------------------------------
    1. 匹配成功率为 {n_matched_pairs/n_treated_before*100:.1f}%，共获得 {n_matched_pairs} 对匹配样本
    2. 匹配后 {n_balanced}/{len(covariates)} 个协变量达到平衡 (|SMD|<0.1)
    3. 匹配后两组倾向性评分分布更加接近，协变量平衡性明显改善
    4. 匹配后的数据可用于后续因果效应分析

八、输出文件说明
--------------------------------------------------------------------------------
    • propensity_scores.csv - 所有患者的倾向性评分
    • matched_data.csv      - 匹配后的患者数据
    • matched_pairs.csv     - 匹配对详情
    • balance_check.csv     - 协变量平衡性检验结果
    • PSM_Report.png        - 可视化分析图表
    • PSM_Report.txt        - 详细文本报告

================================================================================
"""

with open(f'{RESULT_DIR}/PSM_Report.txt', 'w', encoding='utf-8') as f:
    f.write(report)
print(f"✓ 详细文本报告已保存: {RESULT_DIR}/PSM_Report.txt")

print(report)
plt.show()
