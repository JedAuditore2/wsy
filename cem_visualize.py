"""
粗化精确匹配 (CEM) 可视化报告
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'KaiTi', 'FangSong']
plt.rcParams['axes.unicode_minus'] = False

RESULT_DIR = 'result/CEM'

def main():
    print("=" * 60)
    print("粗化精确匹配 (CEM) 可视化报告")
    print("=" * 60)
    
    # 读取数据
    try:
        df_matched = pd.read_csv(os.path.join(RESULT_DIR, 'matched_data.csv'))
        df_balance = pd.read_csv(os.path.join(RESULT_DIR, 'balance_check.csv'))
        df_strata = pd.read_csv(os.path.join(RESULT_DIR, 'strata_info.csv'))
        df_outcome = pd.read_csv(os.path.join(RESULT_DIR, 'outcome_analysis.csv'))
        df_summary = pd.read_csv(os.path.join(RESULT_DIR, 'summary.csv'))
    except FileNotFoundError as e:
        print(f"错误: 找不到结果文件，请先运行 cem_analysis.py")
        print(f"详情: {e}")
        return
    
    # 创建可视化
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('粗化精确匹配 (Coarsened Exact Matching) 分析报告', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # ----- 图1: 匹配层样本分布 -----
    ax1 = fig.add_subplot(2, 2, 1)
    
    # 显示前15个最大的层
    strata_sorted = df_strata.sort_values(by=['治疗组人数', '对照组人数'], ascending=False).head(15)
    
    x = np.arange(len(strata_sorted))
    width = 0.35
    
    ax1.bar(x - width/2, strata_sorted['治疗组人数'], width, label='治疗组', color='#e74c3c', alpha=0.7)
    ax1.bar(x + width/2, strata_sorted['对照组人数'], width, label='对照组', color='#3498db', alpha=0.7)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'层{i+1}' for i in range(len(strata_sorted))], rotation=45, fontsize=9)
    ax1.set_ylabel('样本数', fontsize=11)
    ax1.set_title(f'匹配层样本分布 (前15个层，共{len(df_strata)}层)', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # ----- 图2: 协变量平衡性对比 -----
    ax2 = fig.add_subplot(2, 2, 2)
    covariates = df_balance['协变量'].tolist()
    smd_before = df_balance['匹配前SMD'].abs().tolist()
    smd_after = df_balance['匹配后SMD'].abs().tolist()
    
    y_pos = np.arange(len(covariates))
    height = 0.35
    
    ax2.barh(y_pos - height/2, smd_before, height, label='匹配前', color='#e74c3c', alpha=0.7)
    ax2.barh(y_pos + height/2, smd_after, height, label='匹配后', color='#27ae60', alpha=0.7)
    
    ax2.axvline(x=0.1, color='#2c3e50', linestyle='--', linewidth=2, label='平衡阈值 (0.1)')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(covariates, fontsize=9)
    ax2.set_xlabel('|SMD|', fontsize=11)
    ax2.set_title('协变量平衡性 (Love Plot)', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # ----- 图3: 结果指标变化对比 -----
    ax3 = fig.add_subplot(2, 2, 3)
    
    if len(df_outcome) > 0:
        plot_data = df_outcome.head(6)
        indicators = plot_data['指标'].tolist()
        x = np.arange(len(indicators))
        width = 0.35
        
        treat_change = plot_data['治疗组_变化均值'].fillna(0)
        control_change = plot_data['对照组_变化均值'].fillna(0)
        
        ax3.bar(x - width/2, treat_change, width, label='治疗组', color='#e74c3c', alpha=0.7)
        ax3.bar(x + width/2, control_change, width, label='对照组', color='#3498db', alpha=0.7)
        
        ax3.set_xticks(x)
        ax3.set_xticklabels(indicators, rotation=45, ha='right', fontsize=9)
        ax3.set_ylabel('变化值 (终点-基线)', fontsize=11)
        ax3.set_title('主要指标治疗前后变化对比', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # ----- 图4: 分析摘要 -----
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    
    n_balanced = sum(1 for x in smd_after if x < 0.1)
    
    summary_text = f"""
============================================
     粗化精确匹配 (CEM) 分析摘要
============================================

【原始样本】
  原始样本量:        {df_summary['原始样本量'].values[0]:>6} 人
  原始治疗组:        {df_summary['原始治疗组人数'].values[0]:>6} 人
  原始对照组:        {df_summary['原始对照组人数'].values[0]:>6} 人

【匹配结果】
  匹配后样本量:      {df_summary['匹配后样本量'].values[0]:>6} 人
  匹配后治疗组:      {df_summary['匹配后治疗组人数'].values[0]:>6} 人
  匹配后对照组:      {df_summary['匹配后对照组人数'].values[0]:>6} 人
  有效匹配层数:      {df_summary['有效匹配层数'].values[0]:>6} 层

【平衡性评估】
  协变量总数:        {len(covariates):>6} 个
  平衡协变量:        {n_balanced:>6} 个 (|SMD|<0.1)
  平衡比例:          {n_balanced/len(covariates)*100:>5.1f}%

【方法说明】
  CEM通过将连续变量粗化为类别，
  然后在每个层内进行精确匹配，
  保证匹配样本具有相似的协变量值。
============================================
"""
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.savefig(os.path.join(RESULT_DIR, 'CEM_Report.png'), dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n✓ 可视化报告已保存: {RESULT_DIR}/CEM_Report.png")
    
    plt.show()

if __name__ == '__main__':
    main()
