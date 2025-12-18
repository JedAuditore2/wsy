"""
机器学习增强的倾向性得分匹配 (ML-PSM) 可视化报告
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

RESULT_DIR = 'result/ML-PSM'

def main():
    print("=" * 60)
    print("机器学习增强PSM (ML-PSM) 可视化报告")
    print("=" * 60)
    
    # 读取数据
    try:
        df_scores = pd.read_csv(os.path.join(RESULT_DIR, 'propensity_scores.csv'))
        df_matched = pd.read_csv(os.path.join(RESULT_DIR, 'matched_data.csv'))
        df_balance = pd.read_csv(os.path.join(RESULT_DIR, 'balance_check.csv'))
        df_importance = pd.read_csv(os.path.join(RESULT_DIR, 'feature_importance.csv'))
        df_summary = pd.read_csv(os.path.join(RESULT_DIR, 'summary.csv'))
    except FileNotFoundError as e:
        print(f"错误: 找不到结果文件，请先运行 ml_psm_analysis.py")
        print(f"详情: {e}")
        return
    
    # 创建可视化
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle('机器学习增强倾向性得分匹配 (ML-PSM) 分析报告', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # ----- 图1: 匹配前倾向性评分分布 -----
    ax1 = fig.add_subplot(2, 3, 1)
    treated_before = df_scores[df_scores['treatment']==1]['propensity_score']
    control_before = df_scores[df_scores['treatment']==0]['propensity_score']
    
    ax1.hist(treated_before, bins=20, alpha=0.6, label=f'治疗组 (n={len(treated_before)})', 
             color='#e74c3c', density=True)
    ax1.hist(control_before, bins=20, alpha=0.6, label=f'对照组 (n={len(control_before)})', 
             color='#3498db', density=True)
    ax1.set_xlabel('倾向性评分', fontsize=11)
    ax1.set_ylabel('密度', fontsize=11)
    ax1.set_title('匹配前: 倾向性评分分布', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # ----- 图2: 匹配后倾向性评分分布 -----
    ax2 = fig.add_subplot(2, 3, 2)
    treated_after = df_matched[df_matched['treatment']==1]['propensity_score']
    control_after = df_matched[df_matched['treatment']==0]['propensity_score']
    
    ax2.hist(treated_after, bins=20, alpha=0.6, label=f'治疗组 (n={len(treated_after)})', 
             color='#e74c3c', density=True)
    ax2.hist(control_after, bins=20, alpha=0.6, label=f'对照组 (n={len(control_after)})', 
             color='#3498db', density=True)
    ax2.set_xlabel('倾向性评分', fontsize=11)
    ax2.set_ylabel('密度', fontsize=11)
    ax2.set_title('匹配后: 倾向性评分分布', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # ----- 图3: 特征重要性 -----
    ax3 = fig.add_subplot(2, 3, 3)
    importance_sorted = df_importance.sort_values('重要性', ascending=True)
    
    y_pos = np.arange(len(importance_sorted))
    ax3.barh(y_pos, importance_sorted['重要性'], color='#9b59b6', alpha=0.7)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(importance_sorted['协变量'], fontsize=9)
    ax3.set_xlabel('特征重要性', fontsize=11)
    ax3.set_title('ML模型特征重要性', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # ----- 图4: 协变量平衡性对比 (Love Plot) -----
    ax4 = fig.add_subplot(2, 3, 4)
    covariates = df_balance['协变量'].tolist()
    smd_before = df_balance['匹配前SMD'].abs().tolist()
    smd_after = df_balance['匹配后SMD'].abs().tolist()
    
    y_pos = np.arange(len(covariates))
    height = 0.35
    
    ax4.barh(y_pos - height/2, smd_before, height, label='匹配前', color='#e74c3c', alpha=0.7)
    ax4.barh(y_pos + height/2, smd_after, height, label='匹配后', color='#27ae60', alpha=0.7)
    
    ax4.axvline(x=0.1, color='#2c3e50', linestyle='--', linewidth=2, label='平衡阈值 (0.1)')
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(covariates, fontsize=9)
    ax4.set_xlabel('|SMD|', fontsize=11)
    ax4.set_title('协变量平衡性 (Love Plot)', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=8)
    ax4.grid(True, alpha=0.3, axis='x')
    
    # ----- 图5: SMD变化散点图 -----
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.scatter(smd_before, smd_after, c='#3498db', s=100, alpha=0.7, edgecolors='white')
    
    # 添加参考线
    max_val = max(max(smd_before), max(smd_after)) * 1.1
    ax5.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='无变化线')
    ax5.axhline(y=0.1, color='red', linestyle=':', alpha=0.7, label='|SMD|=0.1')
    ax5.axvline(x=0.1, color='red', linestyle=':', alpha=0.7)
    
    ax5.set_xlabel('匹配前 |SMD|', fontsize=11)
    ax5.set_ylabel('匹配后 |SMD|', fontsize=11)
    ax5.set_title('SMD改善情况', fontsize=12, fontweight='bold')
    ax5.legend(loc='upper right')
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(0, max_val)
    ax5.set_ylim(0, max_val)
    
    # ----- 图6: 分析摘要 -----
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    n_balanced = sum(1 for x in smd_after if x < 0.1)
    
    summary_text = f"""
============================================
     ML-PSM 分析摘要
============================================

【模型信息】
  算法: {df_summary['方法'].values[0].split('(')[1].replace(')', '')}
  模型AUC: {df_summary['模型AUC'].values[0]:.4f}

【样本信息】
  原始样本量:    {df_summary['原始样本量'].values[0]:>6} 人
  治疗组人数:    {df_summary['治疗组人数'].values[0]:>6} 人
  对照组人数:    {df_summary['对照组人数'].values[0]:>6} 人

【匹配结果】
  匹配成功对数:  {df_summary['匹配成功对数'].values[0]:>6} 对
  匹配成功率:    {df_summary['匹配成功率'].values[0]:>6}

【平衡性评估】
  协变量总数:    {len(covariates):>6} 个
  平衡协变量:    {n_balanced:>6} 个 (|SMD|<0.1)
  平衡比例:      {n_balanced/len(covariates)*100:>5.1f}%

【最重要特征】
  {importance_sorted.iloc[-1]['协变量']}: {importance_sorted.iloc[-1]['重要性']:.3f}
  {importance_sorted.iloc[-2]['协变量']}: {importance_sorted.iloc[-2]['重要性']:.3f}
============================================
"""
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.savefig(os.path.join(RESULT_DIR, 'ML_PSM_Report.png'), dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n✓ 可视化报告已保存: {RESULT_DIR}/ML_PSM_Report.png")
    
    plt.show()

if __name__ == '__main__':
    main()
