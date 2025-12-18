"""
熵平衡匹配 (Entropy Balancing) 可视化报告
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'KaiTi', 'FangSong']
plt.rcParams['axes.unicode_minus'] = False

RESULT_DIR = 'result/EB'

def main():
    print("=" * 60)
    print("熵平衡匹配 (EB) 可视化报告")
    print("=" * 60)
    
    # 读取数据
    try:
        df_weighted = pd.read_csv(os.path.join(RESULT_DIR, 'weighted_data.csv'))
        df_balance = pd.read_csv(os.path.join(RESULT_DIR, 'balance_check.csv'))
        df_outcome = pd.read_csv(os.path.join(RESULT_DIR, 'outcome_analysis.csv'))
        df_summary = pd.read_csv(os.path.join(RESULT_DIR, 'summary.csv'))
    except FileNotFoundError as e:
        print(f"错误: 找不到结果文件，请先运行 eb_analysis.py")
        print(f"详情: {e}")
        return
    
    # 创建可视化
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('熵平衡匹配 (Entropy Balancing) 分析报告', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # ----- 图1: 权重分布 -----
    ax1 = fig.add_subplot(2, 2, 1)
    control_weights = df_weighted[df_weighted['treatment'] == 0]['eb_weight']
    
    ax1.hist(control_weights, bins=30, color='#3498db', alpha=0.7, edgecolor='white')
    ax1.axvline(x=1, color='red', linestyle='--', linewidth=2, label='原始权重=1')
    ax1.axvline(x=control_weights.mean(), color='green', linestyle='-', linewidth=2, 
                label=f'均值={control_weights.mean():.2f}')
    ax1.set_xlabel('权重', fontsize=11)
    ax1.set_ylabel('频数', fontsize=11)
    ax1.set_title('对照组熵平衡权重分布', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # ----- 图2: 协变量平衡性对比 (Love Plot) -----
    ax2 = fig.add_subplot(2, 2, 2)
    covariates = df_balance['协变量'].tolist()
    smd_before = df_balance['匹配前SMD'].abs().tolist()
    smd_after = df_balance['匹配后SMD'].abs().tolist()
    
    y_pos = np.arange(len(covariates))
    height = 0.35
    
    bars1 = ax2.barh(y_pos - height/2, smd_before, height, label='加权前', color='#e74c3c', alpha=0.7)
    bars2 = ax2.barh(y_pos + height/2, smd_after, height, label='加权后', color='#27ae60', alpha=0.7)
    
    ax2.axvline(x=0.1, color='#2c3e50', linestyle='--', linewidth=2, label='平衡阈值 (0.1)')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(covariates, fontsize=9)
    ax2.set_xlabel('|标准化均值差异| (|SMD|)', fontsize=11)
    ax2.set_title('协变量平衡性对比 (Love Plot)', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # ----- 图3: 主要结果指标变化对比 -----
    ax3 = fig.add_subplot(2, 2, 3)
    
    if len(df_outcome) > 0:
        # 选择前6个指标
        plot_data = df_outcome.head(6)
        indicators = plot_data['指标'].tolist()
        x = np.arange(len(indicators))
        width = 0.35
        
        ax3.bar(x - width/2, plot_data['治疗组_变化均值'], width, label='治疗组', color='#e74c3c', alpha=0.7)
        ax3.bar(x + width/2, plot_data['对照组_变化均值'], width, label='对照组', color='#3498db', alpha=0.7)
        
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
     熵平衡匹配 (EB) 分析摘要
============================================

【样本信息】
  原始样本量:    {df_summary['原始样本量'].values[0]:>6} 人
  治疗组人数:    {df_summary['治疗组人数'].values[0]:>6} 人
  对照组人数:    {df_summary['对照组人数'].values[0]:>6} 人

【权重分布】
  权重范围: [{control_weights.min():.3f}, {control_weights.max():.3f}]
  权重均值:      {control_weights.mean():.4f}
  权重中位数:    {control_weights.median():.4f}

【平衡性评估】
  协变量总数:    {len(covariates):>6} 个
  平衡协变量:    {n_balanced:>6} 个 (|SMD|<0.1)
  平衡比例:      {n_balanced/len(covariates)*100:>5.1f}%

【方法说明】
  熵平衡通过最优化对照组权重，
  使其协变量矩与治疗组完全匹配，
  无需进行样本丢弃。
============================================
"""
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.savefig(os.path.join(RESULT_DIR, 'EB_Report.png'), dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n✓ 可视化报告已保存: {RESULT_DIR}/EB_Report.png")
    
    plt.show()

if __name__ == '__main__':
    main()
