"""
双重差分法 (DID) 可视化报告
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

RESULT_DIR = 'result/DID'

def main():
    print("=" * 60)
    print("双重差分法 (DID) 可视化报告")
    print("=" * 60)
    
    # 读取数据
    try:
        df_did = pd.read_csv(os.path.join(RESULT_DIR, 'did_analysis.csv'))
        df_baseline = pd.read_csv(os.path.join(RESULT_DIR, 'baseline_comparability.csv'))
        df_summary = pd.read_csv(os.path.join(RESULT_DIR, 'summary.csv'))
        df_long = pd.read_csv(os.path.join(RESULT_DIR, 'long_format_data.csv'))
    except FileNotFoundError as e:
        print(f"错误: 找不到结果文件，请先运行 did_analysis.py")
        print(f"详情: {e}")
        return
    
    # 创建可视化
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle('双重差分法 (Difference-in-Differences) 分析报告', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # ----- 图1-4: DID趋势图（选择4个主要指标） -----
    main_outcomes = ['ALT', 'AST', '白蛋白', '肝硬度值']
    
    for i, outcome in enumerate(main_outcomes):
        ax = fig.add_subplot(2, 3, i+1)
        
        outcome_data = df_long[df_long['outcome'] == outcome]
        
        if len(outcome_data) > 0:
            # 计算各组各时间点均值
            means = outcome_data.groupby(['treatment', 'time'])['value'].mean().unstack()
            
            if means.shape[1] >= 2:
                # 治疗组
                if 1 in means.index:
                    ax.plot([0, 1], means.loc[1].values, 'o-', color='#e74c3c', 
                           linewidth=2, markersize=10, label='治疗组')
                
                # 对照组
                if 0 in means.index:
                    ax.plot([0, 1], means.loc[0].values, 's-', color='#3498db', 
                           linewidth=2, markersize=10, label='对照组')
                
                # 获取DID结果
                did_row = df_did[df_did['指标'] == outcome]
                if len(did_row) > 0:
                    did_est = did_row['DID估计量'].values[0]
                    did_p = did_row['DID_P值'].values[0]
                    sig = '*' if did_p < 0.05 else ''
                    ax.text(0.5, 0.95, f'DID={did_est:.2f}, P={did_p:.3f}{sig}', 
                           transform=ax.transAxes, fontsize=10, ha='center',
                           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['基线', '12个月'])
        ax.set_ylabel(f'{outcome}', fontsize=11)
        ax.set_title(f'{outcome}的DID分析', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    # ----- 图5: DID估计量森林图 -----
    ax5 = fig.add_subplot(2, 3, 5)
    
    # 筛选有效的DID结果
    valid_did = df_did.dropna(subset=['DID估计量', 'DID_95%CI下限', 'DID_95%CI上限'])
    
    if len(valid_did) > 0:
        y_pos = np.arange(len(valid_did))
        
        # 画误差线
        ax5.errorbar(valid_did['DID估计量'], y_pos,
                    xerr=[valid_did['DID估计量'] - valid_did['DID_95%CI下限'],
                          valid_did['DID_95%CI上限'] - valid_did['DID估计量']],
                    fmt='o', color='#2c3e50', capsize=5, capthick=2, markersize=8)
        
        # 添加零线
        ax5.axvline(x=0, color='red', linestyle='--', linewidth=2, label='无效应')
        
        ax5.set_yticks(y_pos)
        ax5.set_yticklabels(valid_did['指标'], fontsize=10)
        ax5.set_xlabel('DID估计量 (95% CI)', fontsize=11)
        ax5.set_title('DID估计量森林图', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='x')
        ax5.legend(loc='upper right')
    
    # ----- 图6: 分析摘要 -----
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    # 统计显著结果
    sig_count = sum(df_did['DID_P值'] < 0.05) if 'DID_P值' in df_did.columns else 0
    comparable_count = sum(df_baseline['P值'] > 0.05) if 'P值' in df_baseline.columns else 0
    
    summary_text = f"""
============================================
     双重差分法 (DID) 分析摘要
============================================

【样本信息】
  总样本量:          {df_summary['样本量'].values[0]:>6} 人
  治疗组人数:        {df_summary['治疗组人数'].values[0]:>6} 人
  对照组人数:        {df_summary['对照组人数'].values[0]:>6} 人

【分析结果】
  分析指标数:        {df_summary['分析指标数'].values[0]:>6} 个
  DID显著(P<0.05):  {sig_count:>6} 个
  基线可比(P>0.05): {comparable_count:>6} 个

【DID模型】
  Y = β₀ + β₁×Treatment + β₂×Post 
      + β₃×(Treatment×Post) + ε
  
  β₃ = DID估计量
  表示治疗组相对于对照组的额外变化

【解读】
  • DID > 0: 治疗组变化大于对照组
  • DID < 0: 治疗组变化小于对照组
  • P < 0.05: 差异具有统计学意义
============================================
"""
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.savefig(os.path.join(RESULT_DIR, 'DID_Report.png'), dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n✓ 可视化报告已保存: {RESULT_DIR}/DID_Report.png")
    
    # 打印DID结果表格
    print("\n" + "=" * 80)
    print("DID分析结果汇总")
    print("=" * 80)
    print(df_did[['指标', '治疗组_变化', '对照组_变化', 'DID估计量', 'DID_P值']].to_string(index=False))
    print("=" * 80)
    
    plt.show()

if __name__ == '__main__':
    main()
