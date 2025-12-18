"""
熵平衡匹配 (Entropy Balancing) 分析程序
通过重新加权使对照组的协变量矩与治疗组匹配
"""
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy import stats
import os
import sys

# 导入共享模块
from data_preprocessing import (
    load_and_preprocess_data, ensure_result_dir, calculate_smd,
    analyze_outcomes, COVARIATES, OUTCOMES_PAIRS
)

def entropy_balancing(X_control, X_treated, max_iter=500):
    """
    熵平衡算法：为对照组计算权重，使其协变量矩与治疗组匹配
    
    Parameters:
    -----------
    X_control : array
        对照组协变量矩阵
    X_treated : array
        治疗组协变量矩阵
    max_iter : int
        最大迭代次数
    
    Returns:
    --------
    weights : array
        对照组的权重
    """
    n_control = X_control.shape[0]
    n_treated = X_treated.shape[0]
    
    # 目标矩：治疗组的均值
    target_moments = X_treated.mean(axis=0)
    
    # 标准化
    X_std = X_control.std(axis=0)
    X_std[X_std == 0] = 1  # 避免除零
    X_control_norm = (X_control - X_control.mean(axis=0)) / X_std
    target_norm = (target_moments - X_control.mean(axis=0)) / X_std
    
    # 初始化拉格朗日乘子
    n_moments = X_control.shape[1]
    lambda_init = np.zeros(n_moments)
    
    def objective(lam):
        """目标函数：最小化熵"""
        weights = np.exp(X_control_norm @ lam)
        weights = weights / weights.sum()  # 归一化
        
        # 约束违反程度
        moment_diff = (X_control_norm.T @ weights) - target_norm / n_control
        
        # 熵 + 惩罚项
        entropy = np.sum(weights * np.log(weights + 1e-10))
        penalty = 1000 * np.sum(moment_diff ** 2)
        
        return entropy + penalty
    
    def constraint_moments(lam):
        """矩约束"""
        weights = np.exp(X_control_norm @ lam)
        weights = weights / weights.sum()
        weighted_mean = X_control @ weights
        return weighted_mean - target_moments
    
    # 优化
    result = minimize(
        objective,
        lambda_init,
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'disp': False}
    )
    
    # 计算最终权重
    weights = np.exp(X_control_norm @ result.x)
    weights = weights / weights.sum() * n_control  # 归一化使权重和为n_control
    
    return weights

def main():
    print("=" * 60)
    print("熵平衡匹配 (Entropy Balancing) 分析")
    print("=" * 60)
    
    # 加载数据
    df = load_and_preprocess_data(fill_missing=True)
    result_dir = ensure_result_dir('EB')
    
    # 分离治疗组和对照组
    treated = df[df['treatment'] == 1].copy()
    control = df[df['treatment'] == 0].copy()
    
    print(f"治疗组人数: {len(treated)}, 对照组人数: {len(control)}")
    
    # 准备协变量矩阵
    X_treated = treated[COVARIATES].values
    X_control = control[COVARIATES].values
    
    # 计算熵平衡权重
    print("正在计算熵平衡权重...")
    try:
        eb_weights = entropy_balancing(X_control, X_treated)
        print(f"权重范围: [{eb_weights.min():.4f}, {eb_weights.max():.4f}]")
        print(f"权重和: {eb_weights.sum():.2f}")
    except Exception as e:
        print(f"熵平衡计算失败: {e}")
        print("使用均匀权重作为替代...")
        eb_weights = np.ones(len(control))
    
    # 保存权重
    control_with_weights = control.copy()
    control_with_weights['eb_weight'] = eb_weights
    treated_with_weights = treated.copy()
    treated_with_weights['eb_weight'] = 1.0  # 治疗组权重为1
    
    df_weighted = pd.concat([treated_with_weights, control_with_weights])
    df_weighted.to_csv(os.path.join(result_dir, 'weighted_data.csv'), index=False, encoding='utf-8-sig')
    
    # 计算平衡性
    print("正在进行平衡性检验...")
    
    # 匹配前SMD
    smd_before = calculate_smd(df, COVARIATES)
    
    # 匹配后SMD (加权)
    full_weights = np.concatenate([np.ones(len(treated)), eb_weights])
    smd_after = calculate_smd(df_weighted, COVARIATES, weights=full_weights)
    
    # 合并结果
    balance_df = pd.merge(
        smd_before[['协变量', 'SMD']], 
        smd_after[['协变量', 'SMD']], 
        on='协变量', 
        suffixes=('_before', '_after')
    )
    balance_df = balance_df.rename(columns={'SMD_before': '匹配前SMD', 'SMD_after': '匹配后SMD'})
    balance_df.to_csv(os.path.join(result_dir, 'balance_check.csv'), index=False, encoding='utf-8-sig')
    
    print("\n平衡性检验结果:")
    print(balance_df.to_string(index=False))
    
    # 结果分析
    print("\n正在进行结果分析...")
    outcome_df = analyze_outcomes(df_weighted)
    outcome_df.to_csv(os.path.join(result_dir, 'outcome_analysis.csv'), index=False, encoding='utf-8-sig')
    
    # 保存汇总信息
    summary = {
        '方法': '熵平衡匹配 (Entropy Balancing)',
        '原始样本量': len(df),
        '治疗组人数': len(treated),
        '对照组人数': len(control),
        '协变量数量': len(COVARIATES),
        '平衡协变量数(|SMD|<0.1)': sum(abs(balance_df['匹配后SMD']) < 0.1),
        '权重最小值': eb_weights.min(),
        '权重最大值': eb_weights.max(),
        '权重均值': eb_weights.mean()
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(os.path.join(result_dir, 'summary.csv'), index=False, encoding='utf-8-sig')
    
    print("\n" + "=" * 60)
    print("熵平衡匹配分析完成！")
    print(f"结果已保存至: {result_dir}")
    print("=" * 60)

if __name__ == '__main__':
    main()
