"""
机器学习增强的倾向性得分匹配 (ML-PSM) 分析程序
使用XGBoost估计倾向性得分
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

# 尝试导入XGBoost
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("警告: XGBoost未安装，将使用随机森林作为替代")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# 导入共享模块
from data_preprocessing import (
    load_and_preprocess_data, ensure_result_dir, calculate_smd,
    analyze_outcomes, COVARIATES, OUTCOMES_PAIRS
)

def train_propensity_model(X, y, method='xgboost'):
    """
    训练倾向性得分模型
    
    Parameters:
    -----------
    X : array
        协变量矩阵
    y : array
        治疗指示变量
    method : str
        'xgboost', 'random_forest', 'gradient_boosting'
    
    Returns:
    --------
    model : fitted model
    propensity_scores : array
    cv_auc : float
    """
    if method == 'xgboost' and HAS_XGBOOST:
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
    elif method == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_leaf=5,
            random_state=42
        )
    else:
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        )
    
    # 交叉验证评估
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
    cv_auc = cv_scores.mean()
    
    # 训练最终模型
    model.fit(X, y)
    propensity_scores = model.predict_proba(X)[:, 1]
    
    return model, propensity_scores, cv_auc

def nearest_neighbor_matching(ps_treated, ps_control, caliper=0.05):
    """
    基于倾向性得分的最近邻匹配
    
    Returns:
    --------
    matched_treated_idx : list
    matched_control_idx : list
    """
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
    nbrs.fit(ps_control.reshape(-1, 1))
    distances, indices = nbrs.kneighbors(ps_treated.reshape(-1, 1))
    
    matched_treated_idx = []
    matched_control_idx = []
    used_control = set()
    
    # 按距离排序
    order = np.argsort(distances.flatten())
    
    for i in order:
        dist = distances[i, 0]
        ctrl_idx = indices[i, 0]
        
        if dist < caliper and ctrl_idx not in used_control:
            matched_treated_idx.append(i)
            matched_control_idx.append(ctrl_idx)
            used_control.add(ctrl_idx)
    
    return matched_treated_idx, matched_control_idx

def main():
    print("=" * 60)
    print("机器学习增强的倾向性得分匹配 (ML-PSM) 分析")
    print("=" * 60)
    
    # 加载数据
    df = load_and_preprocess_data(fill_missing=True)
    result_dir = ensure_result_dir('ML-PSM')
    
    # 准备数据
    X = df[COVARIATES].values
    y = df['treatment'].values
    
    # 选择方法
    if HAS_XGBOOST:
        method = 'xgboost'
        print("使用XGBoost估计倾向性得分...")
    else:
        method = 'random_forest'
        print("使用随机森林估计倾向性得分...")
    
    # 训练模型
    model, ps_scores, cv_auc = train_propensity_model(X, y, method=method)
    df['propensity_score'] = ps_scores
    
    print(f"模型交叉验证AUC: {cv_auc:.4f}")
    
    # 保存倾向性得分
    df.to_csv(os.path.join(result_dir, 'propensity_scores.csv'), index=False, encoding='utf-8-sig')
    
    # 特征重要性
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            '协变量': COVARIATES,
            '重要性': model.feature_importances_
        }).sort_values('重要性', ascending=False)
        importance_df.to_csv(os.path.join(result_dir, 'feature_importance.csv'), index=False, encoding='utf-8-sig')
        print("\n特征重要性:")
        print(importance_df.to_string(index=False))
    
    # 分离治疗组和对照组
    treated = df[df['treatment'] == 1]
    control = df[df['treatment'] == 0]
    
    print(f"\n治疗组人数: {len(treated)}, 对照组人数: {len(control)}")
    
    # 进行匹配
    print("正在进行1:1匹配...")
    ps_treated = treated['propensity_score'].values
    ps_control = control['propensity_score'].values
    
    matched_t_idx, matched_c_idx = nearest_neighbor_matching(ps_treated, ps_control, caliper=0.05)
    
    print(f"匹配成功对数: {len(matched_t_idx)}")
    
    # 构建匹配后数据
    matched_treated = treated.iloc[matched_t_idx]
    matched_control = control.iloc[matched_c_idx]
    df_matched = pd.concat([matched_treated, matched_control])
    
    df_matched.to_csv(os.path.join(result_dir, 'matched_data.csv'), index=False, encoding='utf-8-sig')
    
    # 保存匹配对
    pairs_data = []
    for t_idx, c_idx in zip(matched_t_idx, matched_c_idx):
        t_row = treated.iloc[t_idx]
        c_row = control.iloc[c_idx]
        pairs_data.append({
            'Treated_ID': t_row['病历号'],
            'Control_ID': c_row['病历号'],
            'Treated_PS': t_row['propensity_score'],
            'Control_PS': c_row['propensity_score'],
            'Distance': abs(t_row['propensity_score'] - c_row['propensity_score'])
        })
    pairs_df = pd.DataFrame(pairs_data)
    pairs_df.to_csv(os.path.join(result_dir, 'matched_pairs.csv'), index=False, encoding='utf-8-sig')
    
    # 平衡性检验
    print("正在进行平衡性检验...")
    smd_before = calculate_smd(df, COVARIATES)
    smd_after = calculate_smd(df_matched, COVARIATES)
    
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
    outcome_df = analyze_outcomes(df_matched)
    outcome_df.to_csv(os.path.join(result_dir, 'outcome_analysis.csv'), index=False, encoding='utf-8-sig')
    
    # 保存汇总
    summary = {
        '方法': f'机器学习增强PSM ({method})',
        '原始样本量': len(df),
        '治疗组人数': len(treated),
        '对照组人数': len(control),
        '匹配成功对数': len(matched_t_idx),
        '匹配成功率': f"{len(matched_t_idx)/len(treated)*100:.1f}%",
        '模型AUC': cv_auc,
        '协变量数量': len(COVARIATES),
        '平衡协变量数(|SMD|<0.1)': sum(abs(balance_df['匹配后SMD']) < 0.1)
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(os.path.join(result_dir, 'summary.csv'), index=False, encoding='utf-8-sig')
    
    print("\n" + "=" * 60)
    print("ML-PSM分析完成！")
    print(f"结果已保存至: {result_dir}")
    print("=" * 60)

if __name__ == '__main__':
    main()
