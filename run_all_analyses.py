"""
主运行脚本 - 运行所有匹配方法的分析和可视化
"""
import subprocess
import sys
import os

def run_script(script_name, description):
    """运行Python脚本"""
    print("\n" + "=" * 70)
    print(f"正在运行: {description}")
    print("=" * 70)
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=False, text=True)
        if result.returncode == 0:
            print(f"✓ {description} 完成")
        else:
            print(f"✗ {description} 运行时出现问题")
    except Exception as e:
        print(f"✗ 运行失败: {e}")

def main():
    print("=" * 70)
    print("       慢性乙型肝炎治疗效果分析 - 多方法对比研究")
    print("=" * 70)
    print("\n本程序将依次运行以下分析方法:")
    print("  1. PSM     - 倾向性评分匹配 (Propensity Score Matching)")
    print("  2. EB      - 熵平衡匹配 (Entropy Balancing)")
    print("  3. ML-PSM  - 机器学习增强PSM (XGBoost/随机森林)")
    print("  4. CEM     - 粗化精确匹配 (Coarsened Exact Matching)")
    print("  5. DID     - 双重差分法 (Difference-in-Differences)")
    print("\n结果将保存在 result/ 目录下的各子文件夹中。")
    
    input("\n按回车键开始运行所有分析...")
    
    # 1. PSM分析
    run_script("psm_analysis.py", "倾向性评分匹配 (PSM) 分析")
    run_script("visualized_report.py", "PSM 可视化报告")
    
    # 2. 熵平衡匹配
    run_script("eb_analysis.py", "熵平衡匹配 (EB) 分析")
    run_script("eb_visualize.py", "EB 可视化报告")
    
    # 3. ML-PSM
    run_script("ml_psm_analysis.py", "机器学习增强PSM (ML-PSM) 分析")
    run_script("ml_psm_visualize.py", "ML-PSM 可视化报告")
    
    # 4. CEM
    run_script("cem_analysis.py", "粗化精确匹配 (CEM) 分析")
    run_script("cem_visualize.py", "CEM 可视化报告")
    
    # 5. DID
    run_script("did_analysis.py", "双重差分法 (DID) 分析")
    run_script("did_visualize.py", "DID 可视化报告")
    
    print("\n" + "=" * 70)
    print("所有分析完成！")
    print("=" * 70)
    print("\n结果文件位置:")
    print("  • result/PSM/     - 倾向性评分匹配结果")
    print("  • result/EB/      - 熵平衡匹配结果")
    print("  • result/ML-PSM/  - 机器学习增强PSM结果")
    print("  • result/CEM/     - 粗化精确匹配结果")
    print("  • result/DID/     - 双重差分法结果")
    print("\n每个文件夹包含:")
    print("  - 分析数据文件 (.csv)")
    print("  - 可视化报告 (.png)")
    print("  - 汇总信息 (summary.csv)")

if __name__ == '__main__':
    main()
