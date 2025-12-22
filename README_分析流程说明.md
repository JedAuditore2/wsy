# 慢性乙型肝炎治疗效果分析 - 代码结构与使用说明

## 一、项目结构总览

```
wsy/
├── data/                           # 数据目录
│   ├── data.csv                    # 原始数据文件
│   ├── readme_of_data.txt          # 数据说明
│   └── background_and_method.txt   # 背景和方法说明
│
├── result/                         # 分析结果输出目录
│   ├── Comprehensive/              # 综合分析结果
│   │   ├── Raw/                    # 原始数据分析
│   │   ├── PSM/                    # PSM匹配后分析
│   │   └── CEM/                    # CEM匹配后分析（主要使用）
│   ├── CEM_Standalone/             # CEM独立分析结果
│   ├── PSM/                        # PSM分析结果
│   ├── DID/                        # 双重差分分析
│   ├── EB/                         # 熵平衡分析
│   └── ML-PSM/                     # 机器学习PSM分析
│
├── 核心分析模块 ─────────────────────────────────
│   ├── data_preprocessing.py       # 【入口1】数据预处理（最重要）
│   ├── comprehensive_analysis.py   # 【入口2】综合分析主程序
│   ├── cem_standalone_analysis.py  # 【入口3】CEM独立分析
│   └── lancet_style.py             # 柳叶刀图表风格
│
├── 其他匹配方法 ─────────────────────────────────
│   ├── psm_analysis.py             # 倾向性评分匹配(PSM)
│   ├── cem_analysis.py             # CEM匹配分析
│   ├── eb_analysis.py              # 熵平衡(EB)分析
│   ├── did_analysis.py             # 双重差分(DID)分析
│   └── ml_psm_analysis.py          # 机器学习PSM
│
├── 可视化模块 ───────────────────────────────────
│   ├── cem_visualize.py            # CEM可视化
│   ├── did_visualize.py            # DID可视化
│   ├── eb_visualize.py             # EB可视化
│   └── ml_psm_visualize.py         # ML-PSM可视化
│
├── 配置与辅助 ───────────────────────────────────
│   ├── clinical_config.py          # 临床配置参数
│   ├── clinical_stratified_analysis.py  # 临床分层分析
│   ├── multivariate_analysis.py    # 多变量分析
│   └── run_all_analyses.py         # 运行所有分析
│
└── 报告文件 ─────────────────────────────────────
    ├── report.md                   # 分析报告（Markdown）
    ├── CEM_Report.md               # CEM专项报告
    └── 正常值范围统计分析指南.md      # 临床指南
```

---

## 二、核心数据流程

```
┌─────────────────────────────────────────────────────────────────┐
│                         数据分析流程                              │
└─────────────────────────────────────────────────────────────────┘

    ┌──────────────┐
    │  data.csv    │  原始数据
    └──────┬───────┘
           │
           ▼
    ┌──────────────────────────┐
    │  data_preprocessing.py   │  数据预处理
    │  - 读取CSV               │
    │  - 清洗数据               │
    │  - 定义变量               │
    └──────────┬───────────────┘
               │
               ▼
    ┌──────────────────────────┐
    │  CEM/PSM 匹配            │  倾向性匹配
    │  - 粗化精确匹配(CEM)      │
    │  - 倾向性评分匹配(PSM)    │
    └──────────┬───────────────┘
               │
               ▼
    ┌──────────────────────────┐
    │  comprehensive_analysis  │  10项综合分析
    │  1. 基线可比性检验        │
    │  2. 肝功能差异分析        │
    │  3. 非劣效性检验          │
    │  4. 箱式图比较            │
    │  5. 柱状图比较            │
    │  6. 肝硬度值分析          │
    │  7. 治愈速度分析          │
    │  8. 临床分层分析          │
    │  9. 相关性分析            │
    │  10. 多因素回归           │
    └──────────┬───────────────┘
               │
               ▼
    ┌──────────────────────────┐
    │  result/ 目录            │  输出结果
    │  - CSV数据表             │
    │  - PNG图表               │
    │  - 分析报告              │
    └──────────────────────────┘
```

---

## 三、快速开始：运行分析

### 方式1：运行综合分析（推荐）
```bash
python comprehensive_analysis.py
```
这会执行完整的10项分析，输出到 `result/Comprehensive/` 目录。

### 方式2：运行CEM独立分析
```bash
python cem_standalone_analysis.py
```
专门针对CEM匹配方法，输出到 `result/CEM_Standalone/` 目录。

### 方式3：运行所有匹配方法
```bash
python run_all_analyses.py
```
运行PSM、CEM、EB、DID等所有匹配方法。

---

## 四、如何更换数据集

### 步骤1：准备数据文件

将你的数据保存为 `data/data.csv`，格式要求：
- **第1行**：可以是标题说明（会被跳过）
- **第2行**：列名（变量名称）
- **第3行起**：数据

### 步骤2：修改 `data_preprocessing.py`

这是最关键的配置文件，需要修改以下内容：

```python
# ========== 1. 数据路径 ==========
DATA_PATH = 'data/your_data.csv'  # 修改为你的数据文件名

# ========== 2. 治疗变量 ==========
TREATMENT_COL = '治疗方案'  # 修改为你数据中的治疗分组列名
# 注意：代码假设 1=治疗组, 2=对照组，如不同需修改 load_and_preprocess_data() 函数

# ========== 3. 协变量（用于匹配的基线变量）==========
COVARIATES = [
    '性别', '年龄', '饮酒史', '吸烟史', '是否合并脂肪肝', 
    '基线白蛋白', '基线胆红素', '基线ALT', '基线AST', '基线GGT', '基线ALP'
]
# 修改为你数据中的协变量列名

# ========== 4. 主要结局变量（重点分析）==========
PRIMARY_OUTCOMES_PAIRS = [
    ('基线ALT', 'ALT12个月'),    # (基线列名, 终点列名)
    ('基线AST', 'AST12个月'),
]
# 修改为你的主要结局变量

# ========== 5. 次要结局变量 ==========
SECONDARY_OUTCOMES_PAIRS = [
    ('基线GGT', 'GGT12个月'),
    ('基线ALP', 'ALP12个月'),
    ('基线白蛋白', '白蛋白12个月'),
    # ... 添加其他指标
]

# ========== 6. 临床正常范围 ==========
NORMAL_RANGES = {
    '白蛋白': (40, 55, 'g/L'),      # (下限, 上限, 单位)
    'ALT': (7, 40, 'U/L'),
    'AST': (13, 35, 'U/L'),
    # ... 根据你的指标修改
}
```

### 步骤3：修改 `comprehensive_analysis.py` 中的扩展正常范围

```python
EXTENDED_NORMAL_RANGES = {
    '白蛋白': {'lower': 40, 'upper': 55, 'unit': 'g/L', 'direction': 'higher_better'},
    'ALT': {'lower': 7, 'upper': 40, 'unit': 'U/L', 'direction': 'lower_better'},
    # ... 根据你的指标修改
    # direction: 'higher_better' 表示值越高越好（如白蛋白）
    # direction: 'lower_better' 表示值越低越好（如ALT、AST）
}
```

### 步骤4：运行分析
```bash
python comprehensive_analysis.py
```

---

## 五、10项分析功能说明

| 序号 | 分析项目 | 功能说明 | 输出文件 |
|------|----------|----------|----------|
| 1 | 基线可比性检验 | 验证两组基线是否平衡 | baseline_comparability.csv |
| 2 | 肝功能差异分析 | 比较变化值差异（含分层） | liver_function_difference.csv |
| 3 | 非劣效性检验 | 验证治疗组不劣于对照组 | non_inferiority_test.csv |
| 4 | 箱式图比较 | 展示指标分布 | boxplot_*.png |
| 5 | 柱状图比较 | 治疗前后对比 | barplot_*.png |
| 6 | 肝硬度值分析 | 专项分析肝纤维化 | liver_stiffness_analysis.csv/png |
| 7 | 治愈速度分析 | 比较复常率 | cure_speed_analysis.csv |
| 8 | 临床分层分析 | 按基线状态分层 | stratified_analysis.csv/png |
| 9 | 相关性分析 | 指标间相关性 | correlation_*.csv/png |
| 10 | 多因素回归 | 控制混杂因素 | regression_forest_plot.png |

---

## 六、关键函数说明

### data_preprocessing.py
```python
load_and_preprocess_data()      # 加载并预处理数据
calculate_smd()                  # 计算标准化均值差异
analyze_outcomes()               # 分析结局变量
```

### comprehensive_analysis.py
```python
perform_cem_matching()           # CEM匹配
perform_psm_matching()           # PSM匹配
baseline_comparability_test()    # 基线可比性检验
liver_function_difference_analysis()  # 肝功能差异分析
non_inferiority_test()           # 非劣效性检验
boxplot_comparison()             # 箱式图
barplot_comparison()             # 柱状图
liver_stiffness_analysis()       # 肝硬度分析
cure_speed_analysis()            # 治愈速度分析
stratified_analysis()            # 分层分析
correlation_analysis()           # 相关性分析
multivariate_regression_analysis()  # 多因素回归
run_comprehensive_analysis()     # 主函数，运行所有分析
```

---

## 七、数据格式示例

你的 `data.csv` 应该类似这样的结构：

| 治疗方案 | 性别 | 年龄 | 基线ALT | ALT12个月 | 基线AST | AST12个月 | ... |
|----------|------|------|---------|-----------|---------|-----------|-----|
| 1 | 1 | 45 | 35 | 28 | 30 | 25 | ... |
| 2 | 2 | 52 | 42 | 38 | 35 | 32 | ... |
| 1 | 1 | 38 | 28 | 25 | 22 | 20 | ... |

其中：
- `治疗方案`: 1=治疗组, 2=对照组
- `性别`: 1=男, 2=女（或其他编码）
- 数值型变量直接填数值
- 缺失值可以留空或填NA

---

## 八、常见问题

### Q1: 如何只运行部分分析？
修改 `comprehensive_analysis.py` 中的 `run_comprehensive_analysis()` 函数，注释掉不需要的分析调用。

### Q2: 如何修改图表样式？
修改 `lancet_style.py` 中的颜色和字体设置。

### Q3: 如何添加新的分析指标？
在 `data_preprocessing.py` 中添加到 `OUTCOMES_PAIRS` 列表，并在 `NORMAL_RANGES` 中添加正常范围。

### Q4: 非劣效界值如何设置？
在 `comprehensive_analysis.py` 的 `non_inferiority_test()` 函数中修改 `margins` 字典。

---

## 九、依赖库

```bash
pip install pandas numpy scipy scikit-learn statsmodels matplotlib seaborn
```

---

*文档更新时间：2025年12月22日*
