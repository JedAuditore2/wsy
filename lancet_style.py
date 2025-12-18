"""
柳叶刀(Lancet)论文风格图表配置
参考Lancet系列期刊的图表设计规范
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# ==================== 柳叶刀配色方案 ====================
# 基于Lancet期刊常用的配色
LANCET_COLORS = {
    'primary': '#00468B',      # 深蓝色 - 主色
    'secondary': '#ED0000',    # 红色 - 次色
    'tertiary': '#42B540',     # 绿色
    'quaternary': '#0099B4',   # 青色
    'quinary': '#925E9F',      # 紫色
    'senary': '#FDAF91',       # 浅橙色
    'septenary': '#AD002A',    # 深红色
    'octonary': '#ADB6B6',     # 灰色
}

# 用于分组比较的配色（治疗组 vs 对照组）
TREATMENT_COLOR = '#ED0000'   # 红色 - 治疗组
CONTROL_COLOR = '#00468B'     # 蓝色 - 对照组

# 用于治疗前后的配色
BASELINE_ALPHA = 0.5          # 基线较浅
ENDPOINT_ALPHA = 1.0          # 终点较深

# 柳叶刀调色板列表
LANCET_PALETTE = [
    '#00468B', '#ED0000', '#42B540', '#0099B4', 
    '#925E9F', '#FDAF91', '#AD002A', '#ADB6B6'
]

def setup_lancet_style():
    """
    设置柳叶刀论文风格的matplotlib全局参数
    字体大小为1.5倍
    """
    # 基础设置
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 字体设置 - 1.5倍大小
    plt.rcParams.update({
        # 字体
        'font.family': 'sans-serif',
        'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 15,  # 10 * 1.5
        
        # 坐标轴
        'axes.labelsize': 16.5,  # 11 * 1.5
        'axes.titlesize': 18,    # 12 * 1.5
        'axes.titleweight': 'bold',
        'axes.linewidth': 1.2,
        'axes.edgecolor': '#333333',
        'axes.labelcolor': '#333333',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.facecolor': 'white',
        'axes.grid': False,
        'axes.axisbelow': True,
        
        # 网格
        'grid.color': '#E5E5E5',
        'grid.linewidth': 0.5,
        'grid.alpha': 0.7,
        
        # 刻度 - 1.5倍
        'xtick.labelsize': 13.5,  # 9 * 1.5
        'ytick.labelsize': 13.5,  # 9 * 1.5
        'xtick.color': '#333333',
        'ytick.color': '#333333',
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        
        # 图例 - 1.5倍
        'legend.fontsize': 13.5,  # 9 * 1.5
        'legend.frameon': False,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '#CCCCCC',
        'legend.fancybox': False,
        
        # 图形
        'figure.facecolor': 'white',
        'figure.edgecolor': 'white',
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.facecolor': 'white',
        'savefig.edgecolor': 'white',
        
        # 线条
        'lines.linewidth': 2.0,
        'lines.markersize': 9,
        
        # 负号
        'axes.unicode_minus': False,
    })

def get_lancet_colors(n=2):
    """获取n个柳叶刀风格颜色"""
    if n == 2:
        return [TREATMENT_COLOR, CONTROL_COLOR]
    return LANCET_PALETTE[:n]

def format_pvalue(p):
    """格式化P值显示"""
    if p < 0.001:
        return 'P<0.001'
    elif p < 0.01:
        return f'P={p:.3f}'
    elif p < 0.05:
        return f'P={p:.3f}'
    else:
        return f'P={p:.2f}'

def add_significance_bar(ax, x1, x2, y, p_value, height=0.02):
    """
    添加显著性标注横线
    """
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    bar_height = y_range * height
    
    # 画横线
    ax.plot([x1, x1, x2, x2], [y, y + bar_height, y + bar_height, y], 
            color='#333333', linewidth=1)
    
    # 添加P值或星号
    if p_value < 0.001:
        sig_text = '***'
    elif p_value < 0.01:
        sig_text = '**'
    elif p_value < 0.05:
        sig_text = '*'
    else:
        sig_text = 'ns'
    
    ax.text((x1 + x2) / 2, y + bar_height, sig_text, 
            ha='center', va='bottom', fontsize=10, fontweight='bold')

def create_lancet_boxplot(ax, data_list, labels, colors=None, show_points=True):
    """
    创建柳叶刀风格的箱式图
    
    Parameters:
    -----------
    ax : matplotlib axes
    data_list : list of arrays
    labels : list of str
    colors : list of colors
    show_points : bool, 是否显示散点
    """
    if colors is None:
        colors = get_lancet_colors(len(data_list))
    
    bp = ax.boxplot(data_list, patch_artist=True, labels=labels,
                    widths=0.6,
                    medianprops=dict(color='white', linewidth=2),
                    whiskerprops=dict(color='#333333', linewidth=1),
                    capprops=dict(color='#333333', linewidth=1),
                    flierprops=dict(marker='o', markerfacecolor='#333333', 
                                   markersize=4, alpha=0.5))
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('#333333')
        patch.set_linewidth(1)
    
    # 添加抖动散点
    if show_points:
        for i, (data, color) in enumerate(zip(data_list, colors)):
            # 添加抖动
            x = np.random.normal(i + 1, 0.04, len(data))
            ax.scatter(x, data, alpha=0.4, s=15, color=color, 
                      edgecolors='none', zorder=3)
    
    return bp

def create_lancet_barplot(ax, x, heights, yerr=None, labels=None, colors=None, 
                          width=0.35, show_values=True):
    """
    创建柳叶刀风格的柱状图
    """
    if colors is None:
        colors = get_lancet_colors(len(heights) if isinstance(heights[0], (list, np.ndarray)) else 2)
    
    bars = ax.bar(x, heights, width, yerr=yerr, 
                  color=colors[0] if not isinstance(heights[0], (list, np.ndarray)) else colors,
                  edgecolor='#333333', linewidth=0.8,
                  capsize=3, error_kw={'linewidth': 1, 'capthick': 1})
    
    if show_values and yerr is None:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    return bars

def create_lancet_forest_plot(ax, names, effects, ci_lower, ci_upper, 
                               null_value=0, colors=None):
    """
    创建柳叶刀风格的森林图
    """
    n = len(names)
    y_pos = np.arange(n)
    
    if colors is None:
        colors = []
        for i, (eff, ci_l, ci_u) in enumerate(zip(effects, ci_lower, ci_upper)):
            if ci_u < null_value:
                colors.append('#42B540')  # 优效 - 绿色
            elif ci_l > null_value:
                colors.append('#ED0000')  # 劣效 - 红色
            else:
                colors.append('#00468B')  # 无差异 - 蓝色
    
    for i, (y, eff, ci_l, ci_u, color) in enumerate(zip(y_pos, effects, ci_lower, ci_upper, colors)):
        # 置信区间线
        ax.plot([ci_l, ci_u], [y, y], color=color, linewidth=2, zorder=2)
        # 端点
        ax.plot([ci_l, ci_u], [y, y], '|', color=color, markersize=8, zorder=3)
        # 点估计
        ax.plot(eff, y, 'D', color=color, markersize=8, zorder=4)
    
    # 参考线
    ax.axvline(x=null_value, color='#333333', linestyle='-', linewidth=1, zorder=1)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    
    return ax

def save_lancet_figure(fig, filepath, dpi=300):
    """保存柳叶刀风格图表"""
    import matplotlib
    # 使用Agg后端避免PIL冲突
    backend = matplotlib.get_backend()
    try:
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight', 
                    facecolor='white', edgecolor='none', format='png')
    except Exception as e:
        print(f"保存图片警告: {e}")
        # 尝试使用plt.savefig
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
    finally:
        plt.close(fig)


# 初始化样式
setup_lancet_style()
