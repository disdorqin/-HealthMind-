"""
HealthMind 健康风险管理平台

基于 Streamlit + Pyecharts 的健康风险预测与管理系统
功能：
1. 多维度健康风险趋势曲线（24 小时/7 天）
2. 目标管理进度条（步数/睡眠）
3. SHAP 归因瀑布图
4. 健康积分排行榜
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pyecharts import options as opts
from pyecharts.charts import Line, Gauge, Bar, Pie, Radar, Graph
import json

def st_pyecharts(chart, height="400px", key=None):
    """将 pyecharts 图表转换为 streamlit-echarts 格式并显示"""
    from streamlit_echarts import st_echarts
    chart_config = json.loads(chart.dump_options())
    return st_echarts(options=chart_config, height=height, key=key)

# 后端服务
# 注意：src/models.py 与 src/models/ 目录存在命名冲突
# 直接从服务层导入所需组件
from src.services.service_layer import (
    HealthMindService, SHAPExplainer, DecisionEngine, 
    HealthPointsSystem, PredictionResult, RiskFactor
)

# ==================== 页面配置 ====================

st.set_page_config(
    page_title="HealthMind 健康风险管理",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 自定义样式 ====================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@300;400;500;700&display=swap');
    
    :root {
        --primary-color: #E53935;
        --secondary-color: #EF5350;
        --accent-color: #FFA726;
        --success-color: #66BB6A;
        --warning-color: #FFA726;
        --background-color: #FFEBEE;
        --card-bg: #FFFFFF;
        --text-color: #B71C1C;
    }
    
    * {
        font-family: 'Noto Sans SC', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(180deg, #FFEBEE 0%, #FCE4EC 100%);
    }
    
    h1, h2, h3 {
        color: var(--text-color) !important;
        font-weight: 600;
    }
    
    /* 卡片样式 */
    .health-card {
        background: var(--card-bg);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(229, 57, 53, 0.1);
        border-left: 4px solid var(--primary-color);
        margin: 1rem 0;
    }
    
    .stat-card {
        background: linear-gradient(135deg, #EF5350, #E53935);
        color: white;
        border-radius: 16px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(229, 57, 53, 0.2);
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
    }
    
    .stat-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* 进度条样式 */
    .progress-container {
        background: #f0f0f0;
        border-radius: 10px;
        height: 24px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    
    .progress-bar {
        height: 100%;
        border-radius: 10px;
        transition: width 0.3s ease;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 600;
        font-size: 0.85rem;
    }
    
    .progress-success { background: linear-gradient(90deg, #66BB6A, #43A047); }
    .progress-warning { background: linear-gradient(90deg, #FFA726, #FB8C00); }
    .progress-danger { background: linear-gradient(90deg, #EF5350, #E53935); }
    
    /* 排行榜样式 */
    .rank-table {
        width: 100%;
        border-collapse: collapse;
        border-radius: 12px;
        overflow: hidden;
    }
    
    .rank-table th {
        background: linear-gradient(135deg, #E53935, #EF5350);
        color: white;
        padding: 12px;
        text-align: center;
    }
    
    .rank-table td {
        padding: 10px;
        text-align: center;
        border-bottom: 1px solid #ffebee;
    }
    
    .rank-table tr:nth-child(1) td { background: linear-gradient(90deg, #FFF8E1, #FFFFFF); }
    .rank-table tr:nth-child(2) td { background: linear-gradient(90deg, #F5F5F5, #FFFFFF); }
    .rank-table tr:nth-child(3) td { background: linear-gradient(90deg, #FFF3E0, #FFFFFF); }
    
    .medal { font-size: 1.3rem; }
    
    /* 风险等级标签 */
    .risk-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    .risk-low { background: #E8F5E9; color: #2E7D32; }
    .risk-medium { background: #FFF3E0; color: #EF6C00; }
    .risk-high { background: #FFEBEE; color: #C62828; }
    
    /* SHAP 瀑布图容器 */
    .shap-container {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ==================== 初始化服务 ====================

@st.cache_resource
def get_health_service():
    """初始化 HealthMind 服务"""
    try:
        service = HealthMindService()
        return service
    except Exception as e:
        st.error(f"服务初始化失败：{e}")
        return None


@st.cache_resource
def get_sample_data():
    """生成示例健康数据"""
    np.random.seed(42)
    
    # 24 小时风险趋势
    hours = list(range(24))
    cardio_risk = [0.3 + 0.15 * np.sin((h - 6) * np.pi / 12) + np.random.normal(0, 0.05) 
                   for h in hours]
    bp_risk = [0.25 + 0.2 * np.sin((h - 8) * np.pi / 12) + np.random.normal(0, 0.04) 
               for h in hours]
    glucose_risk = [0.2 + 0.25 * np.sin((h - 12) * np.pi / 12) + np.random.normal(0, 0.03) 
                    for h in hours]
    
    # 7 天风险趋势
    days = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
    week_cardio = [0.35, 0.32, 0.38, 0.34, 0.41, 0.29, 0.31]
    week_bp = [0.30, 0.28, 0.33, 0.31, 0.36, 0.25, 0.27]
    
    # SHAP 归因数据
    shap_data = {
        '收缩压 (ap_hi)': 0.18,
        '舒张压 (ap_lo)': 0.12,
        '年龄': 0.08,
        'BMI': 0.06,
        '血糖': 0.04,
        '胆固醇': 0.03,
        '吸烟': 0.02,
        '饮酒': 0.01,
        '运动': -0.05,
        '基础风险': 0.15
    }
    
    return {
        'hours': hours,
        'cardio_risk': cardio_risk,
        'bp_risk': bp_risk,
        'glucose_risk': glucose_risk,
        'days': days,
        'week_cardio': week_cardio,
        'week_bp': week_bp,
        'shap_data': shap_data
    }


# ==================== 可视化组件 ====================

def render_risk_gauge(value: float, title: str, max_val: float = 1.0) -> None:
    """渲染风险仪表盘"""
    if value < 0.3:
        color = "#66BB6A"
        risk_level = "低风险"
    elif value < 0.6:
        color = "#FFA726"
        risk_level = "中风险"
    else:
        color = "#EF5350"
        risk_level = "高风险"
    
    gauge = (
        Gauge()
        .add(
            "",
            [("当前风险", value)],
            min_=0,
            max_=max_val,
            axisline_opts=opts.AxisLineOpts(
                linestyle_opts=opts.LineStyleOpts(
                    color=[
                        [0.3, "#66BB6A"],
                        [0.6, "#FFA726"],
                        [1, "#EF5350"]
                    ],
                    width=10
                )
            ),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title=f"{title}",
                subtitle=f"{risk_level} ({value:.1%})",
            )
        )
    )
    st_pyecharts(gauge, height="220px", key=f"gauge_{title}")


def render_24h_risk_chart(data: dict) -> None:
    """渲染 24 小时风险趋势图"""
    line = (
        Line()
        .add_xaxis(data['hours'])
        .add_yaxis(
            "心血管风险",
            data['cardio_risk'],
            is_smooth=True,
            linestyle_opts=opts.LineStyleOpts(width=3),
            itemstyle_opts=opts.ItemStyleOpts(color="#E53935")
        )
        .add_yaxis(
            "血压风险",
            data['bp_risk'],
            is_smooth=True,
            linestyle_opts=opts.LineStyleOpts(width=3),
            itemstyle_opts=opts.ItemStyleOpts(color="#FFA726")
        )
        .add_yaxis(
            "血糖风险",
            data['glucose_risk'],
            is_smooth=True,
            linestyle_opts=opts.LineStyleOpts(width=3),
            itemstyle_opts=opts.ItemStyleOpts(color="#66BB6A")
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="📈 24 小时健康风险趋势"),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            legend_opts=opts.LegendOpts(pos_top="10%"),
            xaxis_opts=opts.AxisOpts(name="时间", type_="category"),
            yaxis_opts=opts.AxisOpts(name="风险概率", min_=0, max_=1)
        )
    )
    st_pyecharts(line, height="400px", key="24h_risk")


def render_week_risk_chart(data: dict) -> None:
    """渲染周风险趋势图"""
    bar = (
        Bar()
        .add_xaxis(data['days'])
        .add_yaxis(
            "心血管风险",
            data['week_cardio'],
            bar_width="35%",
            itemstyle_opts=opts.ItemStyleOpts(color="#E53935")
        )
        .add_yaxis(
            "血压风险",
            data['week_bp'],
            bar_width="35%",
            itemstyle_opts=opts.ItemStyleOpts(color="#FFA726")
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="📅 本周健康风险趋势"),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            legend_opts=opts.LegendOpts(pos_top="10%"),
            yaxis_opts=opts.AxisOpts(name="风险概率", min_=0, max_=1)
        )
    )
    st_pyecharts(bar, height="350px", key="week_risk")


def render_shap_waterfall(shap_data: dict) -> None:
    """渲染 SHAP 归因瀑布图"""
    # 排序并计算累积值
    sorted_items = sorted(shap_data.items(), key=lambda x: -abs(x[1]))
    labels = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]
    
    # 计算累积值
    base_risk = 0.15  # 基础风险
    cumulative = [base_risk]
    for v in values[:-1]:  # 排除基础风险
        cumulative.append(cumulative[-1] + v)
    
    # 创建瀑布图
    bar = (
        Bar()
        .add_xaxis(labels)
        .add_yaxis(
            "风险贡献",
            values,
            label_opts=opts.LabelOpts(position="top", formatter="{c}"),
            itemstyle_opts=opts.ItemStyleOpts(
                color=lambda x: "#EF5350" if x['value'] > 0 else "#66BB6A"
            )
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="🔍 SHAP 风险归因分析"),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            yaxis_opts=opts.AxisOpts(name="风险贡献值")
        )
    )
    st_pyecharts(bar, height="400px", key="shap_waterfall")


def render_goal_progress(current: float, target: float, unit: str, icon: str) -> None:
    """渲染目标进度条"""
    progress = min(1.0, current / target)
    percentage = int(progress * 100)
    
    if progress >= 1.0:
        color_class = "progress-success"
        status = "✅ 已达成"
    elif progress >= 0.7:
        color_class = "progress-warning"
        status = "⏳ 进行中"
    else:
        color_class = "progress-danger"
        status = "💪 加油"
    
    st.markdown(f"""
    <div style="margin: 1rem 0;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-size: 1.1rem;">{icon} 目标</span>
            <span style="color: #666;">{current:,} / {target:,} {unit}</span>
        </div>
        <div class="progress-container">
            <div class="progress-bar {color_class}" style="width: {percentage}%">
                {percentage}%
            </div>
        </div>
        <div style="text-align: right; font-size: 0.9rem; color: #666;">{status}</div>
    </div>
    """, unsafe_allow_html=True)


def render_leaderboard() -> None:
    """渲染健康积分排行榜"""
    # 示例数据
    leaderboard_data = [
        {"rank": 1, "name": "健康达人", "points": 2580, "level": "🏆 铂金"},
        {"rank": 2, "name": "运动先锋", "points": 1890, "level": "🥈 黄金"},
        {"rank": 3, "name": "早睡早起", "points": 1250, "level": "🥉 白银"},
        {"rank": 4, "name": "当前用户", "points": 850, "level": "🌿 青铜", "is_current": True},
        {"rank": 5, "name": "减脂小队", "points": 620, "level": "🌿 青铜"},
    ]
    
    html = '<table class="rank-table"><thead><tr><th>排名</th><th>用户</th><th>积分</th><th>等级</th></tr></thead><tbody>'
    
    for user in leaderboard_data:
        medal = ["🥇", "🥈", "🥉", "", ""][user['rank']-1] if user['rank'] <= 3 else f"#{user['rank']}"
        row_style = 'style="background: linear-gradient(90deg, #E8F5E9, #FFFFFF); font-weight: bold;"' if user.get('is_current') else ''
        html += f'<tr {row_style}><td class="medal">{medal}</td><td>{user["name"]}</td><td>{user["points"]}</td><td>{user["level"]}</td></tr>'
    
    html += '</tbody></table>'
    st.markdown(html, unsafe_allow_html=True)


# ==================== 主应用 ====================

def main():
    # 初始化服务
    service = get_health_service()
    data = get_sample_data()
    
    # 侧边栏
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/heart-health.png", width=80)
        st.title("HealthMind")
        st.caption("❤️ 健康风险管理平台")
        
        st.markdown("---")
        
        # 用户信息
        st.subheader("👤 用户信息")
        st.markdown("""
        <div style="background: linear-gradient(135deg, #EF5350, #E53935); color: white; padding: 1rem; border-radius: 12px;">
            <div style="font-size: 1.2rem; font-weight: 600;">张先生</div>
            <div style="font-size: 0.9rem; opacity: 0.9;">ID: HM2024001</div>
            <div style="font-size: 0.9rem; opacity: 0.9; margin-top: 0.5rem;">
                📅 45 岁 | ♂️ 男性 | 📍 上海
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # 健康积分
        st.subheader("🏆 健康积分")
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <div style="font-size: 2.5rem; font-weight: 700; color: #E53935;">850</div>
            <div style="color: #666;">总积分</div>
            <div style="margin-top: 0.5rem;">
                <span class="risk-badge" style="background: #FFF3E0; color: #EF6C00;">🌿 青铜</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 进度
        next_level = 1000
        progress = 850 / next_level
        st.progress(progress)
        st.caption(f"距离黄金等级还需 {next_level - 850} 积分")
        
        st.markdown("---")
        
        # 排行榜
        st.subheader("📊 积分排行榜")
        render_leaderboard()
        
        st.markdown("---")
        
        # 快捷操作
        st.subheader("⚡ 快捷操作")
        if st.button("📝 记录今日数据", use_container_width=True):
            st.session_state['show_record_modal'] = True
        if st.button("📋 查看健康报告", use_container_width=True):
            st.session_state['show_report'] = True
        if st.button("💡 获取健康建议", use_container_width=True):
            st.session_state['show_tips'] = True
    
    # 主内容区
    st.title("❤️ HealthMind 健康风险管理")
    st.markdown("基于 AI 的心血管疾病风险预测与健康管理")
    
    # 核心指标卡片
    cols = st.columns(4)
    with cols[0]:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{0.42:.1%}</div>
            <div class="stat-label">当前风险</div>
        </div>
        """, unsafe_allow_html=True)
    with cols[1]:
        st.markdown(f"""
        <div class="stat-card" style="background: linear-gradient(135deg, #FFA726, #FB8C00);">
            <div class="stat-value">{145/180:.0%}</div>
            <div class="stat-label">步数目标</div>
        </div>
        """, unsafe_allow_html=True)
    with cols[2]:
        st.markdown(f"""
        <div class="stat-card" style="background: linear-gradient(135deg, #66BB6A, #43A047);">
            <div class="stat-value">{6.5/8:.0%}</div>
            <div class="stat-label">睡眠质量</div>
        </div>
        """, unsafe_allow_html=True)
    with cols[3]:
        st.markdown(f"""
        <div class="stat-card" style="background: linear-gradient(135deg, #AB47BC, #8E24AA);">
            <div class="stat-value">+35</div>
            <div class="stat-label">今日积分</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 选项卡
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 风险趋势",
        "🎯 目标管理",
        "🔍 风险归因",
        "💡 健康建议"
    ])
    
    # Tab 1: 风险趋势
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            render_24h_risk_chart(data)
        
        with col2:
            st.subheader("实时风险")
            render_risk_gauge(0.42, "心血管")
            render_risk_gauge(0.35, "高血压")
            render_risk_gauge(0.28, "高血糖")
        
        st.markdown("---")
        render_week_risk_chart(data)
    
    # Tab 2: 目标管理
    with tab2:
        st.subheader("🎯 每日健康目标")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="health-card">
                <h3 style="margin-top: 0;">🚶 步数目标</h3>
            </div>
            """, unsafe_allow_html=True)
            render_goal_progress(14500, 18000, "步", "🚶")
            
            # 步数趋势
            step_data = [8000, 12000, 10000, 15000, 11000, 13000, 14500]
            step_chart = (
                Line()
                .add_xaxis(['周一', '周二', '周三', '周四', '周五', '周六', '周日'])
                .add_yaxis(
                    "每日步数",
                    step_data,
                    is_smooth=True,
                    markline_opts=opts.MarkLineOpts(
                        data=[opts.MarkLineItem(y=10000, name="目标线")]
                    )
                )
                .set_global_opts(
                    title_opts=opts.TitleOpts(title="周步数趋势"),
                    yaxis_opts=opts.AxisOpts(name="步数")
                )
            )
            st_pyecharts(step_chart, height="300px", key="steps")
        
        with col2:
            st.markdown("""
            <div class="health-card">
                <h3 style="margin-top: 0;">😴 睡眠目标</h3>
            </div>
            """, unsafe_allow_html=True)
            render_goal_progress(6.5, 8, "小时", "😴")
            
            # 睡眠趋势
            sleep_data = [7.5, 6.0, 8.0, 5.5, 7.0, 8.5, 6.5]
            sleep_chart = (
                Line()
                .add_xaxis(['周一', '周二', '周三', '周四', '周五', '周六', '周日'])
                .add_yaxis(
                    "睡眠时长",
                    sleep_data,
                    is_smooth=True,
                    markline_opts=opts.MarkLineOpts(
                        data=[opts.MarkLineItem(y=8, name="目标线")]
                    ),
                    areastyle_opts=opts.AreaStyleOpts(opacity=0.3)
                )
                .set_global_opts(
                    title_opts=opts.TitleOpts(title="周睡眠趋势"),
                    yaxis_opts=opts.AxisOpts(name="小时", min_=0, max_=12)
                )
            )
            st_pyecharts(sleep_chart, height="300px", key="sleep")
        
        st.markdown("---")
        
        # 更多目标
        st.subheader("📋 其他健康目标")
        
        goal_cols = st.columns(3)
        with goal_cols[0]:
            render_goal_progress(5, 7, "份", "🥗")
            st.caption("蔬菜摄入")
        
        with goal_cols[1]:
            render_goal_progress(1800, 2500, "ml", "💧")
            st.caption("饮水量")
        
        with goal_cols[2]:
            render_goal_progress(30, 60, "分钟", "🧘")
            st.caption("冥想时间")
    
    # Tab 3: 风险归因
    with tab3:
        st.subheader("🔍 SHAP 风险归因分析")
        
        st.markdown("""
        <div class="shap-container">
            <p>通过 SHAP 值分析，量化各因素对今日健康风险的贡献度。</p>
            <p>红色柱表示增加风险的因素，绿色柱表示降低风险的保护因素。</p>
        </div>
        """, unsafe_allow_html=True)
        
        render_shap_waterfall(data['shap_data'])
        
        st.markdown("---")
        
        # 风险因素详情
        st.subheader("📊 风险因素详情")
        
        factor_data = pd.DataFrame({
            '因素': ['收缩压', '舒张压', '年龄', 'BMI', '血糖', '胆固醇'],
            '当前值': [145, 92, 45, 26.5, 108, 2.1],
            '正常范围': ['90-120', '60-80', '-', '18.5-24', '70-100', '<2.0'],
            '单位': ['mmHg', 'mmHg', '岁', '', 'mg/dL', 'mmol/L'],
            '风险等级': ['⚠️ 偏高', '⚠️ 偏高', '✅ 正常', '⚠️ 偏高', '⚠️ 偏高', '✅ 正常']
        })
        
        st.dataframe(
            factor_data,
            use_container_width=True,
            hide_index=True
        )
        
        # 归因饼图
        pie_data = [
            ('收缩压', 0.18),
            ('舒张压', 0.12),
            ('年龄', 0.08),
            ('BMI', 0.06),
            ('血糖', 0.04),
            ('其他', 0.07)
        ]
        
        pie = (
            Pie()
            .add(
                "",
                pie_data,
                radius=["40%", "70%"]
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(title="风险贡献分布"),
                legend_opts=opts.LegendOpts(pos_top="5%", pos_left="5%")
            )
            .set_series_opts(
                label_opts=opts.LabelOpts(formatter="{b}: {c} ({d}%)")
            )
        )
        st_pyecharts(pie, height="400px", key="risk_pie")
    
    # Tab 4: 健康建议
    with tab4:
        st.subheader("💡 个性化健康建议")
        
        # 风险等级提示
        risk_level = "中风险"
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #FFF3E0, #FFE0B2); border-left: 4px solid #FFA726; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
            <h3 style="margin: 0;">⚠️ 当前风险等级：{risk_level}</h3>
            <p style="margin: 0.5rem 0 0 0;">您的心血管风险处于中等水平，建议采取以下干预措施。</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 建议列表
        st.subheader("📋 干预建议清单")
        
        recommendations = [
            {
                'priority': 1,
                'factor': '血压管理',
                'action': '减少盐分摄入',
                'description': '每日盐摄入量控制在 6g 以下，避免腌制食品',
                'expected_effect': '预期风险降低 5%'
            },
            {
                'priority': 2,
                'factor': '血压管理',
                'action': '增加有氧运动',
                'description': '每周至少 150 分钟中等强度有氧运动',
                'expected_effect': '预期风险降低 8%'
            },
            {
                'priority': 3,
                'factor': '体重管理',
                'action': '控制热量摄入',
                'description': '每日减少 500 大卡，优先选择低 GI 食物',
                'expected_effect': '预期风险降低 6%'
            },
            {
                'priority': 4,
                'factor': '生活方式',
                'action': '增加每日步数',
                'description': '目标每日 10000 步，可使用计步器追踪',
                'expected_effect': '预期风险降低 4%'
            }
        ]
        
        for i, rec in enumerate(recommendations):
            with st.expander(f"{'🔴' if rec['priority'] <= 2 else '🟡'} 建议 {i+1}: {rec['factor']} - {rec['action']}", expanded=(i < 2)):
                st.markdown(f"""
                **具体行动**: {rec['description']}
                
                **预期效果**: {rec['expected_effect']}
                
                <div style="display: flex; justify-content: space-between; margin-top: 1rem;">
                    <button style="background: #66BB6A; color: white; border: none; padding: 8px 16px; border-radius: 6px; cursor: pointer;">✅ 标记完成</button>
                    <button style="background: #E53935; color: white; border: none; padding: 8px 16px; border-radius: 6px; cursor: pointer;">📝 记录进展</button>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # 每日小贴士
        st.subheader("🌟 每日健康小贴士")
        
        tips = [
            "🥗 多吃富含钾的食物（如香蕉、土豆）有助于控制血压",
            "💧 保持充足水分，每日饮水 2000-2500ml",
            "😴 保证 7-8 小时优质睡眠，有助于心血管健康",
            "🚶 饭后散步 15 分钟，有助于血糖控制",
            "🧘 每日冥想 10 分钟，降低压力对心脏的影响"
        ]
        
        for tip in tips:
            st.markdown(f'<div style="background: #E8F5E9; padding: 0.8rem; border-radius: 8px; margin: 0.5rem 0;">{tip}</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # 积分奖励
        st.subheader("🏆 完成建议获得积分")
        
        point_cols = st.columns(4)
        points = [
            ("完成每日步数目标", "+20 分"),
            ("达成睡眠目标", "+15 分"),
            ("记录饮食数据", "+10 分"),
            ("连续 7 天打卡", "+50 分")
        ]
        
        for i, (action, points_val) in enumerate(points):
            point_cols[i].markdown(f"""
            <div style="text-align: center; padding: 1rem; background: white; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                <div style="font-size: 1.5rem; color: #E53935;">{points_val}</div>
                <div style="font-size: 0.9rem; color: #666;">{action}</div>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()