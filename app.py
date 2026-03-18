import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
from pyecharts import options as opts
from pyecharts.charts import Line, Gauge, Bar, Radar
from streamlit_echarts import st_pyecharts

# Backend Services
from src.services.prediction_service import PredictionService
from src.services.carbon_engine import CarbonEngine
from src.core.config.config_manager import get_config
from src.services.user_api_client import UserAPIClient
from src.core.utils.logger import logger

# 1. Page Configuration
st.set_page_config(
    page_title="EcoLife - 个人碳足迹管理",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS (Eco Green Theme - Enhanced)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@300;400;500;700&display=swap');
    
    :root {
        --primary-color: #2E7D32;
        --secondary-color: #66BB6A;
        --accent-color: #FFA726;
        --background-color: #F1F8E9;
        --card-bg: #FFFFFF;
        --text-color: #1B5E20;
        --text-secondary: #666;
    }
    
    * {
        font-family: 'Noto Sans SC', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(180deg, #F1F8E9 0%, #E8F5E9 100%);
    }
    
    h1, h2, h3 {
        color: var(--text-color) !important;
        font-weight: 600;
    }
    
    /* Navigation Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #E8F5E9;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #66BB6A, #2E7D32);
        color: white;
    }
    
    /* Cards */
    .eco-card {
        background: var(--card-bg);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin: 1rem 0;
    }
    .eco-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    }
    
    /* Tip Cards */
    .tip-card {
        background: linear-gradient(135deg, #E8F5E9, #C8E6C9);
        border-left: 4px solid var(--primary-color);
        padding: 1rem 1.2rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        transition: transform 0.2s ease;
    }
    .tip-card:hover {
        transform: translateX(4px);
    }
    
    /* Diet Cards */
    .diet-card {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin: 0.5rem 0;
        transition: all 0.2s ease;
    }
    .diet-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
        transform: translateY(-2px);
    }
    
    /* Rank Table */
    .rank-table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
        border-radius: 12px;
        overflow: hidden;
    }
    .rank-table th {
        background: linear-gradient(135deg, #2E7D32, #66BB6A);
        color: white;
        padding: 12px;
        text-align: center;
        font-weight: 500;
    }
    .rank-table td {
        padding: 12px;
        text-align: center;
        border-bottom: 1px solid #E0E0E0;
    }
    .rank-table tr:nth-child(1) td { background: linear-gradient(90deg, #FFF8E1, #FFFFFF); }
    .rank-table tr:nth-child(2) td { background: linear-gradient(90deg, #F5F5F5, #FFFFFF); }
    .rank-table tr:nth-child(3) td { background: linear-gradient(90deg, #FFF3E0, #FFFFFF); }
    .rank-table tr.current-user td {
        background: linear-gradient(90deg, #E8F5E9, #FFFFFF) !important;
        font-weight: 600;
    }
    
    /* Countdown */
    .countdown {
        font-size: 2.5rem;
        font-weight: 700;
        color: #FFA726;
        text-align: center;
        padding: 0.5rem;
    }
    
    /* Achievement Cards */
    .achievement-card {
        background: white;
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin: 0.5rem 0;
        text-align: center;
        transition: all 0.2s ease;
    }
    .achievement-card:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 15px rgba(0,0,0,0.12);
    }
    
    /* User Info Display */
    .user-info-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    .user-info-item {
        background: linear-gradient(135deg, #F1F8E9, #E8F5E9);
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
    }
    .user-info-label {
        font-size: 0.85rem;
        color: var(--text-secondary);
        margin-bottom: 0.3rem;
    }
    .user-info-value {
        font-size: 1.2rem;
        font-weight: 600;
        color: var(--primary-color);
    }
</style>
""", unsafe_allow_html=True)

# 3. Initialize Services (Singleton-ish in Streamlit via Cache)
@st.cache_resource
def get_services():
    try:
        predictor = PredictionService()
        predictor.load_models()
        engine = CarbonEngine(baseline_kg=12.5)
        return predictor, engine
    except Exception as e:
        st.error(f"Failed to load services: {e}")
        return None, None

predictor, engine = get_services()


@st.cache_resource
def get_user_client():
    config = get_config()
    base_url = f"http://{config.api_host}:{config.api_port}"
    return UserAPIClient(base_url=base_url)


def _format_user_label(user_data):
    if not user_data:
        return "未登录"
    return f"{user_data.get('username', '')} ({user_data.get('user_id', '')})"


def _age_stage(age):
    if age is None:
        return "未知"
    if age < 18:
        return "青少年"
    if age < 30:
        return "青年"
    if age < 45:
        return "中青年"
    if age < 60:
        return "中年"
    return "银龄"


def _personalized_plan(user_data):
    profile = user_data.get("profile", {}) or {}
    age = profile.get("age")
    gender = str(profile.get("gender") or "").strip()
    diet_plan = ["早餐：燕麦 + 无糖豆浆 + 水果", "午餐：时蔬沙拉 + 糙米 + 豆腐", "晚餐：番茄意面 + 清炒蔬菜"]
    tips = ["优先选择公共交通和步行，持续减少通勤碳排放。"]

    if isinstance(age, int):
        if age < 18:
            diet_plan = ["早餐：全麦面包 + 牛奶 + 鸡蛋", "午餐：番茄牛肉饭 + 青菜", "晚餐：鱼类 + 杂粮 + 蔬菜"]
            tips.append("青少年更适合均衡能量摄入，避免极端节食。")
        elif age < 30:
            diet_plan = ["早餐：酸奶杯 + 燕麦 + 香蕉", "午餐：轻食沙拉 + 鸡胸肉", "晚餐：低油高蛋白便当"]
            tips.append("青年群体适合高蛋白、低油脂的轻负担餐单。")
        elif age < 45:
            diet_plan = ["早餐：豆浆 + 鸡蛋 + 全麦", "午餐：杂粮饭 + 时蔬 + 豆制品", "晚餐：蒸鱼 + 绿叶菜"]
            tips.append("中青年建议控制外卖频次，减少高碳加工食品。")
        elif age < 60:
            diet_plan = ["早餐：小米粥 + 鸡蛋羹", "午餐：低盐家常菜 + 杂粮", "晚餐：清蒸蔬菜 + 豆腐汤"]
            tips.append("中年阶段建议规律饮食与适度步行相结合。")
        else:
            diet_plan = ["早餐：粥类 + 鸡蛋 + 豆浆", "午餐：少油少盐的家常菜", "晚餐：清淡蔬菜 + 豆制品"]
            tips.append("银龄用户建议少油少盐，并保持轻量运动。")

    if gender in {"女", "女性", "female", "f"}:
        tips.append("女性用户可优先选择补铁、钙质更丰富的低碳餐单。")
    elif gender in {"男", "男性", "male", "m"}:
        tips.append("男性用户可增加高蛋白低脂餐比例，兼顾饱腹与减碳。")

    return {
        "age_stage": _age_stage(age if isinstance(age, int) else None),
        "diet_plan": diet_plan,
        "tips": tips,
    }


def _render_auth_page(user_client):
    st.markdown(
        """
        <style>
            .auth-hero {
                background: linear-gradient(135deg, rgba(46,125,50,0.92), rgba(129,199,132,0.9));
                color: white;
                border-radius: 24px;
                padding: 2.2rem 2rem;
                box-shadow: 0 18px 40px rgba(0,0,0,0.12);
                margin-bottom: 1.5rem;
            }
            .auth-card {
                background: rgba(255,255,255,0.88);
                backdrop-filter: blur(10px);
                border-radius: 20px;
                padding: 1.5rem 1.5rem 1rem 1.5rem;
                box-shadow: 0 12px 30px rgba(0,0,0,0.08);
                border: 1px solid rgba(46,125,50,0.12);
            }
            .auth-pill {
                display: inline-block;
                padding: 0.35rem 0.8rem;
                border-radius: 999px;
                background: rgba(255,255,255,0.18);
                margin-right: 0.5rem;
                margin-bottom: 0.5rem;
                font-size: 0.9rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="auth-hero">
            <h1 style="margin:0 0 0.4rem 0;">🌿 EcoLife 个人碳足迹管理</h1>
            <p style="margin:0 0 1rem 0;font-size:1.05rem;">先登录或注册，再进入你的个性化碳管理仪表盘。</p>
            <span class="auth-pill">年龄与性别个性化建议</span>
            <span class="auth-pill">前后端分离 API</span>
            <span class="auth-pill">本地 JSON 存储</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    auth_col1, auth_col2 = st.columns([1.15, 0.85], gap="large")

    with auth_col1:
        st.markdown('<div class="auth-card">', unsafe_allow_html=True)
        st.subheader("👤 首次注册 / 登录")
        st.caption("首次注册请补全基本资料，系统会据此生成更准确的减碳建议和饮食计划。")

        with st.form("auth_form", clear_on_submit=False):
            username = st.text_input("姓名 / 昵称", placeholder="例如：张三")
            custom_user_id = st.text_input("用户 ID（可选）", placeholder="留空自动生成")
            age = st.slider("年龄", min_value=6, max_value=90, value=24, step=1)
            gender = st.selectbox("性别", ["男", "女", "其他"], index=0)
            email = st.text_input("邮箱", placeholder="name@example.com")
            address = st.text_input("住址 / 常住城市", placeholder="例如：上海市浦东新区")
            submit = st.form_submit_button("注册并进入仪表盘", use_container_width=True)

        if submit:
            if not username.strip():
                st.error("用户名不能为空")
            else:
                login_result = user_client.login(
                    username=username.strip(),
                    user_id=custom_user_id.strip() or None,
                    age=int(age),
                    gender=gender,
                    email=email.strip() or None,
                    address=address.strip() or None,
                )
                if login_result.get("success"):
                    st.session_state["current_user"] = login_result.get("data", {})
                    st.success(login_result.get("message", "登录成功"))
                    st.rerun()
                else:
                    st.error(login_result.get("message", "登录失败"))

        st.markdown('</div>', unsafe_allow_html=True)

    with auth_col2:
        st.markdown('<div class="auth-card">', unsafe_allow_html=True)
        st.subheader("✨ 登录后你可以获得")
        st.markdown(
            """
            - 个性化碳足迹仪表盘
            - 基于年龄/性别的饮食建议
            - 可保存的碳排放记录
            - 积分成长与历史趋势
            - 前后端分离 API 调用能力
            """
        )
        st.info("提示：如果后端 API 未启动，系统会自动使用本地 JSON 存储模式，仍可完成演示。")
        st.markdown('</div>', unsafe_allow_html=True)


current_user = st.session_state.get("current_user")
if not current_user:
    _render_auth_page(get_user_client())
    st.stop()

# --- Sidebar: Carbon Budget & Profile ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/leaf.png", width=80)
    st.title("EcoLife 碳管理")
    st.caption(_format_user_label(current_user))
    profile = current_user.get("profile", {}) or {}
    profile_cols = st.columns(2)
    profile_cols[0].metric("年龄", profile.get("age", "-"))
    profile_cols[1].metric("性别", profile.get("gender", "-"))

    if st.button("退出登录", use_container_width=True):
        st.session_state.clear()
        st.rerun()
    
    st.markdown("---")
    st.subheader("📊 碳预算设置")
    budget = st.slider("本月碳排放预算 (kg)", 200, 600, 350)
    
    current_usage = 210.5 
    remaining = budget - current_usage
    percent_used = (current_usage / budget) * 100
    
    st.metric("本月已用", f"{current_usage} kg", delta=f"{remaining:.1f} kg 剩余", delta_color="normal")
    
    if percent_used > 80:
        st.warning(f"⚠️ 警告：已使用 {percent_used:.1f}% 预算！")
    else:
        st.success(f"✅ 状态良好：使用率 {percent_used:.1f}%")
        
    st.markdown("---")
    st.info("💡 每日小贴士：乘坐公共交通可减少约 2.6kg 碳排放。")

# --- Main Area ---
st.title("🌿 EcoLife 个人环境足迹仪表盘")

profile = current_user.get("profile", {}) or {}
summary_cols = st.columns(4)
summary_cols[0].metric("用户名", current_user.get("username", "-"))
summary_cols[1].metric("年龄阶段", _age_stage(profile.get("age") if isinstance(profile.get("age"), int) else None))
summary_cols[2].metric("总积分", current_user.get("total_credits", 0))
summary_cols[3].metric("记录数", current_user.get("record_count", 0))

with st.container():
    st.markdown(
        f"""
        <div style="background: rgba(255,255,255,0.88); padding: 1.5rem; border-radius: 16px; margin: 1rem 0 1.2rem 0; box-shadow: 0 4px 15px rgba(0,0,0,0.08);">
            <h3 style="margin-top:0;">👋 欢迎回来，{current_user.get('username', '')}</h3>
            <p style="margin-bottom:0.35rem;">年龄：{profile.get('age', '-')} ｜ 性别：{profile.get('gender', '-')} ｜ 邮箱：{profile.get('email', '-')}</p>
            <p style="margin-bottom:0;">住址：{profile.get('address', '-')}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

user_features = {
    'Transport': 'Public', 
    'Vehicle Distance Km': 15, 
    'Diet Type': 'Omnivore',
    'Heating': 'Gas'
}

# 4. Tabs for Functionality
tab1, tab2, tab3, tab4 = st.tabs(["📈 智能预测", "🥗 减碳计划", "🏆 碳积分荣誉", "👤 用户管理"])

# --- Tab 1: Prediction ---
with tab1:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("未来碳排放趋势预测")
        scale = st.radio("时间维度", ["Daily (天)", "Weekly (周)", "Monthly (月)"], horizontal=True)
        
        base_pred = 0.0
        if predictor:
            preds = predictor.predict_next_cycle("data/personal_carbon_footprint_behavior.csv") 
            base_pred = preds.get('xgboost', preds.get('ensemble_meta', 10.0))
        
        if "Day" in scale:
            x_data = [f"{i}:00" for i in range(24)]
            y_data = [max(0, base_pred/24 * (1 + 0.5*np.sin((i-12)/4))) for i in range(24)]
            title = "24 小时碳排放预测 (kg/h)"
        elif "Week" in scale:
            x_data = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            y_data = [max(0, base_pred + np.random.normal(0, 1)) for _ in range(7)]
            title = "未来 7 天碳排放预测 (kg/day)"
        else:
            x_data = [f"Week {i+1}" for i in range(4)]
            y_data = [max(0, base_pred * 7 + np.random.normal(0, 5)) for _ in range(4)]
            title = "未来一月碳排放预测 (kg/week)"

        c = (
            Line()
            .add_xaxis(x_data)
            .add_yaxis("预测排放量", y_data, is_smooth=True, 
                       itemstyle_opts=opts.ItemStyleOpts(color="#2E7D32"),
                       areastyle_opts=opts.AreaStyleOpts(opacity=0.3, color="#81C784"))
            .set_global_opts(
                title_opts=opts.TitleOpts(title=title),
                tooltip_opts=opts.TooltipOpts(trigger="axis"),
                yaxis_opts=opts.AxisOpts(name="排放量 (kg)"),
            )
        )
        st_pyecharts(c, height="400px")
        
    with col2:
        st.subheader("模型贡献度")
        if predictor:
            p_data = predictor.predict_next_cycle("data/personal_carbon_footprint_behavior.csv")
            radar_data = [[p_data.get('lstm', 0), p_data.get('xgboost', 0), p_data.get('moirai', 0)]]
            radar = (
                Radar()
                .add_schema(
                    schema=[
                        opts.RadarIndicatorItem(name="LSTM (时序)", max_=20),
                        opts.RadarIndicatorItem(name="XGBoost (特征)", max_=20),
                        opts.RadarIndicatorItem(name="Moirai (趋势)", max_=20),
                    ]
                )
                .add("模型预测值", radar_data, color="#1B5E20")
                .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
                .set_global_opts(title_opts=opts.TitleOpts(title="多模型融合视角"))
            )
            st_pyecharts(radar, height="300px")
            st.info(f"融合预测结果：{p_data.get('ensemble_meta', 0):.2f} kg")

# --- Tab 2: Diet & Reduction Plan ---
with tab2:
    personalized = _personalized_plan(current_user)
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #66BB6A, #2E7D32); color: white; padding: 1.5rem; border-radius: 16px; margin-bottom: 1.5rem; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
        <h2 style="margin:0;">🥗 个性化减碳饮食计划</h2>
        <p style="margin:0.5rem 0 0 0; opacity:0.9;">为{profile.get('username', '您')}定制的专属方案 · {personalized['age_stage']} · {profile.get('gender', '未知')}性</p>
    </div>
    """, unsafe_allow_html=True)
    
    col_a, col_b = st.columns([1.1, 0.9])
    
    with col_a:
        st.markdown("### 📅 今日建议")
        
        if engine:
            recs = engine.generate_recommendations(user_features)
            for i, rec in enumerate(recs):
                st.markdown(f'<div class="tip-card">✅ <b>建议 {i+1}:</b> {rec}</div>', unsafe_allow_html=True)
        
        for i, tip in enumerate(personalized["tips"]):
            icon = ["🌱", "💡", "🎯", "⭐"][i % 4]
            st.markdown(f'<div class="tip-card">{icon} {tip}</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 🥦 无肉日提醒")
        today = datetime.now()
        
        if today.weekday() == 0:
            st.markdown('<div class="tip-card" style="background: linear-gradient(135deg, #FFF3E0, #FFE0B2); border-left-color: #FF9800;">🔔 <b>今天是周一无肉日！</b><br>尝试素食可减少约 2.5kg 碳排放。</div>', unsafe_allow_html=True)
        else:
            days_until_monday = (7 - today.weekday()) % 7
            st.markdown(f'<div class="tip-card">📅 距离下个无肉日还有 <b>{days_until_monday}</b> 天</div>', unsafe_allow_html=True)
            
    with col_b:
        st.markdown("### 🍽️ 推荐食谱")
        
        diet_plan = personalized["diet_plan"]
        meal_icons = ["🌅", "☀️", "🌙"]
        meal_names = ["早餐", "午餐", "晚餐"]
        
        for i, meal in enumerate(diet_plan[:3]):
            icon = meal_icons[i] if i < len(meal_icons) else "🍽️"
            name = meal_names[i] if i < len(meal_names) else f"餐{i+1}"
            st.markdown(f'<div class="diet-card"><b>{icon} {name}</b><br>{meal}</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 👤 个人信息")
        if profile.get("age"):
            st.markdown(f"年龄：{profile.get('age')} 岁")
        if profile.get("gender"):
            st.markdown(f"性别：{profile.get('gender')}")
        if profile.get("address"):
            st.markdown(f"地区：{profile.get('address')}")

# --- Tab 3: Carbon Credits & Gamification ---
with tab3:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #FFD700, #FFA000); color: white; padding: 1.5rem; border-radius: 16px; margin-bottom: 1.5rem; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
        <h2 style="margin:0;">🏆 碳积分排行榜</h2>
        <p style="margin:0.5rem 0 0 0; opacity:0.9;">看看谁是最强环保达人！</p>
    </div>
    """, unsafe_allow_html=True)
    
    current_user_name = current_user.get('username', '未知用户')
    current_user_credits = current_user.get('total_credits', 0)
    
    all_users = [
        {"rank": 1, "name": "环保先锋", "credits": 2580, "level": "🌳 森林之王"},
        {"rank": 2, "name": "绿色达人", "credits": 1890, "level": "🌲 森林守护者"},
        {"rank": 3, "name": current_user_name, "credits": current_user_credits, "level": "🌿 环保新手"},
        {"rank": 4, "name": "低碳生活", "credits": max(0, current_user_credits - 200), "level": "🌱 环保新人"},
        {"rank": 5, "name": "减碳先锋", "credits": max(0, current_user_credits - 500), "level": "🌱 环保新人"},
    ]
    
    all_users.sort(key=lambda x: x['credits'], reverse=True)
    for i, user in enumerate(all_users):
        user['rank'] = i + 1
    
    rank_html = '<table class="rank-table"><thead><tr><th>排名</th><th>用户</th><th>总积分</th><th>等级</th></tr></thead><tbody>'
    
    for user in all_users:
        medal = ["🥇", "🥈", "🥉", "", ""][user['rank']-1] if user['rank'] <= 3 else f"#{user['rank']}"
        is_current = user['name'] == current_user_name
        row_style = 'style="background: linear-gradient(90deg, #E8F5E9, #FFFFFF); font-weight: bold;"' if is_current else ''
        rank_html += f'<tr {row_style}><td class="medal">{medal}</td><td>{user["name"]}</td><td>{user["credits"]}</td><td>{user["level"]}</td></tr>'
    
    rank_html += '</tbody></table>'
    st.markdown(rank_html, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <div style="background: linear-gradient(135deg, #66BB6A, #2E7D32); color: white; padding: 1.5rem; border-radius: 16px; margin-bottom: 1.5rem; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
        <h2 style="margin:0;">🎖️ 我的成就</h2>
        <p style="margin:0.5rem 0 0 0; opacity:0.9;">记录你的环保足迹</p>
    </div>
    """, unsafe_allow_html=True)
    
    achievements = []
    
    if current_user_credits >= 2000:
        achievements.append(("🌳 森林之王", "gold", "积分达到 2000"))
    if current_user_credits >= 1000:
        achievements.append(("🌲 森林守护者", "silver", "积分达到 1000"))
    if current_user_credits >= 500:
        achievements.append(("🌿 环保达人", "bronze", "积分达到 500"))
    if current_user_credits >= 100:
        achievements.append(("🌱 环保新人", "green", "积分达到 100"))
    
    record_count = current_user.get('record_count', 0)
    if record_count >= 30:
        achievements.append(("📊 坚持一月", "green", "连续记录 30 天"))
    if record_count >= 7:
        achievements.append(("📅 第一周", "green", "连续记录 7 天"))
    
    if achievements:
        cols = st.columns(min(len(achievements), 4))
        for i, (name, badge_type, desc) in enumerate(achievements):
            with cols[i % len(cols)]:
                st.markdown(f'<div class="achievement-card"><span style="font-size:2rem;">{name.split()[0]}</span><br><b>{name}</b><br><small style="color:#666;">{desc}</small></div>', unsafe_allow_html=True)
    else:
        st.info("🎯 继续努力，解锁更多成就徽章！")
    
    st.markdown("---")
    next_level_credits = 500
    progress = min(1.0, current_user_credits / next_level_credits)
    st.progress(progress)
    st.caption(f"距离下一等级 🌿 环保达人 还需 {max(0, next_level_credits - current_user_credits)} 积分")
    
    if engine and predictor:
        pred_val = predictor.predict_next_cycle("data/personal_carbon_footprint_behavior.csv").get('ensemble_meta', 10.0)
        actual_val = 9.5
        credit_info = engine.calculate_credits(actual_kg=actual_val, predicted_kg=pred_val)
        
        cols = st.columns(3)
        cols[0].metric("📈 本周获得", f"+{credit_info['total_credits']}", "表现优异")
        cols[1].metric("💰 总积分", f"{current_user_credits:,}", "+持续累积中")
        cols[2].metric("🏅 当前等级", "🌿 环保新手")

# --- Tab 4: User Management ---
with tab4:
    st.subheader("👤 用户管理系统")
    st.caption("前后端分离：默认优先调用 Flask API，若后端未启动则自动切换到本地 JSON 存储模式。")

    user_client = get_user_client()
    st.info(f"当前 API 基地址：{user_client.base_url}")

    left, right = st.columns([1, 1])

    with left:
        st.markdown("### 登录 / 注册")
        with st.form("user_login_form"):
            username = st.text_input("用户名", placeholder="例如：张三")
            custom_user_id = st.text_input("可选用户 ID", placeholder="留空自动生成")
            login_submit = st.form_submit_button("登录 / 注册")

        if login_submit:
            if not username.strip():
                st.error("用户名不能为空")
            else:
                login_result = user_client.login(username=username.strip(), user_id=custom_user_id.strip() or None)
                if login_result.get("success"):
                    st.session_state["current_user"] = login_result.get("data", {})
                    st.success(login_result.get("message", "登录成功"))
                else:
                    st.error(login_result.get("message", "登录失败"))

    with right:
        st.markdown("### 当前用户状态")
        current_user = st.session_state.get("current_user")
        if current_user:
            st.success(_format_user_label(current_user))
            c1, c2, c3 = st.columns(3)
            c1.metric("总积分", current_user.get("total_credits", 0))
            c2.metric("记录数", current_user.get("record_count", 0))
            c3.metric("创建时间", current_user.get("created_at", "-"))
        else:
            st.info("尚未登录，请先完成登录 / 注册。")

    current_user = st.session_state.get("current_user")
    if current_user:
        st.markdown("---")
        record_col, credit_col = st.columns(2)

        with record_col:
            st.markdown("### 添加碳排放记录")
            with st.form("carbon_record_form"):
                record_date = st.date_input("日期", value=datetime.now().date())
                carbon_value = st.number_input("实际碳排放 (kg)", min_value=0.0, value=10.0, step=0.1)
                predicted_value = st.number_input("预测碳排放 (kg，可选)", min_value=0.0, value=10.5, step=0.1)
                use_predicted_value = st.checkbox("启用预测值", value=True)
                record_submit = st.form_submit_button("保存记录")

            if record_submit:
                result = user_client.add_record(
                    user_id=current_user["user_id"],
                    date=record_date.isoformat(),
                    carbon_value=float(carbon_value),
                    predicted_value=float(predicted_value) if use_predicted_value else None,
                )
                if result.get("success"):
                    st.session_state["current_user"] = {
                        **current_user,
                        "record_count": current_user.get("record_count", 0) + 1,
                    }
                    st.success(result.get("message", "记录保存成功"))
                    st.json(result.get("data", {}))
                else:
                    st.error(result.get("message", "记录保存失败"))

        with credit_col:
            st.markdown("### 添加积分")
            with st.form("credit_form"):
                credits = st.number_input("积分数量", min_value=1, value=10, step=1)
                reason = st.text_input("积分原因", value="低碳行为奖励")
                credit_submit = st.form_submit_button("发放积分")

            if credit_submit:
                result = user_client.add_credits(
                    user_id=current_user["user_id"],
                    credits=int(credits),
                    reason=reason,
                )
                if result.get("success"):
                    st.success(result.get("message", "积分添加成功"))
                    updated_user = {
                        **current_user,
                        "total_credits": result.get("data", {}).get("new_total", current_user.get("total_credits", 0)),
                    }
                    st.session_state["current_user"] = updated_user
                else:
                    st.error(result.get("message", "积分添加失败"))

        st.markdown("### 历史记录")
        history_days = st.slider("查询天数", min_value=7, max_value=180, value=30, step=1)
        if st.button("刷新历史记录"):
            history_result = user_client.get_history(current_user["user_id"], days=history_days)
            if history_result.get("success"):
                records = history_result.get("data", {}).get("records", [])
                if records:
                    st.dataframe(pd.DataFrame(records), use_container_width=True)
                else:
                    st.info("当前没有历史记录。")
            else:
                st.error(history_result.get("message", "获取历史记录失败"))

