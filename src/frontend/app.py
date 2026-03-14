
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import streamlit as st
from streamlit_echarts import st_echarts

# 页面配置
st.set_page_config(
    page_title="风芒可测 - 电力预测与交易优化系统",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# API 服务地址（从环境变量读取）
import os
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv('API_BASE_URL', "http://localhost:5000")
BACKEND_TIMEOUT = 30


# ============================================================
# 1. 后端连接容错
# ============================================================

@st.cache_resource
def check_backend_health() -> bool:
    """检查后端健康状态"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/health", timeout=5)
        return response.status_code == 200
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        return False


def ensure_backend_ready():
    """确保后端就绪，否则显示加载动画和重试提示"""
    if not check_backend_health():
        st.warning("⚠️ 正在连接后端服务...")
        
        with st.spinner("🔄 正在启动 Flask 后端..."):
            for attempt in range(3):
                time.sleep(2)
                if check_backend_health():
                    st.success("✅ 后端已就绪！")
                    st.rerun()
                    return
        
        st.error("""
        ❌ **后端服务未就绪**
        
        请按以下步骤启动后端：
        
        ```bash
        python src/backend/api.py
        ```
        
        或使用 main.py 启动完整系统：
        
        ```bash
        python main.py --mode api
        ```
        
        启动后，点击下方按钮刷新页面：
        """)
        
        if st.button("🔄 刷新页面", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()
        
        st.stop()


# ============================================================
# 2. 异步缓存 - 避免重复调用耗时操作
# ============================================================

@st.cache_data(ttl=300)
def cached_api_request(endpoint: str, method: str = "GET", 
                      json_data: Optional[Dict] = None) -> Optional[Dict]:
    """缓存的 API 请求函数（5分钟缓存）"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        
        if method == "GET":
            response = requests.get(url, timeout=BACKEND_TIMEOUT)
        else:
            response = requests.post(url, json=json_data, timeout=BACKEND_TIMEOUT)
        
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.ConnectionError:
        st.error("❌ 无法连接到 API 服务，请确保后端已启动")
        return None
    except requests.exceptions.Timeout:
        st.error(f"❌ 请求超时（{BACKEND_TIMEOUT}s），请检查后端性能")
        return None
    except Exception as e:
        st.error(f"❌ 请求失败：{str(e)}")
        return None


def make_api_request(endpoint: str, method: str = "GET", 
                    json_data: Optional[Dict] = None, 
                    use_cache: bool = True) -> Optional[Dict]:
    """灵活的 API 请求函数（支持缓存和非缓存）"""
    if use_cache:
        return cached_api_request(endpoint, method, json_data)
    else:
        # 直接请求不缓存
        try:
            url = f"{API_BASE_URL}{endpoint}"
            if method == "GET":
                response = requests.get(url, timeout=BACKEND_TIMEOUT)
            else:
                response = requests.post(url, json=json_data, timeout=BACKEND_TIMEOUT)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"❌ 请求失败：{str(e)}")
            return None


def run_full_pipeline_via_api() -> bool:
    """
    通过 API 触发后端运行完整管道
    
    Returns:
        True 如果成功，False 如果失败
    """
    try:
        with st.status("⏳ 正在执行完整管道...", expanded=True) as status:
            st.write("📊 [1/5] 数据层检测...")
            response = requests.post(f"{API_BASE_URL}/api/run_full_pipeline", timeout=BACKEND_TIMEOUT)
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get('status') == 'success':
                    st.write("✓ [1/5] 数据层检测完成")
                    st.write("✓ [2/5] 模型校验完成")
                    st.write("✓ [3/5] 特征工程完成")
                    st.write("✓ [4/5] 模型训练完成")
                    st.write("✓ [5/5] 结果保存完成")
                    
                    status.update(label="✅ 完整管道执行成功", state="complete")
                    return True
                else:
                    st.error(f"管道执行失败：{result.get('message', '未知错误')}")
                    status.update(label="❌ 管道执行失败", state="error")
                    return False
            else:
                st.error(f"API 请求失败：{response.status_code}")
                status.update(label="❌ API 请求失败", state="error")
                return False
    
    except requests.exceptions.Timeout:
        st.error("❌ 请求超时，后端可能在处理中，请稍候...")
        return False
    except Exception as e:
        st.error(f"❌ 执行失败：{str(e)}")
        return False


def render_sync_and_predict():
    """渲染一键同步/预测页面"""
    st.title("⚡ 一键同步与预测")
    st.markdown("点击下方按钮可一次性完成数据同步、模型验证和预测任务")
    
    # 大按钮
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 执行完整流程", use_container_width=True, key="run_pipeline"):
            if run_full_pipeline_via_api():
                st.balloons()
                st.success("✅ 所有操作完成！数据已更新到最新。")
                time.sleep(2)
                st.rerun()
            else:
                st.error("❌ 流程执行失败，请查看错误信息")
    
    st.markdown("---")
    
    # 执行流程说明
    st.markdown("""
    ### 📋 执行步骤说明
    
    点击 **执行完整流程** 按钮后，系统会自动执行以下步骤：
    
    1. ✓ **数据层检测** - 验证数据文件完整性
    2. ✓ **模型校验** - 确保模型文件有效
    3. ✓ **特征工程** - 对原始数据进行特征处理
    4. ✓ **模型训练** - 使用最新数据训练模型
    5. ✓ **结果保存** - 保存训练结果和预测数据
    
    整个过程需要 **2-5 分钟**，具体时间取决于数据量和服务器性能。
    
    ### ⚖️ 系统状态检查
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔍 检查数据文件", use_container_width=True):
            try:
                response = requests.get(f"{API_BASE_URL}/api/data/status", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    st.success(f"✓ 数据文件状态: {data.get('status', 'OK')}")
                    if 'size' in data:
                        st.info(f"📁 文件大小: {data['size']}")
                    if 'rows' in data:
                        st.info(f"📊 数据行数: {data['rows']}")
                else:
                    st.error("❌ 无法获取数据文件状态")
            except Exception as e:
                st.error(f"❌ 检查失败: {str(e)}")
    
    with col2:
        if st.button("🤖 检查模型文件", use_container_width=True):
            try:
                response = requests.get(f"{API_BASE_URL}/api/model/status", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    st.success(f"✓ 模型状态: {data.get('status', 'OK')}")
                    if 'models' in data:
                        for model_name, model_info in data['models'].items():
                            st.info(f"📦 {model_name}: {model_info}")
                else:
                    st.error("❌ 无法获取模型文件状态")
            except Exception as e:
                st.error(f"❌ 检查失败: {str(e)}")


def render_sidebar() -> str:
    """渲染侧边栏导航"""
    st.sidebar.title("⚡ 风芒可测")
    st.sidebar.markdown("---")
    
    menu = st.sidebar.radio(
        "导航",
        ["� 一键同步", "�📊 实时监控", "📈 多维预测", "💰 交易助手"],
        index=0
    )
    
    st.sidebar.markdown("---")
    
    # 显示后端状态
    if check_backend_health():
        st.sidebar.success("✅ 后端：在线")
    else:
        st.sidebar.error("❌ 后端：离线")
    
    return menu


def render_realtime_monitoring():
    """渲染实时监控页面"""
    st.title("📊 实时监控")
    st.markdown("实时功率监测与预测对比")
    
    # 一键更新按钮
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("#### 功率预测曲线")
    with col2:
        if st.button("🔄 刷新数据", use_container_width=True):
            st.cache_data.clear()  # 清空所有缓存
            with st.spinner("📊 正在加载最新数据..."):
                result = make_api_request("/api/run_all", method="POST", use_cache=False)
                if result and result.get('status') == 'success':
                    st.success("✅ 数据刷新完成！")
                    st.rerun()
    
    # 获取绘图数据（使用缓存）
    plot_data = make_api_request("/api/data/plot")
    
    if plot_data and plot_data.get('status') == 'success':
        data = plot_data['data']
        
        # 截断数据用于展示（最多 96 点）
        max_points = 96
        time_axis = data['time_axis'][:max_points]
        actual_values = data['actual_values'][:max_points]
        predicted_values = data.get('predicted_values', [])[:max_points] if data.get('predicted_values') else []
        
        # 构建 ECharts 配置
        series = [
            {
                "name": "实际功率",
                "type": "line",
                "smooth": True,
                "data": actual_values,
                "markPoint": {
                    "data": [
                        {"type": "max", "name": "最大值"},
                        {"type": "min", "name": "最小值"}
                    ]
                }
            }
        ]
        
        if predicted_values:
            series.append({
                "name": "预测功率",
                "type": "line",
                "smooth": True,
                "data": predicted_values,
                "lineStyle": {"type": "dashed"}
            })
        
        options = {
            "title": {"text": "功率对比"},
            "tooltip": {"trigger": "axis"},
            "legend": {"data": ["实际功率", "预测功率"] if predicted_values else ["实际功率"], "top": "5%"},
            "xAxis": {
                "type": "category",
                "data": [str(i) for i in range(len(time_axis))],
                "name": "时间",
                "axisLabel": {"rotate": 45}
            },
            "yAxis": {"type": "value", "name": "功率 (kW)"},
            "series": series
        }
        
        st_echarts(options, height="500px")
        
        # 展示数据表格
        with st.expander("📋 查看详细数据"):
            df = pd.DataFrame({
                '时间': time_axis,
                '实际功率': actual_values,
            })
            if predicted_values:
                df['预测功率'] = predicted_values
            st.dataframe(df, use_container_width=True)
    
    else:
        st.warning("⚠️ 暂无数据，请点击刷新按钮加载")
    
    # 模型指标（使用缓存）
    st.markdown("---")
    st.markdown("#### 模型评估指标")
    
    metrics_data = make_api_request("/api/metrics")
    
    if metrics_data and metrics_data.get('status') == 'success':
        m = metrics_data['data']
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("MAE", f"{m['mae']:.2f}", delta="平均绝对误差")
        with col2:
            st.metric("RMSE", f"{m['rmse']:.2f}", delta="均方根误差")
        with col3:
            st.metric("R²", f"{m['r2']:.4f}", delta="决定系数")
        with col4:
            st.metric("MAPE", f"{m['mape']:.2f}%", delta="平均百分比误差")
    else:
        st.info("📊 请先运行模型训练以获取评估指标")


def render_multi_scale_forecast():
    """渲染多维预测页面"""
    st.title("📈 多维预测")
    st.markdown("多时间尺度功率预测")
    
    # 尺度选择按钮
    st.markdown("#### 预测尺度选择")
    col1, col2, col3 = st.columns(3)
    
    scale_map = {
        "天": "day",
        "周": "week",
        "月": "month"
    }
    
    with col1:
        day_btn = st.button("📅 天预测", use_container_width=True, key="day")
    with col2:
        week_btn = st.button("📆 周预测", use_container_width=True, key="week")
    with col3:
        month_btn = st.button("📅 月预测", use_container_width=True, key="month")
    
    selected_scale = None
    if day_btn:
        selected_scale = "day"
    elif week_btn:
        selected_scale = "week"
    elif month_btn:
        selected_scale = "month"
    
    if selected_scale:
        with st.spinner(f"正在加载{selected_scale}预测数据..."):
            forecast_data = make_api_request(f"/api/forecast/{selected_scale}")
            
            if forecast_data and forecast_data.get('status') == 'success':
                data = forecast_data['data']
                
                # 显示统计信息
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("数据点数", data['count'])
                with col2:
                    st.metric("最小值", f"{data['min']:.2f} kW")
                with col3:
                    st.metric("最大值", f"{data['max']:.2f} kW")
                
                # 构建 ECharts 配置
                options = {
                    "title": {"text": f"{selected_scale}功率预测"},
                    "tooltip": {"trigger": "axis"},
                    "xAxis": {
                        "type": "category",
                        "data": [str(i) for i in range(len(data['predictions']))],
                        "name": "小时"
                    },
                    "yAxis": {"type": "value", "name": "功率 (kW)"},
                    "series": [{
                        "name": "预测功率",
                        "type": "line",
                        "smooth": True,
                        "data": data['predictions'],
                        "markLine": {
                            "data": [{"type": "average", "name": "平均值"}]
                        }
                    }]
                }
                
                st_echarts(options, height="500px")
                
            else:
                st.error("❌ 获取预测数据失败，请确保模型已训练")
    
    else:
        st.info("👆 请选择预测尺度查看预测结果")


def render_trade_assistant():
    """渲染交易助手页面"""
    st.title("💰 交易助手")
    st.markdown("基于AI预测和分时电价的交易优化建议")
    
    # 三列标签页
    tab1, tab2, tab3 = st.tabs(["📊 交易建议", "📈 收益指标", "⚠️ 风险分析"])
    
    # ============================================================
    # Tab 1: 交易建议
    # ============================================================
    with tab1:
        st.subheader("🤖 AI 生成的交易建议")
        
        # 获取交易建议（后端完全处理）
        with st.spinner("⏳ 正在计算交易优化方案..."):
            trade_advice = make_api_request("/api/trade/advice", method="GET")
        
        if trade_advice and trade_advice.get('status') == 'success':
            data = trade_advice['data']
            
            # 核心指标
            st.markdown("#### 📊 核心指标")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("预期收益", f"¥{data['expected_revenue']:.2f}")
            with col2:
                st.metric("成本节约", f"¥{data['cost_saving']:.2f}")
            with col3:
                st.metric("削峰功率", f"{data['peak_shaving_power']:.2f} kW")
            with col4:
                st.metric("填谷功率", f"{data['valley_filling_power']:.2f} kW")
            
            # 电价信息
            st.markdown("---")
            st.markdown("#### ⚡ 分时电价")
            summary = data['summary']
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info(f"**高峰电价**\n\n¥{summary['peak_price']:.3f}/kWh\n\n11-14点，18-23点")
            with col2:
                st.info(f"**平段电价**\n\n¥{summary['flat_price']:.3f}/kWh\n\n其他时段")
            with col3:
                st.info(f"**低谷电价**\n\n¥{summary['valley_price']:.3f}/kWh\n\n23-7点")
            
            # 交易建议表格
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🔴 建议买入（充电储能）")
                buy_df = pd.DataFrame(data['buy_advice'])
                if not buy_df.empty:
                    buy_df = buy_df[['hour_str', 'power', 'price', 'reason', 'priority']]
                    buy_df.columns = ['时间', '功率(kW)', '电价(¥/kWh)', '原因', '优先级']
                    st.dataframe(buy_df, use_container_width=True)
                else:
                    st.write("暂无买入建议")
            
            with col2:
                st.markdown("#### 🟢 建议卖出（放电自用）")
                sell_df = pd.DataFrame(data['sell_advice'])
                if not sell_df.empty:
                    sell_df = sell_df[['hour_str', 'power', 'price', 'reason', 'priority']]
                    sell_df.columns = ['时间', '功率(kW)', '电价(¥/kWh)', '原因', '优先级']
                    st.dataframe(sell_df, use_container_width=True)
                else:
                    st.write("暂无卖出建议")
            
        else:
            st.error("❌ 获取交易建议失败，请重试")
    
    # ============================================================
    # Tab 2: 收益指标
    # ============================================================
    with tab2:
        st.subheader("💰 收益分析")
        
        with st.spinner("📊 正在计算收益指标..."):
            metrics_data = make_api_request("/api/trade/metrics", method="GET")
        
        if metrics_data and metrics_data.get('status') == 'success':
            m = metrics_data['data']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("日发电量", f"{m['daily_energy']:.2f} kWh")
            with col2:
                st.metric("平均成本", f"¥{m['average_cost_per_kwh']:.3f}/kWh")
            with col3:
                st.metric("收益潜力", f"¥{m['revenue_potential']:.2f}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("削峰效益", f"¥{m['peak_shaving_benefit']:.2f}")
            with col2:
                st.metric("填谷效益", f"¥{m['valley_filling_benefit']:.2f}")
            with col3:
                st.metric("总效益", f"¥{m['total_benefit']:.2f}", 
                         delta=f"+¥{m['total_benefit']:.2f}")
            
            st.markdown("---")
            st.info("""
            💡 **指标说明**
            
            - **日发电量**: 24小时预测发电总量
            - **平均成本**: 加权平均电价（按预测功率加权）
            - **收益潜力**: 按高峰电价计算的最大收益
            - **削峰效益**: 高峰时段通过储能优化可节省的成本
            - **填谷效益**: 低谷时段通过储能利用可节省的成本
            - **总效益**: 削峰填谷的总经济效果
            """)
        
        else:
            st.error("❌ 获取收益指标失败，请重试")
    
    # ============================================================
    # Tab 3: 风险分析
    # ============================================================
    with tab3:
        st.subheader("⚠️ 风险评估")
        
        with st.spinner("🔍 正在分析交易风险..."):
            risk_data = make_api_request("/api/trade/risk", method="GET")
        
        if risk_data and risk_data.get('status') == 'success':
            d = risk_data['data']
            
            # 风险等级和评分
            st.markdown("#### 🎯 风险评级")
            
            risk_level_colors = {
                'VERY_LOW': '🟢 极低风险',
                'LOW': '🟢 低风险',
                'MEDIUM': '🟡 中等风险',
                'HIGH': '🔴 高风险',
                'VERY_HIGH': '🔴 极高风险'
            }
            
            col1, col2, col3 = st.columns(3)
            with col1:
                # 风险分数（0-10）
                st.metric(
                    "风险分数",
                    f"{d['risk_score']:.1f}/10",
                    delta="0分为最安全"
                )
            with col2:
                st.markdown(f"**{risk_level_colors[d['risk_level']]}**")
            with col3:
                st.write("")
            
            # 建议
            st.warning(f"💬 {d['recommendation']}")
            
            # 统计信息
            st.markdown("---")
            st.markdown("#### 📊 统计分析")
            
            stats = d['statistics']
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("平均功率", f"{stats['mean']:.2f} kW")
            with col2:
                st.metric("标准差", f"{stats['std']:.2f}")
            with col3:
                st.metric("波动系数", f"{stats['coefficient_of_variation']:.3f}")
            with col4:
                st.metric("偏度", f"{stats['skewness']:.3f}")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("最小值", f"{stats['min']:.2f} kW")
            with col2:
                st.metric("中位数", f"{stats['median']:.2f} kW")
            with col3:
                st.metric("最大值", f"{stats['max']:.2f} kW")
            with col4:
                st.metric("极值范围", f"{stats['range']:.2f} kW")
            
            # 风险指标
            st.markdown("---")
            st.markdown("#### ⚠️ 风险指标")
            
            risk_ind = d['risk_indicators']
            col1, col2 = st.columns(2)
            with col1:
                st.metric("波动性评分", f"{risk_ind['volatility_score']:.1f}/10", 
                         delta=risk_ind['volatility_warning'])
            with col2:
                st.metric("极值范围评分", f"{risk_ind['range_score']:.1f}/10",
                         delta=risk_ind['range_warning'])
        
        else:
            st.error("❌ 获取风险分析失败，请重试")


def main():
    """主函数"""
    # 首先检查后端是否就绪
    ensure_backend_ready()
    
    # 渲染侧边栏
    menu = render_sidebar()
    
    # 根据选择渲染不同页面
    if menu == "� 一键同步":
        render_sync_and_predict()
    elif menu == "�📊 实时监控":
        render_realtime_monitoring()
    elif menu == "📈 多维预测":
        render_multi_scale_forecast()
    elif menu == "💰 交易助手":
        render_trade_assistant()


if __name__ == "__main__":
    main()