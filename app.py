"""
风芒可测 - 电力预测与交易优化系统
Streamlit 主应用入口（Streamlit Cloud 兼容）

特性：
- 无 Flask 依赖，直接调用 Python 业务逻辑
- 自愈式运行：自动检查模型，缺失时自动训练
- 路径跨平台兼容：Windows/Linux
- 环境隔离：所有敏感信息通过 st.secrets 读取
- 缓存机制：@st.cache_resource 缓存模型，@st.cache_data 缓存数据
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_echarts import st_echarts

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.core.utils.logger import logger
from src.logic import BusinessLogic, TradeOptimizer, TimeOfUsePrice


# ============================================================
# 1. Streamlit 页面配置
# ============================================================

st.set_page_config(
    page_title="风芒可测 - 电力预测与交易优化系统",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "电力预测与交易优化系统 v1.0.0"
    }
)

# 自定义样式
st.markdown("""
    <style>
    .main {padding: 2rem;}
    .metric-card {background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0;}
    </style>
""", unsafe_allow_html=True)


# ============================================================
# 2. 缓存和辅助函数
# ============================================================

@st.cache_resource
def get_config() -> Dict[str, Any]:
    """获取应用配置"""
    return {
        'data_path': st.secrets.get('model', {}).get('data_path', 'data/data.csv'),
        'model_path': st.secrets.get('model', {}).get('model_path', 'models/lstm_forecaster.pth'),
        'epochs': st.secrets.get('model', {}).get('epochs', 50),
        'batch_size': st.secrets.get('model', {}).get('batch_size', 32),
        'peak_price': st.secrets.get('trade', {}).get('peak_price', 1.2),
        'flat_price': st.secrets.get('trade', {}).get('flat_price', 0.8),
        'valley_price': st.secrets.get('trade', {}).get('valley_price', 0.4),
    }


def ensure_model_exists() -> bool:
    """
    自愈式运行：确保模型存在
    
    如果模型不存在，自动触发训练
    """
    config = get_config()
    model_path = Path(config['model_path'])
    
    if model_path.exists():
        logger.info(f"✓ 模型文件存在：{model_path}")
        return True
    
    logger.warning(f"模型文件不存在，自动触发训练：{model_path}")
    
    with st.status("🚀 首次运行检测到模型缺失，正在自动训练模型...", expanded=True) as status:
        st.write("📊 [1/3] 加载数据...")
        
        # 创建模型目录
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # 运行完整管道
            result = BusinessLogic.run_full_pipeline(
                data_path=config['data_path'],
                model_path=config['model_path'],
                epochs=config['epochs'],
                batch_size=config['batch_size']
            )
            
            if result['status'] == 'success':
                st.write("✓ [1/3] 数据加载完成")
                st.write("✓ [2/3] 模型训练完成")
                st.write("✓ [3/3] 模型验证完成")
                status.update(label="✅ 模型自动训练成功！", state="complete")
                time.sleep(1)
                return True
            else:
                st.error(f"模型训练失败：{result.get('message', '未知错误')}")
                return False
        except Exception as e:
            st.error(f"自动训练出错：{str(e)}")
            logger.error(f"自动训练异常：{str(e)}")
            return False


@st.cache_data(ttl=300)
def load_predictions(data_path: str = None, model_path: str = None) -> Optional[Dict[str, Any]]:
    """加载预测数据（缓存5分钟）"""
    config = get_config()
    data_path = data_path or config['data_path']
    model_path = model_path or config['model_path']
    
    result = BusinessLogic.predict(data_path=data_path, model_path=model_path)
    return result


@st.cache_data(ttl=300)
def load_trade_advice(data_path: str = None, model_path: str = None) -> Optional[Dict[str, Any]]:
    """加载交易建议（缓存5分钟）"""
    config = get_config()
    data_path = data_path or config['data_path']
    model_path = model_path or config['model_path']
    
    result = BusinessLogic.get_trade_advice(data_path=data_path, model_path=model_path)
    return result


# ============================================================
# 3. 主应用
# ============================================================

def main():
    """主应用入口"""
    
    # 头部
    col1, col2 = st.columns([1, 5])
    with col1:
        st.markdown("# ⚡")
    with col2:
        st.markdown("# 风芒可测 - 电力预测与交易优化系统")
    
    st.markdown("---")
    
    # 侧边栏：导航
    with st.sidebar:
        st.markdown("## 📑 导航")
        page = st.radio(
            "选择功能",
            ["📊 数据预览", "🤖 模型训练", "🔮 预测结果", "💰 交易建议", "📈 交易指标", "⚠️ 风险分析"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("## ⚙️ 系统配置")
        
        config = get_config()
        show_config = st.checkbox("展示配置信息", value=False, help="显示当前系统配置")
        if show_config:
            st.json({
                'data_path': str(config['data_path']),
                'model_path': str(config['model_path']),
                'epochs': config['epochs'],
                'batch_size': config['batch_size'],
            })
        
        st.markdown("---")
        st.markdown("### 📝 关于")
        st.markdown("""
        **版本**: 1.0.0  
        **框架**: Streamlit  
        **算法**: LSTM + XGBoost Stacking  
        **部署**: Streamlit Cloud 兼容
        """)
    
    # ============================================================
    # 第1页：数据预览
    # ============================================================
    if page == "📊 数据预览":
        st.header("📊 数据预览")
        
        config = get_config()
        data_path = Path(config['data_path'])
        
        if not data_path.exists():
            st.error(f"❌ 数据文件不存在：{data_path}")
            st.info("请确保数据文件位置正确")
            return
        
        try:
            # 显示文件信息
            file_size = data_path.stat().st_size / (1024 * 1024)
            st.metric("📦 文件大小", f"{file_size:.2f} MB")
            
            # 读取数据
            df = pd.read_csv(data_path)
            st.success(f"✓ 数据文件加载成功，包含 {len(df)} 行，{len(df.columns)} 列")
            
            # 显示数据统计
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("行数", f"{len(df):,}")
            with col2:
                st.metric("列数", len(df.columns))
            with col3:
                st.metric("缺失值", int(df.isnull().sum().sum()))
            
            # 显示前几行
            st.subheader("数据样本")
            st.dataframe(df.head(10), use_container_width=True)
            
            # 数据统计
            st.subheader("数据统计")
            st.dataframe(df.describe(), use_container_width=True)
            
            # 列信息
            st.subheader("列信息")
            col_info = pd.DataFrame({
                '列名': df.columns,
                '数据类型': df.dtypes.astype(str),
                '非空数': df.count(),
                '缺失数': df.isnull().sum(),
                '缺失率%': (df.isnull().sum() / len(df) * 100).round(2)
            })
            st.dataframe(col_info, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ 读取数据失败：{str(e)}")
    
    # ============================================================
    # 第2页：模型训练
    # ============================================================
    elif page == "🤖 模型训练":
        st.header("🤖 模型训练")
        
        config = get_config()
        
        st.markdown("""
        在此页面可以训练或重新训练预测模型。
        
        - **模型类型**: LSTM 长短期记忆网络
        - **输入**: 历史电力数据
        - **输出**: 未来24小时的功率预测
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            epochs = st.number_input("训练轮数 (Epochs)", min_value=10, max_value=200, value=config['epochs'])
        with col2:
            batch_size = st.number_input("批次大小 (Batch Size)", min_value=16, max_value=256, value=config['batch_size'])
        
        if st.button("🚀 开始训练", use_container_width=True, type="primary"):
            with st.status("⏳ 正在训练模型...", expanded=True) as status:
                st.write("📊 [1/3] 加载数据...")
                
                try:
                    result = BusinessLogic.train_model(
                        data_path=config['data_path'],
                        model_path=config['model_path'],
                        epochs=epochs,
                        batch_size=batch_size
                    )
                    
                    if result['status'] == 'success':
                        st.write("✓ [1/3] 数据加载完成")
                        st.write("✓ [2/3] 模型训练完成")
                        st.write("✓ [3/3] 模型验证完成")
                        
                        status.update(label="✅ 模型训练成功！", state="complete")
                        st.success("模型已成功保存")
                        
                        # 显示训练结果
                        if 'result' in result:
                            st.json(result['result'])
                    else:
                        st.error(f"训练失败：{result.get('message', '未知错误')}")
                except Exception as e:
                    st.error(f"训练出错：{str(e)}")
        
        # 模型文件检查
        st.markdown("---")
        st.subheader("📋 模型文件状态")
        
        model_path = Path(config['model_path'])
        if model_path.exists():
            file_size = model_path.stat().st_size / (1024 * 1024)
            st.success(f"✓ 模型文件存在")
            st.metric("模型大小", f"{file_size:.2f} MB")
        else:
            st.warning(f"⚠️ 模型文件不存在：{model_path}")
            st.info("请先运行训练来生成模型")
    
    # ============================================================
    # 第3页：预测结果
    # ============================================================
    elif page == "🔮 预测结果":
        st.header("🔮 预测结果")
        
        config = get_config()
        
        if not Path(config['model_path']).exists():
            st.error("❌ 模型文件不存在，请先进行模型训练")
            return
        
        with st.spinner("加载预测数据..."):
            result = load_predictions(config['data_path'], config['model_path'])
        
        if result and result['status'] == 'success':
            pred_data = result['result']
            
            st.success("✓ 预测成功")
            
            # 显示预测统计
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("预测数据点", int(pred_data['count']))
            with col2:
                st.metric("最小值 (kW)", f"{pred_data['min']:.2f}")
            with col3:
                st.metric("最大值 (kW)", f"{pred_data['max']:.2f}")
            with col4:
                st.metric("平均值 (kW)", f"{pred_data['mean']:.2f}")
            
            # 绘制预测曲线
            st.subheader("24小时功率预测曲线")
            
            predictions = pred_data['predictions'][:24]  # 只显示前24小时
            hours = list(range(len(predictions)))
            
            # 使用 ECharts 绘制
            option = {
                "xAxis": {
                    "type": "category",
                    "data": [f"{h:02d}:00" for h in hours]
                },
                "yAxis": {
                    "type": "value"
                },
                "series": [
                    {
                        "data": [round(p, 2) for p in predictions],
                        "type": "line",
                        "smooth": True,
                        "itemStyle": {"color": "#1f77b4"},
                        "areaStyle": {}
                    }
                ],
                "tooltip": {
                    "trigger": "axis"
                }
            }
            st_echarts(option, height=400)
            
            # 显示预测值表格
            st.subheader("预测值详情")
            pred_df = pd.DataFrame({
                '时刻': [f"{h:02d}:00" for h in hours],
                '功率 (kW)': [round(p, 2) for p in predictions]
            })
            st.dataframe(pred_df, use_container_width=True)
        else:
            st.error("❌ 预测失败，请重试")
    
    # ============================================================
    # 第4页：交易建议
    # ============================================================
    elif page == "💰 交易建议":
        st.header("💰 交易建议")
        
        config = get_config()
        
        if not Path(config['model_path']).exists():
            st.error("❌ 模型文件不存在，请先进行模型训练")
            return
        
        with st.spinner("生成交易建议..."):
            result = load_trade_advice(config['data_path'], config['model_path'])
        
        if result and result['status'] == 'success':
            advice = result['data']
            
            st.success("✓ 交易建议生成成功")
            
            # 显示经济指标
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("预期收益 (¥)", f"¥{advice['expected_revenue']:.2f}")
            with col2:
                st.metric("成本节约 (¥)", f"¥{advice['cost_saving']:.2f}")
            with col3:
                st.metric("削峰功率 (kW)", f"{advice['peak_shaving_power']:.2f}")
            with col4:
                st.metric("填谷功率 (kW)", f"{advice['valley_filling_power']:.2f}")
            
            # 显示买入建议
            st.subheader("💳 买入建议（低谷充电）")
            buy_advice = advice['buy_advice']
            if buy_advice:
                buy_df = pd.DataFrame({
                    '时刻': [a['hour_str'] for a in buy_advice],
                    '功率 (kW)': [round(a['power'], 2) for a in buy_advice],
                    '电价 (¥/kWh)': [round(a['price'], 2) for a in buy_advice],
                    '优先级': [a['priority'] for a in buy_advice],
                    '理由': [a['reason'] for a in buy_advice]
                })
                st.dataframe(buy_df, use_container_width=True)
            else:
                st.info("暂无买入建议")
            
            # 显示卖出建议
            st.subheader("💳 卖出建议（高峰放电）")
            sell_advice = advice['sell_advice']
            if sell_advice:
                sell_df = pd.DataFrame({
                    '时刻': [a['hour_str'] for a in sell_advice],
                    '功率 (kW)': [round(a['power'], 2) for a in sell_advice],
                    '电价 (¥/kWh)': [round(a['price'], 2) for a in sell_advice],
                    '优先级': [a['priority'] for a in sell_advice],
                    '理由': [a['reason'] for a in sell_advice]
                })
                st.dataframe(sell_df, use_container_width=True)
            else:
                st.info("暂无卖出建议")
            
            # 显示电价信息
            st.subheader("⚡ 分时电价信息")
            summary = advice['summary']
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("高峰电价", f"¥{summary['peak_price']:.2f}/kWh")
            with col2:
                st.metric("平段电价", f"¥{summary['flat_price']:.2f}/kWh")
            with col3:
                st.metric("低谷电价", f"¥{summary['valley_price']:.2f}/kWh")
            with col4:
                st.metric("平均电价", f"¥{summary['mean_price']:.2f}/kWh")
        else:
            st.error("❌ 生成交易建议失败，请重试")
    
    # ============================================================
    # 第5页：交易指标
    # ============================================================
    elif page == "📈 交易指标":
        st.header("📈 交易指标")
        
        config = get_config()
        
        if not Path(config['model_path']).exists():
            st.error("❌ 模型文件不存在，请先进行模型训练")
            return
        
        try:
            result = BusinessLogic.get_trade_metrics(
                data_path=config['data_path'],
                model_path=config['model_path']
            )
            
            if result and result['status'] == 'success':
                metrics = result['data']
                
                st.success("✓ 交易指标计算成功")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("日发电量", f"{metrics['daily_energy']:.2f} kWh")
                with col2:
                    st.metric("平均成本", f"¥{metrics['average_cost_per_kwh']:.2f}/kWh")
                with col3:
                    st.metric("收益潜力", f"¥{metrics['revenue_potential']:.2f}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("削峰效益", f"¥{metrics['peak_shaving_benefit']:.2f}")
                with col2:
                    st.metric("填谷效益", f"¥{metrics['valley_filling_benefit']:.2f}")
                with col3:
                    st.metric("总效益", f"¥{metrics['total_benefit']:.2f}")
            else:
                st.error("❌ 指标计算失败")
        except Exception as e:
            st.error(f"❌ 错误：{str(e)}")
    
    # ============================================================
    # 第6页：风险分析
    # ============================================================
    elif page == "⚠️ 风险分析":
        st.header("⚠️ 风险分析")
        
        config = get_config()
        
        if not Path(config['model_path']).exists():
            st.error("❌ 模型文件不存在，请先进行模型训练")
            return
        
        try:
            result = BusinessLogic.get_trade_risk(
                data_path=config['data_path'],
                model_path=config['model_path']
            )
            
            if result and result['status'] == 'success':
                risk = result['data']
                
                st.success("✓ 风险分析完成")
                
                # 风险评级
                risk_score = risk['risk_score']
                risk_level = risk['risk_level']
                
                # 颜色编码
                level_colors = {
                    'VERY_LOW': '#00cc44',
                    'LOW': '#44dd44',
                    'MEDIUM': '#ffaa00',
                    'HIGH': '#ff6600',
                    'VERY_HIGH': '#ff0000'
                }
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown(f"""
                    <div style="background-color: {level_colors.get(risk_level, '#cccccc')}; 
                                padding: 2rem; border-radius: 1rem; text-align: center; color: white;">
                        <h3 style="margin: 0;">风险等级</h3>
                        <h2 style="margin: 0.5rem 0;">{risk_level}</h2>
                        <p style="margin: 0;">分数: {risk_score:.2f}/10</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"**建议**: {risk['recommendation']}")
                
                # 统计信息
                st.subheader("📊 统计信息")
                stats = risk['statistics']
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("平均值", f"{stats['mean']:.2f}")
                with col2:
                    st.metric("标准差", f"{stats['std']:.2f}")
                with col3:
                    st.metric("最小值", f"{stats['min']:.2f}")
                with col4:
                    st.metric("最大值", f"{stats['max']:.2f}")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("25分位数", f"{stats['q25']:.2f}")
                with col2:
                    st.metric("中位数", f"{stats['q50']:.2f}")
                with col3:
                    st.metric("75分位数", f"{stats['q75']:.2f}")
                with col4:
                    st.metric("四分位距", f"{stats['iqr']:.2f}")
                
                # 风险指标
                st.subheader("⚠️ 风险指标")
                indicators = risk['risk_indicators']
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("波动性评分", f"{indicators['volatility_score']:.2f}")
                with col2:
                    st.metric("极值范围评分", f"{indicators['range_score']:.2f}")
                with col3:
                    st.metric("变异系数", f"{indicators['coefficient_of_variation']:.4f}")
            else:
                st.error("❌ 风险分析失败")
        except Exception as e:
            st.error(f"❌ 错误：{str(e)}")


# ============================================================
# 4. 入口函数
# ============================================================

if __name__ == "__main__":
    # 确保模型存在（自愈式运行）
    if not ensure_model_exists():
        st.error("""
        ❌ **模型初始化失败**
        
        请尝试以下步骤：
        1. 检查 `data/data.csv` 文件是否存在
        2. 确保有足够的磁盘空间
        3. 查看日志文件获取更详细的错误信息
        """)
        st.stop()
    
    # 运行主应用
    main()
