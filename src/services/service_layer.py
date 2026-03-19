"""
HealthMind 服务层模块

实现：
1. SHAP 解释性工程：量化各因素对健康风险的边际贡献
2. 决策引擎：基于预测结果的'识别 - 推荐'闭环
3. 激励匹配：健康积分系统
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import json

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============== 数据结构 ==============

@dataclass
class RiskFactor:
    """风险因素数据"""
    name: str
    value: float
    unit: str
    shap_value: float = 0.0  # SHAP 贡献值
    risk_level: str = 'normal'  # normal, warning, high
    
@dataclass
class PredictionResult:
    """预测结果"""
    user_id: str
    prediction_date: datetime
    risk_probability: float  # 风险概率 (0-1)
    risk_level: str  # low, medium, high
    risk_factors: List[RiskFactor]
    shap_contributions: Dict[str, float]
    
@dataclass
class InterventionRecommendation:
    """干预建议"""
    factor: str
    current_value: float
    target_value: float
    action: str  # increase, decrease, maintain
    priority: int  # 1-5, 1 最高
    description: str
    expected_risk_reduction: float  # 预期风险降低比例

@dataclass
class HealthPoints:
    """健康积分"""
    user_id: str
    total_points: int = 0
    earned_points: int = 0
    spent_points: int = 0
    level: str = 'bronze'  # bronze, silver, gold, platinum
    streak_days: int = 0
    last_activity: Optional[datetime] = None
    
@dataclass
class UserHealthProfile:
    """用户健康档案"""
    user_id: str
    created_at: datetime
    predictions: List[PredictionResult] = field(default_factory=list)
    recommendations: List[InterventionRecommendation] = field(default_factory=list)
    health_points: HealthPoints = None
    execution_records: List[Dict] = field(default_factory=list)
    
    def __post_init__(self):
        if self.health_points is None:
            self.health_points = HealthPoints(user_id=self.user_id)


# ============== SHAP 解释性引擎 ==============

class SHAPExplainer:
    """
    SHAP 解释性引擎
    
    量化步数、血压、饮食等因素对当日健康风险的边际贡献
    """
    
    # 特征名称映射
    FEATURE_NAMES = {
        0: 'age',
        1: 'gender', 
        2: 'height',
        3: 'weight',
        4: 'bmi',
        5: 'ap_hi',      # 收缩压
        6: 'ap_lo',      # 舒张压
        7: 'bp_diff',    # 血压差
        8: 'cholesterol',
        9: 'gluc',       # 血糖
        10: 'smoke',
        11: 'alco',
        12: 'active',    # 运动活跃度
    }
    
    # 风险阈值
    RISK_THRESHOLDS = {
        'bp_hi': {'warning': 140, 'high': 160},
        'bp_lo': {'warning': 90, 'high': 100},
        'bmi': {'warning': 25, 'high': 30},
        'gluc': {'warning': 100, 'high': 126},
        'cholesterol': {'warning': 2, 'high': 3},  # 分级编码
    }
    
    def __init__(self, model=None):
        """
        初始化 SHAP 解释器
        
        Args:
            model: 已训练的模型（用于获取预测）
        """
        self.model = model
        self._shap_available = False
        self._try_import_shap()
        
    def _try_import_shap(self):
        """尝试导入 SHAP 库"""
        try:
            import shap
            self.shap = shap
            self._shap_available = True
            logger.info("SHAP library loaded successfully")
        except ImportError:
            logger.warning("SHAP library not available. Using fallback explanation method.")
            self._shap_available = False
    
    def explain(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
        background_data: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        生成 SHAP 解释
        
        Args:
            X: 特征矩阵 (n_samples, n_features)
            feature_names: 特征名称列表
            background_data: 背景数据（用于计算基准）
            
        Returns:
            解释结果字典
        """
        if self._shap_available and self.model is not None:
            return self._explain_with_shap(X, feature_names, background_data)
        else:
            return self._explain_fallback(X, feature_names)
    
    def _explain_with_shap(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
        background_data: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """使用 SHAP 库进行解释"""
        # 创建解释器
        if hasattr(self.model, 'model'):
            # 集成模型
            model_to_explain = self.model.model if hasattr(self.model, 'model') else self.model
        else:
            model_to_explain = self.model
        
        # 使用 Kernel SHAP（适用于任何模型）
        if background_data is None:
            background_data = X[:100] if len(X) > 100 else X
        
        explainer = self.shap.KernelExplainer(
            lambda x: self.model.predict_proba(x)[:, 1] if hasattr(self.model, 'predict_proba') else self.model.predict(x),
            background_data
        )
        
        # 计算 SHAP 值（采样以提高速度）
        n_samples = min(len(X), 10)
        shap_values = explainer.shap_values(X[:n_samples], nsamples=100)
        
        # 获取平均绝对 SHAP 值（特征重要性）
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        
        # 获取单个样本的 SHAP 值
        sample_shap = shap_values if isinstance(shap_values, np.ndarray) else shap_values[1]
        
        return {
            'method': 'kernel_shap',
            'feature_names': feature_names or list(range(X.shape[1])),
            'mean_abs_shap': mean_abs_shap,
            'sample_shap_values': sample_shap,
            'feature_importance_ranking': np.argsort(-mean_abs_shap).tolist()
        }
    
    def _explain_fallback(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        回退解释方法（当 SHAP 不可用时）
        
        基于规则的特征贡献评估
        """
        feature_names = feature_names or [f'feature_{i}' for i in range(X.shape[1])]
        
        contributions = {}
        risk_levels = {}
        
        for i, name in enumerate(feature_names):
            if i >= X.shape[1]:
                continue
                
            values = X[:, i] if len(X.shape) > 1 else X
            mean_val = np.mean(values)
            
            # 基于规则的风险评估
            risk_level = 'normal'
            contribution = 0.0
            
            if 'ap_hi' in name or 'ap_lo' in name or 'bp' in name:
                # 血压因素
                if mean_val > self.RISK_THRESHOLDS['bp_hi']['high']:
                    risk_level = 'high'
                    contribution = 0.3
                elif mean_val > self.RISK_THRESHOLDS['bp_hi']['warning']:
                    risk_level = 'warning'
                    contribution = 0.15
                    
            elif 'bmi' in name or 'weight' in name:
                # BMI/体重因素
                if 'bmi' in name:
                    if mean_val > self.RISK_THRESHOLDS['bmi']['high']:
                        risk_level = 'high'
                        contribution = 0.25
                    elif mean_val > self.RISK_THRESHOLDS['bmi']['warning']:
                        risk_level = 'warning'
                        contribution = 0.12
                        
            elif 'gluc' in name:
                # 血糖因素
                if mean_val > self.RISK_THRESHOLDS['gluc']['high']:
                    risk_level = 'high'
                    contribution = 0.2
                elif mean_val > self.RISK_THRESHOLDS['gluc']['warning']:
                    risk_level = 'warning'
                    contribution = 0.1
                    
            elif 'smoke' in name:
                # 吸烟因素
                if mean_val > 0.5:
                    risk_level = 'high'
                    contribution = 0.2
                    
            elif 'active' in name:
                # 运动因素（负相关）
                if mean_val < 0.5:
                    risk_level = 'warning'
                    contribution = -0.1  # 负贡献表示保护因素
            
            contributions[name] = contribution
            risk_levels[name] = risk_level
        
        # 按贡献度排序
        sorted_features = sorted(contributions.items(), key=lambda x: -abs(x[1]))
        
        return {
            'method': 'rule_based',
            'feature_names': feature_names,
            'contributions': contributions,
            'risk_levels': risk_levels,
            'feature_importance_ranking': [f[0] for f in sorted_features]
        }
    
    def get_top_risk_factors(
        self,
        X_sample: np.ndarray,
        feature_names: Optional[List[str]] = None,
        top_k: int = 5
    ) -> List[RiskFactor]:
        """
        获取 Top-K 风险因素
        
        Args:
            X_sample: 单个样本特征
            feature_names: 特征名称
            top_k: 返回前 K 个因素
            
        Returns:
            风险因素列表
        """
        explanation = self.explain(X_sample.reshape(1, -1), feature_names)
        
        risk_factors = []
        
        if explanation['method'] == 'kernel_shap':
            shap_values = explanation['sample_shap_values'][0] if len(explanation['sample_shap_values'].shape) > 1 else explanation['sample_shap_values']
            for i in range(min(len(shap_values), len(feature_names or []))):
                factor = RiskFactor(
                    name=feature_names[i] if feature_names else f'feature_{i}',
                    value=float(X_sample[i]),
                    unit='',
                    shap_value=float(shap_values[i]),
                    risk_level='high' if abs(shap_values[i]) > 0.1 else 'normal'
                )
                risk_factors.append(factor)
        else:
            contributions = explanation['contributions']
            risk_levels = explanation['risk_levels']
            for name in explanation['feature_importance_ranking'][:top_k]:
                factor = RiskFactor(
                    name=name,
                    value=float(X_sample[list(contributions.keys()).index(name)]) if name in contributions else 0,
                    unit='',
                    shap_value=contributions.get(name, 0),
                    risk_level=risk_levels.get(name, 'normal')
                )
                risk_factors.append(factor)
        
        return sorted(risk_factors, key=lambda x: -abs(x.shap_value))[:top_k]


# ============== 决策引擎 ==============

class DecisionEngine:
    """
    健康干预决策引擎
    
    基于预测结果生成个性化干预建议
    """
    
    # 干预策略库
    INTERVENTION_STRATEGIES = {
        'blood_pressure': {
            'actions': [
                {
                    'action': 'decrease',
                    'target': 'salt_intake',
                    'description': '减少盐分摄入（每日<6g）',
                    'expected_reduction': 0.05
                },
                {
                    'action': 'increase',
                    'target': 'potassium_intake',
                    'description': '增加钾摄入（香蕉、土豆等）',
                    'expected_reduction': 0.03
                },
                {
                    'action': 'increase',
                    'target': 'aerobic_exercise',
                    'description': '增加有氧运动（每周 150 分钟）',
                    'expected_reduction': 0.08
                }
            ]
        },
        'bmi': {
            'actions': [
                {
                    'action': 'decrease',
                    'target': 'calorie_intake',
                    'description': '控制热量摄入（每日减少 500 大卡）',
                    'expected_reduction': 0.06
                },
                {
                    'action': 'increase',
                    'target': 'physical_activity',
                    'description': '增加日常活动量',
                    'expected_reduction': 0.05
                },
                {
                    'action': 'maintain',
                    'target': 'sleep_schedule',
                    'description': '保持规律作息（7-8 小时睡眠）',
                    'expected_reduction': 0.02
                }
            ]
        },
        'blood_glucose': {
            'actions': [
                {
                    'action': 'decrease',
                    'target': 'sugar_intake',
                    'description': '减少精制糖摄入',
                    'expected_reduction': 0.07
                },
                {
                    'action': 'increase',
                    'target': 'fiber_intake',
                    'description': '增加膳食纤维摄入',
                    'expected_reduction': 0.04
                }
            ]
        },
        'lifestyle': {
            'actions': [
                {
                    'action': 'decrease',
                    'target': 'smoking',
                    'description': '减少或戒烟',
                    'expected_reduction': 0.15
                },
                {
                    'action': 'decrease',
                    'target': 'alcohol',
                    'description': '限制酒精摄入',
                    'expected_reduction': 0.05
                },
                {
                    'action': 'increase',
                    'target': 'steps',
                    'description': '增加每日步数（目标 10000 步）',
                    'expected_reduction': 0.04
                }
            ]
        }
    }
    
    # 风险级别映射
    RISK_LEVEL_MAP = {
        'low': (0, 0.3),
        'medium': (0.3, 0.6),
        'high': (0.6, 1.0)
    }
    
    def __init__(self):
        self.strategy_db = self.INTERVENTION_STRATEGIES
    
    def generate_recommendations(
        self,
        prediction: PredictionResult,
        user_profile: Optional[UserHealthProfile] = None
    ) -> List[InterventionRecommendation]:
        """
        生成干预建议清单
        
        Args:
            prediction: 预测结果
            user_profile: 用户档案（可选）
            
        Returns:
            干预建议列表
        """
        recommendations = []
        
        # 根据风险级别决定干预强度
        risk_level = prediction.risk_level
        risk_prob = prediction.risk_probability
        
        # 分析各风险因素的贡献
        for factor in prediction.risk_factors:
            if factor.shap_value <= 0:  # 只关注正向风险因素
                continue
                
            # 匹配干预策略
            strategy = self._match_strategy(factor.name)
            if strategy:
                for action in strategy['actions']:
                    rec = InterventionRecommendation(
                        factor=factor.name,
                        current_value=factor.value,
                        target_value=self._calculate_target(factor.name, factor.value),
                        action=action['action'],
                        priority=self._calculate_priority(factor, risk_prob),
                        description=action['description'],
                        expected_risk_reduction=action['expected_reduction']
                    )
                    recommendations.append(rec)
        
        # 按优先级排序
        recommendations.sort(key=lambda x: x.priority)
        
        return recommendations
    
    def _match_strategy(self, factor_name: str) -> Optional[Dict]:
        """匹配干预策略"""
        factor_lower = factor_name.lower()
        
        if any(x in factor_lower for x in ['ap_hi', 'ap_lo', 'bp', 'blood_pressure']):
            return self.strategy_db.get('blood_pressure')
        elif any(x in factor_lower for x in ['bmi', 'weight']):
            return self.strategy_db.get('bmi')
        elif any(x in factor_lower for x in ['gluc', 'glucose', 'blood_glucose']):
            return self.strategy_db.get('blood_glucose')
        else:
            return self.strategy_db.get('lifestyle')
    
    def _calculate_target(self, factor_name: str, current_value: float) -> float:
        """计算目标值"""
        factor_lower = factor_name.lower()
        
        if 'ap_hi' in factor_lower:
            return min(current_value, 120)
        elif 'ap_lo' in factor_lower:
            return min(current_value, 80)
        elif 'bmi' in factor_lower:
            return min(current_value, 24)
        elif 'gluc' in factor_lower:
            return min(current_value, 100)
        else:
            return current_value * 0.9  # 默认降低 10%
    
    def _calculate_priority(self, factor: RiskFactor, risk_prob: float) -> int:
        """计算优先级（1-5，1 最高）"""
        # 基于 SHAP 值和风险概率
        impact = abs(factor.shap_value) * risk_prob
        
        if impact > 0.2:
            return 1
        elif impact > 0.1:
            return 2
        elif impact > 0.05:
            return 3
        elif impact > 0.02:
            return 4
        else:
            return 5
    
    def create_intervention_plan(
        self,
        prediction: PredictionResult,
        max_recommendations: int = 5
    ) -> Dict[str, Any]:
        """
        创建完整干预计划
        
        Args:
            prediction: 预测结果
            max_recommendations: 最大建议数
            
        Returns:
            干预计划字典
        """
        recommendations = self.generate_recommendations(prediction)
        
        plan = {
            'user_id': prediction.user_id,
            'prediction_date': prediction.prediction_date.isoformat(),
            'risk_level': prediction.risk_level,
            'risk_probability': prediction.risk_probability,
            'recommendations': [
                {
                    'factor': r.factor,
                    'action': r.action,
                    'description': r.description,
                    'priority': r.priority,
                    'expected_risk_reduction': r.expected_risk_reduction
                }
                for r in recommendations[:max_recommendations]
            ],
            'total_expected_reduction': sum(
                r.expected_risk_reduction for r in recommendations[:max_recommendations]
            ),
            'next_review_date': (prediction.prediction_date + timedelta(days=7)).isoformat()
        }
        
        return plan


# ============== 激励匹配系统 ==============

class HealthPointsSystem:
    """
    健康积分激励系统
    
    根据用户执行情况发放健康积分
    """
    
    # 积分规则
    POINT_RULES = {
        'daily_prediction': 10,      # 每日预测
        'complete_recommendation': 20,  # 完成建议
        'weekly_streak': 50,         # 连续 7 天
        'risk_improvement': 30,      # 风险改善
        'goal_achieved': 100,        # 达成目标
    }
    
    # 等级阈值
    LEVEL_THRESHOLDS = {
        'bronze': (0, 100),
        'silver': (100, 500),
        'gold': (500, 2000),
        'platinum': (2000, float('inf'))
    }
    
    def __init__(self):
        self.point_rules = self.POINT_RULES
        self.user_profiles: Dict[str, UserHealthProfile] = {}
    
    def get_or_create_profile(self, user_id: str) -> UserHealthProfile:
        """获取或创建用户档案"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserHealthProfile(
                user_id=user_id,
                created_at=datetime.now()
            )
        return self.user_profiles[user_id]
    
    def record_prediction(
        self,
        user_id: str,
        prediction: PredictionResult
    ) -> int:
        """
        记录预测并发放积分
        
        Returns:
            获得的积分
        """
        profile = self.get_or_create_profile(user_id)
        profile.predictions.append(prediction)
        
        # 发放积分
        points_earned = self.point_rules['daily_prediction']
        profile.health_points.earned_points += points_earned
        profile.health_points.total_points += points_earned
        profile.health_points.last_activity = datetime.now()
        
        # 更新等级
        self._update_level(profile.health_points)
        
        return points_earned
    
    def record_execution(
        self,
        user_id: str,
        recommendation_idx: int,
        completed: bool,
        details: Optional[Dict] = None
    ) -> int:
        """
        记录建议执行情况并发放积分
        
        Returns:
            获得的积分
        """
        profile = self.get_or_create_profile(user_id)
        
        record = {
            'recommendation_idx': recommendation_idx,
            'completed': completed,
            'timestamp': datetime.now(),
            'details': details or {}
        }
        profile.execution_records.append(record)
        
        points_earned = 0
        if completed:
            points_earned = self.point_rules['complete_recommendation']
            profile.health_points.earned_points += points_earned
            profile.health_points.total_points += points_earned
            
            # 更新连续天数
            self._update_streak(profile)
        
        # 更新等级
        self._update_level(profile.health_points)
        
        return points_earned
    
    def evaluate_risk_improvement(
        self,
        user_id: str,
        previous_risk: float,
        current_risk: float
    ) -> int:
        """
        评估风险改善并发放积分
        
        Returns:
            获得的积分
        """
        if current_risk < previous_risk:
            profile = self.get_or_create_profile(user_id)
            points_earned = self.point_rules['risk_improvement']
            profile.health_points.earned_points += points_earned
            profile.health_points.total_points += points_earned
            self._update_level(profile.health_points)
            return points_earned
        return 0
    
    def _update_level(self, health_points: HealthPoints) -> None:
        """更新用户等级"""
        total = health_points.total_points
        
        for level, (min_pts, max_pts) in self.LEVEL_THRESHOLDS.items():
            if min_pts <= total < max_pts:
                health_points.level = level
                break
    
    def _update_streak(self, profile: UserHealthProfile) -> None:
        """更新连续天数"""
        last_activity = profile.health_points.last_activity
        if last_activity:
            days_since = (datetime.now() - last_activity).days
            if days_since <= 1:
                profile.health_points.streak_days += 1
                # 周连续奖励
                if profile.health_points.streak_days % 7 == 0:
                    bonus = self.point_rules['weekly_streak']
                    profile.health_points.earned_points += bonus
                    profile.health_points.total_points += bonus
            else:
                profile.health_points.streak_days = 1
    
    def get_user_summary(self, user_id: str) -> Dict[str, Any]:
        """获取用户摘要"""
        profile = self.get_or_create_profile(user_id)
        hp = profile.health_points
        
        return {
            'user_id': user_id,
            'total_points': hp.total_points,
            'earned_points': hp.earned_points,
            'spent_points': hp.spent_points,
            'level': hp.level,
            'streak_days': hp.streak_days,
            'last_activity': hp.last_activity.isoformat() if hp.last_activity else None,
            'predictions_count': len(profile.predictions),
            'execution_count': len(profile.execution_records)
        }
    
    def spend_points(
        self,
        user_id: str,
        points: int,
        reason: str
    ) -> bool:
        """
        消费积分
        
        Returns:
            是否成功
        """
        profile = self.get_or_create_profile(user_id)
        
        if profile.health_points.total_points >= points:
            profile.health_points.total_points -= points
            profile.health_points.spent_points += points
            return True
        return False


# ============== 服务层主类 ==============

class HealthMindService:
    """
    HealthMind 服务层
    
    整合 SHAP 解释、决策引擎和激励系统
    """
    
    def __init__(self, model=None):
        """
        初始化服务层
        
        Args:
            model: 预测模型
        """
        self.shap_explainer = SHAPExplainer(model)
        self.decision_engine = DecisionEngine()
        self.points_system = HealthPointsSystem()
        self.model = model
    
    def predict_and_explain(
        self,
        user_id: str,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> PredictionResult:
        """
        预测并生成解释
        
        Args:
            user_id: 用户 ID
            X: 特征矩阵
            feature_names: 特征名称
            
        Returns:
            预测结果
        """
        # 获取预测
        if hasattr(self.model, 'predict_proba'):
            risk_prob = float(self.model.predict_proba(X)[0, 1])
        else:
            risk_prob = float(self.model.predict(X)[0])
        
        # 风险级别
        if risk_prob < 0.3:
            risk_level = 'low'
        elif risk_prob < 0.6:
            risk_level = 'medium'
        else:
            risk_level = 'high'
        
        # 获取风险因素
        risk_factors = self.shap_explainer.get_top_risk_factors(
            X[0] if len(X.shape) > 1 else X,
            feature_names,
            top_k=5
        )
        
        # SHAP 贡献
        shap_contributions = {f.name: f.shap_value for f in risk_factors}
        
        prediction = PredictionResult(
            user_id=user_id,
            prediction_date=datetime.now(),
            risk_probability=risk_prob,
            risk_level=risk_level,
            risk_factors=risk_factors,
            shap_contributions=shap_contributions
        )
        
        # 记录并发放积分
        self.points_system.record_prediction(user_id, prediction)
        
        return prediction
    
    def generate_intervention_plan(
        self,
        prediction: PredictionResult
    ) -> Dict[str, Any]:
        """
        生成干预计划
        
        Args:
            prediction: 预测结果
            
        Returns:
            干预计划字典
        """
        return self.decision_engine.create_intervention_plan(prediction)
    
    def record_user_action(
        self,
        user_id: str,
        recommendation_idx: int,
        completed: bool,
        details: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        记录用户行为
        
        Args:
            user_id: 用户 ID
            recommendation_idx: 建议索引
            completed: 是否完成
            details: 详细信息
            
        Returns:
            积分更新结果
        """
        points = self.points_system.record_execution(
            user_id, recommendation_idx, completed, details
        )
        
        summary = self.points_system.get_user_summary(user_id)
        summary['points_earned'] = points
        
        return summary
    
    def get_user_dashboard(self, user_id: str) -> Dict[str, Any]:
        """
        获取用户仪表板
        
        Args:
            user_id: 用户 ID
            
        Returns:
            仪表板数据
        """
        profile = self.points_system.get_or_create_profile(user_id)
        
        # 最近预测
        recent_predictions = profile.predictions[-5:] if profile.predictions else []
        
        # 风险趋势
        risk_trend = [p.risk_probability for p in recent_predictions]
        
        # 积分摘要
        points_summary = self.points_system.get_user_summary(user_id)
        
        return {
            'user_id': user_id,
            'current_risk': recent_predictions[-1].risk_probability if recent_predictions else None,
            'risk_level': recent_predictions[-1].risk_level if recent_predictions else None,
            'risk_trend': risk_trend,
            'points_summary': points_summary,
            'top_risk_factors': [
                {
                    'name': f.name,
                    'shap_value': f.shap_value,
                    'risk_level': f.risk_level
                }
                for f in recent_predictions[-1].risk_factors
            ] if recent_predictions else [],
            'pending_recommendations': [
                {
                    'factor': r.factor,
                    'description': r.description,
                    'priority': r.priority
                }
                for r in profile.recommendations
            ] if profile.recommendations else []
        }


# ============== 便捷函数 ==============

def create_service(model=None) -> HealthMindService:
    """创建服务层实例"""
    return HealthMindService(model)


# ============== 主函数 ==============

if __name__ == '__main__':
    # 示例用法
    print("=" * 60)
    print("HealthMind 服务层 - 示例运行")
    print("=" * 60)
    
    # 创建模拟数据
    np.random.seed(42)
    n_features = 17
    
    # 模拟特征
    X_sample = np.random.randn(1, n_features)
    X_sample[0, 0] = 55  # 年龄
    X_sample[0, 5] = 150  # 收缩压
    X_sample[0, 6] = 95   # 舒张压
    
    feature_names = [
        'age', 'gender', 'height', 'weight', 'bmi',
        'ap_hi', 'ap_lo', 'bp_diff', 'cholesterol', 'gluc',
        'smoke', 'alco', 'active', 'bmi_encoded', 'bp_encoded'
    ]
    
    # 创建服务（无模型模式）
    service = create_service()
    
    # 模拟预测结果
    prediction = PredictionResult(
        user_id='user_001',
        prediction_date=datetime.now(),
        risk_probability=0.45,
        risk_level='medium',
        risk_factors=[
            RiskFactor(name='ap_hi', value=150, unit='mmHg', shap_value=0.15, risk_level='warning'),
            RiskFactor(name='ap_lo', value=95, unit='mmHg', shap_value=0.12, risk_level='warning'),
            RiskFactor(name='age', value=55, unit='years', shap_value=0.08, risk_level='normal'),
            RiskFactor(name='bmi', value=27, unit='', shap_value=0.06, risk_level='warning'),
        ],
        shap_contributions={'ap_hi': 0.15, 'ap_lo': 0.12, 'age': 0.08, 'bmi': 0.06}
    )
    
    print(f"\n用户：{prediction.user_id}")
    print(f"风险概率：{prediction.risk_probability:.2%}")
    print(f"风险级别：{prediction.risk_level}")
    
    print("\n风险因素分析:")
    for factor in prediction.risk_factors:
        print(f"  - {factor.name}: {factor.value} (SHAP={factor.shap_value:.3f}, {factor.risk_level})")
    
    # 生成干预计划
    plan = service.decision_engine.create_intervention_plan(prediction)
    
    print("\n干预计划:")
    for i, rec in enumerate(plan['recommendations'], 1):
        print(f"  {i}. [{rec['priority']}] {rec['description']}")
    
    print(f"\n预期风险降低：{plan['total_expected_reduction']:.1%}")
    
    # 积分系统
    points_summary = service.points_system.get_user_summary('user_001')
    
    print("\n健康积分:")
    print(f"  总积分：{points_summary['total_points']}")
    print(f"  等级：{points_summary['level']}")
    print(f"  连续天数：{points_summary['streak_days']}")
    
    print("\n" + "=" * 60)
    print("示例运行完成!")
    print("=" * 60)