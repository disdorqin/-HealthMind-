"""
HealthMind 心血管数据处理模块

本模块提供心血管数据的加载、特征工程、时序模拟和数据管道功能，
支持 XGBoost 和深度学习（LSTM/GRU）模型的训练需求。
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


# ============== 健康基线库 ==============
class HealthBaseline:
    """健康指标基线库，用于 BMI 等指标的分箱处理"""
    
    # BMI 分类标准（WHO 标准）
    BMI_BINS = {
        'underweight': (-float('inf'), 18.5),
        'normal': (18.5, 24.9),
        'overweight': (24.9, 29.9),
        'obese': (29.9, float('inf'))
    }
    
    # 血压分类标准（mmHg）
    BP_BINS = {
        'optimal': (0, 120, 80),      # (收缩压上限，舒张压上限)
        'normal': (120, 130, 80, 85),
        'high_normal': (130, 140, 85, 90),
        'hypertension_1': (140, 160, 90, 100),
        'hypertension_2': (160, float('inf'), 100, float('inf'))
    }
    
    # 年龄分段
    AGE_BINS = {
        'young': (0, 35),
        'middle': (35, 50),
        'senior': (50, 65),
        'elderly': (65, float('inf'))
    }


@dataclass
class DataConfig:
    """数据配置类"""
    data_path: str = 'data/cardio_train.csv'
    separator: str = ';'
    time_window: int = 7  # 时序窗口大小（天）
    random_seed: int = 42
    test_size: float = 0.2


class CardioDataProcessor:
    """
    心血管数据处理器
    
    功能：
    1. 数据加载
    2. 特征转换（年龄、BMI、血压差等）
    3. 时序模拟（滑动窗口）
    4. 数据管道（XGBoost 和深度学习）
    """
    
    def __init__(self, config: Optional[DataConfig] = None):
        """
        初始化数据处理器
        
        Args:
            config: 数据配置对象，默认使用 DataConfig 默认值
        """
        self.config = config or DataConfig()
        self.data: Optional[pd.DataFrame] = None
        self.feature_columns: List[str] = []
        self.baseline = HealthBaseline()
        self.feature_stats: Dict = {}
        
    # ============== 1. 数据加载 ==============
    
    def load_data(self, path: Optional[str] = None) -> pd.DataFrame:
        """
        加载心血管数据集
        
        Args:
            path: 数据文件路径，默认使用配置中的路径
            
        Returns:
            加载的 DataFrame
            
        Raises:
            FileNotFoundError: 文件不存在时抛出
            pd.errors.EmptyDataError: 文件为空时抛出
        """
        data_path = path or self.config.data_path
        
        print(f"[INFO] 正在加载数据：{data_path}")
        self.data = pd.read_csv(data_path, sep=self.config.separator)
        
        print(f"[INFO] 数据加载完成，形状：{self.data.shape}")
        print(f"[INFO] 列名：{list(self.data.columns)}")
        
        return self.data
    
    # ============== 2. 特征转换 ==============
    
    def convert_age_to_years(self, inplace: bool = True) -> pd.DataFrame:
        """
        将年龄从天数转换为岁数
        
        Args:
            inplace: 是否就地修改数据
            
        Returns:
            处理后的 DataFrame
        """
        if self.data is None:
            raise ValueError("请先调用 load_data() 加载数据")
            
        df = self.data if inplace else self.data.copy()
        
        # 将天数转换为年数（保留 2 位小数）
        df['age_years'] = (df['age'] / 365.25).round(2)
        
        # 添加年龄分段
        df['age_group'] = pd.cut(
            df['age_years'],
            bins=[0, 35, 50, 65, 100],
            labels=['young', 'middle', 'senior', 'elderly']
        )
        
        if not inplace:
            return df
        self.data = df
        return df
    
    def calculate_bmi(self, inplace: bool = True) -> pd.DataFrame:
        """
        计算 BMI 指数并进行分箱处理
        
        BMI = weight / (height/100)^2
        
        Args:
            inplace: 是否就地修改数据
            
        Returns:
            处理后的 DataFrame
        """
        if self.data is None:
            raise ValueError("请先调用 load_data() 加载数据")
            
        df = self.data if inplace else self.data.copy()
        
        # 计算 BMI：体重 (kg) / 身高 (m)^2
        height_m = df['height'] / 100
        df['bmi'] = (df['weight'] / (height_m ** 2)).round(2)
        
        # BMI 分箱
        bmi_bins = [-float('inf'), 18.5, 24.9, 29.9, float('inf')]
        bmi_labels = ['underweight', 'normal', 'overweight', 'obese']
        
        df['bmi_category'] = pd.cut(
            df['bmi'],
            bins=bmi_bins,
            labels=bmi_labels
        )
        
        # BMI 数值编码（用于模型训练）
        bmi_encoding = {
            'underweight': 0,
            'normal': 1,
            'overweight': 2,
            'obese': 3
        }
        df['bmi_encoded'] = df['bmi_category'].map(bmi_encoding)
        
        if not inplace:
            return df
        self.data = df
        return df
    
    def extract_blood_pressure_features(self, inplace: bool = True) -> pd.DataFrame:
        """
        提取血压相关特征
        
        包括：
        - 血压差（脉压差）= 收缩压 - 舒张压
        - 平均动脉压 = (收缩压 + 2*舒张压) / 3
        - 血压分类
        
        Args:
            inplace: 是否就地修改数据
            
        Returns:
            处理后的 DataFrame
        """
        if self.data is None:
            raise ValueError("请先调用 load_data() 加载数据")
            
        df = self.data if inplace else self.data.copy()
        
        # 血压差（脉压差）
        df['bp_diff'] = df['ap_hi'] - df['ap_lo']
        
        # 平均动脉压
        df['map'] = ((df['ap_hi'] + 2 * df['ap_lo']) / 3).round(2)
        
        # 收缩压/舒张压比值
        df['bp_ratio'] = (df['ap_hi'] / df['ap_lo']).round(2)
        
        # 血压分类
        def classify_bp(row):
            hi, lo = row['ap_hi'], row['ap_lo']
            if hi < 120 and lo < 80:
                return 'optimal'
            elif hi < 130 and lo < 85:
                return 'normal'
            elif hi < 140 and lo < 90:
                return 'high_normal'
            elif hi < 160 and lo < 100:
                return 'hypertension_1'
            else:
                return 'hypertension_2'
        
        df['bp_category'] = df.apply(classify_bp, axis=1)
        
        # 血压分类编码
        bp_encoding = {
            'optimal': 0,
            'normal': 1,
            'high_normal': 2,
            'hypertension_1': 3,
            'hypertension_2': 4
        }
        df['bp_encoded'] = df['bp_category'].map(bp_encoding)
        
        if not inplace:
            return df
        self.data = df
        return df
    
    def process_all_features(self) -> pd.DataFrame:
        """
        处理所有特征
        
        Returns:
            处理后的 DataFrame
        """
        print("[INFO] 开始特征处理...")
        
        self.convert_age_to_years()
        print("  [OK] 年龄转换完成")
        
        self.calculate_bmi()
        print("  [OK] BMI 计算完成")
        
        self.extract_blood_pressure_features()
        print("  [OK] 血压特征提取完成")
        
        # 定义特征列
        self.feature_columns = [
            'age_years', 'gender', 'height', 'weight', 'bmi',
            'ap_hi', 'ap_lo', 'bp_diff', 'map', 'bp_ratio',
            'cholesterol', 'gluc', 'smoke', 'alco', 'active',
            'bmi_encoded', 'bp_encoded'
        ]
        
        print(f"[INFO] 特征处理完成，共 {len(self.feature_columns)} 个特征")
        
        return self.data
    
    # ============== 3. 时序模拟 ==============
    
    def simulate_time_series(
        self,
        window_size: int = 7,
        n_samples_per_id: int = 5,
        noise_level: float = 0.05
    ) -> pd.DataFrame:
        """
        使用滑动窗口算法为每个 ID 生成模拟的时序序列
        
        由于原始数据是横断面数据，此方法通过添加小幅随机扰动
        来模拟时间序列数据，以支持 LSTM/GRU 模型的训练。
        
        Args:
            window_size: 时间窗口大小（天数）
            n_samples_per_id: 每个 ID 生成的样本数
            noise_level: 噪声水平（标准差比例）
            
        Returns:
            模拟的时序数据 DataFrame
        """
        if self.data is None:
            raise ValueError("请先调用 load_data() 加载数据")
            
        print(f"[INFO] 开始时序模拟，窗口大小：{window_size} 天")
        
        np.random.seed(self.config.random_seed)
        
        # 选择用于时序模拟的数值特征
        numeric_features = [
            'age_years', 'bmi', 'ap_hi', 'ap_lo', 'bp_diff',
            'cholesterol', 'gluc'
        ]
        
        simulated_data = []
        
        for idx, row in self.data.iterrows():
            # 为每个样本生成多个时间步
            for sample_idx in range(n_samples_per_id):
                time_step_data = []
                
                for t in range(window_size):
                    # 复制当前行
                    row_copy = row.copy()
                    
                    # 为数值特征添加时间相关的微小扰动
                    for feat in numeric_features:
                        if feat in row_copy.index:
                            # 使用绝对值确保标准差为正，并设置最小值
                            scale = max(abs(row_copy[feat]) * noise_level, 0.01)
                            noise = np.random.normal(0, scale)
                            row_copy[feat] = max(0, row_copy[feat] + noise)
                    
                    # 添加时间步信息
                    row_copy['time_step'] = t
                    row_copy['sample_id'] = f"{row['id']}_{sample_idx}"
                    
                    time_step_data.append(row_copy)
                
                simulated_data.extend(time_step_data)
        
        simulated_df = pd.DataFrame(simulated_data)
        print(f"[INFO] 时序模拟完成，生成 {len(simulated_df)} 条记录")
        
        return simulated_df
    
    def create_sliding_window_sequences(
        self,
        window_size: int = 7,
        stride: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建滑动窗口序列（用于深度学习模型）
        
        Args:
            window_size: 窗口大小
            stride: 滑动步长
            
        Returns:
            (X_sequences, y_labels) 元组
        """
        if self.data is None:
            raise ValueError("请先调用 load_data() 加载数据")
            
        print(f"[INFO] 创建滑动窗口序列，窗口大小：{window_size}")
        
        # 获取特征矩阵和标签
        X = self.data[self.feature_columns].values
        y = self.data['cardio'].values
        
        sequences = []
        labels = []
        
        # 由于是横断面数据，我们创建重叠的窗口
        for i in range(0, len(X) - window_size + 1, stride):
            seq = X[i:i + window_size]
            # 使用该窗口内样本的平均标签
            label = int(np.mean(y[i:i + window_size]) > 0.5)
            sequences.append(seq)
            labels.append(label)
        
        X_seq = np.array(sequences)
        y_seq = np.array(labels)
        
        print(f"[INFO] 创建完成，序列形状：{X_seq.shape}")
        
        return X_seq, y_seq
    
    # ============== 4. 数据管道 ==============
    
    def prepare_xgboost_data(
        self,
        include_categorical: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        准备 XGBoost 模型数据（2D 展平特征）
        
        Args:
            include_categorical: 是否包含分类特征编码
            
        Returns:
            (X, y, feature_names) 元组
        """
        if self.data is None:
            raise ValueError("请先调用 load_data() 加载数据")
            
        print("[INFO] 准备 XGBoost 数据...")
        
        # 选择特征列
        feature_cols = self.feature_columns.copy()
        
        if not include_categorical:
            # 移除编码后的分类特征
            feature_cols = [c for c in feature_cols if not c.endswith('_encoded')]
        
        X = self.data[feature_cols].values
        y = self.data['cardio'].values
        
        print(f"[INFO] XGBoost 数据准备完成")
        print(f"  - 特征形状：{X.shape}")
        print(f"  - 标签形状：{y.shape}")
        
        return X, y, feature_cols
    
    def prepare_deep_learning_data(
        self,
        window_size: int = 7,
        include_categorical: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        准备深度学习模型数据（3D 张量：samples, timesteps, features）
        
        Args:
            window_size: 时间窗口大小
            include_categorical: 是否包含分类特征编码
            
        Returns:
            (X_3d, y, feature_names) 元组
        """
        if self.data is None:
            raise ValueError("请先调用 load_data() 加载数据")
            
        print(f"[INFO] 准备深度学习数据，窗口大小：{window_size}")
        
        # 获取 2D 特征
        X_2d, y, feature_cols = self.prepare_xgboost_data(include_categorical)
        
        # 转换为 3D 张量
        n_samples = len(X_2d)
        n_features = X_2d.shape[1]
        
        # 创建 3D 张量 (samples, timesteps, features)
        X_3d = np.zeros((n_samples, window_size, n_features))
        
        for i in range(n_samples):
            for t in range(window_size):
                # 为每个时间步添加轻微扰动以模拟时序变化
                noise = np.random.normal(0, 0.01, n_features)
                X_3d[i, t, :] = X_2d[i] + noise
        
        print(f"[INFO] 深度学习数据准备完成")
        print(f"  - 输入形状：{X_3d.shape} (samples, timesteps, features)")
        print(f"  - 标签形状：{y.shape}")
        
        return X_3d, y, feature_cols
    
    def get_data_pipeline(
        self,
        model_type: str = 'xgboost',
        window_size: int = 7,
        **kwargs
    ) -> Dict:
        """
        获取完整的数据管道
        
        Args:
            model_type: 模型类型 ('xgboost' 或 'deep_learning')
            window_size: 时间窗口大小（仅对深度学习有效）
            **kwargs: 其他参数
            
        Returns:
            包含数据管道各组件的字典
        """
        pipeline = {
            'config': self.config,
            'feature_columns': self.feature_columns,
            'feature_stats': self.feature_stats
        }
        
        if model_type == 'xgboost':
            X, y, features = self.prepare_xgboost_data(**kwargs)
            pipeline.update({
                'model_type': 'xgboost',
                'X': X,
                'y': y,
                'feature_names': features,
                'input_shape': X.shape
            })
            
        elif model_type == 'deep_learning':
            X_3d, y, features = self.prepare_deep_learning_data(
                window_size=window_size,
                **kwargs
            )
            pipeline.update({
                'model_type': 'deep_learning',
                'X': X_3d,
                'y': y,
                'feature_names': features,
                'input_shape': X_3d.shape,
                'window_size': window_size
            })
        else:
            raise ValueError(f"不支持的模型类型：{model_type}")
        
        return pipeline
    
    # ============== 5. 数据探索与统计 ==============
    
    def compute_feature_statistics(self) -> Dict:
        """
        计算特征统计信息
        
        Returns:
            统计信息字典
        """
        if self.data is None:
            raise ValueError("请先调用 load_data() 加载数据")
        
        self.feature_stats = {}
        
        for col in self.feature_columns:
            if col in self.data.columns:
                # 跳过分类列（非数值类型）
                if self.data[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                    self.feature_stats[col] = {
                        'mean': float(self.data[col].mean()),
                        'std': float(self.data[col].std()),
                        'min': float(self.data[col].min()),
                        'max': float(self.data[col].max()),
                        'median': float(self.data[col].median())
                    }
                else:
                    # 对于分类列，只记录唯一值和频次
                    value_counts = self.data[col].value_counts().to_dict()
                    self.feature_stats[col] = {
                        'dtype': str(self.data[col].dtype),
                        'unique_values': list(value_counts.keys())[:5],
                        'value_counts': {str(k): int(v) for k, v in list(value_counts.items())[:5]}
                    }
        
        return self.feature_stats
    
    def get_data_summary(self) -> str:
        """
        获取数据摘要
        
        Returns:
            数据摘要字符串
        """
        if self.data is None:
            return "请先加载数据"
        
        summary = []
        summary.append("=" * 50)
        summary.append("HealthMind 心血管数据摘要")
        summary.append("=" * 50)
        summary.append(f"总样本数：{len(self.data)}")
        summary.append(f"特征数：{len(self.feature_columns)}")
        summary.append(f"正样本（患病）: {self.data['cardio'].sum()} ({self.data['cardio'].mean()*100:.2f}%)")
        summary.append(f"负样本（健康）: {(1-self.data['cardio']).sum()} ({(1-self.data['cardio'].mean())*100:.2f}%)")
        summary.append("")
        summary.append("特征列表:")
        for i, col in enumerate(self.feature_columns, 1):
            summary.append(f"  {i}. {col}")
        summary.append("=" * 50)
        
        return "\n".join(summary)


# ============== 便捷函数 ==============

def load_and_process(
    data_path: str = 'data/cardio_train.csv',
    model_type: str = 'xgboost',
    window_size: int = 7
) -> Dict:
    """
    一键加载和处理数据的便捷函数
    
    Args:
        data_path: 数据文件路径
        model_type: 模型类型 ('xgboost' 或 'deep_learning')
        window_size: 时间窗口大小
        
    Returns:
        数据管道字典
    """
    config = DataConfig(data_path=data_path)
    processor = CardioDataProcessor(config)
    
    # 加载和处理数据
    processor.load_data()
    processor.process_all_features()
    processor.compute_feature_statistics()
    
    # 获取数据管道
    pipeline = processor.get_data_pipeline(
        model_type=model_type,
        window_size=window_size
    )
    
    print(processor.get_data_summary())
    
    return pipeline


# ============== 主函数 ==============

if __name__ == '__main__':
    # 示例用法
    print("=" * 60)
    print("HealthMind 心血管数据处理模块 - 示例运行")
    print("=" * 60)
    
    # 创建处理器
    config = DataConfig(
        data_path='data/cardio_train.csv',
        random_seed=42
    )
    processor = CardioDataProcessor(config)
    
    # 加载数据
    processor.load_data()
    
    # 处理特征
    processor.process_all_features()
    
    # 计算统计信息
    processor.compute_feature_statistics()
    
    # 打印数据摘要
    print(processor.get_data_summary())
    
    # 准备 XGBoost 数据
    print("\n[测试] XGBoost 数据管道")
    X_xgb, y_xgb, features = processor.prepare_xgboost_data()
    print(f"XGBoost 输入形状：{X_xgb.shape}")
    
    # 准备深度学习数据
    print("\n[测试] 深度学习数据管道")
    X_dl, y_dl, features = processor.prepare_deep_learning_data(window_size=7)
    print(f"深度学习输入形状：{X_dl.shape}")
    
    # 时序模拟测试（仅测试前 100 条样本以加快运行速度）
    print("\n[测试] 时序模拟 (采样测试)")
    processor.data = processor.data.head(100)
    ts_data = processor.simulate_time_series(window_size=7, n_samples_per_id=2)
    print(f"时序数据形状：{ts_data.shape}")
    
    print("\n" + "=" * 60)
    print("示例运行完成!")
    print("=" * 60)
