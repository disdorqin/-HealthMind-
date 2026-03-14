#!/usr/bin/env python3
"""
本地一键测试脚本 - 数据导入->模型训练->启动前端

功能：
1. 检查数据文件
2. 自动训练模型（如果不存在）
3. 启动 Streamlit 应用

使用：
    python main.py              # 默认启动 Streamlit 应用
    python main.py --train-only # 仅训练模型
    python main.py --help       # 显示帮助
"""

import argparse
import sys
import subprocess
import time
from pathlib import Path
from typing import Tuple

import pandas as pd

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.core.utils.logger import logger
from src.logic import BusinessLogic


# ============================================================
# 1. 路径配置（跨平台兼容）
# ============================================================

def get_paths() -> dict:
    """获取所有重要的路径"""
    return {
        'root': project_root,
        'data': project_root / 'data' / 'data.csv',
        'model': project_root / 'models' / 'lstm_forecaster.pth',
        'app': project_root / 'app.py',
    }


# ============================================================
# 2. 数据层检测
# ============================================================

def check_data_layer() -> bool:
    """
    检查数据层 - 验证数据文件是否存在
    
    Returns:
        True 如果数据文件存在，False 否则
    """
    logger.info("\n" + "="*80)
    logger.info("[1/4] 数据层检测 - 验证数据文件")
    logger.info("="*80)
    
    paths = get_paths()
    data_path = paths['data']
    
    if data_path.exists():
        logger.info(f"✓ 数据文件存在：{data_path}")
        
        # 获取文件信息
        file_size = data_path.stat().st_size / (1024 * 1024)
        logger.info(f"  文件大小：{file_size:.2f} MB")
        
        # 尝试读取前几行验证格式
        try:
            df = pd.read_csv(data_path, nrows=5)
            logger.info(f"  数据形状：{df.shape}")
            logger.info(f"  列名：{list(df.columns)[:5]}")
            logger.info("✓ 数据层检测通过")
            return True
        except Exception as e:
            logger.error(f"✗ 数据文件格式错误：{str(e)}")
            return False
    else:
        logger.error(f"✗ 数据文件不存在：{data_path}")
        logger.warning("  请确保数据文件位置正确")
        return False


# ============================================================
# 3. 模型层检测
# ============================================================

def check_model_layer() -> bool:
    """
    检查模型层 - 验证模型文件是否完整
    
    Returns:
        True 如果模型文件存在，False 否则
    """
    logger.info("\n" + "="*80)
    logger.info("[2/4] 模型校验 - 检查模型文件")
    logger.info("="*80)
    
    paths = get_paths()
    model_path = paths['model']
    
    if model_path.exists():
        logger.info(f"✓ 模型文件存在：{model_path}")
        file_size = model_path.stat().st_size / (1024 * 1024)
        logger.info(f"  文件大小：{file_size:.2f} MB")
        logger.info("✓ 模型校验通过")
        return True
    else:
        logger.warning(f"⚠️  模型文件不存在：{model_path}")
        logger.info("  需要进行模型训练")
        return False


# ============================================================
# 4. 模型自动训练
# ============================================================

def auto_train_model() -> bool:
    """
    自动训练模型 - 如果模型不存在则进行训练
    
    Returns:
        True 如果训练成功，False 否则
    """
    logger.info("\n" + "="*80)
    logger.info("[2/4] 模型训练 - 自动训练模型")
    logger.info("="*80)
    
    paths = get_paths()
    data_path = str(paths['data'])
    model_path = str(paths['model'])
    
    # 创建模型目录
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    
    try:
        logger.info("启动模型训练...")
        
        result = BusinessLogic.run_full_pipeline(
            data_path=data_path,
            model_path=model_path,
            epochs=50,
            batch_size=32
        )
        
        if result['status'] == 'success':
            logger.info("✓ 模型训练成功")
            if 'training' in result:
                logger.info(f"  训练结果：{result.get('training')}")
            if 'prediction' in result:
                logger.info(f"  预测统计：{result.get('prediction')}")
            return True
        else:
            logger.error(f"✗ 模型训练失败：{result.get('message')}")
            return False
    
    except Exception as e:
        logger.error(f"✗ 训练异常：{str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================
# 5. 完整检查
# ============================================================

def check_all_layers() -> Tuple[bool, bool, bool]:
    """
    完整的层级检查
    
    Returns:
        (data_ok, model_ok, training_ok)
    """
    logger.info("\n" + "="*80)
    logger.info("风芒可测 - 电力预测与交易优化系统")
    logger.info("本地一键测试流程启动")
    logger.info("="*80)
    
    # 数据层检测
    data_ok = check_data_layer()
    
    if not data_ok:
        logger.error("\n✗ 数据层检测失败，无法继续")
        return False, False, False
    
    # 模型层检测
    model_ok = check_model_layer()
    
    # 如果模型不存在，自动训练
    training_ok = True
    if not model_ok:
        logger.info("\n开始自动训练模型...")
        training_ok = auto_train_model()
        
        if not training_ok:
            logger.error("\n✗ 模型训练失败")
            return data_ok, model_ok, False
        
        model_ok = check_model_layer()  # 重新检查
    
    return data_ok, model_ok, training_ok


# ============================================================
# 6. Streamlit 启动
# ============================================================

def start_streamlit_app(port: int = 8501) -> bool:
    """
    启动 Streamlit 应用
    
    Args:
        port: Streamlit 端口（默认 8501）
    
    Returns:
        True 如果启动成功
    """
    logger.info("\n" + "="*80)
    logger.info("[4/4] 前端启动 - 启动 Streamlit 应用")
    logger.info("="*80)
    
    paths = get_paths()
    app_path = paths['app']
    
    if not app_path.exists():
        logger.error(f"✗ 应用文件不存在：{app_path}")
        return False
    
    try:
        # 启动 Streamlit
        cmd = [
            sys.executable, '-m', 'streamlit', 'run', str(app_path),
            '--server.port', str(port),
            '--server.headless', 'false'
        ]
        
        logger.info(f"执行命令：{' '.join(cmd)}")
        logger.info(f"访问地址：http://localhost:{port}")
        logger.info("按 Ctrl+C 停止服务")
        
        # 使用 subprocess 运行（阻塞式）
        process = subprocess.run(cmd, cwd=str(project_root))
        
        return process.returncode == 0
    
    except KeyboardInterrupt:
        logger.info("应用已停止")
        return True
    except Exception as e:
        logger.error(f"✗ 启动失败：{str(e)}")
        return False


# ============================================================
# 7. 命令行接口
# ============================================================

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='风芒可测 - 电力预测与交易优化系统 本地测试脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例：
    python main.py                  # 启动完整系统
    python main.py --train-only    # 仅训练模型
    python main.py --data-only     # 仅检查数据
    python main.py --port 9000     # 在端口 9000 启动应用
        '''
    )
    
    parser.add_argument(
        '--train-only',
        action='store_true',
        help='仅训练模型，不启动应用'
    )
    
    parser.add_argument(
        '--data-only',
        action='store_true',
        help='仅检查数据，不训练和启动'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=8501,
        help='Streamlit 应用端口（默认：8501）'
    )
    
    parser.add_argument(
        '--skip-train',
        action='store_true',
        help='跳过模型训练，直接启动应用（模型必须已存在）'
    )
    
    args = parser.parse_args()
    
    # ==================== 执行检查 ====================
    
    data_ok, model_ok, training_ok = check_all_layers()
    
    if not data_ok:
        logger.error("\n✗ 数据检查失败，请检查数据文件")
        sys.exit(1)
    
    # ==================== 仅数据检查 ====================
    
    if args.data_only:
        logger.info("\n✓ 数据层检查完成")
        sys.exit(0)
    
    # ==================== 仅训练模型 ====================
    
    if args.train_only:
        if not training_ok:
            logger.error("\n✗ 模型训练失败")
            sys.exit(1)
        
        logger.info("\n✓ 模型训练完成")
        sys.exit(0)
    
    # ==================== 检查模型是否存在（如果跳过训练） ====================
    
    if args.skip_train and not model_ok:
        logger.error("\n✗ 跳过训练但模型不存在")
        sys.exit(1)
    
    # ==================== 启动应用 ====================
    
    logger.info("\n" + "="*80)
    logger.info("✓ 所有检查通过，准备启动应用")
    logger.info("="*80)
    
    time.sleep(1)
    
    # 启动 Streamlit
    success = start_streamlit_app(args.port)
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
