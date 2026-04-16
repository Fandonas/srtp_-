#!/usr/bin/env python3
"""
测试TEXT_COFF参数日志输出的简单脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from yacs.config import CfgNode as CN
import utils.logging as logging
from models.base.few_shot import CNN_OTAM_CLIPFSAR

# 设置日志
logging.setup_logging()
logger = logging.get_logger(__name__)

def test_text_coff_logging():
    """测试TEXT_COFF参数的日志输出"""
    
    # 创建一个完整的配置
    cfg = CN()
    
    # VIDEO配置
    cfg.VIDEO = CN()
    cfg.VIDEO.HEAD = CN()
    cfg.VIDEO.HEAD.BACKBONE_NAME = "ViT-B/16"
    
    # DATA配置 - 这是关键的缺失部分
    cfg.DATA = CN()
    cfg.DATA.NUM_INPUT_FRAMES = 8  # 必需参数
    cfg.DATA.TRAIN_CROP_SIZE = 224
    cfg.DATA.NUM_INPUT_CHANNELS = 3
    cfg.DATA.SAMPLING_RATE = 1
    
    # TRAIN配置
    cfg.TRAIN = CN()
    cfg.TRAIN.CLASS_NAME = ["test_class_1", "test_class_2", "test_class_3"]
    cfg.TRAIN.TEXT_COFF = 0.7  # 设置TEXT_COFF参数
    
    # TEST配置
    cfg.TEST = CN()
    cfg.TEST.CLASS_NAME = ["test_class_1", "test_class_2", "test_class_3"]
    
    logger.info("开始测试TEXT_COFF参数日志输出...")
    logger.info(f"配置中的TEXT_COFF值: {cfg.TRAIN.TEXT_COFF}")
    
    try:
        # 创建模型实例，这应该会触发__init__方法中的日志输出
        logger.info("正在初始化CNN_OTAM_CLIPFSAR模型...")
        model = CNN_OTAM_CLIPFSAR(cfg)
        logger.info("模型初始化完成")
        
    except Exception as e:
        logger.error(f"模型初始化失败: {e}")
        return False
    
    return True

if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("TEXT_COFF参数日志测试开始")
    logger.info("=" * 50)
    
    success = test_text_coff_logging()
    
    logger.info("=" * 50)
    if success:
        logger.info("测试完成！请检查上面的日志输出中是否包含TEXT_COFF参数信息")
    else:
        logger.info("测试失败！")
    logger.info("=" * 50)