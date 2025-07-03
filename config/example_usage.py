#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置解析器使用示例
演示如何使用ConfigParser来解析和使用ShapeNet配置文件
"""

from config_parser import ConfigParser


def example_basic_usage():
    """基本使用示例"""
    print("=== 基本使用示例 ===")
    
    # 创建解析器实例
    parser = ConfigParser("exp-shapenet-car/specs_shapenet_all.json")
    
    # 解析配置
    config = parser.parse_config()
    
    # 访问配置信息
    print(f"模型类型: {config.model}")
    print(f"数据集: {config.input.dataset_name}")
    print(f"训练轮数: {config.train.num_epochs}")
    print(f"学习率: {config.train.learning_params.init}")


def example_config_summary():
    """配置摘要示例"""
    print("\n=== 配置摘要示例 ===")
    
    parser = ConfigParser("exp-shapenet-car/specs_shapenet_all.json")
    summary = parser.get_config_summary()
    
    for key, value in summary.items():
        print(f"{key}: {value}")


def example_path_validation():
    """路径验证示例"""
    print("\n=== 路径验证示例 ===")
    
    parser = ConfigParser("exp-shapenet-car/specs_shapenet_all.json")
    invalid_paths = parser.validate_paths()
    
    if invalid_paths:
        print("发现无效路径:")
        for path in invalid_paths:
            print(f"  - {path}")
    else:
        print("所有路径都有效")


def example_training_config():
    """训练配置示例"""
    print("\n=== 训练配置示例 ===")
    
    parser = ConfigParser("exp-shapenet-car/specs_shapenet_all.json")
    config = parser.parse_config()
    
    # 训练相关配置
    train_config = config.train
    print(f"训练轮数: {train_config.num_epochs}")
    print(f"批次大小: {train_config.b_size}")
    print(f"梯度累积: {train_config.gradient_acc}")
    print(f"保存检查点间隔: {train_config.save_checkpoint_iter}")
    
    # 学习参数
    learning_params = train_config.learning_params
    print(f"初始学习率: {learning_params.init}")
    print(f"SDF衰减: {learning_params.sdf_decay}")
    print(f"RGB衰减: {learning_params.rgb_decay}")
    
    # 损失权重
    loss_weights = train_config.loss_wgt
    print(f"RGB损失权重: {loss_weights.lmbd_rgb}")
    print(f"拉普拉斯正则化权重: {loss_weights.lmbd_reg_lap}")
    print(f"KL散度权重: {loss_weights.kl}")


def example_architecture_config():
    """架构配置示例"""
    print("\n=== 架构配置示例 ===")
    
    parser = ConfigParser("exp-shapenet-car/specs_shapenet_all.json")
    config = parser.parse_config()
    
    # 架构规格
    arch_specs = config.arch_specs
    print(f"UNet类型: {arch_specs.unet_type}")
    print(f"自编码器类型: {arch_specs.autoencoder_type}")
    print(f"潜在维度: {arch_specs.latent_dim}")
    print(f"使用3D感知: {arch_specs.use_3D_aware}")
    print(f"特征连接: {arch_specs.fea_concat}")
    
    # 编码器规格
    encoder_specs = config.encoder_specs
    print(f"编码器潜在大小: {encoder_specs.latent_size}")
    print(f"编码器隐藏维度: {encoder_specs.hidden_dim}")
    print(f"颜色输出维度: {encoder_specs.color_output}")
    print(f"平面分辨率: {encoder_specs.plane_resolution}")
    
    # 解码器规格
    decoder_specs = config.decoder_specs
    print(f"解码器条件维度: {decoder_specs.c_dim}")
    print(f"解码器平面分辨率: {decoder_specs.plane_resolution}")


def example_diffusion_config():
    """扩散配置示例"""
    print("\n=== 扩散配置示例 ===")
    
    parser = ConfigParser("exp-shapenet-car/specs_shapenet_all.json")
    config = parser.parse_config()
    
    # 扩散规格
    diff_specs = config.diff_specs
    print(f"扩散模式: {diff_specs.sd_mode}")
    print(f"扩散配置: {diff_specs.sd_cfg}")
    print(f"扩散lambda: {diff_specs.sd_lambda}")
    print(f"批次大小: {diff_specs.batch_size}")
    print(f"学习率: {diff_specs.lr}")
    print(f"学习率调度器: {diff_specs.lr_scheduler_type}")
    print(f"beta调度: {diff_specs.beta_schedule}")
    print(f"VAE缩放因子: {diff_specs.vae_scaling_factor}")
    print(f"噪声偏移: {diff_specs.noise_offset}")
    print(f"指导尺度: {diff_specs.guidance_scale}")


def example_input_config():
    """输入配置示例"""
    print("\n=== 输入配置示例 ===")
    
    parser = ConfigParser("exp-shapenet-car/specs_shapenet_all.json")
    config = parser.parse_config()
    
    # 输入配置
    input_config = config.input
    print(f"任务模式: {input_config.task_mode}")
    print(f"数据集名称: {input_config.dataset_name}")
    print(f"数据集类别: {input_config.class_name}")
    print(f"图像数量: {input_config.img_num}")
    print(f"数据集类型: {input_config.dataset_type}")
    print(f"块大小: {input_config.chunk_size}")
    print(f"缩放因子: {input_config.scale}")
    print(f"半径: {input_config.radius}")
    print(f"RGB分辨率: {input_config.reso_rgb}")
    print(f"深度分辨率: {input_config.reso_dpt}")
    
    # 路径信息
    print(f"PCD目录: {input_config.pcd_dir}")
    print(f"网格目录: {input_config.mesh_dir}")
    print(f"渲染图像目录: {input_config.render_img_dir}")


def main():
    """主函数 - 运行所有示例"""
    try:
        example_basic_usage()
        example_config_summary()
        example_path_validation()
        example_training_config()
        example_architecture_config()
        example_diffusion_config()
        example_input_config()
        
    except FileNotFoundError as e:
        print(f"错误: 配置文件未找到 - {e}")
    except Exception as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    main() 