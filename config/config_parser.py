#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShapeNet配置文件解析器
用于解析config/exp-shapenet-car/specs_shapenet_all.json配置文件
"""

import json
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path


@dataclass
class InputConfig:
    """输入配置类"""
    task_mode: str
    dataset_name: str
    pcd_dir: str
    mesh_dir: str
    render_img_dir: str
    blender_transform: str
    prompt_path: str
    img_num: int
    split_dir: str
    class_name: str  # 使用class_name避免与Python关键字冲突
    tet_grid_size: int
    dmtet_input_file: str
    mem_val_num: int
    dataset_type: str
    chunk_path: str
    chunk_size: int
    scale: float
    radius: float
    reso_dpt: List[int]
    reso_rgb: List[int]


@dataclass
class PretrainConfig:
    """预训练配置类"""
    mode: Optional[str]
    warm_up: int
    batch_size: int
    warmup_iter: int
    decay: float
    num_iter: int
    sdf_threshold: float
    sdf_scale: int
    batch_infer: bool
    lr: float
    radius: float


@dataclass
class LearningParams:
    """学习参数配置类"""
    init: float
    sdf_decay: float
    rgb_decay: float


@dataclass
class SupervisionConfig:
    """监督配置类"""
    geo_sup_mode: str
    geo_sig_src: str
    tex_sup_mode: str
    smpl_cam_num: int
    view_num: int


@dataclass
class LossWeights:
    """损失权重配置类"""
    lmbd_rgb: float
    lmbd_reg_lap: float
    kl: float
    commit: float
    lmbd_kl: float
    lmbd_lpips: float
    lmbd_ulip_loss: bool
    lmbd_dssim: float


@dataclass
class RandomnessConfig:
    """随机性配置类"""
    lrn_ae_pts_num: int
    qry_tf_pts_num: int
    xyz_noise_scale: float


@dataclass
class TrainConfig:
    """训练配置类"""
    num_epochs: int
    warm_up: int
    decay: float
    learning_params: LearningParams
    sup_cfg: SupervisionConfig
    loss_wgt: LossWeights
    randomness: RandomnessConfig
    b_size: int
    gradient_acc: int
    eva_sub_iter: int
    eva_all_iter: int
    save_checkpoint_iter: int
    exp_uv_mesh: bool
    random_bg: bool
    shift: int
    init_mdl_path: Optional[str]
    finetune_mdl_path: str
    ulip_flag: bool
    ulip_model_path: Optional[str]


@dataclass
class ArchSpecs:
    """架构规格配置类"""
    unet_type: str
    autoencoder_type: str
    use_3D_aware: bool
    fea_concat: bool
    tri_enc: str
    fused_type: str
    mlp_bias: bool
    geo_type: str
    q_plane: bool
    random_downsample: bool
    latent_dim: int


@dataclass
class DiffSpecs:
    """扩散规格配置类"""
    sd_mode: Optional[str]
    sd_cfg: int
    sd_lambda: int
    num_epochs: int
    warm_up: int
    batch_size: int
    eva_iter: int
    prediction: str
    gradient_acc: int
    lr: float
    decay: float
    lr_scheduler_type: str
    betas_scale: Optional[float]
    beta_schedule: str
    diffusion_zero_snr: bool
    diffusion_clip_flag: bool
    vae_scaling_factor: float
    noise_offset: float
    input_perturbation: float
    snr_gamma: Optional[float]
    clip_grad_norm: Optional[float]
    model_name: str
    image_dir: str
    txt_dir: str
    random_view: bool
    image_resolution: int
    guidance_scale: float


@dataclass
class UnetKwargs:
    """UNet参数配置类"""
    depth: int
    merge_mode: str
    start_filts: int


@dataclass
class EncoderSpecs:
    """编码器规格配置类"""
    latent_size: int
    hidden_dim: int
    color_output: int
    unet_kwargs: UnetKwargs
    plane_resolution: int


@dataclass
class DecoderSpecs:
    """解码器规格配置类"""
    c_dim: int
    unet_kwargs: UnetKwargs
    plane_resolution: int


@dataclass
class OutputConfig:
    """输出配置类"""
    exp_name: str


@dataclass
class ShapeNetConfig:
    """ShapeNet完整配置类"""
    model: str
    input: InputConfig
    pretrain: PretrainConfig
    train: TrainConfig
    arch_specs: ArchSpecs
    diff_specs: DiffSpecs
    encoder_specs: EncoderSpecs
    decoder_specs: DecoderSpecs
    output: OutputConfig


class ConfigParser:
    """配置文件解析器"""
    
    def __init__(self, config_path: str):
        """
        初始化解析器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = Path(config_path)
        self.config_data = None
        self.parsed_config = None
    
    def load_config(self) -> Dict[str, Any]:
        """
        加载JSON配置文件
        
        Returns:
            配置字典
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config_data = json.load(f)
        
        return self.config_data
    
    def parse_config(self) -> ShapeNetConfig:
        """
        解析配置文件为结构化对象
        
        Returns:
            解析后的配置对象
        """
        if self.config_data is None:
            self.load_config()
        
        # 解析各个配置部分
        input_config = InputConfig(
            task_mode=self.config_data["Input"]["task_mode"],
            dataset_name=self.config_data["Input"]["dataset_name"],
            pcd_dir=self.config_data["Input"]["pcd_dir"],
            mesh_dir=self.config_data["Input"]["mesh_dir"],
            render_img_dir=self.config_data["Input"]["render_img_dir"],
            blender_transform=self.config_data["Input"]["blender_transform"],
            prompt_path=self.config_data["Input"]["prompt_path"],
            img_num=self.config_data["Input"]["img_num"],
            split_dir=self.config_data["Input"]["split_dir"],
            class_name=self.config_data["Input"]["class"],
            tet_grid_size=self.config_data["Input"]["tet_grid_size"],
            dmtet_input_file=self.config_data["Input"]["dmtet_input_file"],
            mem_val_num=self.config_data["Input"]["mem_val_num"],
            dataset_type=self.config_data["Input"]["dataset_type"],
            chunk_path=self.config_data["Input"]["chunk_path"],
            chunk_size=self.config_data["Input"]["chunk_size"],
            scale=self.config_data["Input"]["scale"],
            radius=self.config_data["Input"]["radius"],
            reso_dpt=self.config_data["Input"]["reso_dpt"],
            reso_rgb=self.config_data["Input"]["reso_rgb"]
        )
        
        pretrain_config = PretrainConfig(
            mode=self.config_data["Pretrain"]["mode"],
            warm_up=self.config_data["Pretrain"]["warm_up"],
            batch_size=self.config_data["Pretrain"]["batch_size"],
            warmup_iter=self.config_data["Pretrain"]["warmup_iter"],
            decay=self.config_data["Pretrain"]["decay"],
            num_iter=self.config_data["Pretrain"]["num_iter"],
            sdf_threshold=self.config_data["Pretrain"]["sdf_threshold"],
            sdf_scale=self.config_data["Pretrain"]["sdf_scale"],
            batch_infer=self.config_data["Pretrain"]["batch_infer"],
            lr=self.config_data["Pretrain"]["lr"],
            radius=self.config_data["Pretrain"]["radius"]
        )
        
        # 解析学习参数
        learning_params = LearningParams(
            init=self.config_data["Train"]["learning_params"]["init"],
            sdf_decay=self.config_data["Train"]["learning_params"]["sdf_decay"],
            rgb_decay=self.config_data["Train"]["learning_params"]["rgb_decay"]
        )
        
        # 解析监督配置
        sup_cfg = SupervisionConfig(
            geo_sup_mode=self.config_data["Train"]["sup_cfg"]["geo_sup_mode"],
            geo_sig_src=self.config_data["Train"]["sup_cfg"]["geo_sig_src"],
            tex_sup_mode=self.config_data["Train"]["sup_cfg"]["tex_sup_mode"],
            smpl_cam_num=self.config_data["Train"]["sup_cfg"]["smpl_cam_num"],
            view_num=self.config_data["Train"]["sup_cfg"]["view_num"]
        )
        
        # 解析损失权重
        loss_wgt = LossWeights(
            lmbd_rgb=self.config_data["Train"]["loss_wgt"]["lmbd_rgb"],
            lmbd_reg_lap=self.config_data["Train"]["loss_wgt"]["lmbd_reg_lap"],
            kl=self.config_data["Train"]["loss_wgt"]["kl"],
            commit=self.config_data["Train"]["loss_wgt"]["commit"],
            lmbd_kl=self.config_data["Train"]["loss_wgt"]["lmbd_kl"],
            lmbd_lpips=self.config_data["Train"]["loss_wgt"]["lmbd_lpips"],
            lmbd_ulip_loss=self.config_data["Train"]["loss_wgt"]["lmbd_ulip_loss"],
            lmbd_dssim=self.config_data["Train"]["loss_wgt"]["lmbd_dssim"]
        )
        
        # 解析随机性配置
        randomness = RandomnessConfig(
            lrn_ae_pts_num=self.config_data["Train"]["randomness"]["lrn_ae_pts_num"],
            qry_tf_pts_num=self.config_data["Train"]["randomness"]["qry_tf_pts_num"],
            xyz_noise_scale=self.config_data["Train"]["randomness"]["xyz_noise_scale"]
        )
        
        train_config = TrainConfig(
            num_epochs=self.config_data["Train"]["num_epochs"],
            warm_up=self.config_data["Train"]["warm_up"],
            decay=self.config_data["Train"]["decay"],
            learning_params=learning_params,
            sup_cfg=sup_cfg,
            loss_wgt=loss_wgt,
            randomness=randomness,
            b_size=self.config_data["Train"]["b_size"],
            gradient_acc=self.config_data["Train"]["gradient_acc"],
            eva_sub_iter=self.config_data["Train"]["eva_sub_iter"],
            eva_all_iter=self.config_data["Train"]["eva_all_iter"],
            save_checkpoint_iter=self.config_data["Train"]["save_checkpoint_iter"],
            exp_uv_mesh=self.config_data["Train"]["exp_uv_mesh"],
            random_bg=self.config_data["Train"]["random_bg"],
            shift=self.config_data["Train"]["shift"],
            init_mdl_path=self.config_data["Train"]["init_mdl_path"],
            finetune_mdl_path=self.config_data["Train"]["finetune_mdl_path"],
            ulip_flag=self.config_data["Train"]["ulip_flag"],
            ulip_model_path=self.config_data["Train"]["ulip_model_path"]
        )
        
        arch_specs = ArchSpecs(
            unet_type=self.config_data["ArchSpecs"]["unet_type"],
            autoencoder_type=self.config_data["ArchSpecs"]["autoencoder_type"],
            use_3D_aware=self.config_data["ArchSpecs"]["use_3D_aware"],
            fea_concat=self.config_data["ArchSpecs"]["fea_concat"],
            tri_enc=self.config_data["ArchSpecs"]["tri_enc"],
            fused_type=self.config_data["ArchSpecs"]["fused_type"],
            mlp_bias=self.config_data["ArchSpecs"]["mlp_bias"],
            geo_type=self.config_data["ArchSpecs"]["geo_type"],
            q_plane=self.config_data["ArchSpecs"]["q_plane"],
            random_downsample=self.config_data["ArchSpecs"]["random_downsample"],
            latent_dim=self.config_data["ArchSpecs"]["latent_dim"]
        )
        
        diff_specs = DiffSpecs(
            sd_mode=self.config_data["DiffSpecs"]["sd_mode"],
            sd_cfg=self.config_data["DiffSpecs"]["sd_cfg"],
            sd_lambda=self.config_data["DiffSpecs"]["sd_lambda"],
            num_epochs=self.config_data["DiffSpecs"]["num_epochs"],
            warm_up=self.config_data["DiffSpecs"]["warm_up"],
            batch_size=self.config_data["DiffSpecs"]["batch_size"],
            eva_iter=self.config_data["DiffSpecs"]["eva_iter"],
            prediction=self.config_data["DiffSpecs"]["prediction"],
            gradient_acc=self.config_data["DiffSpecs"]["gradient_acc"],
            lr=self.config_data["DiffSpecs"]["lr"],
            decay=self.config_data["DiffSpecs"]["decay"],
            lr_scheduler_type=self.config_data["DiffSpecs"]["lr_scheduler_type"],
            betas_scale=self.config_data["DiffSpecs"]["betas_scale"],
            beta_schedule=self.config_data["DiffSpecs"]["beta_schedule"],
            diffusion_zero_snr=self.config_data["DiffSpecs"]["diffusion_zero_snr"],
            diffusion_clip_flag=self.config_data["DiffSpecs"]["diffusion_clip_flag"],
            vae_scaling_factor=self.config_data["DiffSpecs"]["vae_scaling_factor"],
            noise_offset=self.config_data["DiffSpecs"]["noise_offset"],
            input_perturbation=self.config_data["DiffSpecs"]["input_perturbation"],
            snr_gamma=self.config_data["DiffSpecs"]["snr_gamma"],
            clip_grad_norm=self.config_data["DiffSpecs"]["clip_grad_norm"],
            model_name=self.config_data["DiffSpecs"]["model_name"],
            image_dir=self.config_data["DiffSpecs"]["image_dir"],
            txt_dir=self.config_data["DiffSpecs"]["txt_dir"],
            random_view=self.config_data["DiffSpecs"]["random_view"],
            image_resolution=self.config_data["DiffSpecs"]["image_resolution"],
            guidance_scale=self.config_data["DiffSpecs"]["guidance_scale"]
        )
        
        # 解析UNet参数
        encoder_unet_kwargs = UnetKwargs(
            depth=self.config_data["EncoderSpecs"]["unet_kwargs"]["depth"],
            merge_mode=self.config_data["EncoderSpecs"]["unet_kwargs"]["merge_mode"],
            start_filts=self.config_data["EncoderSpecs"]["unet_kwargs"]["start_filts"]
        )
        
        decoder_unet_kwargs = UnetKwargs(
            depth=self.config_data["DecoderSpecs"]["unet_kwargs"]["depth"],
            merge_mode=self.config_data["DecoderSpecs"]["unet_kwargs"]["merge_mode"],
            start_filts=self.config_data["DecoderSpecs"]["unet_kwargs"]["start_filts"]
        )
        
        encoder_specs = EncoderSpecs(
            latent_size=self.config_data["EncoderSpecs"]["latent_size"],
            hidden_dim=self.config_data["EncoderSpecs"]["hidden_dim"],
            color_output=self.config_data["EncoderSpecs"]["color_output"],
            unet_kwargs=encoder_unet_kwargs,
            plane_resolution=self.config_data["EncoderSpecs"]["plane_resolution"]
        )
        
        decoder_specs = DecoderSpecs(
            c_dim=self.config_data["DecoderSpecs"]["c_dim"],
            unet_kwargs=decoder_unet_kwargs,
            plane_resolution=self.config_data["DecoderSpecs"]["plane_resolution"]
        )
        
        output_config = OutputConfig(
            exp_name=self.config_data["Output"]["exp_name"]
        )
        
        self.parsed_config = ShapeNetConfig(
            model=self.config_data["Model"],
            input=input_config,
            pretrain=pretrain_config,
            train=train_config,
            arch_specs=arch_specs,
            diff_specs=diff_specs,
            encoder_specs=encoder_specs,
            decoder_specs=decoder_specs,
            output=output_config
        )
        
        return self.parsed_config
    
    def get_config_summary(self) -> Dict[str, Any]:
        """
        获取配置摘要
        
        Returns:
            配置摘要字典
        """
        if self.parsed_config is None:
            self.parse_config()
        
        return {
            "模型": self.parsed_config.model,
            "数据集": self.parsed_config.input.dataset_name,
            "类别": self.parsed_config.input.class_name,
            "训练轮数": self.parsed_config.train.num_epochs,
            "批次大小": self.parsed_config.train.b_size,
            "学习率": self.parsed_config.train.learning_params.init,
            "实验名称": self.parsed_config.output.exp_name,
            "潜在维度": self.parsed_config.arch_specs.latent_dim,
            "图像数量": self.parsed_config.input.img_num
        }
    
    def validate_paths(self) -> List[str]:
        """
        验证配置文件中的路径是否存在
        
        Returns:
            不存在的路径列表
        """
        if self.parsed_config is None:
            self.parse_config()
        
        invalid_paths = []
        
        # 检查输入路径
        paths_to_check = [
            ("PCD目录", self.parsed_config.input.pcd_dir),
            ("网格目录", self.parsed_config.input.mesh_dir),
            ("渲染图像目录", self.parsed_config.input.render_img_dir),
            ("分割目录", self.parsed_config.input.split_dir),
            ("块路径", self.parsed_config.input.chunk_path)
        ]
        
        for name, path in paths_to_check:
            if path and not os.path.exists(path):
                invalid_paths.append(f"{name}: {path}")
        
        return invalid_paths


def main():
    """主函数 - 演示如何使用配置解析器"""
    config_path = "config/exp-shapenet-car/specs_shapenet_all.json"
    
    try:
        # 创建解析器实例
        parser = ConfigParser(config_path)
        
        # 解析配置
        config = parser.parse_config()
        
        # 打印配置摘要
        print("=== ShapeNet配置摘要 ===")
        summary = parser.get_config_summary()
        for key, value in summary.items():
            print(f"{key}: {value}")
        
        print("\n=== 详细配置信息 ===")
        print(f"模型类型: {config.model}")
        print(f"数据集名称: {config.input.dataset_name}")
        print(f"数据集类别: {config.input.class_name}")
        print(f"训练轮数: {config.train.num_epochs}")
        print(f"批次大小: {config.train.b_size}")
        print(f"学习率: {config.train.learning_params.init}")
        print(f"潜在维度: {config.arch_specs.latent_dim}")
        print(f"图像分辨率: {config.input.reso_rgb}")
        print(f"深度分辨率: {config.input.reso_dpt}")
        
        # 验证路径
        print("\n=== 路径验证 ===")
        invalid_paths = parser.validate_paths()
        if invalid_paths:
            print("以下路径不存在:")
            for path in invalid_paths:
                print(f"  - {path}")
        else:
            print("所有路径都有效")
        
    except FileNotFoundError as e:
        print(f"错误: {e}")
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
    except Exception as e:
        print(f"解析错误: {e}")


if __name__ == "__main__":
    main() 