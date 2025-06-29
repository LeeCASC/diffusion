import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from diffusers import DDPMScheduler, DDIMScheduler
from ddim_model import DDIMDiffusionModel


def compare_ddpm_vs_ddim():
    """对比DDPM和DDIM的差异"""
    print("=== DDPM vs DDIM 对比分析 ===\n")
    
    # 创建测试数据
    test_features = torch.randn(1, 256, 16, 48)
    print(f"测试特征形状: {test_features.shape}")
    
    # 初始化调度器
    ddpm_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        prediction_type="epsilon",
    )
    
    ddim_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        prediction_type="epsilon",
    )
    
    print("1. 训练过程对比")
    print("-" * 40)
    
    # 模拟训练过程
    timesteps = torch.randint(0, 1000, (1,))
    noise = torch.randn_like(test_features)
    
    # DDPM添加噪声
    ddpm_noisy = ddpm_scheduler.add_noise(test_features, noise, timesteps)
    
    # DDIM添加噪声（与DDPM相同）
    ddim_noisy = ddim_scheduler.add_noise(test_features, noise, timesteps)
    
    print(f"时间步: {timesteps.item()}")
    print(f"DDPM噪声特征均值: {ddpm_noisy.mean():.6f}")
    print(f"DDIM噪声特征均值: {ddim_noisy.mean():.6f}")
    print(f"差异: {torch.abs(ddpm_noisy - ddim_noisy).max():.8f}")
    print("结论: 训练时DDPM和DDIM添加噪声的方式完全相同\n")
    
    print("2. 采样过程对比")
    print("-" * 40)
    
    # 设置采样步数
    num_inference_steps = 50
    ddpm_scheduler.set_timesteps(num_inference_steps)
    ddim_scheduler.set_timesteps(num_inference_steps)
    
    print(f"采样步数: {num_inference_steps}")
    print(f"DDPM时间步: {ddpm_scheduler.timesteps[:5].tolist()}...")
    print(f"DDIM时间步: {ddim_scheduler.timesteps[:5].tolist()}...")
    print("结论: DDIM可以跳过更多中间步骤\n")
    
    print("3. 采样随机性对比")
    print("-" * 40)
    
    # 模拟噪声预测（简化版本）
    def mock_noise_pred(x, t):
        """模拟噪声预测函数"""
        return torch.randn_like(x) * 0.1
    
    # DDPM采样（随机）
    x_ddpm = torch.randn(1, 256, 16, 48)
    for t in ddpm_scheduler.timesteps:
        noise_pred = mock_noise_pred(x_ddpm, t)
        x_ddpm = ddpm_scheduler.step(noise_pred, t, x_ddpm).prev_sample
    
    # DDIM采样（确定性，eta=0）
    x_ddim_det = torch.randn(1, 256, 16, 48)
    for t in ddim_scheduler.timesteps:
        noise_pred = mock_noise_pred(x_ddim_det, t)
        x_ddim_det = ddim_scheduler.step(noise_pred, t, x_ddim_det, eta=0.0).prev_sample
    
    # DDIM采样（随机，eta=1.0）
    x_ddim_stoch = torch.randn(1, 256, 16, 48)
    for t in ddim_scheduler.timesteps:
        noise_pred = mock_noise_pred(x_ddim_stoch, t)
        x_ddim_stoch = ddim_scheduler.step(noise_pred, t, x_ddim_stoch, eta=1.0).prev_sample
    
    print(f"DDPM采样结果均值: {x_ddpm.mean():.6f}")
    print(f"DDIM确定性采样均值: {x_ddim_det.mean():.6f}")
    print(f"DDIM随机采样均值: {x_ddim_stoch.mean():.6f}")
    print("结论: DDIM可以通过eta参数控制随机性\n")


def demonstrate_training_strategy():
    """演示为什么训练时使用DDPM策略"""
    print("=== 训练策略分析 ===\n")
    
    print("1. 为什么训练时使用DDPM策略？")
    print("-" * 50)
    
    reasons = [
        "稳定性: DDPM在每个时间步添加随机噪声，训练更稳定",
        "完整性: 学习完整的噪声分布，而不是部分分布",
        "泛化性: 训练时的不确定性有助于模型泛化",
        "数学一致性: DDIM论文证明训练过程可以与采样过程解耦",
        "兼容性: 使用DDPM训练的模型可以用DDIM进行推理"
    ]
    
    for i, reason in enumerate(reasons, 1):
        print(f"{i}. {reason}")
    
    print("\n2. 训练vs推理的分离")
    print("-" * 50)
    
    print("训练阶段:")
    print("  - 使用DDPM调度器添加噪声")
    print("  - 学习预测噪声")
    print("  - 目标是最小化噪声预测误差")
    
    print("\n推理阶段:")
    print("  - 使用DDIM调度器进行采样")
    print("  - 可以跳过中间步骤")
    print("  - 通过eta参数控制随机性")
    
    print("\n3. 实际代码示例")
    print("-" * 50)
    
    code_example = '''
# 训练时（ddim_model.py中的train_step方法）
def train_step(self, clean_features, optimizer):
    # 使用DDPM调度器添加噪声
    noise = torch.randn_like(clean_features)
    timesteps = torch.randint(0, self.train_scheduler.num_train_timesteps, (batch_size,), device=self.device)
    noisy_features = self.train_scheduler.add_noise(clean_features, noise, timesteps)  # DDPM策略
    
    # 预测噪声
    noise_pred = self.model(noisy_features, timesteps).sample
    loss = F.mse_loss(noise_pred, noise)
    
# 推理时（ddim_model.py中的sample方法）
def sample(self, batch_size=1, num_inference_steps=50, eta=0.0):
    # 使用DDIM调度器进行采样
    self.noise_scheduler.set_timesteps(num_inference_steps)  # DDIM策略
    
    for t in self.noise_scheduler.timesteps:
        noise_pred = self.model(x, t).sample
        x = self.noise_scheduler.step(noise_pred, t, x, eta=eta).prev_sample  # DDIM步骤
    '''
    
    print(code_example)


def visualize_differences():
    """可视化DDPM和DDIM的差异"""
    print("\n=== 可视化差异 ===\n")
    
    # 创建时间步对比图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # DDPM时间步（完整1000步）
    ddpm_steps = np.arange(1000, 0, -1)
    ax1.plot(ddpm_steps, label='DDPM (1000步)', color='blue')
    ax1.set_title('DDPM采样时间步')
    ax1.set_xlabel('采样步数')
    ax1.set_ylabel('时间步')
    ax1.legend()
    ax1.grid(True)
    
    # DDIM时间步（50步）
    ddim_steps = np.linspace(1000, 1, 50, dtype=int)
    ax2.plot(ddim_steps, label='DDIM (50步)', color='red', marker='o')
    ax2.set_title('DDIM采样时间步')
    ax2.set_xlabel('采样步数')
    ax2.set_ylabel('时间步')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('ddpm_vs_ddim_comparison.png', dpi=300, bbox_inches='tight')
    print("对比图已保存为: ddpm_vs_ddim_comparison.png")
    
    # 创建eta参数影响图
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    eta_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(eta_values)))
    
    for eta, color in zip(eta_values, colors):
        # 模拟不同eta值的影响
        x = np.linspace(0, 1, 100)
        y = x + eta * np.random.normal(0, 0.1, 100)  # 添加随机性
        ax.plot(x, y, color=color, label=f'η={eta}', alpha=0.7)
    
    ax.set_title('DDIM eta参数对随机性的影响')
    ax.set_xlabel('确定性程度')
    ax.set_ylabel('随机性程度')
    ax.legend()
    ax.grid(True)
    
    plt.savefig('ddim_eta_effect.png', dpi=300, bbox_inches='tight')
    print("eta参数影响图已保存为: ddim_eta_effect.png")


def main():
    """主函数"""
    print("DDPM vs DDIM 详细对比分析")
    print("=" * 60)
    
    # 运行对比分析
    compare_ddpm_vs_ddim()
    demonstrate_training_strategy()
    visualize_differences()
    
    print("\n=== 总结 ===")
    print("1. 训练时使用DDPM策略是为了稳定性和完整性")
    print("2. 推理时使用DDIM策略是为了速度和可控性")
    print("3. 这种组合既保证了训练质量，又提高了推理效率")
    print("4. DDIM的eta参数允许在确定性和随机性之间平衡")


if __name__ == "__main__":
    main() 