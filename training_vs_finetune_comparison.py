import torch
import matplotlib.pyplot as plt
import numpy as np
from ddim_model import DDIMDiffusionModel
from data_generator import create_dataloader


def compare_training_strategies():
    """对比训练和微调策略"""
    print("=== 训练 vs 微调策略对比 ===\n")
    
    # 1. 学习率对比
    print("1. 学习率策略对比")
    print("-" * 40)
    
    training_lr = 1e-4
    finetune_lr = 1e-5
    
    print(f"普通训练学习率: {training_lr}")
    print(f"微调学习率: {finetune_lr}")
    print(f"学习率比例: {finetune_lr/training_lr:.1%}")
    print("微调使用更小的学习率，避免破坏预训练知识\n")
    
    # 2. 训练轮数对比
    print("2. 训练轮数对比")
    print("-" * 40)
    
    training_epochs = 100
    finetune_epochs = 10
    
    print(f"普通训练轮数: {training_epochs}")
    print(f"微调轮数: {finetune_epochs}")
    print(f"轮数比例: {finetune_epochs/training_epochs:.1%}")
    print("微调使用更少的轮数，避免过拟合\n")
    
    # 3. 数据量对比
    print("3. 数据量对比")
    print("-" * 40)
    
    training_samples = 50000
    finetune_samples = 1000
    
    print(f"普通训练样本数: {training_samples}")
    print(f"微调样本数: {finetune_samples}")
    print(f"样本比例: {finetune_samples/training_samples:.1%}")
    print("微调使用更少的数据，因为模型已经有基础能力\n")
    
    # 4. 优化器对比
    print("4. 优化器策略对比")
    print("-" * 40)
    
    print("普通训练:")
    print("  - 优化器: Adam")
    print("  - 权重衰减: 1e-6")
    print("  - 梯度裁剪: 无")
    
    print("\n微调:")
    print("  - 优化器: AdamW")
    print("  - 权重衰减: 1e-6")
    print("  - 梯度裁剪: max_norm=1.0")
    print("  - 目的: 防止参数变化过大\n")


def visualize_learning_curves():
    """可视化学习曲线对比"""
    print("=== 学习曲线对比 ===\n")
    
    # 模拟训练和微调的学习曲线
    training_epochs = 100
    finetune_epochs = 10
    
    # 普通训练：从高损失开始，逐渐下降
    training_losses = []
    for epoch in range(training_epochs):
        # 模拟训练损失下降
        base_loss = 2.0 * np.exp(-epoch / 30) + 0.1 + np.random.normal(0, 0.05)
        training_losses.append(base_loss)
    
    # 微调：从较低损失开始，小幅下降
    finetune_losses = []
    for epoch in range(finetune_epochs):
        # 模拟微调损失小幅下降
        base_loss = 0.15 * np.exp(-epoch / 5) + 0.08 + np.random.normal(0, 0.02)
        finetune_losses.append(base_loss)
    
    # 创建对比图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 损失曲线对比
    ax1.plot(range(training_epochs), training_losses, 'b-', label='普通训练', linewidth=2)
    ax1.plot(range(finetune_epochs), finetune_losses, 'r-', label='微调', linewidth=2)
    ax1.set_title('损失曲线对比')
    ax1.set_xlabel('训练轮数')
    ax1.set_ylabel('损失')
    ax1.legend()
    ax1.grid(True)
    
    # 2. 学习率对比
    training_lr = [1e-4] * training_epochs
    finetune_lr = [1e-5] * finetune_epochs
    
    ax2.plot(range(training_epochs), training_lr, 'b-', label='普通训练', linewidth=2)
    ax2.plot(range(finetune_epochs), finetune_lr, 'r-', label='微调', linewidth=2)
    ax2.set_title('学习率对比')
    ax2.set_xlabel('训练轮数')
    ax2.set_ylabel('学习率')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True)
    
    # 3. 参数变化对比
    # 模拟参数变化
    training_param_changes = [0.1 * np.exp(-epoch / 20) for epoch in range(training_epochs)]
    finetune_param_changes = [0.01 * np.exp(-epoch / 3) for epoch in range(finetune_epochs)]
    
    ax3.plot(range(training_epochs), training_param_changes, 'b-', label='普通训练', linewidth=2)
    ax3.plot(range(finetune_epochs), finetune_param_changes, 'r-', label='微调', linewidth=2)
    ax3.set_title('参数变化幅度对比')
    ax3.set_xlabel('训练轮数')
    ax3.set_ylabel('参数变化幅度')
    ax3.legend()
    ax3.grid(True)
    
    # 4. 收敛速度对比
    # 模拟收敛指标
    training_convergence = [1 - 0.9 * np.exp(-epoch / 15) for epoch in range(training_epochs)]
    finetune_convergence = [0.8 + 0.2 * (1 - np.exp(-epoch / 3)) for epoch in range(finetune_epochs)]
    
    ax4.plot(range(training_epochs), training_convergence, 'b-', label='普通训练', linewidth=2)
    ax4.plot(range(finetune_epochs), finetune_convergence, 'r-', label='微调', linewidth=2)
    ax4.set_title('收敛速度对比')
    ax4.set_xlabel('训练轮数')
    ax4.set_ylabel('收敛指标')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_vs_finetune_comparison.png', dpi=300, bbox_inches='tight')
    print("学习曲线对比图已保存为: training_vs_finetune_comparison.png")


def demonstrate_parameter_behavior():
    """演示参数行为差异"""
    print("=== 参数行为差异演示 ===\n")
    
    # 模拟参数变化
    print("1. 普通训练参数变化:")
    print("   - 从随机初始化开始")
    print("   - 参数变化幅度大")
    print("   - 需要学习所有特征")
    print("   - 收敛时间长")
    
    print("\n2. 微调参数变化:")
    print("   - 从预训练参数开始")
    print("   - 参数变化幅度小")
    print("   - 只需要适应新数据")
    print("   - 收敛时间短")
    
    # 创建参数变化可视化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 普通训练参数变化
    epochs = 100
    training_params = np.random.randn(epochs) * 0.1 + np.exp(-np.arange(epochs) / 20)
    ax1.plot(training_params, 'b-', linewidth=2)
    ax1.set_title('普通训练参数变化')
    ax1.set_xlabel('训练轮数')
    ax1.set_ylabel('参数值')
    ax1.grid(True)
    
    # 微调参数变化
    finetune_epochs = 10
    finetune_params = np.random.randn(finetune_epochs) * 0.01 + 0.5 + np.exp(-np.arange(finetune_epochs) / 3)
    ax2.plot(finetune_params, 'r-', linewidth=2)
    ax2.set_title('微调参数变化')
    ax2.set_xlabel('训练轮数')
    ax2.set_ylabel('参数值')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('parameter_behavior_comparison.png', dpi=300, bbox_inches='tight')
    print("参数行为对比图已保存为: parameter_behavior_comparison.png")


def compare_training_objectives():
    """对比训练目标"""
    print("=== 训练目标对比 ===\n")
    
    print("1. 普通训练目标:")
    print("   - 学习数据的基本分布")
    print("   - 建立特征表示能力")
    print("   - 从头学习所有模式")
    print("   - 需要大量数据和时间")
    
    print("\n2. 微调目标:")
    print("   - 适应特定数据分布")
    print("   - 保持原有能力")
    print("   - 增量学习新模式")
    print("   - 需要少量数据和时间")
    
    print("\n3. 损失函数差异:")
    print("   普通训练: L = E[||ε - ε_θ(x_t, t)||²]")
    print("   微调: L = E[||ε - ε_θ(x_t, t)||²] + λ||θ - θ₀||²")
    print("   (其中θ₀是预训练参数，λ是正则化系数)")


def practical_differences():
    """实际差异总结"""
    print("=== 实际差异总结 ===\n")
    
    differences = [
        {
            "方面": "初始化",
            "普通训练": "随机初始化",
            "微调": "预训练参数",
            "影响": "微调起点更高"
        },
        {
            "方面": "学习率",
            "普通训练": "1e-4",
            "微调": "1e-5",
            "影响": "微调更保守"
        },
        {
            "方面": "训练轮数",
            "普通训练": "100轮",
            "微调": "10轮",
            "影响": "微调更快"
        },
        {
            "方面": "数据需求",
            "普通训练": "5万个样本",
            "微调": "1千个样本",
            "影响": "微调更高效"
        },
        {
            "方面": "收敛时间",
            "普通训练": "数小时",
            "微调": "数分钟",
            "影响": "微调更快速"
        },
        {
            "方面": "风险",
            "普通训练": "可能不收敛",
            "微调": "可能过拟合",
            "影响": "需要不同策略"
        }
    ]
    
    print(f"{'方面':<12} {'普通训练':<15} {'微调':<15} {'影响':<15}")
    print("-" * 60)
    for diff in differences:
        print(f"{diff['方面']:<12} {diff['普通训练']:<15} {diff['微调']:<15} {diff['影响']:<15}")


def code_comparison():
    """代码层面的对比"""
    print("\n=== 代码层面对比 ===\n")
    
    print("1. 普通训练代码:")
    print("```python")
    print("# 创建新模型")
    print("model = DDIMDiffusionModel()")
    print("optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)")
    print("")
    print("# 训练循环")
    print("for epoch in range(100):")
    print("    for batch in dataloader:")
    print("        loss = model.train_step(batch, optimizer)")
    print("```")
    
    print("\n2. 微调代码:")
    print("```python")
    print("# 加载预训练模型")
    print("finetune_model = FineTuneDiffusionModel(pretrained_path)")
    print("optimizer = torch.optim.AdamW(finetune_model.parameters(), lr=1e-5)")
    print("")
    print("# 微调循环")
    print("for epoch in range(10):")
    print("    for batch in finetune_dataloader:")
    print("        loss = finetune_model.finetune_step(batch, optimizer)")
    print("        # 包含梯度裁剪")
    print("        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)")
    print("```")


def main():
    """主函数"""
    print("训练 vs 微调详细对比分析")
    print("=" * 60)
    
    # 运行对比分析
    compare_training_strategies()
    visualize_learning_curves()
    demonstrate_parameter_behavior()
    compare_training_objectives()
    practical_differences()
    code_comparison()
    
    print("\n=== 总结 ===")
    print("1. 微调是建立在预训练基础上的增量学习")
    print("2. 微调使用更保守的学习策略")
    print("3. 微调需要更少的数据和时间")
    print("4. 微调的目标是适应而不是重建")
    print("5. 微调需要特殊的正则化策略")


if __name__ == "__main__":
    main() 