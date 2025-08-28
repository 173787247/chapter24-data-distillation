# 第二十四章数据蒸馏实验指南

##  实验概述
本实验使用Unsloth框架对DeepSeek-R1-Distill-Qwen-1.5B模型进行QLoRA微调，在医疗推理数据集上验证数据蒸馏效果。

##  实验目标
1. 掌握Unsloth框架的使用方法
2. 理解QLoRA微调原理
3. 在医疗数据集上验证模型性能
4. 对比微调前后的生成质量

##  环境要求
- Python 3.8+
- CUDA支持的GPU
- 至少16GB显存
- 稳定的网络连接

##  参考资源
- 原始代码: https://github.com/DjangoPeng/deepseek-quickstart/blob/main/distill/qwen_1.5B_lora.ipynb
- Unsloth文档: https://github.com/unslothai/unsloth
- 数据集: https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT

##  实验步骤

### 步骤1: 环境准备
`ash
# 安装依赖
pip install -r requirements.txt

# 验证安装
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
`

### 步骤2: 数据集准备
`python
from datasets import load_dataset

# 加载医疗推理数据集
dataset = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT")
print(f"数据集大小: {len(dataset['train'])} 训练样本")
`

### 步骤3: 模型初始化
`python
from unsloth import FastLanguageModel

# 加载和配置模型
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="deepseek-ai/deepseek-coder-1.3b-instruct",
    max_seq_length=2048,
    load_in_4bit=True,
    token=your_hf_token
)
`

### 步骤4: QLoRA配置
`python
# 配置LoRA参数
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)
`

### 步骤5: 训练配置
`python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./models/distilled_model",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_steps=100,
    logging_steps=10,
    save_steps=500,
    evaluation_strategy="steps",
    eval_steps=500,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)
`

### 步骤6: 开始训练
`python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer
)

# 开始训练
trainer.train()
`

### 步骤7: 性能评估
`python
# 评估微调后的模型
eval_results = trainer.evaluate()
print(f"评估结果: {eval_results}")

# 生成测试样本
test_prompts = [
    "什么是糖尿病？",
    "高血压患者应该注意什么？",
    "感冒的症状有哪些？"
]

for prompt in test_prompts:
    response = generate_response(model, tokenizer, prompt)
    print(f"问题: {prompt}")
    print(f"回答: {response}\\n")
`

##  预期结果
- 微调后的模型在医疗推理任务上表现更好
- 回答更加准确和专业
- 推理逻辑更加清晰
- 医疗知识应用更加准确

##  注意事项
1. 训练时间可能很长，请耐心等待
2. 定期保存检查点，避免训练中断
3. 监控GPU显存使用情况
4. 注意数据集的版权和使用条款

##  故障排除
- 如果显存不足，减少batch_size
- 如果网络问题，使用镜像源
- 如果依赖冲突，创建虚拟环境

##  实验报告
完成实验后，请记录：
1. 训练配置参数
2. 训练过程中的关键指标
3. 微调前后的性能对比
4. 遇到的问题和解决方案
5. 实验结论和改进建议

##  完成标准
-  成功完成QLoRA微调
-  在医疗数据集上验证效果
-  对比微调前后的性能
-  生成完整的实验报告

祝实验顺利！
