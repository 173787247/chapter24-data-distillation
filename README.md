# 第二十四章作业：数据蒸馏实践

##  作业目标
按照本节课教授的数据蒸馏方法，在 FreedomIntelligence/medical-o1-reasoning-SFT 数据集上，使用 Unsloth 对 DeepSeek-R1-Distill-Qwen-1.5B 模型进行 QLoRA 微调，并对比微调前后的模型生成结果。

##  项目概述
本项目实现了基于Unsloth框架的QLoRA微调，用于医疗推理任务的数据蒸馏。通过对比微调前后的模型性能，验证数据蒸馏的效果。

##  项目结构
`
chapter24-data-distillation/
 README.md                           # 项目说明文档
 requirements.txt                     # Python依赖包
 config/                             # 配置文件目录
    training_config.yaml            # 训练配置文件
 data/                               # 数据目录
    medical_dataset/                # 医疗数据集
 models/                             # 模型目录
    base_model/                     # 基础模型
    distilled_model/                # 蒸馏后模型
 results/                            # 结果目录
    training_logs/                  # 训练日志
    generated_samples/              # 生成样本
    performance_comparison/         # 性能对比
 medical_distillation_experiment.py  # 主要实验脚本
 EXPERIMENT_GUIDE.md                 # 实验指南
`

##  技术实现
- **模型**: DeepSeek-R1-Distill-Qwen-1.5B
- **框架**: Unsloth (高效QLoRA微调)
- **数据集**: FreedomIntelligence/medical-o1-reasoning-SFT
- **微调方法**: QLoRA (Quantized Low-Rank Adaptation)
- **任务类型**: 医疗推理问答

##  快速开始

### 1. 环境准备
`ash
pip install -r requirements.txt
`

### 2. 配置设置
编辑 config/training_config.yaml 文件，设置模型参数和训练配置。

### 3. 运行实验
`ash
python medical_distillation_experiment.py
`

##  实验设计
1. **基础模型评估**: 在医疗数据集上测试原始模型性能
2. **QLoRA微调**: 使用Unsloth进行高效微调
3. **性能对比**: 对比微调前后的生成质量
4. **结果分析**: 分析数据蒸馏的改进效果

##  评估指标
- **生成质量**: 回答准确性、逻辑性
- **推理能力**: 医疗知识应用、问题解决
- **效率提升**: 训练时间、资源消耗对比

##  学习价值
- 掌握Unsloth框架的使用
- 理解QLoRA微调原理
- 学习数据蒸馏实践方法
- 提升医疗AI模型性能

##  注意事项
- 建议使用GPU进行训练
- 微调时间可能较长，请耐心等待
- 及时保存训练过程中的检查点
- 注意模型文件大小和存储空间

##  实验总结
**数据蒸馏实验100%完成！**

成功实现了：
-  使用Unsloth进行QLoRA微调
-  在医疗推理数据集上验证效果
-  对比微调前后的模型性能
-  分析数据蒸馏的改进效果

##  联系方式
- **GitHub**: [你的仓库链接]
- **作业**: 第二十四章 - 数据蒸馏实践
- **完成时间**: 2025年8月

##  许可协议
本项目基于MIT许可协议开源，详见LICENSE文件。
