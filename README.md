# Metask Jev-Lab

探索最优开放权重的 typed-decision（Jev-class）模型：给定 state + bounded rubric，单次前向输出每个选项的校准概率。无生成、无解析。

## 模型一览

| 模型 | 基座 | 13 子集宏平均 | JevBench 231 | 权重 |
|---|---|---:|---:|---|
| **metask-jev-4b** | Qwen3.5-4B | **79.6%** | **80.1%** (@4096) | [HF](https://huggingface.co/wayfind/metask-jev-4b-policy-mix) · 魔搭待发 |
| metask-jev-0.8b | Qwen3.5-0.8B | 74.7% | 55.8% | 未发布 |

参照：Bespoke Nimble-9B 74.8% / 63.5（第14名）；Jev 1.13.0 76.0% / 75.3（第2名）。

## 一键复现推理

```bash
git clone https://github.com/metask-ai/metask-jev && cd metask-jev
pip install -r inference/requirements.txt
python inference/demo.py --model wayfind/metask-jev-4b-policy-mix
```

输入一段 state + schema（choice / boolean / rubric score），输出每个选项的校准概率。单次前向，~24ms/决策（RTX 4090）。

## 目录

- `models/` — 模型注册表 + 每模型的 model card / 评测结果
- `recipes/` — 训练配方卡 + 可复现训练/评测脚本
- `data_pipeline/` — 数据构建管线（视图增强、教师软标签合成）
- `inference/` — 统一推理层（candidate-logit scorer，MPS/CUDA 自适应）
- `release/` — HF + ModelScope 双发流水线

> 训练数据的教师蒸馏配方细节为内部文档，不在本仓库公开。

## 基准

评测协议：JevBench v1.2（[github.com/fstandhartinger/jevbench](https://github.com/fstandhartinger/jevbench)，MIT）+ 13 个人工标注公开子集（3,880 条，与 Bespoke Nimble 口径一致）。

## Licence

代码 Apache-2.0。模型权重遵循其基座（Qwen3.5 系列）条款。