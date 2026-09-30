# meeting-materials

本人的学习笔记库，**整个仓库就是一个 Obsidian vault 的根目录**。

## 在 Obsidian 里打开

1. `git clone https://github.com/cchen473/meeting-materials.git`
2. Obsidian →「打开文件夹作为库 / Open folder as vault」→ 选中 clone 下来的仓库根目录
3. 完成，无需额外配置，各笔记里 `Figs/` 的图片会直接内嵌显示

## 格式约定

- 链接统一使用 Obsidian 原生 wikilink：图片 `![[C2F1.png]]`，笔记之间 `[[Badge]]`
- **不写路径、不做 URL 编码**（不要出现 `Figs/x.png` 或 `%E5%A4%9A…` 这类写法）——Obsidian 按文件名在整个 vault 内解析
- 每篇笔记的配图放在同级的 `Figs/` 目录下，且**图片文件名全局唯一**，否则 wikilink 会产生歧义、可能显示错图
- `.obsidian/`（本地界面配置）与 `.DS_Store` 不入库

## 目录

| 目录 | 内容 |
| --- | --- |
| `RL/` | 强化学习自学笔记（Sutton 中文版），`RLcode/` 放实验代码 |
| `Active_Learning/` | 主动学习论文精读，`MMAL/` 为多模态主动学习子专题 |
| `Multimodal_Learning/` | 多模态学习 |
| `LLM_Inference_Optimization/` | 大模型推理优化 |
| `Basic_ML/`、`Embodied_Intelligence/`、`LLM_and_Agent/` | 预留目录（`占位符.txt`） |
