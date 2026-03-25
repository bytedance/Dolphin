<div align="center">
  <img src="./assets/dolphin.png" width="300">
</div>

<div align="center">
  <a href="https://arxiv.org/abs/2505.14059">
    <img src="https://img.shields.io/badge/论文-arXiv-red">
  </a>
  <a href="https://huggingface.co/ByteDance/Dolphin-v2">
    <img src="https://img.shields.io/badge/HuggingFace-Dolphin-yellow">
  </a>
  <a href="https://github.com/bytedance/Dolphin">
    <img src="https://img.shields.io/badge/代码-Github-green">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/许可证-MIT-lightgray">
  </a>
  <br>
</div>

<br>

<div align="center">
  <img src="./assets/demo.gif" width="800">
</div>

# Dolphin: 基于异构锚点提示的文档图像解析

Dolphin（**Do**cument Image **P**arsing via **H**eterogeneous Anchor Prompt**in**g）是一个创新的多模态文档图像解析模型（**0.3B**），采用"分析-解析"的两阶段范式。本仓库包含Dolphin的演示代码和预训练模型。

## 📑 概述

由于文档图像中文本段落、图表、公式和表格等元素的复杂交织，文档图像解析具有挑战性。Dolphin通过两阶段方法解决这些挑战：

1. **🔍 第一阶段**：通过按自然阅读顺序生成元素序列进行全面的页面级布局分析
2. **🧩 第二阶段**：使用异构锚点和任务特定提示高效并行解析文档元素

<div align="center">
  <img src="./assets/framework.png" width="680">
</div>

Dolphin在多样化的页面级和元素级解析任务中取得了优异的性能，同时通过其轻量级架构和并行解析机制确保了卓越的效率。

## 📅 更新日志
- 🔥 **2025.12.12** *Dolphin-v2* 开源！支持 21 类元素检测、属性字段提取、代码专用解析，以及拍照文档解析。（原1.5版本已迁移至[v1.5分支](https://github.com/bytedance/Dolphin/tree/v1.5)）
- 🔥 **2025.10.16** *Dolphin-1.5* 开源！在保持轻量级0.3B架构的同时，该版本实现了显著的解析性能提升。（原1.0版本已迁移至[v1.0分支](https://github.com/bytedance/Dolphin/tree/v1.0)）
- 🔥 **2025.07.10** *Fox-Page* 基准测试开源。这是原始 [Fox 数据集](https://github.com/ucaslcl/Fox) 人工矫正标注后的版本。下载地址：[百度网盘](https://pan.baidu.com/share/init?surl=t746ULp6iU5bUraVrPlMSw&pwd=fox1) | [Google Drive](https://drive.google.com/file/d/1yZQZqI34QCqvhB4Tmdl3X_XEvYvQyP0q/view?usp=sharing)。
- 🔥 **2025.06.30** 新增[TensorRT-LLM](https://github.com/bytedance/Dolphin/blob/master/deployment/tensorrt_llm/ReadMe.md)支持，提升推理速度！
- 🔥 **2025.06.27** 新增[vLLM](https://github.com/bytedance/Dolphin/blob/master/deployment/vllm/ReadMe.md)支持，提升推理速度！
- 🔥 **2025.06.13** 新增多页PDF文档解析功能。
- 🔥 **2025.05.21** 我们的演示已在 [链接](http://115.190.42.15:8888/dolphin/) 发布。快来体验吧！
- 🔥 **2025.05.20** Dolphin的预训练模型和推理代码已发布。
- 🔥 **2025.05.16** 我们的论文已被ACL 2025接收。论文链接：[arXiv](https://arxiv.org/abs/2505.14059)。

## 📈 性能表现

<table style="width:90%; border-collapse: collapse; text-align: center;">
    <caption>OmniDocBench (v1.5) 测试基准上评估结果</caption>
    <thead>
        <tr>
            <th style="text-align: center !important;">模型</th>
            <th style="text-align: center !important;">参数</th>
            <th style="text-align: center !important;">总体&#x2191;</th>
            <th style="text-align: center !important;">文本<sup>Edit</sup>&#x2193;</th>
            <th style="text-align: center !important;">公式<sup>CDM</sup>&#x2191;</th>
            <th style="text-align: center !important;">表格<sup>TEDS</sup>&#x2191;</th>
            <th style="text-align: center !important;">表格<sup>TEDS-S</sup>&#x2191;</th>
            <th style="text-align: center !important;">阅读顺序<sup>Edit</sup>&#x2193;</th>
        </tr>
    </thead>
        <tr>
            <td>Dolphin</td>
            <td>0.3B</td>
            <td>74.67</td>
            <td>0.125</td>
            <td>67.85</td>
            <td>68.70</td>
            <td>77.77</td>
            <td>0.124</td>
        </tr>
        <tr>
            <td>Dolphin-1.5</td>
            <td>0.3B</td>
            <td>85.06</td>
            <td>0.085</td>
            <td>79.44</td>
            <td>84.25</td>
            <td>88.06</td>
            <td>0.071</td>
        </tr>
        <tr>
            <td>Dolphin-v2</td>
            <td>0.3B</td>
            <td><strong>89.78</strong></td>
            <td><strong>0.054</strong></td>
            <td><strong>87.63</strong></td>
            <td><strong>87.02</strong></td>
            <td><strong>90.48</strong></td>
            <td><strong>0.054</strong></td>
        </tr>
    </tbody>
</table>

## 🛠️ 安装

1. 克隆仓库：
   ```bash
   git clone https://github.com/ByteDance/Dolphin.git
   cd Dolphin
   ```

2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

3. 使用以下选项之一下载 *Dolphin-v2* 的预训练模型：
   访问我们的Huggingface [模型卡片](https://huggingface.co/ByteDance/Dolphin-v2)，或通过以下方式下载模型：
   
   ```bash
   # 从Hugging Face Hub下载模型
   git lfs install
   git clone https://huggingface.co/ByteDance/Dolphin-v2 ./hf_model
   # 或使用Hugging Face CLI
   pip install huggingface_hub
   huggingface-cli download ByteDance/Dolphin-v2 --local-dir ./hf_model
   ```

## ⚡ 推理

Dolphin提供两个推理框架，支持两种解析粒度：
- **页面级解析**：将整个文档页面解析为结构化的JSON和Markdown格式
- **元素级解析**：解析单个文档元素（文本、表格、公式）


### 📄 页面级解析

```bash
# 处理单个文档图像
python demo_page.py --model_path ./hf_model --save_dir ./results \
    --input_path ./demo/page_imgs/page_1.png 

# 处理单个文档PDF
python demo_page.py --model_path ./hf_model --save_dir ./results \
    --input_path ./demo/page_imgs/page_6.pdf 

# 处理目录中的所有文档
python demo_page.py --model_path ./hf_model --save_dir ./results \
    --input_path ./demo/page_imgs 

# 使用自定义批次大小进行并行元素解码
python demo_page.py --model_path ./hf_model --save_dir ./results \
    --input_path ./demo/page_imgs \
    --max_batch_size 8

# 页面解析时忽略识别出的页眉和页脚区域
python demo_page.py --model_path ./hf_model --save_dir ./results \
    --input_path ./demo/page_imgs/page_6.pdf \
    --ignore_header --ignore_footer
```

### 🧩 元素级解析

````bash
# 解析块图像 (支持块图像类型: table, formula, text, or code)
python demo_element.py --model_path ./hf_model --save_dir ./results \
    --input_path  \
    --element_type [table|formula|text|code]
````

### 🎨 元素定位及阅读顺序解析
````bash
# 处理单个文档图像
python demo_layout.py --model_path ./hf_model --save_dir ./results \
    --input_path ./demo/page_imgs/page_1.png \
    
# 处理单个文档PDF
python demo_layout.py --model_path ./hf_model --save_dir ./results \
    --input_path ./demo/page_imgs/page_6.pdf \

# 处理目录中的所有文档
python demo_layout.py --model_path ./hf_model --save_dir ./results \
    --input_path ./demo/page_imgs 
````


## 🌟 主要特性

- 🔄 基于单一VLM的两阶段分析-解析方法
- 📊 在文档解析任务上的优异性能
- 🔍 自然阅读顺序元素序列生成
- 🧩 针对不同文档元素的异构锚点提示
- ⏱️ 高效的并行解析机制
- 🤗 支持Hugging Face Transformers，便于集成


## 📮 通知
**征集不良案例：** 如果您遇到模型表现不佳的案例，我们非常欢迎您在issue中分享。我们正在持续优化和改进模型。


## 💖 致谢

我们要感谢以下开源项目为本工作提供的灵感和参考：
- [OmniDocBench](https://github.com/opendatalab/OmniDocBench)
- [Donut](https://github.com/clovaai/donut/)
- [Nougat](https://github.com/facebookresearch/nougat)
- [GOT](https://github.com/Ucas-HaoranWei/GOT-OCR2.0)
- [MinerU](https://github.com/opendatalab/MinerU/tree/master)
- [Swin](https://github.com/microsoft/Swin-Transformer)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)


## 📝 引用

如果您在研究中发现此代码有用，请使用以下BibTeX条目。

```bibtex
@article{feng2025dolphin,
  title={Dolphin: Document Image Parsing via Heterogeneous Anchor Prompting},
  author={Feng, Hao and Wei, Shu and Fei, Xiang and Shi, Wei and Han, Yingdong and Liao, Lei and Lu, Jinghui and Wu, Binghong and Liu, Qi and Lin, Chunhui and others},
  journal={arXiv preprint arXiv:2505.14059},
  year={2025}
}
```

## 星标历史

[![Star History Chart](https://api.star-history.com/svg?repos=bytedance/Dolphin&type=Date)](https://www.star-history.com/#bytedance/Dolphin&Date)
