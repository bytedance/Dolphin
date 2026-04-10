<div align="center">
  <img src="./assets/dolphin.png" width="300">
</div>

<div align="center">
  <a href="https://arxiv.org/abs/2505.14059">
    <img src="https://img.shields.io/badge/Paper-arXiv-red">
  </a>
  <a href="https://huggingface.co/ByteDance/Dolphin-v2">
    <img src="https://img.shields.io/badge/HuggingFace-Dolphin-yellow">
  </a>
  <a href="https://github.com/bytedance/Dolphin">
    <img src="https://img.shields.io/badge/Code-Github-green">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-lightgray">
  </a>
  <br>
</div>

<br>

<div align="center">
  <img src="./assets/demo.gif" width="800">
</div>

# Dolphin: Document Image Parsing via Heterogeneous Anchor Prompting
Dolphin-v2 is an enhanced universal document parsing model that substantially improves upon the original Dolphin. It seamlessly handles any document type—whether digital-born or photographed—through a document-type-aware two-stage architecture with scalable anchor prompting.


## 📑 Overview

Document image parsing is challenging due to diverse document types and complexly intertwined elements such as text paragraphs, figures, formulas, tables, and code blocks. Dolphin-v2 addresses these challenges through a document-type-aware two-stage approach:

1. **🔍 Stage 1**: Document type classification (digital vs. photographed) + layout analysis with reading order prediction
2. **🧩 Stage 2**: Hybrid parsing strategy - holistic parsing for photographed documents, parallel element-wise parsing for digital documents

<div align="center">
  <img src="./assets/framework.png" width="680">
</div>

Dolphin achieves promising performance across diverse page-level and element-level parsing tasks while ensuring superior efficiency through its lightweight architecture and parallel parsing mechanism.

<!-- ## 🚀 Demo
Try our demo on [Demo-Dolphin](https://huggingface.co/spaces/ByteDance/Dolphin). -->

## 📅 Changelog
- 🔥 **2025.12.12** Released *Dolphin-v2* model. Upgraded to 3B parameters with 21-element detection, attribute field extraction, dedicated formula/code parsing, and robust photographed document parsing. (Dolphin-1.5 moved to [v1.5 branch](https://github.com/bytedance/Dolphin/tree/v1.5))
- 🔥 **2025.10.16** Released *Dolphin-1.5* model. While maintaining the lightweight 0.3B architecture, this version achieves significant parsing improvements. (Dolphin 1.0 moved to [v1.0 branch](https://github.com/bytedance/Dolphin/tree/v1.0))
- 🔥 **2025.07.10** Released the *Fox-Page Benchmark*, a manually refined subset of the original [Fox dataset](https://github.com/ucaslcl/Fox). Download via: [Baidu Yun](https://pan.baidu.com/share/init?surl=t746ULp6iU5bUraVrPlMSw&pwd=fox1) | [Google Drive](https://drive.google.com/file/d/1yZQZqI34QCqvhB4Tmdl3X_XEvYvQyP0q/view?usp=sharing).
- 🔥 **2025.06.30** Added [TensorRT-LLM support](https://github.com/bytedance/Dolphin/blob/master/deployment/tensorrt_llm/ReadMe.md) for accelerated inference！
- 🔥 **2025.06.27** Added [vLLM support](https://github.com/bytedance/Dolphin/blob/master/deployment/vllm/ReadMe.md) for accelerated inference！
- 🔥 **2025.06.13** Added multi-page PDF document parsing capability.
- 🔥 **2025.05.21** Our demo is released at [link](http://115.190.42.15:8888/dolphin/). Check it out!
- 🔥 **2025.05.20** The pretrained model and inference code of Dolphin are released.
- 🔥 **2025.05.16** Our paper has been accepted by ACL 2025. Paper link: [arXiv](https://arxiv.org/abs/2505.14059).

## 📈 Performance

<table style="width:90%; border-collapse: collapse; text-align: center;">
    <caption>Comprehensive evaluation of document parsing on OmniDocBench (v1.5)</caption>
    <thead>
        <tr>
            <th style="text-align: center !important;">Model</th>
            <th style="text-align: center !important;">Size</th>
            <th style="text-align: center !important;">Overall&#x2191;</th>
            <th style="text-align: center !important;">Text<sup>Edit</sup>&#x2193;</th>
            <th style="text-align: center !important;">Formula<sup>CDM</sup>&#x2191;</th>
            <th style="text-align: center !important;">Table<sup>TEDS</sup>&#x2191;</th>
            <th style="text-align: center !important;">Table<sup>TEDS-S</sup>&#x2191;</th>
            <th style="text-align: center !important;">Read Order<sup>Edit</sup>&#x2193;</th>
        </tr>
    </thead>
    <tbody>
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
            <td>3B</td>
            <td><strong>89.78</strong></td>
            <td><strong>0.054</strong></td>
            <td><strong>87.63</strong></td>
            <td><strong>87.02</strong></td>
            <td><strong>90.48</strong></td>
            <td><strong>0.054</strong></td>
        </tr>
    </tbody>
</table>

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ByteDance/Dolphin.git
   cd Dolphin
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the pre-trained models of *Dolphin-v2*:

   Visit our Huggingface [model card](https://huggingface.co/ByteDance/Dolphin-v2), or download model by:
   
   ```bash
   # Download the model from Hugging Face Hub
   git lfs install
   git clone https://huggingface.co/ByteDance/Dolphin-v2 ./hf_model
   # Or use the Hugging Face CLI
   pip install huggingface_hub
   huggingface-cli download ByteDance/Dolphin-v2 --local-dir ./hf_model
   ```

## ⚡ Inference

Dolphin provides two inference frameworks with support for two parsing granularities:
- **Page-level Parsing**: Parse the entire document page into a structured JSON and Markdown format
- **Element-level Parsing**: Parse individual document elements (text, table, formula)


### 📄 Page-level Parsing

```bash
# Process a single document image
python demo_page.py --model_path ./hf_model --save_dir ./results \
    --input_path ./demo/page_imgs/page_1.png 

# Process a single document pdf
python demo_page.py --model_path ./hf_model --save_dir ./results \
    --input_path ./demo/page_imgs/page_6.pdf 

# Process all documents in a directory
python demo_page.py --model_path ./hf_model --save_dir ./results \
    --input_path ./demo/page_imgs 

# Process with custom batch size for parallel element decoding
python demo_page.py --model_path ./hf_model --save_dir ./results \
    --input_path ./demo/page_imgs \
    --max_batch_size 8
```

### 🧩 Element-level Parsing

````bash
# Process element images (specify element_type: table, formula, text, or code)
python demo_element.py --model_path ./hf_model --save_dir ./results \
    --input_path  \
    --element_type [table|formula|text|code]
````

### 🎨 Layout Parsing
````bash
# Process a single document image
python demo_layout.py --model_path ./hf_model --save_dir ./results \
    --input_path ./demo/page_imgs/page_1.png \
    
# Process a single PDF document
python demo_layout.py --model_path ./hf_model --save_dir ./results \
    --input_path ./demo/page_imgs/page_6.pdf \

# Process all documents in a directory
python demo_layout.py --model_path ./hf_model --save_dir ./results \
    --input_path ./demo/page_imgs 
````


## 🌟 Key Features

- 🔄 Two-stage analyze-then-parse approach based on a single VLM
- 📊 Promising performance on document parsing tasks
- 🔍 Natural reading order element sequence generation
- 🧩 Heterogeneous anchor prompting for different document elements
- ⏱️ Efficient parallel parsing mechanism
- 🤗 Support for Hugging Face Transformers for easier integration


## 📮 Notice
**Call for Bad Cases:** If you have encountered any cases where the model performs poorly, we would greatly appreciate it if you could share them in the issue. We are continuously working to optimize and improve the model.

## 💖 Acknowledgement

We would like to acknowledge the following open-source projects that provided inspiration and reference for this work:
- [OmniDocBench](https://github.com/opendatalab/OmniDocBench)
- [Donut](https://github.com/clovaai/donut/)
- [Nougat](https://github.com/facebookresearch/nougat)
- [GOT](https://github.com/Ucas-HaoranWei/GOT-OCR2.0)
- [MinerU](https://github.com/opendatalab/MinerU/tree/master)
- [Swin](https://github.com/microsoft/Swin-Transformer)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)

## 📝 Citation

If you find this code useful for your research, please use the following BibTeX entry.

```bibtex
@article{feng2025dolphin,
  title={Dolphin: Document Image Parsing via Heterogeneous Anchor Prompting},
  author={Feng, Hao and Wei, Shu and Fei, Xiang and Shi, Wei and Han, Yingdong and Liao, Lei and Lu, Jinghui and Wu, Binghong and Liu, Qi and Lin, Chunhui and others},
  journal={arXiv preprint arXiv:2505.14059},
  year={2025}
}
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=bytedance/Dolphin&type=Date)](https://www.star-history.com/#bytedance/Dolphin&Date)
