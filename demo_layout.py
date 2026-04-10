""" 
Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
SPDX-License-Identifier: MIT
"""

import argparse
import glob
import os

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from utils.utils import *


class DOLPHIN:
    def __init__(self, model_id_or_path):
        """Initialize the Hugging Face model
        
        Args:
            model_id_or_path: Path to local model or Hugging Face model ID
        """
        # Load model from local path or Hugging Face hub
        self.processor = AutoProcessor.from_pretrained(model_id_or_path)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id_or_path)
        self.model.eval()
        
        # Set device and precision
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        if self.device == "cuda":
            self.model = self.model.bfloat16()
        else:
            self.model = self.model.float()
        
        # set tokenizer
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.padding_side = "left"

    def chat(self, prompt, image):
        # Check if we're dealing with a batch
        is_batch = isinstance(image, list)
        
        if not is_batch:
            # Single image, wrap it in a list for consistent processing
            images = [image]
            prompts = [prompt]
        else:
            # Batch of images
            images = image
            prompts = prompt if isinstance(prompt, list) else [prompt] * len(images)
        
        assert len(images) == len(prompts)
        
        # preprocess all images
        processed_images = [resize_img(img) for img in images]
        # generate all messages
        all_messages = []
        for img, question in zip(processed_images, prompts):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": img,
                        },
                        {"type": "text", "text": question}
                    ],
                }
            ]
            all_messages.append(messages)

        # prepare all texts
        texts = [
            self.processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            for msgs in all_messages
        ]

        # collect all image inputs
        all_image_inputs = []
        all_video_inputs = None
        for msgs in all_messages:
            image_inputs, video_inputs = process_vision_info(msgs)
            all_image_inputs.extend(image_inputs)

        # prepare model inputs
        inputs = self.processor(
            text=texts,
            images=all_image_inputs if all_image_inputs else None,
            videos=all_video_inputs if all_video_inputs else None,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        # inference
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=4096,
            do_sample=False,
            temperature=None,
            # repetition_penalty=1.05
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        results = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )

        # Return a single result for single image input
        if not is_batch:
            return results[0]
        return results



def process_layout(input_path, model, save_dir):
    """Process layout detection for image or PDF
    
    Args:
        input_path: Path to input image or PDF
        model: DOLPHIN model instance
        save_dir: Directory to save results
    """
    file_ext = os.path.splitext(input_path)[1].lower()
    
    if file_ext == '.pdf':
        # Convert PDF to images
        images = convert_pdf_to_images(input_path)
        if not images:
            raise Exception(f"Failed to convert PDF {input_path} to images")
        
        # Process each page
        for page_idx, pil_image in enumerate(images):
            print(f"\nProcessing page {page_idx + 1}/{len(images)}")
            
            # Generate output name for this page
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            page_name = f"{base_name}_page_{page_idx + 1:03d}"
            
            # Process layout for this page
            process_single_layout(pil_image, model, save_dir, page_name)
    
    else:
        # Process regular image file
        pil_image = Image.open(input_path).convert("RGB")
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        process_single_layout(pil_image, model, save_dir, base_name)


def process_single_layout(pil_image, model, save_dir, image_name):
    """Process layout for a single image
    
    Args:
        pil_image: PIL Image object
        model: DOLPHIN model instance
        save_dir: Directory to save results
        image_name: Name for the output files
    """
    # Parse layout
    print("Parsing layout and reading order...")
    layout_results = model.chat("Parse the reading order of this document.", pil_image)

    # Parse the layout string
    layout_results_list = parse_layout_string(layout_results)
    if not layout_results_list or not (layout_results.startswith("[") and layout_results.endswith("]")):
        layout_results_list = [([0, 0, *pil_image.size], 'distorted_page', [])]
    
    # map bbox to original image coordinates
    recognition_results = []
    reading_order = 0
    for bbox, label, tags in layout_results_list:
        x1, y1, x2, y2 = process_coordinates(bbox, pil_image)
        recognition_results.append({
                        "label": label,
                        "bbox": [x1, y1, x2, y2],
                        "text": "", # empty for now
                        "reading_order": reading_order,
                        "tags": tags,
                    })
        reading_order += 1
    json_path = save_outputs(recognition_results, pil_image, image_name, save_dir)


def main():
    parser = argparse.ArgumentParser(description="Layout detection and visualization using DOLPHIN model")
    parser.add_argument("--model_path", default="./hf_model", help="Path to Hugging Face model")
    parser.add_argument(
        "--input_path", 
        type=str, 
        required=True, 
        help="Path to input image/PDF or directory of files"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Directory to save results (default: same as input directory)",
    )
    args = parser.parse_args()
    
    # Load Model
    print("Loading model...")
    model = DOLPHIN(args.model_path)
    
    # Set save directory
    save_dir = args.save_dir or (
        args.input_path if os.path.isdir(args.input_path) else os.path.dirname(args.input_path)
    )
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Collect files
    if os.path.isdir(args.input_path):
        # Support both image and PDF files
        file_extensions = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG", ".pdf", ".PDF"]
        
        input_files = []
        for ext in file_extensions:
            input_files.extend(glob.glob(os.path.join(args.input_path, f"*{ext}")))
        input_files = sorted(input_files)
    else:
        if not os.path.exists(args.input_path):
            raise FileNotFoundError(f"Input path {args.input_path} does not exist")
        
        # Check if it's a supported file type
        file_ext = os.path.splitext(args.input_path)[1].lower()
        supported_exts = ['.jpg', '.jpeg', '.png', '.pdf']
        
        if file_ext not in supported_exts:
            raise ValueError(f"Unsupported file type: {file_ext}. Supported types: {supported_exts}")
        
        input_files = [args.input_path]
    
    total_files = len(input_files)
    print(f"\nTotal files to process: {total_files}")
    
    # Process files
    for file_path in input_files:
        print(f"\n{'='*60}")
        print(f"Processing: {file_path}")
        print('='*60)
        
        try:
            process_layout(
                input_path=file_path,
                model=model,
                save_dir=save_dir,
            )
            print(f"\n✓ Processing completed for {file_path}")
            
        except Exception as e:
            print(f"\n✗ Error processing {file_path}: {str(e)}")
            continue
    
    print(f"\n{'='*60}")
    print(f"All processing completed. Results saved to {save_dir}")
    print('='*60)


if __name__ == "__main__":
    main()
