"""
Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
SPDX-License-Identifier: MIT
"""

import argparse
import glob
import os

import cv2
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


def process_element(image_path, model, element_type, save_dir=None):
    """Process a single element image (text, table, formula)
    
    Args:
        image_path: Path to the element image
        model: HFModel model instance
        element_type: Type of element ('text', 'table', 'formula')
        save_dir: Directory to save results (default: same as input directory)
        
    Returns:
        Parsed content of the element and recognition results
    """
    # Load and prepare image
    pil_image = Image.open(image_path).convert("RGB")
    # pil_image = crop_margin(pil_image)
    
    # Select appropriate prompt based on element type
    if element_type == "table":
        prompt = "Parse the table in the image."
        label = "tab"
    elif element_type == "formula":
        prompt = "Read formula in the image."
        label = "equ"
    elif element_type == "code":
        prompt = "Read code in the image."
        label = "code"
    else:  # Default to text
        prompt = "Read text in the image."
        label = "para"
    
    # Process the element
    result = model.chat(prompt, pil_image)
    
    # Create recognition result in the same format as the document parser
    recognition_results = [
        {
            "label": label,
            "text": result.strip(),
        }
    ]
    
    # Save results if save_dir is provided
    save_outputs(recognition_results, pil_image, os.path.basename(image_path).split(".")[0], save_dir)
    print(f"Results saved to {save_dir}")
    
    return result, recognition_results


def main():
    parser = argparse.ArgumentParser(description="Element-level processing using DOLPHIN model")
    parser.add_argument("--model_path", default="./hf_model", help="Path to Hugging Face model")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input image or directory of images")
    parser.add_argument(
        "--element_type",
        type=str,
        choices=["text", "table", "formula", "code"],
        default="text",
        help="Type of element to process (text, table, formula)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Directory to save parsing results (default: same as input directory)",
    )
    parser.add_argument("--print_results", action="store_true", help="Print recognition results to console")
    args = parser.parse_args()
    
    # Load Model
    model = DOLPHIN(args.model_path)
    
    # Set save directory
    save_dir = args.save_dir or (
        args.input_path if os.path.isdir(args.input_path) else os.path.dirname(args.input_path)
    )
    setup_output_dirs(save_dir)
    
    # Collect Images
    if os.path.isdir(args.input_path):
        image_files = []
        for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
            image_files.extend(glob.glob(os.path.join(args.input_path, f"*{ext}")))
        image_files = sorted(image_files)
    else:
        if not os.path.exists(args.input_path):
            raise FileNotFoundError(f"Input path {args.input_path} does not exist")
        image_files = [args.input_path]
    
    total_samples = len(image_files)
    print(f"\nTotal samples to process: {total_samples}")
    
    # Process images one by one
    for image_path in image_files:
        print(f"\nProcessing {image_path}")
        try:
            result, recognition_result = process_element(
                image_path=image_path,
                model=model,
                element_type=args.element_type,
                save_dir=save_dir,
            )

            if args.print_results:
                print("\nRecognition result:")
                print(result)
                print("-" * 40)
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            continue


if __name__ == "__main__":
    main()
