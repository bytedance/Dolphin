"""
Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
SPDX-License-Identifier: MIT
"""

import argparse
import glob
import os

import torch
from PIL import Image
from transformers import AutoProcessor, VisionEncoderDecoderModel

from utils.utils import *


class DOLPHIN:
    def __init__(self, model_id_or_path):
        """Initialize the Hugging Face model
        
        Args:
            model_id_or_path: Path to local model or Hugging Face model ID
        """
        # Load model from local path or Hugging Face hub
        self.processor = AutoProcessor.from_pretrained(model_id_or_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_id_or_path)
        self.model.eval()
        
        # Set device and precision
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model = self.model.half()  # Always use half precision by default
        
        # set tokenizer
        self.tokenizer = self.processor.tokenizer
        
    def chat(self, prompt, image):
        """Process an image with the given prompt
        
        Args:
            prompt: Text prompt to guide the model
            image: PIL Image to process
            
        Returns:
            Generated text from the model
        """
        # Prepare image
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.half()
            
        # Prepare prompt
        prompt = f"<s>{prompt} <Answer/>"
        prompt_ids = self.tokenizer(
            prompt, 
            add_special_tokens=False, 
            return_tensors="pt"
        ).input_ids.to(self.device)
        
        decoder_attention_mask = torch.ones_like(prompt_ids)
        
        # Generate text
        outputs = self.model.generate(
            pixel_values=pixel_values.to(self.device),
            decoder_input_ids=prompt_ids,
            decoder_attention_mask=decoder_attention_mask,
            min_length=1,
            max_length=4096,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
            bad_words_ids=[[self.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.1,
            temperature=1.0
        )
        
        # Process the output
        sequence = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=False)[0]
        sequence = sequence.replace(prompt, "").replace("<pad>", "").replace("</s>", "").strip()
        
        return sequence

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
    pil_image = crop_margin(pil_image)
    
    # Select appropriate prompt based on element type
    if element_type == "table":
        prompt = "Parse the table in the image."
        label = "tab"
    elif element_type == "formula":
        prompt = "Read text in the image."
        label = "formula"
    else:  # Default to text
        prompt = "Read text in the image."
        label = "text"
    
    # Process the element
    result = model.chat(prompt, pil_image)
    
    # Create recognition result in the same format as the document parser
    recognition_result = [
        {
            "label": label,
            "text": result.strip(),
        }
    ]
    
    # Save results if save_dir is provided
    if save_dir:
        save_outputs(recognition_result, image_path, save_dir)
        print(f"Results saved to {save_dir}")
    
    return result, recognition_result


def main():
    parser = argparse.ArgumentParser(description="Element-level processing using DOLPHIN model")
    parser.add_argument("--model_path", default="./hf_model", help="Path to Hugging Face model")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input image or directory of images")
    parser.add_argument(
        "--element_type",
        type=str,
        choices=["text", "table", "formula"],
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
