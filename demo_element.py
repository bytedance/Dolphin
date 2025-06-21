""" 
Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
SPDX-License-Identifier: MIT
"""

import argparse
import glob
import os

from omegaconf import OmegaConf
from PIL import Image

import gc
import torch

from chat import DOLPHIN
from utils.utils import *


def flush_mps_cache():
    """Release cached MPS memory on Apple Silicon GPUs and run Python GC."""
    gc.collect()
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        torch.mps.empty_cache()



def process_element(image_path, model, element_type, save_dir=None):
    """Process a single element image (text, table, formula)

    Args:
        image_path: Path to the element image
        model: DOLPHIN model instance
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
    parser.add_argument("--config", default="./config/Dolphin.yaml", help="Path to configuration file")
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
    config = OmegaConf.load(args.config)
    model = DOLPHIN(config)

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

            # Flush cached GPU memory after each processed image
            flush_mps_cache()

        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            continue


if __name__ == "__main__":
    main()
