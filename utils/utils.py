"""
Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
SPDX-License-Identifier: MIT
"""

import io
import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
import pymupdf
from PIL import Image

from utils.markdown_utils import MarkdownConverter


def save_figure_to_local(pil_crop, save_dir, image_name, reading_order):
    """Save cropped figure to local file system

    Args:
        pil_crop: PIL Image object of the cropped figure
        save_dir: Base directory to save results
        image_name: Name of the source image/document
        reading_order: Reading order of the figure in the document

    Returns:
        str: Filename of the saved figure
    """
    try:
        # Create figures directory if it doesn't exist
        figures_dir = os.path.join(save_dir, "markdown", "figures")
        # os.makedirs(figures_dir, exist_ok=True)

        # Generate figure filename
        figure_filename = f"{image_name}_figure_{reading_order:03d}.png"
        figure_path = os.path.join(figures_dir, figure_filename)

        # Save the figure
        pil_crop.save(figure_path, format="PNG", quality=95)

        # print(f"Saved figure: {figure_filename}")
        return figure_filename

    except Exception as e:
        print(f"Error saving figure: {str(e)}")
        # Return a fallback filename
        return f"{image_name}_figure_{reading_order:03d}_error.png"


def convert_pdf_to_images(pdf_path, target_size=896):
    """Convert PDF pages to images

    Args:
        pdf_path: Path to PDF file
        target_size: Target size for the longest dimension

    Returns:
        List of PIL Images
    """
    images = []
    try:
        doc = pymupdf.open(pdf_path)

        for page_num in range(len(doc)):
            page = doc[page_num]

            # Calculate scale to make longest dimension equal to target_size
            rect = page.rect
            scale = target_size / max(rect.width, rect.height)

            # Render page as image
            mat = pymupdf.Matrix(scale, scale)
            pix = page.get_pixmap(matrix=mat)

            # Convert to PIL Image
            img_data = pix.tobytes("png")
            pil_image = Image.open(io.BytesIO(img_data))
            images.append(pil_image)

        doc.close()
        print(f"Successfully converted {len(images)} pages from PDF")
        return images

    except Exception as e:
        print(f"Error converting PDF to images: {str(e)}")
        return []


def save_combined_pdf_results(all_page_results, pdf_path, save_dir):
    """Save combined results for multi-page PDF with both JSON and Markdown

    Args:
        all_page_results: List of results for all pages
        pdf_path: Path to original PDF file
        save_dir: Directory to save results

    Returns:
        Path to saved combined JSON file
    """
    # Create output filename based on PDF name
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]

    # Prepare combined results
    combined_results = {"source_file": pdf_path, "total_pages": len(all_page_results), "pages": all_page_results}

    # Save combined JSON results
    json_filename = f"{base_name}.json"
    json_path = os.path.join(save_dir, "recognition_json", json_filename)
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(combined_results, f, indent=2, ensure_ascii=False)

    # Generate and save combined markdown
    try:
        markdown_converter = MarkdownConverter()

        # Combine all page results into a single list for markdown conversion
        all_elements = []
        for page_data in all_page_results:
            page_elements = page_data.get("elements", [])
            if page_elements:
                # Add page separator if not the first page
                if all_elements:
                    all_elements.append(
                        {"label": "page_separator", "text": f"\n\n---\n\n", "reading_order": len(all_elements)}
                    )
                all_elements.extend(page_elements)

        # Generate markdown content
        markdown_content = markdown_converter.convert(all_elements)

        # Save markdown file
        markdown_filename = f"{base_name}.md"
        markdown_path = os.path.join(save_dir, "markdown", markdown_filename)
        os.makedirs(os.path.dirname(markdown_path), exist_ok=True)

        with open(markdown_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        # print(f"Combined markdown saved to: {markdown_path}")

    except ImportError:
        print("MarkdownConverter not available, skipping markdown generation")
    except Exception as e:
        print(f"Error generating markdown: {e}")

    # print(f"Combined JSON results saved to: {json_path}")
    return json_path


def parse_layout_string(bbox_str):
    """
    Dolphin-V1.5 layout string parsing function
    Parse layout string to extract bbox and category information
    Supports multiple formats:
    1. Original format: [x1,y1,x2,y2] label
    2. New format: [x1,y1,x2,y2][label][PAIR_SEP] or [x1,y1,x2,y2][label][meta_info][PAIR_SEP]
    """
    parsed_results = []
    
    segments = bbox_str.split('[PAIR_SEP]')
    new_segments = []
    for seg in segments:
        new_segments.extend(seg.split('[RELATION_SEP]'))
    segments = new_segments
    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue
            
        coord_pattern = r'\[(\d*\.?\d+),(\d*\.?\d+),(\d*\.?\d+),(\d*\.?\d+)\]'
        label_pattern = r'\]\[([^\]]+)\]'
        
        coord_match = re.search(coord_pattern, segment)
        label_matches = re.findall(label_pattern, segment)
        
        if coord_match and label_matches:
            coords = [float(coord_match.group(i)) for i in range(1, 5)]
            label = label_matches[0].strip()
            parsed_results.append((coords, label))
    
    return parsed_results


@dataclass
class ImageDimensions:
    """Class to store image dimensions"""

    original_w: int
    original_h: int
    padded_w: int
    padded_h: int


def map_to_original_coordinates(x1, y1, x2, y2, dims: ImageDimensions) -> Tuple[int, int, int, int]:
    """Map coordinates from padded image back to original image

    Args:
        x1, y1, x2, y2: Coordinates in padded image
        dims: Image dimensions object

    Returns:
        tuple: (x1, y1, x2, y2) coordinates in original image
    """
    try:
        # Calculate padding offsets
        top = (dims.padded_h - dims.original_h) // 2
        left = (dims.padded_w - dims.original_w) // 2

        # Map back to original coordinates
        orig_x1 = max(0, x1 - left)
        orig_y1 = max(0, y1 - top)
        orig_x2 = min(dims.original_w, x2 - left)
        orig_y2 = min(dims.original_h, y2 - top)

        # Ensure we have a valid box (width and height > 0)
        if orig_x2 <= orig_x1:
            orig_x2 = min(orig_x1 + 1, dims.original_w)
        if orig_y2 <= orig_y1:
            orig_y2 = min(orig_y1 + 1, dims.original_h)

        return int(orig_x1), int(orig_y1), int(orig_x2), int(orig_y2)
    except Exception as e:
        print(f"map_to_original_coordinates error: {str(e)}")
        # Return safe coordinates
        return 0, 0, min(100, dims.original_w), min(100, dims.original_h)


def process_coordinates(coords, padded_image, dims: ImageDimensions, previous_box=None):
    """Process and adjust coordinates

    Args:
        coords: Normalized coordinates [x1, y1, x2, y2]
        padded_image: Padded image
        dims: Image dimensions object
        previous_box: Previous box coordinates for overlap adjustment

    Returns:
        tuple: (x1, y1, x2, y2, orig_x1, orig_y1, orig_x2, orig_y2, new_previous_box)
    """
    try:
        # Convert normalized coordinates to absolute coordinates
        x1, y1 = round(coords[0] / 896. * dims.padded_w), round(coords[1] / 896. * dims.padded_h)
        x2, y2 = round(coords[2] / 896. * dims.padded_w) + 1, round(coords[3] / 896. * dims.padded_h) + 1

        # Ensure coordinates are within image bounds before adjustment
        x1 = max(0, min(x1, dims.padded_w - 1))
        y1 = max(0, min(y1, dims.padded_h - 1))
        x2 = max(0, min(x2, dims.padded_w))
        y2 = max(0, min(y2, dims.padded_h))

        # Ensure width and height are at least 1 pixel
        if x2 <= x1:
            x2 = min(x1 + 1, dims.padded_w)
        if y2 <= y1:
            y2 = min(y1 + 1, dims.padded_h)

        # Ensure coordinates are still within image bounds after adjustment
        x1 = max(0, min(x1, dims.padded_w - 1))
        y1 = max(0, min(y1, dims.padded_h - 1))
        x2 = max(0, min(x2, dims.padded_w))
        y2 = max(0, min(y2, dims.padded_h))

        # Ensure width and height are at least 1 pixel after adjustment
        if x2 <= x1:
            x2 = min(x1 + 1, dims.padded_w)
        if y2 <= y1:
            y2 = min(y1 + 1, dims.padded_h)

        # Check for overlap with previous box and adjust
        if previous_box is not None:
            prev_x1, prev_y1, prev_x2, prev_y2 = previous_box
            if (x1 < prev_x2 and x2 > prev_x1) and (y1 < prev_y2 and y2 > prev_y1):
                y1 = prev_y2
                # Ensure y1 is still valid
                y1 = min(y1, dims.padded_h - 1)
                # Make sure y2 is still greater than y1
                if y2 <= y1:
                    y2 = min(y1 + 1, dims.padded_h)

        # Update previous box
        new_previous_box = [x1, y1, x2, y2]

        # Map to original coordinates
        orig_x1, orig_y1, orig_x2, orig_y2 = map_to_original_coordinates(x1, y1, x2, y2, dims)

        return x1, y1, x2, y2, orig_x1, orig_y1, orig_x2, orig_y2, new_previous_box
    except Exception as e:
        print(f"process_coordinates error: {str(e)}")
        # Return safe values
        orig_x1, orig_y1, orig_x2, orig_y2 = 0, 0, min(100, dims.original_w), min(100, dims.original_h)
        return 0, 0, 100, 100, orig_x1, orig_y1, orig_x2, orig_y2, [0, 0, 100, 100]


def prepare_image(image, pdf_name=None, page_number=None, processed_images_dir=None) -> Tuple[np.ndarray, ImageDimensions]:
    """Load and prepare image with padding while maintaining aspect ratio

    Args:
        image: PIL image

    Returns:
        tuple: (padded_image, image_dimensions)
    """
    try:
        # Convert PIL image to OpenCV format
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        original_h, original_w = image.shape[:2]

        # Calculate padding to make square image
        max_size = max(original_h, original_w)
        top = (max_size - original_h) // 2
        bottom = max_size - original_h - top
        left = (max_size - original_w) // 2
        right = max_size - original_w - left

        # Apply padding
        padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

         # Save the processed padded image with organized filename
        try:
            if processed_images_dir is None:
                processed_images_dir = "./processed_images"
            # Create PDF-specific subdirectory if pdf_name is provided
            if pdf_name:
                pdf_dir = os.path.join(processed_images_dir, pdf_name)
                os.makedirs(pdf_dir, exist_ok=True)
                
                # Generate organized filename
                if page_number is not None:
                    unique_filename = f"page-{page_number}.png"
                else:
                    unique_filename = f"{pdf_name}.png"
                    
                processed_image_path = os.path.join(pdf_dir, unique_filename)
            else:
                # Fallback to original naming scheme for non-PDF files
                os.makedirs(processed_images_dir, exist_ok=True)
                timestamp = int(time.time() * 1000)  # milliseconds since epoch
                unique_id = str(uuid.uuid4())[:8]  # first 8 characters of UUID
                unique_filename = f"processed_{timestamp}_{unique_id}.png"
                processed_image_path = os.path.join(processed_images_dir, unique_filename)

            cv2.imwrite(processed_image_path, padded_image)
            print(f"✓ Saved processed padded image: {unique_filename}")
            
        except Exception as save_error:
            # Don't let saving errors affect the main functionality
            print(f"Warning: Could not save processed image: {str(save_error)}")


        padded_h, padded_w = padded_image.shape[:2]

        dimensions = ImageDimensions(original_w=original_w, original_h=original_h, padded_w=padded_w, padded_h=padded_h)

        return padded_image, dimensions
    except Exception as e:
        print(f"prepare_image error: {str(e)}")
        # Create a minimal valid image and dimensions
        h, w = image.height, image.width
        dimensions = ImageDimensions(original_w=w, original_h=h, padded_w=w, padded_h=h)
        # Return a black image of the same size
        return np.zeros((h, w, 3), dtype=np.uint8), dimensions


def setup_output_dirs(save_dir):
    """Create necessary output directories"""
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "markdown"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "recognition_json"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "markdown", "figures"), exist_ok=True)


def save_outputs(recognition_results, image_path, save_dir):
    """Save JSON and markdown outputs"""
    basename = os.path.splitext(os.path.basename(image_path))[0]

    # Save JSON file
    json_path = os.path.join(save_dir, "recognition_json", f"{basename}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(recognition_results, f, ensure_ascii=False, indent=2)

    # Generate and save markdown file
    markdown_converter = MarkdownConverter()
    markdown_content = markdown_converter.convert(recognition_results)
    markdown_path = os.path.join(save_dir, "markdown", f"{basename}.md")
    with open(markdown_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    return json_path


def crop_margin(img: Image.Image) -> Image.Image:
    """Crop margins from image"""
    try:
        width, height = img.size
        if width == 0 or height == 0:
            print("Warning: Image has zero width or height")
            return img

        data = np.array(img.convert("L"))
        data = data.astype(np.uint8)
        max_val = data.max()
        min_val = data.min()
        if max_val == min_val:
            return img
        data = (data - min_val) / (max_val - min_val) * 255
        gray = 255 * (data < 200).astype(np.uint8)

        coords = cv2.findNonZero(gray)  # Find all non-zero points (text)
        if coords is None:
            return img
        a, b, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box

        # Ensure crop coordinates are within image bounds
        a = max(0, a)
        b = max(0, b)
        w = min(w, width - a)
        h = min(h, height - b)

        # Only crop if we have a valid region
        if w > 0 and h > 0:
            return img.crop((a, b, a + w, b + h))
        return img
    except Exception as e:
        print(f"crop_margin error: {str(e)}")
        return img  # Return original image on error

def visualize_layout(image_path, layout_results, save_path, alpha=0.3, original_image=None):
    """Visualize layout detection results on the image
    
    Args:
        image_path: Path to the input image
        layout_results: List of (bbox, label) tuples with coordinates in 896x896 space
        save_path: Path to save the visualization
        alpha: Transparency of the overlay (0-1, lower = more transparent)
        original_image: Original PIL image (for coordinate mapping)
    """
    # Read image
    if isinstance(image_path, str):
        image = cv2.imread(image_path)
        if original_image is None:
            original_image = Image.open(image_path).convert("RGB")
    else:
        # If it's already a PIL Image
        image = cv2.cvtColor(np.array(image_path), cv2.COLOR_RGB2BGR)
        original_image = image_path
    
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}")
    
    # Get padded image and dimensions using the same function as document processing
    padded_image, dims = prepare_image(original_image)
    
    # Assign colors to all elements at once
    element_colors = assign_colors_to_elements(len(layout_results))
    
    # Create overlay
    overlay = image.copy()
    
    # Draw each layout element
    for idx, (bbox, label) in enumerate(layout_results):
        coords = [float(c) for c in bbox]
        
        # Use the same coordinate processing function as document parsing
        try:
            _, _, _, _, orig_x1, orig_y1, orig_x2, orig_y2, _ = process_coordinates(
                coords, padded_image, dims, previous_box=None
            )
        except Exception as e:
            print(f"Error processing coordinates for element {idx}: {str(e)}")
            continue
        
        # Get color for this element (assigned by order, not by label)
        color = element_colors[idx]
        
        # Draw filled rectangle with transparency
        cv2.rectangle(overlay, (orig_x1, orig_y1), (orig_x2, orig_y2), color, -1)
        
        # Draw border
        cv2.rectangle(image, (orig_x1, orig_y1), (orig_x2, orig_y2), color, 3)
        
        # Add label text with background at the top-left corner (outside the box)
        label_text = f"{idx+1}: {label}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, font, font_scale, thickness
        )
        
        # Position text above the box (outside)
        text_x = orig_x1
        text_y = orig_y1 - 5  # 5 pixels above the box
        
        # If text would go outside the image at the top, put it inside the box instead
        if text_y - text_height < 0:
            text_y = orig_y1 + text_height + 5
        
        # Draw text background
        cv2.rectangle(
            image,
            (text_x - 2, text_y - text_height - 2),
            (text_x + text_width + 2, text_y + baseline + 2),
            (255, 255, 255),
            -1
        )
        
        # Draw text
        cv2.putText(
            image,
            label_text,
            (text_x, text_y),
            font,
            font_scale,
            (0, 0, 0),
            thickness
        )
    
    # Blend the overlay with the original image
    result = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    
    # Save the result
    cv2.imwrite(save_path, result)
    print(f"Layout visualization saved to {save_path}")


def save_layout_json(layout_results, image_path, save_dir, original_image=None):
    """Save layout results to JSON file
    
    Args:
        layout_results: List of (bbox, label) tuples with coordinates in 896x896 space
        image_path: Path to the input image
        save_dir: Directory to save the JSON file
        original_image: Original PIL image (for coordinate mapping)
    """
    # Get original image if not provided
    if original_image is None and isinstance(image_path, str):
        original_image = Image.open(image_path).convert("RGB")
    elif original_image is None:
        original_image = image_path
    
    # Get padded image and dimensions using the same function as document processing
    padded_image, dims = prepare_image(original_image)
    
    # Prepare JSON structure
    layout_data = {
        "image": os.path.basename(image_path) if isinstance(image_path, str) else "pdf_page",
        "image_width": dims.original_w,
        "image_height": dims.original_h,
        "num_elements": len(layout_results),
        "elements": []
    }
    
    for idx, (bbox, label) in enumerate(layout_results):
        coords = [float(c) for c in bbox]
        try:
            _, _, _, _, orig_x1, orig_y1, orig_x2, orig_y2, _ = process_coordinates(
                coords, padded_image, dims, previous_box=None
            )
            element = {
                "label": label,
                "bbox": [int(orig_x1), int(orig_y1), int(orig_x2), int(orig_y2)],
                "reading_order": idx
            }
            layout_data["elements"].append(element)
        except Exception as e:
            print(f"Error processing coordinates for element {idx}: {str(e)}")
            continue
    
    # Save JSON
    base_name = os.path.splitext(os.path.basename(image_path) if isinstance(image_path, str) else "page")[0]
    json_path = os.path.join(save_dir, f"{base_name}_layout.json")
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(layout_data, f, indent=2, ensure_ascii=False)
    
    print(f"Layout JSON saved to {json_path}")
    return json_path


def get_color_palette():
    """Get a visually pleasing color palette for layout visualization
    
    Returns:
        List of BGR color tuples (semi-transparent, good for overlay)
    """
    # Carefully selected color palette with good visual distinction
    # Colors are chosen to be light, pleasant, and distinguishable
    color_palette = [
        (200, 255, 255),  # Light cyan
        (255, 200, 255),  # Light magenta
        (255, 255, 200),  # Light yellow
        (200, 255, 200),  # Light green
        (255, 220, 200),  # Light orange
        (220, 200, 255),  # Light purple
        (200, 240, 255),  # Light sky blue
        (255, 240, 220),  # Light peach
        (220, 255, 240),  # Light mint
        (255, 220, 240),  # Light pink
        (240, 255, 200),  # Light lime
        (240, 220, 255),  # Light lavender
        (200, 255, 240),  # Light turquoise
        (255, 240, 200),  # Light apricot
        (220, 240, 255),  # Light periwinkle
        (255, 200, 220),  # Light rose
        (220, 255, 220),  # Light jade
        (255, 230, 200),  # Light salmon
        (210, 230, 255),  # Light cornflower
        (255, 210, 230),  # Light carnation
    ]
    return color_palette


def assign_colors_to_elements(num_elements):
    """Assign colors to elements in order
    
    Args:
        num_elements: Number of elements to assign colors to
        
    Returns:
        List of color tuples, one for each element
    """
    palette = get_color_palette()
    colors = []
    
    for i in range(num_elements):
        # Cycle through the palette if we have more elements than colors
        color_idx = i % len(palette)
        colors.append(palette[color_idx])
    
    return colors