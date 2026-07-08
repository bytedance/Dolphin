"""
Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
SPDX-License-Identifier: MIT
"""

import io
import json
import os
import re

import cv2
import numpy as np
import pymupdf
from PIL import Image
from qwen_vl_utils import smart_resize

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


def save_combined_pdf_results(all_page_results, pdf_path, save_dir, post_process=False):
    """Save combined results for multi-page PDF with both JSON and Markdown

    Args:
        all_page_results: List of results for all pages
        pdf_path: Path to original PDF file
        save_dir: Directory to save results
        post_process: Whether to apply post-processing to the markdown

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
        markdown_converter = MarkdownConverter(post_process)

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


def extract_labels_from_string(text):
    """
    from [202,217,921,325][para][author] extract para and author
    """
    all_matches = re.findall(r'\[([^\]]+)\]', text)
    
    labels = []
    for match in all_matches:
        if not re.match(r'^\d+,\d+,\d+,\d+$', match):
            labels.append(match)
    
    return labels


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
        coord_match = re.search(coord_pattern, segment)
        label_matches = extract_labels_from_string(segment)
        
        if coord_match and label_matches:
            coords = [float(coord_match.group(i)) for i in range(1, 5)]
            label = label_matches[0].strip()
            parsed_results.append((coords, label, label_matches[1:])) # label_matches[1:] 是 tags
    
    return parsed_results


def process_coordinates(coords, pil_image):
    original_w, original_h = pil_image.size[:2]
    # use the same resize logic as the model
    resized_pil = resize_img(pil_image)
    resized_image = np.array(resized_pil)
    resized_h, resized_w = resized_image.shape[:2]
    resized_h, resized_w = smart_resize(resized_h, resized_w, factor=28, min_pixels=784, max_pixels=2560000)

    w_ratio, h_ratio = original_w / resized_w, original_h / resized_h
    x1 = int(coords[0] * w_ratio)
    y1 = int(coords[1] * h_ratio)
    x2 = int(coords[2] * w_ratio)
    y2 = int(coords[3] * h_ratio)

    x1 = max(0, min(x1, original_w - 1))
    y1 = max(0, min(y1, original_h - 1))
    x2 = max(x1 + 1, min(x2, original_w))
    y2 = max(y1 + 1, min(y2, original_h))
    return x1, y1, x2, y2


def setup_output_dirs(save_dir):
    """Create necessary output directories"""
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "markdown"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "output_json"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "markdown", "figures"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "layout_visualization"), exist_ok=True)


def save_outputs(recognition_results, image, image_name, save_dir, post_process=False):
    """Save JSON and markdown outputs"""

    # Save JSON file
    json_path = os.path.join(save_dir, "output_json", f"{image_name}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(recognition_results, f, ensure_ascii=False, indent=2)

    # Generate and save markdown file
    markdown_converter = MarkdownConverter(post_process)
    markdown_content = markdown_converter.convert(recognition_results)
    markdown_path = os.path.join(save_dir, "markdown", f"{image_name}.md")
    with open(markdown_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    # visualize layout
    # Save visualization (pass original PIL image for coordinate mapping)
    vis_path = os.path.join(save_dir, "layout_visualization", f"{image_name}_layout.png")

    visualize_layout(image, recognition_results, vis_path)
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

def visualize_layout(image_path, layout_results, save_path, alpha=0.3):
    """Visualize layout detection results on the image
    
    Args:
        image_path: Path to the input image
        layout_results: List of (bbox, label, tags) dict
        save_path: Path to save the visualization
        alpha: Transparency of the overlay (0-1, lower = more transparent)
    """
    # Read image
    if isinstance(image_path, str):
        image = cv2.imread(image_path)
    else:
        # If it's already a PIL Image
        image = cv2.cvtColor(np.array(image_path), cv2.COLOR_RGB2BGR)
    
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}")
    
    # Assign colors to all elements at once
    element_colors = assign_colors_to_elements(len(layout_results))
    
    # Create overlay
    overlay = image.copy()
    
    # Draw each layout element
    for idx, layout_res in enumerate(layout_results):
        if "bbox" not in layout_res:
            return
        bbox, label, reading_order, tags = layout_res["bbox"], layout_res["label"], layout_res["reading_order"], layout_res["tags"]
       
        x1,y1,x2,y2 = bbox
        
        # Get color for this element (assigned by order, not by label)
        color = element_colors[idx]
        
        # Draw filled rectangle with transparency
        cv2.rectangle(overlay, (x1,y1), (x2,y2), color, -1)
        
        # Draw border
        cv2.rectangle(image, (x1,y1), (x2,y2), color, 3)
        
        # Add label text with background at the top-left corner (outside the box)
        label_text = f"{reading_order}: {label} | {tags}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, font, font_scale, thickness
        )
        
        # Position text above the box (outside)
        text_x = x1
        text_y = y1 - 5  # 5 pixels above the box
        
        # If text would go outside the image at the top, put it inside the box instead
        if text_y - text_height < 0:
            text_y = y1 + text_height + 5
        
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
    # print(f"Layout visualization saved to {save_path}")


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

def resize_img(image, max_size=1600, min_size=28):
    width, height = image.size
    if max(width, height) < max_size and min(width, height) >= 28:
        return image
    
    if max(width, height) > max_size:
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        image = image.resize((new_width, new_height))
        width, height = image.size
    
    if min(width, height) < 28:
        if width < height:
            new_width = min_size
            new_height = int(height * (min_size / width))
        else:
            new_height = min_size
            new_width = int(width * (min_size / height))
        image = image.resize((new_width, new_height))

    return image


def calculate_iou_matrix(boxes):
    """Vectorized IoU matrix calculation [N, N]
    
    Args:
        boxes: List of bounding boxes in [x1, y1, x2, y2] format
        
    Returns:
        numpy.ndarray: IoU matrix of shape [N, N]
    """
    boxes = np.array(boxes)  # [N, 4]
    
    # Calculate areas
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])  # [N]
    
    # Broadcast to calculate intersection
    lt = np.maximum(boxes[:, None, :2], boxes[None, :, :2])  # [N, N, 2]
    rb = np.minimum(boxes[:, None, 2:], boxes[None, :, 2:])  # [N, N, 2]
    
    wh = np.clip(rb - lt, 0, None)  # [N, N, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, N]
    
    # Calculate IoU
    union = areas[:, None] + areas[None, :] - inter
    iou = inter / np.clip(union, 1e-6, None)
    
    return iou


def check_bbox_overlap(layout_results_list, image, iou_threshold=0.1, overlap_box_ratio=0.25):
    """Check if bounding boxes have significant overlaps, indicating a distorted/photographed document
    
    If more than 60% of boxes have overlaps (IoU > threshold with at least 1 other box),
    treat as photographed document.
    
    Args:
        layout_results_list: List of (bbox, label, tags) tuples
        image: PIL Image object
        iou_threshold: IoU threshold to consider two boxes as overlapping (default: 0.3)
        overlap_box_ratio: Ratio threshold of boxes with overlaps (default: 0.6, i.e., 60%)
    
    Returns:
        bool: True if significant overlap detected (should treat as distorted_page)
    """
    if len(layout_results_list) <= 1:
        return False
    
    # Convert to absolute coordinates
    bboxes = []
    for bbox, label, tags in layout_results_list:
        x1, y1, x2, y2 = process_coordinates(bbox, image)
        bboxes.append([x1, y1, x2, y2])
    
    # Vectorized IoU matrix calculation
    iou_matrix = calculate_iou_matrix(bboxes)
    
    # Check if each box has overlap with any other box (excluding itself)
    overlap_mask = iou_matrix > iou_threshold
    np.fill_diagonal(overlap_mask, False)  # Exclude self
    has_overlap = overlap_mask.any(axis=1)  # Whether each box has overlap
    
    # Count boxes with overlaps
    overlap_count = has_overlap.sum()
    total_boxes = len(bboxes)
    overlap_ratio = overlap_count / total_boxes
    
    # print(f"Overlap detection: {overlap_count}/{total_boxes} boxes have overlaps (ratio: {overlap_ratio:.2%})")
    
    # If more than 60% boxes have overlaps, treat as photographed document
    if overlap_ratio > overlap_box_ratio:
        print(f"⚠️ High overlap detected ({overlap_ratio:.2%} > {overlap_box_ratio:.2%}), treating as distorted/photographed document")
        return True
    
    return False

if __name__ == "__main__":
    bbox_str = "[210,136,910,172][sec_0][PAIR_SEP][202,217,921,325][para][author][PAIR_SEP][520,341,604,367][para][PAIR_SEP][290,404,384,432][sec_1][paper_abstract][PAIR_SEP][156,448,520,723][para][paper_abstract][PAIR_SEP][125,740,290,768][sec_1][PAIR_SEP][125,781,552,1143][para][PAIR_SEP][125,1144,552,1400][para][RELATION_SEP][573,406,1000,561][para][PAIR_SEP][573,581,1001,943][para][PAIR_SEP][573,962,1001,1222][para][PAIR_SEP][573,1241,1001,1475][para][PAIR_SEP][126,1410,551,1470][fnote][PAIR_SEP][21,499,63,1163][watermark][meta_num]"
    print(parse_layout_string(bbox_str))