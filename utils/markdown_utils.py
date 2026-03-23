""" 
Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
SPDX-License-Identifier: MIT
"""

import base64
import re
from typing import Any, Dict, List, Optional


def extract_table_from_html(html_string):
    """Extract and clean table tags from HTML string"""
    try:
        table_pattern = re.compile(r'<table.*?>.*?</table>', re.DOTALL)
        tables = table_pattern.findall(html_string)
        tables = [re.sub(r'<table[^>]*>', '<table>', table) for table in tables]
        return '\n'.join(tables)
    except Exception as e:
        print(f"extract_table_from_html error: {str(e)}")
        return f"<table><tr><td>Error extracting table: {str(e)}</td></tr></table>"


def remove_numeric_quad_ending(s: str) -> str:
    pattern = r"\\quad\([^)]*\)"
    result = re.sub(pattern, "", s)
    return result


def aligned_to_array(latex: str) -> str:
    pattern = re.compile(r"\\begin\{aligned\}(.*?)\\end\{aligned\}", flags=re.DOTALL)

    def repl(match):
        content = match.group(1).strip()
        new_content = content.replace("&", "")
        return f"\\begin{{array}}{{l}}\n{new_content}\n\\end{{array}}"

    converted = pattern.sub(repl, latex)
    return converted


def gathered_to_aligned(latex: str) -> str:
    latex = latex.replace(r"\begin{gathered}", r"\begin{aligned}")
    latex = latex.replace(r"\end{gathered}", r"\end{aligned}")
    return latex


def replace_repeated_cdots(latex):
    latex = re.sub(r"(\\cdots\s*){3,}", r"\\cdots$$", latex)
    return latex


def truncate_repeated_tail(s, threshold=20, keep=1):
    if not s:
        return s

    max_pattern_len = min(100, len(s) // threshold)

    for pattern_len in range(1, max_pattern_len + 1):
        if len(s) < pattern_len:
            break

        pattern = s[-pattern_len:]

        count = 0
        pos = len(s)

        while pos >= pattern_len:
            if s[pos - pattern_len : pos] == pattern:
                count += 1
                pos -= pattern_len
            else:
                break

        if count > threshold:
            non_repeat_part = s[:pos]
            kept_repeats = pattern * keep
            return non_repeat_part + kept_repeats

    return s


class MarkdownConverter:
    """Convert structured recognition results to Markdown format"""
    
    def __init__(self, post_process: Optional[bool] = False):
        # Define heading levels for different section types
        self.post_process = post_process
        self.heading_levels = {
            'sec_0': '#',
            'sec_1': '##',
            'sec_2': '###',
            'sec_3': '###',
            'sec_4': '###',
            'sec_5': '###',
        }
        
        # Define which labels need special handling
        self.special_labels = {
            'sec_0', 'sec_1', 'sec_2', 'sec_3', 'sec_4', 'sec_5',
            'list', 'equ', 'tab', 'fig'
        }

        # Define replacements for special formulas
        self.replace_dict = {
            '\\bm': '\mathbf ',
            '\eqno': '\quad ',
            '\quad': '\quad ',
            '\leq': '\leq ',
            '\pm': '\pm ',
            '\\varmathbb': '\mathbb ',
            '\in fty': '\infty',
            '\mu': '\mu ',
            '\cdot': '\cdot ',
            '\langle': '\langle ',
            '\pm': '\pm '
        }
    
    def try_remove_newline(self, text: str) -> str:
        try:
            # Preprocess text to handle line breaks
            text = text.strip()
            text = text.replace('-\n', '')
            
            # Handle Chinese text line breaks
            def is_chinese(char):
                return '\u4e00' <= char <= '\u9fff'

            lines = text.split('\n')
            processed_lines = []
            
            # Process all lines except the last one
            for i in range(len(lines)-1):
                current_line = lines[i].strip()
                next_line = lines[i+1].strip()
                
                # Always add the current line, but determine if we need a newline
                if current_line:  # If current line is not empty
                    if next_line:  # If next line is not empty
                        # For Chinese text handling
                        if is_chinese(current_line[-1]) and is_chinese(next_line[0]):
                            processed_lines.append(current_line)
                        else:
                            processed_lines.append(current_line + ' ')
                    else:
                        # Next line is empty, add current line with newline
                        processed_lines.append(current_line + '\n')
                else:
                    # Current line is empty, add an empty line
                    processed_lines.append('\n')
            
            # Add the last line
            if lines and lines[-1].strip():
                processed_lines.append(lines[-1].strip())
            
            text = ''.join(processed_lines)
            return text
        
        except Exception as e:
            print(f"try_remove_newline error: {str(e)}")
            return text  # Return original text on error

    def _handle_text(self, text: str) -> str:
        """
        Process regular text content, preserving paragraph structure
        """
        try:
            if not text:
                return ""
            
            # Process formulas in text before handling other text processing
            text = self._process_formulas_in_text(text)
            if self.post_process:
                text = replace_repeated_cdots(text)
            text = self.try_remove_newline(text)
            return text
        except Exception as e:
            print(f"_handle_text error: {str(e)}")
            return text  # Return original text on error
    
    def _process_formulas_in_text(self, text: str) -> str:
        """
        Process mathematical formulas in text by iteratively finding and replacing formulas.
        - Identify inline and block formulas
        - Replace newlines within formulas with \\
        """
        try:
            text = text.replace(r'\upmu', r'\mu')
            for key, value in self.replace_dict.items():
                text = text.replace(key, value)
            return text
        
        except Exception as e:
            print(f"_process_formulas_in_text error: {str(e)}")
            return text  # Return original text on error
    
    def _remove_newline_in_heading(self, text: str) -> str:
        """
        Remove newline in heading
        """
        try:
            # Handle Chinese text line breaks
            def is_chinese(char):
                return '\u4e00' <= char <= '\u9fff'
            
            # Check if the text contains Chinese characters
            if any(is_chinese(char) for char in text):
                return text.replace('\n', '')
            else:
                return text.replace('\n', ' ')

        except Exception as e:
            print(f"_remove_newline_in_heading error: {str(e)}")
            return text
    
    def _handle_heading(self, text: str, label: str) -> str:
        """
        Convert section headings to appropriate markdown format
        """
        try:
            level = self.heading_levels.get(label, '#')
            text = text.strip()
            text = self._remove_newline_in_heading(text)
            text = self._handle_text(text)
            return f"{level} {text}\n\n"
        
        except Exception as e:
            print(f"_handle_heading error: {str(e)}")
            return f"# Error processing heading: {text}\n\n"
    
    def _handle_list_item(self, text: str) -> str:
        """
        Convert list items to markdown list format
        """
        try:
            return f"- {text.strip()}\n"
        except Exception as e:
            print(f"_handle_list_item error: {str(e)}")
            return f"- Error processing list item: {text}\n"
    
    def _handle_figure(self, text: str, section_count: int) -> str:
        """
        Handle figure content
        """
        try:
            # Check if it's a file path starting with "figures/"
            if text.startswith("figures/"):
                # Convert to relative path from markdown directory to figures directory
                relative_path = f"../{text}"
                return f"![Figure {section_count}]({relative_path})\n\n"

            # Check if it's already a markdown format image link
            if text.startswith("!["):
                # Already in markdown format, return directly
                return f"{text}\n\n"

            # If it's still base64 format, maintain original logic
            if text.startswith("data:image/"):
                return f"![Figure {section_count}]({text})\n\n"
            elif ";" in text and "," in text:
                return f"![Figure {section_count}]({text})\n\n"
            else:
                # Assume it's raw base64, convert to data URI
                img_format = "png"
                data_uri = f"data:image/{img_format};base64,{text}"
                return f"![Figure {section_count}]({data_uri})\n\n"
                
        except Exception as e:
            print(f"_handle_figure error: {str(e)}")
            return f"*[Error processing figure: {str(e)}]*\n\n"

    def _handle_table(self, text: str) -> str:
        """
        Convert table content to markdown format
        """
        try:
            if self.post_process:
                markdown_table = text
            else:
                markdown_table = extract_table_from_html(text)
            return markdown_table + '\n\n'
        
        except Exception as e:
            print(f"_handle_table error: {str(e)}")
            return f"*[Error processing table: {str(e)}]*\n\n"

    def _handle_formula(self, text: str) -> str:
        """
        Handle formula-specific content
        """
        try:
            if self.post_process:
                text = remove_numeric_quad_ending(text)
                text = gathered_to_aligned(text)
                text = aligned_to_array(text)
                text = replace_repeated_cdots(text)
                return f"{text}\n\n"

            text = text.strip('$').rstrip("\ ").replace(r'\upmu', r'\mu')
            for key, value in self.replace_dict.items():
                text = text.replace(key, value)
            processed_text = '$$' + text + '$$'
            return f"{processed_text}\n\n"
        
        except Exception as e:
            print(f"_handle_formula error: {str(e)}")
            return f"*[Error processing formula: {str(e)}]*\n\n"

    def convert(self, recognition_results: List[Dict[str, Any]]) -> str:
        """
        Convert recognition results to markdown format
        """
        try:
            markdown_content = []
            
            for section_count, result in enumerate(recognition_results):
                try:
                    label = result.get('label', '')
                    text = result.get('text', '').strip()
                    
                    if self.post_process:
                        text = truncate_repeated_tail(text, threshold=20, keep=1)
                        
                    # Skip empty text
                    if not text:
                        continue
                        
                    # Handle different content types
                    if label in {'sec_0', 'sec_1', 'sec_2', 'sec_3', 'sec_4', 'sec_5', 'sec'}:
                        markdown_content.append(self._handle_heading(text, label))
                    elif label == 'fig':
                        markdown_content.append(self._handle_figure(text, section_count))
                    elif label == 'tab':
                        markdown_content.append(self._handle_table(text))
                    elif label == 'equ':
                        markdown_content.append(self._handle_formula(text))
                    elif label == 'list':
                        markdown_content.append(self._handle_list_item(text))
                    elif label == 'code':
                        markdown_content.append(f"```bash\n{text}\n```\n\n")
                    elif label == 'distorted_page':
                        markdown_content.append(f"{text}\n\n")
                    else:
                        # Handle regular text (paragraphs, etc.)
                        processed_text = self._handle_text(text)
                        markdown_content.append(f"{processed_text}\n\n")

                except Exception as e:
                    print(f"Error processing item {section_count}: {str(e)}")
                    # Add a placeholder for the failed item
                    markdown_content.append(f"*[Error processing content]*\n\n")
            
            # Join all content and apply post-processing
            result = ''.join(markdown_content)
            return result
        
        except Exception as e:
            print(f"convert error: {str(e)}")
            return f"Error generating markdown content: {str(e)}"
