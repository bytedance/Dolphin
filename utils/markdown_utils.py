""" 
Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
SPDX-License-Identifier: MIT
"""

import re
import base64
from typing import List, Dict, Any, Optional


"""
    Example input:
    [
        {"label": "tab", "bbox": [0.176, 0.74, 0.824, 0.82], "text": "<table><tr><td></td><td>HellaSwag</td><td>Obqa</td><td>WinoGrande</td><td>ARC-c</td><td>ARC-e</td><td>boolq</td><td>piqa</td><td>Avg</td></tr><tr><td>OPT-1.3B</td><td>53.65</td><td>33.40</td><td>59.59</td><td>29.44</td><td>50.80</td><td>60.83</td><td>72.36</td><td>51.44</td></tr><tr><td>Pythia-1.0B</td><td>47.16</td><td>31.40</td><td>53.43</td><td>27.05</td><td>48.99</td><td>57.83</td><td>69.21</td><td>48.30</td></tr><tr><td>Pythia-1.4B</td><td>52.01</td><td>33.20</td><td>57.38</td><td>28.50</td><td>54.00</td><td>63.27</td><td>70.95</td><td>51.33</td></tr><tr><td>TinyLlama-1.1B</td><td>59.20</td><td>36.00</td><td>59.12</td><td>30.10</td><td>55.25</td><td>57.83</td><td>73.29</td><td>52.99</td></tr></table>", "reading_order": 6},
        {"label": "cap", "bbox": [0.28, 0.729, 0.711, 0.74], "text": "Table 2: Zero-shot performance on commonsense reasoning tasks", "reading_order": 7}, 
        {"label": "para", "bbox": [0.176, 0.848, 0.826, 0.873], "text": "We of performance during training We tracked the accuracy of TinyLlama on common-\nsense reasoning benchmarks during its pre-training, as shown in Fig. 2 . Generally, the performance of", "reading_order": 8}, 
        {"label": "fnote", "bbox": [0.176, 0.88, 0.824, 0.912], "text": "${ }^{4}$ Due to a bug in the config file, the learning rate did not decrease immediately after warmup and remained at\nthe maximum value for several steps before we fixed this.", "reading_order": 9}, 
        {"label": "foot", "bbox": [0.496, 0.939, 0.501, 0.95], "text": "14", "reading_order": 10}
    ]
"""


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


class MarkdownConverter:
    """Convert structured recognition results to Markdown format"""
    
    def __init__(self):
        # Define heading levels for different section types
        self.heading_levels = {
            'title': '#',
            'sec': '##',
            'sub_sec': '###'
        }
        
        # Define which labels need special handling
        self.special_labels = {
            'tab', 'fig', 'title', 'sec', 'sub_sec', 
            'list', 'formula', 'reference', 'alg'
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
            
            if text.strip().startswith("\\begin{array}") and text.strip().endswith("\\end{array}"):
                text = "$$" + text + "$$"
            elif ("_{" in text or "^{" in text or "\\" in text or "_ {" in text or "^ {" in text) and ("$" not in text) and ("\\begin" not in text):
                text = "$" + text + "$"

            # Process formulas in text before handling other text processing
            text = self._process_formulas_in_text(text)
            
            text = self.try_remove_newline(text)
            
            # Return processed text
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
            # Define formula delimiters and their corresponding patterns
            delimiters = [
                ('$$', '$$'),  # Block formula with $$
                ('\\[', '\\]'),  # Block formula with \[ \]
                ('$', '$'),  # Inline formula with $
                ('\\(', '\\)')  # Inline formula with \( \)
            ]
            
            # Process the text by iterating through each delimiter type
            result = text
            
            for start_delim, end_delim in delimiters:
                # Create a pattern that matches from start to end delimiter
                # Using a custom approach to avoid issues with nested delimiters
                current_pos = 0
                processed_parts = []
                
                while current_pos < len(result):
                    # Find the next start delimiter
                    start_pos = result.find(start_delim, current_pos)
                    if start_pos == -1:
                        # No more formulas of this type
                        processed_parts.append(result[current_pos:])
                        break
                    
                    # Add text before the formula
                    processed_parts.append(result[current_pos:start_pos])
                    
                    # Find the matching end delimiter
                    end_pos = result.find(end_delim, start_pos + len(start_delim))
                    if end_pos == -1:
                        # No matching end delimiter, treat as regular text
                        processed_parts.append(result[start_pos:])
                        break
                    
                    # Extract the formula content (without delimiters)
                    formula_content = result[start_pos + len(start_delim):end_pos]
                    
                    # Process the formula content - replace newlines with \\
                    processed_formula = formula_content.replace('\n', ' \\\\ ')
                    
                    # Add the processed formula with its delimiters
                    processed_parts.append(f"{start_delim}{processed_formula}{end_delim}")
                    
                    # Move past this formula
                    current_pos = end_pos + len(end_delim)
                
                # Update the result with processed text
                result = ''.join(processed_parts)
            return result
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
            markdown_content = []
            if '<table' in text.lower() or '<tr' in text.lower():
                markdown_table = extract_table_from_html(text)
                markdown_content.append(markdown_table + "\n")
            else:
                table_lines = text.split('\n')
                if table_lines:
                    col_count = len(table_lines[0].split()) if table_lines[0] else 1
                    header = '| ' + ' | '.join(table_lines[0].split()) + ' |'
                    markdown_content.append(header)
                    markdown_content.append('| ' + ' | '.join(['---'] * col_count) + ' |')
                    for line in table_lines[1:]:
                        cells = line.split()
                        while len(cells) < col_count:
                            cells.append('')
                        markdown_content.append('| ' + ' | '.join(cells) + ' |')
            return '\n'.join(markdown_content) + '\n\n'
        except Exception as e:
            print(f"_handle_table error: {str(e)}")
            return f"*[Error processing table: {str(e)}]*\n\n"

    def _handle_algorithm(self, text: str) -> str:
        """
        Process algorithm blocks with proper formatting
        """
        try:
            # Remove algorithm environment tags if present
            text = re.sub(r'\\begin\{algorithm\}(.*?)\\end\{algorithm\}', r'\1', text, flags=re.DOTALL)
            text = text.replace('\\begin{algorithm}', '').replace('\\end{algorithm}', '')
            text = text.replace('\\begin{algorithmic}', '').replace('\\end{algorithmic}', '')
            
            # Process the algorithm text
            lines = text.strip().split('\n')
            
            # Check if there's a caption or label
            caption = ""
            algorithm_text = []
            
            for line in lines:
                if '\\caption' in line:
                    # Extract caption text
                    caption_match = re.search(r'\\caption\{(.*?)\}', line)
                    if caption_match:
                        caption = f"**{caption_match.group(1)}**\n\n"
                    continue
                elif '\\label' in line:
                    continue  # Skip label lines
                else:
                    algorithm_text.append(line)
            
            # Join the algorithm text and wrap in code block
            formatted_text = '\n'.join(algorithm_text)
            
            # Return the formatted algorithm with caption
            return f"{caption}```\n{formatted_text}\n```\n\n"
        except Exception as e:
            print(f"_handle_algorithm error: {str(e)}")
            return f"*[Error processing algorithm: {str(e)}]*\n\n{text}\n\n"

    def _handle_formula(self, text: str) -> str:
        """
        Handle formula-specific content
        """
        try:
            # Process the formula content
            processed_text = self._process_formulas_in_text(text)
            
            # For formula blocks, ensure they're properly formatted in markdown
            if '$$' not in processed_text and '\\[' not in processed_text:
                # If no block formula delimiters are present, wrap in $$ for block formula
                processed_text = f'$${processed_text}$$'
            
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
                    
                    # Skip empty text
                    if not text:
                        continue
                        
                    # Handle different content types
                    if label in {'title', 'sec', 'sub_sec'}:
                        markdown_content.append(self._handle_heading(text, label))
                    elif label == 'list':
                        markdown_content.append(self._handle_list_item(text))
                    elif label == 'fig':
                        markdown_content.append(self._handle_figure(text, section_count))
                    elif label == 'tab':
                        markdown_content.append(self._handle_table(text))
                    elif label == 'alg':
                        markdown_content.append(self._handle_algorithm(text))
                    elif label == 'formula':
                        markdown_content.append(self._handle_formula(text))
                    elif label not in self.special_labels:
                        # Handle regular text (paragraphs, etc.)
                        processed_text = self._handle_text(text)
                        markdown_content.append(f"{processed_text}\n\n")
                except Exception as e:
                    print(f"Error processing item {section_count}: {str(e)}")
                    # Add a placeholder for the failed item
                    markdown_content.append(f"*[Error processing content]*\n\n")
            
            # Join all content and apply post-processing
            result = ''.join(markdown_content)
            return self._post_process(result)
        except Exception as e:
            print(f"convert error: {str(e)}")
            return f"Error generating markdown content: {str(e)}"

    def _post_process(self, markdown_content: str) -> str:
        """Apply post-processing fixes to the generated markdown content."""
        try:
            # Replace author commands with plain text
            markdown_content = re.sub(r"\\author\{(.*?)\}",
                                      lambda m: self._handle_text(m.group(1)),
                                      markdown_content,
                                      flags=re.DOTALL)

            # Replace abstract environments
            markdown_content = re.sub(r"\\begin\{abstract\}(.*?)\\end\{abstract\}",
                                      r"**Abstract** \1",
                                      markdown_content,
                                      flags=re.DOTALL)
            markdown_content = re.sub(r"\\begin\{abstract\}", r"**Abstract**", markdown_content)

            # Convert equation numbers to tags
            markdown_content = re.sub(r"\\eqno\{\((.*?)\)\}", r"\\tag{\1}", markdown_content)

            # Normalise display‐math delimiters
            markdown_content = markdown_content.replace(r"\[ \\", r"$$ \\")

            # Find the ending tag of the formula (ensure this is the only ending tag)
            markdown_content = markdown_content.replace(r"\\ \"]", r"\\ $$")

            # Misc clean-ups
            markdown_content = re.sub(r"_ {", r"_{", markdown_content)
            markdown_content = re.sub(r"\^ {", r"^{", markdown_content)
            markdown_content = re.sub(r"\n{3,}", r"\n\n", markdown_content)

            return markdown_content
        except Exception as e:
            print(f"_post_process error: {e}")
            return markdown_content
        try:
            # Handle author information
            author_pattern = re.compile(r'\\author\{(.*?)\}', re.DOTALL)
            
            def process_author_match(match):
                # Extract author content
                author_content = match.group(1)
                # Process the author content
                return self._handle_text(author_content)
            
            # Replace \author{...} with processed content
            markdown_content = author_pattern.sub(process_author_match, markdown_content)
            
            # Handle special case where author is inside math environment
            math_author_pattern = re.compile(r'\$(\\author\{.*?\})\$', re.DOTALL)
            match = math_author_pattern.search(markdown_content)
            if match:
                # Extract the author command
                author_cmd = match.group(1)
                # Extract content from author command
                author_content_match = re.search(r'\\author\{(.*?)\}', author_cmd, re.DOTALL)
                if author_content_match:
                    # Get author content and process it
                    author_content = author_content_match.group(1)
                    processed_content = self._handle_text(author_content)
                    # Replace the entire $\author{...}$ block with processed content
                    markdown_content = markdown_content.replace(match.group(0), processed_content)
            
            # Replace LaTeX abstract environment with plain text
            markdown_content = re.sub(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', 
                                     r'**Abstract** \1', 
                                     markdown_content, 
                                     flags=re.DOTALL)
            
            # Replace standalone \begin{abstract} (without matching end)
            markdown_content = re.sub(r'\\begin\{abstract\}', 
                                     r'**Abstract**', 
                                     markdown_content)
            
            # Replace LaTeX equation numbers with tag format, handling cases with extra backslashes
            markdown_content = re.sub(r'\\eqno\{\((.*?)\)\}',
                                    r'\\tag{\1}',
                                    markdown_content)

            # Find the starting tag of the formula
            markdown_content = markdown_content.replace(r"\\[ \\", r"$$ \\")

            # Find the ending tag of the formula (ensure this is the only ending tag)
            markdown_content = markdown_content.replace(r"\\ \"]", r"\\ $$")

            # Fix other common LaTeX issues
            replacements = [
                # Fix spacing issues in subscripts and superscripts
                (r'_ {', r'_{'),
                (r'^ {', r'^{'),
                
                # Fix potential issues with multiple consecutive newlines
                (r'\n{3,}', r'\n\n')
            ]
            
            for old, new in replacements:
                markdown_content = re.sub(old, new, markdown_content)
            
            return markdown_content
        except Exception as e:
            print(f"_post_process error: {str(e)}")
            return markdown_content  # Return original content if post-processing fails
