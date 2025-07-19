"""
Utility functions for semantic analysis and text processing
"""

import re
import logging
from typing import List, Dict, Any, Tuple, Optional
import difflib
from collections import Counter
import unicodedata

logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """
    Normalize text for better comparison
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    if not text:
        return ""
    
    # Convert to Unicode NFC form
    text = unicodedata.normalize('NFC', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Normalize punctuation
    text = re.sub(r'["""]', '"', text)
    text = re.sub(r"[''']", "'", text)
    text = re.sub(r'[–—]', '-', text)
    text = re.sub(r'[…]', '...', text)
    
    return text


def extract_sentences(text: str) -> List[str]:
    """
    Extract sentences from text
    
    Args:
        text: Input text
        
    Returns:
        List of sentences
    """
    if not text:
        return []
    
    # Simple sentence splitting (can be improved with NLTK or spaCy)
    sentences = re.split(r'[.!?]+\s+', text)
    
    # Clean and filter sentences
    cleaned_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 10:  # Filter very short sentences
            cleaned_sentences.append(sentence)
    
    return cleaned_sentences


def calculate_text_overlap(text1: str, text2: str) -> Dict[str, Any]:
    """
    Calculate detailed overlap between two texts
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Dictionary with overlap information
    """
    if not text1 or not text2:
        return {
            'character_overlap': 0.0,
            'word_overlap': 0.0,
            'sentence_overlap': 0.0,
            'longest_match': 0,
            'total_matches': 0,
            'overlap_ratio': 0.0
        }
    
    # Normalize texts
    norm_text1 = normalize_text(text1)
    norm_text2 = normalize_text(text2)
    
    # Character-level overlap
    matcher = difflib.SequenceMatcher(None, norm_text1, norm_text2)
    char_overlap = matcher.ratio()
    
    # Word-level overlap
    words1 = set(norm_text1.split())
    words2 = set(norm_text2.split())
    common_words = words1.intersection(words2)
    word_overlap = len(common_words) / max(len(words1.union(words2)), 1)
    
    # Sentence-level overlap
    sentences1 = extract_sentences(norm_text1)
    sentences2 = extract_sentences(norm_text2)
    
    sentence_matches = 0
    for s1 in sentences1:
        for s2 in sentences2:
            if difflib.SequenceMatcher(None, s1, s2).ratio() > 0.8:
                sentence_matches += 1
                break
    
    sentence_overlap = sentence_matches / max(len(sentences1), len(sentences2), 1)
    
    # Find longest matching subsequence
    matching_blocks = matcher.get_matching_blocks()
    longest_match = max(block.size for block in matching_blocks) if matching_blocks else 0
    
    return {
        'character_overlap': char_overlap,
        'word_overlap': word_overlap,
        'sentence_overlap': sentence_overlap,
        'longest_match': longest_match,
        'total_matches': len(matching_blocks),
        'overlap_ratio': (char_overlap + word_overlap + sentence_overlap) / 3
    }


def identify_text_boundaries(text1: str, text2: str) -> Dict[str, Any]:
    """
    Identify where texts overlap and their boundaries
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Dictionary with boundary information
    """
    matcher = difflib.SequenceMatcher(None, text1, text2)
    matches = matcher.get_matching_blocks()
    
    if not matches:
        return {
            'has_overlap': False,
            'overlap_type': 'none',
            'text1_overlap_start': -1,
            'text1_overlap_end': -1,
            'text2_overlap_start': -1,
            'text2_overlap_end': -1
        }
    
    # Find the longest match
    longest_match = max(matches, key=lambda x: x.size)
    
    if longest_match.size < 20:  # Very short overlap
        return {
            'has_overlap': True,
            'overlap_type': 'minimal',
            'text1_overlap_start': longest_match.a,
            'text1_overlap_end': longest_match.a + longest_match.size,
            'text2_overlap_start': longest_match.b,
            'text2_overlap_end': longest_match.b + longest_match.size
        }
    
    # Determine overlap type based on position
    text1_start_ratio = longest_match.a / max(len(text1), 1)
    text1_end_ratio = (longest_match.a + longest_match.size) / max(len(text1), 1)
    text2_start_ratio = longest_match.b / max(len(text2), 1)
    text2_end_ratio = (longest_match.b + longest_match.size) / max(len(text2), 1)
    
    if text1_end_ratio > 0.8 and text2_start_ratio < 0.2:
        overlap_type = 'text1_end_text2_start'
    elif text1_start_ratio < 0.2 and text2_end_ratio > 0.8:
        overlap_type = 'text1_start_text2_end'
    elif text1_start_ratio < 0.3 and text2_start_ratio < 0.3:
        overlap_type = 'both_start'
    elif text1_end_ratio > 0.7 and text2_end_ratio > 0.7:
        overlap_type = 'both_end'
    else:
        overlap_type = 'middle'
    
    return {
        'has_overlap': True,
        'overlap_type': overlap_type,
        'text1_overlap_start': longest_match.a,
        'text1_overlap_end': longest_match.a + longest_match.size,
        'text2_overlap_start': longest_match.b,
        'text2_overlap_end': longest_match.b + longest_match.size,
        'overlap_length': longest_match.size
    }


def merge_overlapping_texts(text1: str, text2: str, overlap_info: Dict[str, Any]) -> str:
    """
    Intelligently merge two overlapping texts
    
    Args:
        text1: First text
        text2: Second text
        overlap_info: Overlap information from identify_text_boundaries
        
    Returns:
        Merged text
    """
    if not overlap_info.get('has_overlap'):
        return f"{text1}\n\n{text2}"
    
    overlap_type = overlap_info.get('overlap_type', 'none')
    
    if overlap_type == 'text1_end_text2_start':
        # text1 ends where text2 starts
        overlap_end = overlap_info['text2_overlap_end']
        remaining_text2 = text2[overlap_end:].strip()
        
        if remaining_text2:
            return f"{text1} {remaining_text2}"
        else:
            return text1
    
    elif overlap_type == 'text1_start_text2_end':
        # text2 ends where text1 starts
        overlap_end = overlap_info['text1_overlap_end']
        remaining_text1 = text1[overlap_end:].strip()
        
        if remaining_text1:
            return f"{text2} {remaining_text1}"
        else:
            return text2
    
    elif overlap_type == 'both_start':
        # Both texts start similarly, use the longer one
        return text1 if len(text1) > len(text2) else text2
    
    elif overlap_type == 'both_end':
        # Both texts end similarly, use the longer one
        return text1 if len(text1) > len(text2) else text2
    
    else:
        # For middle overlaps or other cases, use the longer text
        return text1 if len(text1) > len(text2) else text2


def extract_key_phrases(text: str, max_phrases: int = 10) -> List[str]:
    """
    Extract key phrases from text
    
    Args:
        text: Input text
        max_phrases: Maximum number of phrases to return
        
    Returns:
        List of key phrases
    """
    if not text:
        return []
    
    # Normalize text
    normalized = normalize_text(text)
    
    # Extract noun phrases (simple pattern matching)
    noun_phrases = re.findall(r'\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text)
    
    # Extract multi-word terms
    words = normalized.split()
    phrases = []
    
    # 2-word phrases
    for i in range(len(words) - 1):
        phrase = f"{words[i]} {words[i+1]}"
        if len(phrase) > 6:  # Filter short phrases
            phrases.append(phrase)
    
    # 3-word phrases
    for i in range(len(words) - 2):
        phrase = f"{words[i]} {words[i+1]} {words[i+2]}"
        if len(phrase) > 10:  # Filter short phrases
            phrases.append(phrase)
    
    # Count frequency and return most common
    phrase_counts = Counter(phrases + [p.lower() for p in noun_phrases])
    
    return [phrase for phrase, count in phrase_counts.most_common(max_phrases)]


def calculate_semantic_similarity_simple(text1: str, text2: str) -> float:
    """
    Calculate semantic similarity using simple methods (no ML)
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score between 0 and 1
    """
    if not text1 or not text2:
        return 0.0
    
    # Get overlap information
    overlap_info = calculate_text_overlap(text1, text2)
    
    # Extract key phrases
    phrases1 = set(extract_key_phrases(text1))
    phrases2 = set(extract_key_phrases(text2))
    
    # Calculate phrase overlap
    common_phrases = phrases1.intersection(phrases2)
    phrase_similarity = len(common_phrases) / max(len(phrases1.union(phrases2)), 1)
    
    # Combine different similarity measures
    similarity_score = (
        overlap_info['character_overlap'] * 0.3 +
        overlap_info['word_overlap'] * 0.4 +
        overlap_info['sentence_overlap'] * 0.2 +
        phrase_similarity * 0.1
    )
    
    return min(similarity_score, 1.0)


def detect_paragraph_type(text: str) -> str:
    """
    Detect the type of paragraph based on content patterns
    
    Args:
        text: Paragraph text
        
    Returns:
        Paragraph type
    """
    if not text:
        return "empty"
    
    text = text.strip()
    
    # Check for titles/headers
    if len(text) < 100 and text.isupper():
        return "title"
    
    if len(text) < 150 and not text.endswith('.'):
        return "heading"
    
    # Check for lists
    if re.match(r'^\s*[\d\w]\.\s', text) or re.match(r'^\s*[•\-\*]\s', text):
        return "list_item"
    
    # Check for tables (simple pattern)
    if '\t' in text or re.search(r'\s{3,}', text):
        return "table_like"
    
    # Check for formulas
    if re.search(r'[=+\-*/\^()]', text) and len(re.findall(r'[a-zA-Z]', text)) < len(text) * 0.3:
        return "formula"
    
    # Check for captions
    if text.lower().startswith(('figure', 'table', 'chart', 'image')):
        return "caption"
    
    # Default to regular paragraph
    return "paragraph"


def clean_extracted_text(text: str) -> str:
    """
    Clean extracted text from OCR/document parsing
    
    Args:
        text: Raw extracted text
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Fix common OCR errors
    text = re.sub(r'\b([a-z])([A-Z])', r'\1 \2', text)  # Split merged words
    text = re.sub(r'([a-z])(\d)', r'\1 \2', text)  # Space between letter and number
    text = re.sub(r'(\d)([a-z])', r'\1 \2', text)  # Space between number and letter
    
    # Fix punctuation spacing
    text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)  # Space after sentence end
    text = re.sub(r'([a-z])([.!?])', r'\1\2', text)  # Remove space before punctuation
    
    # Remove isolated single characters (common OCR artifacts)
    text = re.sub(r'\b[a-zA-Z]\b', '', text)
    
    # Clean up remaining whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    return text


def segment_text_by_topic(text: str) -> List[Dict[str, Any]]:
    """
    Segment text into topic-based sections
    
    Args:
        text: Input text
        
    Returns:
        List of text segments with metadata
    """
    if not text:
        return []
    
    # Split into paragraphs
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    segments = []
    current_segment = []
    current_topic_words = set()
    
    for i, paragraph in enumerate(paragraphs):
        para_words = set(normalize_text(paragraph).split())
        
        # Calculate topic continuity
        if current_topic_words and para_words:
            continuity = len(current_topic_words.intersection(para_words)) / len(para_words)
        else:
            continuity = 0.0
        
        # Start new segment if topic changes significantly
        if continuity < 0.2 and current_segment:
            segments.append({
                'text': '\n\n'.join(current_segment),
                'paragraph_count': len(current_segment),
                'topic_words': list(current_topic_words)[:10],
                'start_paragraph': i - len(current_segment),
                'end_paragraph': i - 1
            })
            current_segment = []
            current_topic_words = set()
        
        current_segment.append(paragraph)
        current_topic_words.update(para_words)
    
    # Add final segment
    if current_segment:
        segments.append({
            'text': '\n\n'.join(current_segment),
            'paragraph_count': len(current_segment),
            'topic_words': list(current_topic_words)[:10],
            'start_paragraph': len(paragraphs) - len(current_segment),
            'end_paragraph': len(paragraphs) - 1
        })
    
    return segments 