"""
Semantic analyzer for cross-page overlap detection and content merging
Uses sentence transformers and similarity analysis to identify overlapping content
"""

import logging
import asyncio
import re
from typing import List, Dict, Any, Tuple, Optional, Set
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import dataclass
import difflib
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class SemanticMatch:
    """Represents a semantic match between two elements"""
    source_element: Dict[str, Any]
    target_element: Dict[str, Any]
    similarity_score: float
    match_type: str  # "exact", "semantic", "partial"
    confidence: float
    overlap_region: Optional[str] = None


@dataclass
class MergedElement:
    """Represents a merged element from multiple sources"""
    element_id: str
    element_type: str
    merged_content: str
    source_elements: List[Dict[str, Any]]
    confidence_score: float
    page_span: Tuple[int, int]  # (start_page, end_page)


class SemanticAnalyzer:
    """
    Analyzes semantic relationships between document elements across pages
    Identifies overlapping content and merges discrete paragraphs
    """
    
    def __init__(self, dolphin_model=None, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the semantic analyzer
        
        Args:
            dolphin_model: Optional Dolphin model instance for advanced analysis
            model_name: Sentence transformer model name
        """
        self.dolphin_model = dolphin_model
        self.similarity_threshold = 0.8
        self.partial_threshold = 0.6
        self.exact_threshold = 0.95
        
        # Initialize sentence transformer
        try:
            self.sentence_model = SentenceTransformer(model_name)
            logger.info(f"Loaded sentence transformer: {model_name}")
        except Exception as e:
            logger.warning(f"Failed to load sentence transformer: {e}")
            self.sentence_model = None
        
        # Content processing
        self.content_cache = {}
        self.embedding_cache = {}
        self.max_cache_size = 1000
        
        # Statistics
        self.total_comparisons = 0
        self.semantic_matches = 0
        self.exact_matches = 0
        self.partial_matches = 0
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for comparison
        
        Args:
            text: Input text
            
        Returns:
            Cleaned and normalized text
        """
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove common document artifacts
        text = re.sub(r'^\d+[\.\)]?\s*', '', text)  # Remove numbering
        text = re.sub(r'[•\-\*]\s*', '', text)     # Remove bullet points
        
        # Normalize quotes and dashes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        text = text.replace('–', '-').replace('—', '-')
        
        return text.strip()
    
    def _get_text_hash(self, text: str) -> str:
        """Generate hash for text content"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()[:16]
    
    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Get sentence embedding for text
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector or None if model unavailable
        """
        if not self.sentence_model or not text:
            return None
        
        # Check cache
        text_hash = self._get_text_hash(text)
        if text_hash in self.embedding_cache:
            return self.embedding_cache[text_hash]
        
        try:
            embedding = self.sentence_model.encode([text])[0]
            
            # Cache with size limit
            if len(self.embedding_cache) < self.max_cache_size:
                self.embedding_cache[text_hash] = embedding
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        if not text1 or not text2:
            return 0.0
        
        # Preprocess texts
        clean_text1 = self._preprocess_text(text1)
        clean_text2 = self._preprocess_text(text2)
        
        # Check for exact match
        if clean_text1 == clean_text2:
            return 1.0
        
        # Use sequence matcher for basic similarity
        seq_similarity = difflib.SequenceMatcher(None, clean_text1, clean_text2).ratio()
        
        # Try semantic similarity if available
        if self.sentence_model:
            emb1 = self._get_embedding(clean_text1)
            emb2 = self._get_embedding(clean_text2)
            
            if emb1 is not None and emb2 is not None:
                semantic_sim = cosine_similarity([emb1], [emb2])[0][0]
                # Combine sequence and semantic similarity
                return max(seq_similarity, semantic_sim)
        
        return seq_similarity
    
    def _extract_text_content(self, element: Dict[str, Any]) -> str:
        """Extract text content from an element"""
        if 'text' in element:
            return str(element['text'])
        elif 'content' in element:
            return str(element['content'])
        elif isinstance(element, str):
            return element
        else:
            return str(element)
    
    def _categorize_match(self, similarity: float) -> Tuple[str, float]:
        """
        Categorize a similarity score into match type and confidence
        
        Args:
            similarity: Similarity score
            
        Returns:
            Tuple of (match_type, confidence)
        """
        if similarity >= self.exact_threshold:
            return "exact", min(similarity * 1.1, 1.0)
        elif similarity >= self.similarity_threshold:
            return "semantic", similarity
        elif similarity >= self.partial_threshold:
            return "partial", similarity * 0.8
        else:
            return "none", similarity
    
    async def detect_paragraph_overlap(
        self,
        window1_results: List[Dict[str, Any]],
        window2_results: List[Dict[str, Any]],
        window1_id: int,
        window2_id: int
    ) -> List[SemanticMatch]:
        """
        Detect overlapping paragraphs between two windows
        
        Args:
            window1_results: Elements from first window
            window2_results: Elements from second window
            window1_id: First window ID
            window2_id: Second window ID
            
        Returns:
            List of semantic matches
        """
        matches = []
        
        # Extract text elements only
        text_elements1 = [elem for elem in window1_results 
                         if elem.get('label') in ['text', 'para']]
        text_elements2 = [elem for elem in window2_results 
                         if elem.get('label') in ['text', 'para']]
        
        logger.info(f"Comparing {len(text_elements1)} elements from window {window1_id} "
                   f"with {len(text_elements2)} elements from window {window2_id}")
        
        # Compare all pairs
        for elem1 in text_elements1:
            text1 = self._extract_text_content(elem1)
            if len(text1.strip()) < 10:  # Skip very short texts
                continue
            
            for elem2 in text_elements2:
                text2 = self._extract_text_content(elem2)
                if len(text2.strip()) < 10:  # Skip very short texts
                    continue
                
                self.total_comparisons += 1
                
                # Calculate similarity
                similarity = self._calculate_text_similarity(text1, text2)
                match_type, confidence = self._categorize_match(similarity)
                
                if match_type != "none":
                    # Update statistics
                    if match_type == "exact":
                        self.exact_matches += 1
                    elif match_type == "semantic":
                        self.semantic_matches += 1
                    elif match_type == "partial":
                        self.partial_matches += 1
                    
                    match = SemanticMatch(
                        source_element=elem1,
                        target_element=elem2,
                        similarity_score=similarity,
                        match_type=match_type,
                        confidence=confidence,
                        overlap_region=self._identify_overlap_region(text1, text2)
                    )
                    matches.append(match)
                    
                    logger.debug(f"Found {match_type} match (score: {similarity:.3f}) "
                              f"between windows {window1_id} and {window2_id}")
        
        return matches
    
    def _identify_overlap_region(self, text1: str, text2: str) -> str:
        """
        Identify the overlapping region between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Description of overlap region
        """
        matcher = difflib.SequenceMatcher(None, text1, text2)
        matches = matcher.get_matching_blocks()
        
        if not matches:
            return "no_overlap"
        
        # Find the longest matching block
        longest_match = max(matches, key=lambda x: x.size)
        
        if longest_match.size < 20:  # Very short overlap
            return "minimal_overlap"
        
        # Determine position of overlap
        overlap_start1 = longest_match.a / len(text1) if text1 else 0
        overlap_start2 = longest_match.b / len(text2) if text2 else 0
        
        if overlap_start1 < 0.1 and overlap_start2 > 0.8:
            return "text1_start_text2_end"
        elif overlap_start1 > 0.8 and overlap_start2 < 0.1:
            return "text1_end_text2_start"
        elif overlap_start1 < 0.2 and overlap_start2 < 0.2:
            return "both_start"
        elif overlap_start1 > 0.7 and overlap_start2 > 0.7:
            return "both_end"
        else:
            return "middle_overlap"
    
    async def merge_overlapping_content(
        self,
        matches: List[SemanticMatch],
        all_elements: List[Dict[str, Any]]
    ) -> List[MergedElement]:
        """
        Merge overlapping content intelligently
        
        Args:
            matches: List of semantic matches
            all_elements: All elements from all windows
            
        Returns:
            List of merged elements
        """
        merged_elements = []
        processed_element_ids = set()
        
        # Group matches by similarity
        high_similarity_matches = [m for m in matches if m.similarity_score >= self.exact_threshold]
        
        for match in high_similarity_matches:
            source_id = id(match.source_element)
            target_id = id(match.target_element)
            
            # Skip if already processed
            if source_id in processed_element_ids or target_id in processed_element_ids:
                continue
            
            # Create merged element
            merged = await self._create_merged_element(match)
            if merged:
                merged_elements.append(merged)
                processed_element_ids.add(source_id)
                processed_element_ids.add(target_id)
        
        return merged_elements
    
    async def _create_merged_element(self, match: SemanticMatch) -> Optional[MergedElement]:
        """Create a merged element from a semantic match"""
        try:
            source_text = self._extract_text_content(match.source_element)
            target_text = self._extract_text_content(match.target_element)
            
            # Choose the best content based on match type and position
            if match.match_type == "exact":
                # Use the longer version
                merged_text = source_text if len(source_text) > len(target_text) else target_text
            else:
                # Intelligent merging based on overlap region
                merged_text = self._merge_texts_intelligently(source_text, target_text, match.overlap_region)
            
            # Get page information
            source_page = match.source_element.get('page_number', 0)
            target_page = match.target_element.get('page_number', 0)
            
            merged_element = MergedElement(
                element_id=f"merged_{source_page}_{target_page}_{hash(merged_text) % 10000}",
                element_type=match.source_element.get('label', 'text'),
                merged_content=merged_text,
                source_elements=[match.source_element, match.target_element],
                confidence_score=match.confidence,
                page_span=(min(source_page, target_page), max(source_page, target_page))
            )
            
            return merged_element
            
        except Exception as e:
            logger.error(f"Error creating merged element: {e}")
            return None
    
    def _merge_texts_intelligently(self, text1: str, text2: str, overlap_region: str) -> str:
        """
        Intelligently merge two overlapping texts
        
        Args:
            text1: First text
            text2: Second text
            overlap_region: Type of overlap
            
        Returns:
            Merged text
        """
        if overlap_region == "text1_end_text2_start":
            # text1 ends where text2 starts - concatenate with overlap removal
            matcher = difflib.SequenceMatcher(None, text1, text2)
            match = matcher.find_longest_match(0, len(text1), 0, len(text2))
            
            if match.size > 10:  # Meaningful overlap
                # Remove overlap from text2 and concatenate
                overlap_end_in_text2 = match.b + match.size
                remaining_text2 = text2[overlap_end_in_text2:].strip()
                if remaining_text2:
                    return f"{text1} {remaining_text2}"
                else:
                    return text1
            else:
                return f"{text1} {text2}"
        
        elif overlap_region == "text1_start_text2_end":
            # text2 ends where text1 starts
            return self._merge_texts_intelligently(text2, text1, "text1_end_text2_start")
        
        else:
            # For other cases, return the longer text
            return text1 if len(text1) > len(text2) else text2
    
    async def compile_discrete_paragraphs(
        self,
        all_window_results: List[List[Dict[str, Any]]],
        merged_elements: List[MergedElement]
    ) -> List[Dict[str, Any]]:
        """
        Compile discrete paragraphs across all windows
        
        Args:
            all_window_results: Results from all processing windows
            merged_elements: Already merged elements
            
        Returns:
            List of unique paragraphs
        """
        discrete_paragraphs = []
        merged_element_sources = set()
        
        # Track elements that were merged
        for merged in merged_elements:
            for source_elem in merged.source_elements:
                merged_element_sources.add(id(source_elem))
        
        # Collect all elements that weren't merged
        all_elements = []
        for window_results in all_window_results:
            for element in window_results:
                if id(element) not in merged_element_sources:
                    all_elements.append(element)
        
        # Remove duplicates based on content
        seen_content = set()
        for element in all_elements:
            text = self._extract_text_content(element)
            text_hash = self._get_text_hash(self._preprocess_text(text))
            
            if text_hash not in seen_content and len(text.strip()) > 5:
                seen_content.add(text_hash)
                discrete_paragraphs.append(element)
        
        logger.info(f"Compiled {len(discrete_paragraphs)} discrete paragraphs")
        return discrete_paragraphs
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get analyzer statistics"""
        total_matches = self.exact_matches + self.semantic_matches + self.partial_matches
        
        return {
            "total_comparisons": self.total_comparisons,
            "total_matches": total_matches,
            "exact_matches": self.exact_matches,
            "semantic_matches": self.semantic_matches,
            "partial_matches": self.partial_matches,
            "match_rate": total_matches / max(self.total_comparisons, 1),
            "exact_match_rate": self.exact_matches / max(self.total_comparisons, 1),
            "cache_size": len(self.embedding_cache),
            "sentence_model_available": self.sentence_model is not None,
            "thresholds": {
                "similarity": self.similarity_threshold,
                "partial": self.partial_threshold,
                "exact": self.exact_threshold
            }
        }
    
    def reset_statistics(self):
        """Reset all statistics"""
        self.total_comparisons = 0
        self.semantic_matches = 0
        self.exact_matches = 0
        self.partial_matches = 0
        logger.info("Semantic analyzer statistics reset")
    
    def clear_cache(self):
        """Clear all caches"""
        self.content_cache.clear()
        self.embedding_cache.clear()
        logger.info("Semantic analyzer cache cleared") 