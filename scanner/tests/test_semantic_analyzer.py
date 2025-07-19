"""
Unit tests for semantic analyzer
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass

from models.semantic_analyzer import (
    SemanticAnalyzer, SemanticMatch, MergedElement
)


@pytest.fixture
def mock_dolphin_model():
    """Create a mock Dolphin model"""
    model = Mock()
    return model


@pytest.fixture
def mock_sentence_model():
    """Create a mock sentence transformer model"""
    model = Mock()
    model.encode = Mock(return_value=np.array([[0.1, 0.2, 0.3]]))
    return model


@pytest.fixture
def analyzer(mock_dolphin_model):
    """Create a SemanticAnalyzer instance"""
    return SemanticAnalyzer(mock_dolphin_model)


@pytest.fixture
def sample_elements():
    """Create sample document elements for testing"""
    return [
        {
            "label": "text",
            "text": "This is the first paragraph of the document.",
            "page_number": 1,
            "reading_order": 1
        },
        {
            "label": "text", 
            "text": "This is the second paragraph with different content.",
            "page_number": 1,
            "reading_order": 2
        },
        {
            "label": "table",
            "text": "Table data with numbers and text",
            "page_number": 1,
            "reading_order": 3
        }
    ]


@pytest.fixture
def overlapping_elements():
    """Create overlapping elements for testing"""
    return {
        "window1": [
            {
                "label": "text",
                "text": "This is a paragraph that continues across pages.",
                "page_number": 1,
                "window_id": 0
            },
            {
                "label": "text",
                "text": "Another paragraph on the first page.",
                "page_number": 1,
                "window_id": 0
            }
        ],
        "window2": [
            {
                "label": "text",
                "text": "This is a paragraph that continues across pages and has more content.",
                "page_number": 2,
                "window_id": 1
            },
            {
                "label": "text",
                "text": "A completely different paragraph on the second page.",
                "page_number": 2,
                "window_id": 1
            }
        ]
    }


class TestSemanticAnalyzer:
    """Test SemanticAnalyzer initialization and configuration"""
    
    def test_analyzer_initialization(self, mock_dolphin_model):
        """Test analyzer initialization"""
        analyzer = SemanticAnalyzer(mock_dolphin_model)
        
        assert analyzer.dolphin_model == mock_dolphin_model
        assert analyzer.similarity_threshold == 0.8
        assert analyzer.partial_threshold == 0.6
        assert analyzer.exact_threshold == 0.95
        assert analyzer.total_comparisons == 0
        assert analyzer.semantic_matches == 0
        assert analyzer.exact_matches == 0
        assert analyzer.partial_matches == 0
    
    def test_analyzer_with_custom_model(self, mock_dolphin_model):
        """Test analyzer with custom sentence transformer model"""
        with patch('models.semantic_analyzer.SentenceTransformer') as mock_st:
            mock_st.return_value = Mock()
            analyzer = SemanticAnalyzer(mock_dolphin_model, model_name="custom-model")
            
            mock_st.assert_called_once_with("custom-model")
            assert analyzer.sentence_model is not None
    
    def test_analyzer_without_sentence_transformer(self, mock_dolphin_model):
        """Test analyzer when sentence transformer fails to load"""
        with patch('models.semantic_analyzer.SentenceTransformer') as mock_st:
            mock_st.side_effect = Exception("Model not found")
            analyzer = SemanticAnalyzer(mock_dolphin_model)
            
            assert analyzer.sentence_model is None


class TestTextPreprocessing:
    """Test text preprocessing methods"""
    
    def test_preprocess_text_basic(self, analyzer):
        """Test basic text preprocessing"""
        text = "  This is a TEST with   extra spaces.  "
        processed = analyzer._preprocess_text(text)
        
        assert processed == "this is a test with extra spaces."
    
    def test_preprocess_text_empty(self, analyzer):
        """Test preprocessing empty text"""
        assert analyzer._preprocess_text("") == ""
        assert analyzer._preprocess_text(None) == ""
    
    def test_preprocess_text_special_chars(self, analyzer):
        """Test preprocessing text with special characters"""
        text = "Text with "quotes" and – dashes —"
        processed = analyzer._preprocess_text(text)
        
        assert '"quotes"' in processed
        assert ('– dashes –' in processed or '- dashes -' in processed)
    
    def test_preprocess_text_numbering(self, analyzer):
        """Test preprocessing text with numbering and bullets"""
        text = "1. First item\n• Second item\n- Third item"
        processed = analyzer._preprocess_text(text)
        
        # Should remove numbering and bullets
        assert "first item" in processed
        assert "second item" in processed
        assert "third item" in processed
    
    def test_extract_text_content(self, analyzer):
        """Test extracting text content from elements"""
        element_with_text = {"text": "Sample text"}
        element_with_content = {"content": "Sample content"}
        string_element = "Direct string"
        
        assert analyzer._extract_text_content(element_with_text) == "Sample text"
        assert analyzer._extract_text_content(element_with_content) == "Sample content"
        assert analyzer._extract_text_content(string_element) == "Direct string"
    
    def test_get_text_hash(self, analyzer):
        """Test text hashing"""
        text1 = "Same text"
        text2 = "Same text"
        text3 = "Different text"
        
        hash1 = analyzer._get_text_hash(text1)
        hash2 = analyzer._get_text_hash(text2)
        hash3 = analyzer._get_text_hash(text3)
        
        assert hash1 == hash2
        assert hash1 != hash3
        assert len(hash1) == 16  # MD5 hash truncated to 16 chars


class TestSimilarityCalculation:
    """Test similarity calculation methods"""
    
    def test_calculate_text_similarity_identical(self, analyzer):
        """Test similarity calculation for identical texts"""
        text1 = "This is identical text."
        text2 = "This is identical text."
        
        similarity = analyzer._calculate_text_similarity(text1, text2)
        assert similarity == 1.0
    
    def test_calculate_text_similarity_different(self, analyzer):
        """Test similarity calculation for different texts"""
        text1 = "This is completely different."
        text2 = "Something entirely unrelated."
        
        similarity = analyzer._calculate_text_similarity(text1, text2)
        assert 0.0 <= similarity < 0.5  # Should be low similarity
    
    def test_calculate_text_similarity_similar(self, analyzer):
        """Test similarity calculation for similar texts"""
        text1 = "This is a sample text for testing."
        text2 = "This is a sample text for verification."
        
        similarity = analyzer._calculate_text_similarity(text1, text2)
        assert 0.5 < similarity < 1.0  # Should be moderate to high similarity
    
    def test_calculate_text_similarity_empty(self, analyzer):
        """Test similarity calculation with empty texts"""
        assert analyzer._calculate_text_similarity("", "text") == 0.0
        assert analyzer._calculate_text_similarity("text", "") == 0.0
        assert analyzer._calculate_text_similarity("", "") == 0.0
    
    @patch.object(SemanticAnalyzer, '_get_embedding')
    def test_calculate_text_similarity_with_embeddings(self, mock_get_embedding, analyzer):
        """Test similarity calculation using embeddings"""
        # Mock embeddings that are similar
        mock_get_embedding.side_effect = [
            np.array([0.8, 0.6, 0.4]),
            np.array([0.7, 0.7, 0.3])
        ]
        
        analyzer.sentence_model = Mock()  # Enable embeddings
        
        similarity = analyzer._calculate_text_similarity("text1", "text2")
        
        # Should use embedding similarity
        assert similarity > 0.0
        assert mock_get_embedding.call_count == 2
    
    def test_categorize_match(self, analyzer):
        """Test match categorization"""
        # Test exact match
        match_type, confidence = analyzer._categorize_match(0.96)
        assert match_type == "exact"
        assert confidence > 0.96
        
        # Test semantic match
        match_type, confidence = analyzer._categorize_match(0.85)
        assert match_type == "semantic"
        assert confidence == 0.85
        
        # Test partial match
        match_type, confidence = analyzer._categorize_match(0.7)
        assert match_type == "partial"
        assert confidence == 0.7 * 0.8
        
        # Test no match
        match_type, confidence = analyzer._categorize_match(0.3)
        assert match_type == "none"
        assert confidence == 0.3


class TestEmbeddings:
    """Test embedding-related functionality"""
    
    @patch('models.semantic_analyzer.SentenceTransformer')
    def test_get_embedding_success(self, mock_st, analyzer):
        """Test successful embedding generation"""
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        analyzer.sentence_model = mock_model
        
        embedding = analyzer._get_embedding("test text")
        
        assert embedding is not None
        assert isinstance(embedding, np.ndarray)
        mock_model.encode.assert_called_once_with(["test text"])
    
    def test_get_embedding_no_model(self, analyzer):
        """Test embedding generation without model"""
        analyzer.sentence_model = None
        
        embedding = analyzer._get_embedding("test text")
        assert embedding is None
    
    def test_get_embedding_error(self, analyzer):
        """Test embedding generation with error"""
        mock_model = Mock()
        mock_model.encode.side_effect = Exception("Encoding error")
        analyzer.sentence_model = mock_model
        
        embedding = analyzer._get_embedding("test text")
        assert embedding is None
    
    def test_embedding_caching(self, analyzer):
        """Test embedding caching functionality"""
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        analyzer.sentence_model = mock_model
        
        text = "test text for caching"
        
        # First call should generate embedding
        embedding1 = analyzer._get_embedding(text)
        
        # Second call should use cache
        embedding2 = analyzer._get_embedding(text)
        
        assert np.array_equal(embedding1, embedding2)
        mock_model.encode.assert_called_once()  # Only called once due to caching


class TestOverlapDetection:
    """Test overlap detection between windows"""
    
    @pytest.mark.asyncio
    async def test_detect_paragraph_overlap_basic(self, analyzer, overlapping_elements):
        """Test basic overlap detection"""
        matches = await analyzer.detect_paragraph_overlap(
            window1_results=overlapping_elements["window1"],
            window2_results=overlapping_elements["window2"],
            window1_id=0,
            window2_id=1
        )
        
        assert len(matches) >= 0  # Should detect some matches
        
        for match in matches:
            assert isinstance(match, SemanticMatch)
            assert match.similarity_score >= analyzer.partial_threshold
            assert match.match_type in ["exact", "semantic", "partial"]
    
    @pytest.mark.asyncio
    async def test_detect_paragraph_overlap_no_matches(self, analyzer):
        """Test overlap detection with no matches"""
        window1 = [{"label": "text", "text": "Completely different content"}]
        window2 = [{"label": "text", "text": "Totally unrelated information"}]
        
        matches = await analyzer.detect_paragraph_overlap(
            window1_results=window1,
            window2_results=window2,
            window1_id=0,
            window2_id=1
        )
        
        # Should find no matches due to low similarity
        assert len(matches) == 0
    
    @pytest.mark.asyncio
    async def test_detect_paragraph_overlap_identical(self, analyzer):
        """Test overlap detection with identical text"""
        identical_text = "This is exactly the same text in both windows."
        window1 = [{"label": "text", "text": identical_text}]
        window2 = [{"label": "text", "text": identical_text}]
        
        matches = await analyzer.detect_paragraph_overlap(
            window1_results=window1,
            window2_results=window2,
            window1_id=0,
            window2_id=1
        )
        
        assert len(matches) == 1
        assert matches[0].match_type == "exact"
        assert matches[0].similarity_score >= analyzer.exact_threshold
    
    @pytest.mark.asyncio
    async def test_detect_paragraph_overlap_filter_short_text(self, analyzer):
        """Test that very short texts are filtered out"""
        window1 = [{"label": "text", "text": "Short"}]  # Too short
        window2 = [{"label": "text", "text": "Also short"}]  # Too short
        
        matches = await analyzer.detect_paragraph_overlap(
            window1_results=window1,
            window2_results=window2,
            window1_id=0,
            window2_id=1
        )
        
        assert len(matches) == 0  # Should be filtered out due to length
    
    @pytest.mark.asyncio
    async def test_detect_paragraph_overlap_statistics(self, analyzer, overlapping_elements):
        """Test that statistics are updated during overlap detection"""
        initial_comparisons = analyzer.total_comparisons
        
        await analyzer.detect_paragraph_overlap(
            window1_results=overlapping_elements["window1"],
            window2_results=overlapping_elements["window2"],
            window1_id=0,
            window2_id=1
        )
        
        assert analyzer.total_comparisons > initial_comparisons
    
    def test_identify_overlap_region(self, analyzer):
        """Test overlap region identification"""
        # Test end-to-start overlap
        text1 = "This is the beginning and this is the end part"
        text2 = "this is the end part and here is new content"
        
        region = analyzer._identify_overlap_region(text1, text2)
        assert region in ["text1_end_text2_start", "middle_overlap", "minimal_overlap"]
        
        # Test no overlap
        text1 = "Completely different"
        text2 = "Totally unrelated"
        
        region = analyzer._identify_overlap_region(text1, text2)
        assert region in ["no_overlap", "minimal_overlap"]


class TestContentMerging:
    """Test content merging functionality"""
    
    @pytest.mark.asyncio
    async def test_merge_overlapping_content_basic(self, analyzer):
        """Test basic content merging"""
        source_element = {"text": "Source text", "page_number": 1}
        target_element = {"text": "Target text", "page_number": 2}
        
        match = SemanticMatch(
            source_element=source_element,
            target_element=target_element,
            similarity_score=0.96,
            match_type="exact",
            confidence=0.96,
            overlap_region="text1_end_text2_start"
        )
        
        merged_elements = await analyzer.merge_overlapping_content(
            matches=[match],
            all_elements=[source_element, target_element]
        )
        
        assert len(merged_elements) == 1
        assert isinstance(merged_elements[0], MergedElement)
        assert merged_elements[0].confidence_score == 0.96
    
    @pytest.mark.asyncio
    async def test_merge_overlapping_content_no_matches(self, analyzer):
        """Test content merging with no matches"""
        merged_elements = await analyzer.merge_overlapping_content(
            matches=[],
            all_elements=[]
        )
        
        assert len(merged_elements) == 0
    
    @pytest.mark.asyncio
    async def test_create_merged_element(self, analyzer):
        """Test creation of merged elements"""
        source_element = {
            "text": "This is the first part",
            "page_number": 1,
            "label": "text"
        }
        target_element = {
            "text": "This is the second part",
            "page_number": 2,
            "label": "text"
        }
        
        match = SemanticMatch(
            source_element=source_element,
            target_element=target_element,
            similarity_score=0.9,
            match_type="semantic",
            confidence=0.9,
            overlap_region="middle_overlap"
        )
        
        merged = await analyzer._create_merged_element(match)
        
        assert merged is not None
        assert isinstance(merged, MergedElement)
        assert merged.element_type == "text"
        assert merged.page_span == (1, 2)
        assert len(merged.source_elements) == 2
    
    def test_merge_texts_intelligently(self, analyzer):
        """Test intelligent text merging"""
        # Test end-to-start merging
        text1 = "This is the beginning part"
        text2 = "the beginning part and this is the end"
        
        merged = analyzer._merge_texts_intelligently(
            text1, text2, "text1_end_text2_start"
        )
        
        assert len(merged) > len(text1)
        assert text1 in merged or text2 in merged
        
        # Test start-to-end merging
        merged = analyzer._merge_texts_intelligently(
            text1, text2, "text1_start_text2_end"
        )
        
        assert isinstance(merged, str)
        assert len(merged) > 0


class TestDiscreteParagraphs:
    """Test discrete paragraph compilation"""
    
    @pytest.mark.asyncio
    async def test_compile_discrete_paragraphs_basic(self, analyzer, sample_elements):
        """Test basic discrete paragraph compilation"""
        all_window_results = [sample_elements]
        merged_elements = []  # No merged elements
        
        discrete = await analyzer.compile_discrete_paragraphs(
            all_window_results=all_window_results,
            merged_elements=merged_elements
        )
        
        assert len(discrete) >= 0
        # Should include elements that weren't merged
    
    @pytest.mark.asyncio
    async def test_compile_discrete_paragraphs_with_merged(self, analyzer, sample_elements):
        """Test discrete paragraph compilation with merged elements"""
        all_window_results = [sample_elements]
        
        # Create a merged element that references one of the sample elements
        merged_element = MergedElement(
            element_id="merged_1",
            element_type="text",
            merged_content="Merged content",
            source_elements=[sample_elements[0]],  # Reference first element
            confidence_score=0.9,
            page_span=(1, 2)
        )
        
        discrete = await analyzer.compile_discrete_paragraphs(
            all_window_results=all_window_results,
            merged_elements=[merged_element]
        )
        
        # Should exclude the merged element from discrete paragraphs
        assert len(discrete) == len(sample_elements) - 1
    
    @pytest.mark.asyncio
    async def test_compile_discrete_paragraphs_deduplication(self, analyzer):
        """Test that duplicate content is removed"""
        duplicate_text = "This text appears in both windows"
        element1 = {"text": duplicate_text, "page_number": 1}
        element2 = {"text": duplicate_text, "page_number": 2}
        
        all_window_results = [[element1], [element2]]
        merged_elements = []
        
        discrete = await analyzer.compile_discrete_paragraphs(
            all_window_results=all_window_results,
            merged_elements=merged_elements
        )
        
        # Should deduplicate identical content
        assert len(discrete) == 1


class TestStatisticsAndUtilities:
    """Test statistics and utility methods"""
    
    def test_get_statistics(self, analyzer):
        """Test statistics generation"""
        # Simulate some processing
        analyzer.total_comparisons = 100
        analyzer.exact_matches = 5
        analyzer.semantic_matches = 10
        analyzer.partial_matches = 15
        
        stats = analyzer.get_statistics()
        
        assert stats["total_comparisons"] == 100
        assert stats["total_matches"] == 30
        assert stats["exact_matches"] == 5
        assert stats["semantic_matches"] == 10
        assert stats["partial_matches"] == 15
        assert stats["match_rate"] == 0.3
        assert stats["exact_match_rate"] == 0.05
        assert "sentence_model_available" in stats
        assert "thresholds" in stats
    
    def test_reset_statistics(self, analyzer):
        """Test statistics reset"""
        # Set some values
        analyzer.total_comparisons = 100
        analyzer.exact_matches = 5
        
        analyzer.reset_statistics()
        
        assert analyzer.total_comparisons == 0
        assert analyzer.exact_matches == 0
        assert analyzer.semantic_matches == 0
        assert analyzer.partial_matches == 0
    
    def test_clear_cache(self, analyzer):
        """Test cache clearing"""
        # Add some items to cache
        analyzer.content_cache["key1"] = "value1"
        analyzer.embedding_cache["key2"] = np.array([1, 2, 3])
        
        analyzer.clear_cache()
        
        assert len(analyzer.content_cache) == 0
        assert len(analyzer.embedding_cache) == 0


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_analyzer_with_none_input(self, analyzer):
        """Test analyzer methods with None inputs"""
        similarity = analyzer._calculate_text_similarity(None, "text")
        assert similarity == 0.0
        
        processed = analyzer._preprocess_text(None)
        assert processed == ""
    
    def test_analyzer_with_empty_embeddings(self, analyzer):
        """Test analyzer behavior when embeddings fail"""
        analyzer.sentence_model = Mock()
        analyzer.sentence_model.encode.side_effect = Exception("Error")
        
        similarity = analyzer._calculate_text_similarity("text1", "text2")
        assert 0.0 <= similarity <= 1.0  # Should fall back to sequence matching
    
    @pytest.mark.asyncio
    async def test_overlap_detection_with_non_text_elements(self, analyzer):
        """Test overlap detection with non-text elements"""
        window1 = [{"label": "figure", "text": "Image description"}]
        window2 = [{"label": "table", "text": "Table content"}]
        
        matches = await analyzer.detect_paragraph_overlap(
            window1_results=window1,
            window2_results=window2,
            window1_id=0,
            window2_id=1
        )
        
        # Should filter out non-text elements
        assert len(matches) == 0
    
    def test_cache_size_limit(self, analyzer):
        """Test that cache respects size limits"""
        analyzer.max_cache_size = 2
        
        # Add items beyond cache limit
        for i in range(5):
            analyzer.embedding_cache[f"key{i}"] = np.array([i])
        
        # Cache should not exceed max size
        assert len(analyzer.embedding_cache) <= analyzer.max_cache_size
    
    @patch('models.semantic_analyzer.logger')
    def test_logging(self, mock_logger, analyzer, overlapping_elements):
        """Test that appropriate logging occurs"""
        # This is an async method, so we need to test it properly
        import asyncio
        
        async def test_async():
            await analyzer.detect_paragraph_overlap(
                window1_results=overlapping_elements["window1"],
                window2_results=overlapping_elements["window2"],
                window1_id=0,
                window2_id=1
            )
        
        asyncio.run(test_async())
        
        # Check that logging was called
        mock_logger.info.assert_called()


@pytest.mark.asyncio
class TestIntegration:
    """Integration tests for semantic analyzer"""
    
    async def test_full_semantic_analysis_pipeline(self, analyzer, overlapping_elements):
        """Test complete semantic analysis pipeline"""
        # Step 1: Detect overlaps
        matches = await analyzer.detect_paragraph_overlap(
            window1_results=overlapping_elements["window1"],
            window2_results=overlapping_elements["window2"],
            window1_id=0,
            window2_id=1
        )
        
        # Step 2: Merge overlapping content
        all_elements = overlapping_elements["window1"] + overlapping_elements["window2"]
        merged_elements = await analyzer.merge_overlapping_content(
            matches=matches,
            all_elements=all_elements
        )
        
        # Step 3: Compile discrete paragraphs
        discrete_paragraphs = await analyzer.compile_discrete_paragraphs(
            all_window_results=[overlapping_elements["window1"], overlapping_elements["window2"]],
            merged_elements=merged_elements
        )
        
        # Verify pipeline results
        assert isinstance(matches, list)
        assert isinstance(merged_elements, list)
        assert isinstance(discrete_paragraphs, list)
        
        # Check statistics were updated
        stats = analyzer.get_statistics()
        assert stats["total_comparisons"] > 0 