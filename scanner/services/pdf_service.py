"""
PDF processing service that integrates sliding window processing,
semantic analysis, and content merging
"""

import asyncio
import logging
import time
import uuid
from typing import List, Dict, Any, Optional, Union
from io import BytesIO
import tempfile
import os

from models.dolphin_model import DolphinModelWrapper
from models.window_processor import SlidingWindowProcessor, WindowResult
from models.semantic_analyzer import SemanticAnalyzer, SemanticMatch, MergedElement
from schemas.response_models import (
    ProcessingResult, WindowResult as WindowResultSchema, 
    SemanticRelation, CrossPageElement, Paragraph, MergedContent
)

logger = logging.getLogger(__name__)


class PDFProcessingService:
    """
    Main service for processing PDFs with sliding window approach
    Coordinates all components to provide comprehensive document analysis
    """
    
    def __init__(
        self,
        dolphin_model: DolphinModelWrapper,
        window_processor: SlidingWindowProcessor,
        semantic_analyzer: SemanticAnalyzer
    ):
        """
        Initialize the PDF processing service
        
        Args:
            dolphin_model: Dolphin model wrapper
            window_processor: Sliding window processor
            semantic_analyzer: Semantic analyzer
        """
        self.dolphin_model = dolphin_model
        self.window_processor = window_processor
        self.semantic_analyzer = semantic_analyzer
        
        # Processing statistics
        self.total_documents_processed = 0
        self.total_processing_time = 0.0
        self.total_pages_processed = 0
        self.total_windows_processed = 0
        
        logger.info("PDF processing service initialized")
    
    async def process_pdf_with_prompt(
        self,
        file_content: Union[bytes, str],
        prompt: str = "Parse and analyze this document",
        window_size: int = 2,
        overlap: int = 1,
        semantic_threshold: float = 0.8,
        combination_method: str = "vertical",
        max_batch_size: int = 4
    ) -> ProcessingResult:
        """
        Process a PDF with sliding window approach and semantic analysis
        
        Args:
            file_content: PDF file content as bytes
            prompt: Processing prompt
            window_size: Number of pages per window
            overlap: Window overlap
            semantic_threshold: Semantic similarity threshold
            combination_method: How to combine pages in windows
            max_batch_size: Maximum batch size for processing
            
        Returns:
            ProcessingResult with comprehensive analysis
        """
        start_time = time.time()
        document_id = str(uuid.uuid4())
        
        logger.info(f"Starting PDF processing for document {document_id}")
        
        try:
            # Step 1: Convert PDF to images
            if isinstance(file_content, str):
                file_content = file_content.encode('utf-8')
            
            pdf_images = await self.dolphin_model.convert_pdf_to_images_async(file_content)
            if not pdf_images:
                raise ValueError("Failed to convert PDF to images")
            
            total_pages = len(pdf_images)
            logger.info(f"Converted PDF to {total_pages} page images")
            
            # Step 2: Create sliding windows
            window_processor = SlidingWindowProcessor(window_size, overlap)
            windows = window_processor.create_windows(pdf_images)
            
            logger.info(f"Created {len(windows)} sliding windows")
            
            # Step 3: Process all windows
            window_results = await window_processor.process_windows_async(
                windows=windows,
                dolphin_model=self.dolphin_model,
                prompt=prompt,
                combination_method=combination_method,
                max_batch_size=max_batch_size
            )
            
            # Step 4: Semantic analysis and overlap detection
            semantic_matches = await self._analyze_window_overlaps(window_results)
            
            # Step 5: Merge overlapping content
            merged_elements = await self._merge_overlapping_content(semantic_matches, window_results)
            
            # Step 6: Compile discrete paragraphs
            discrete_paragraphs = await self._compile_discrete_paragraphs(window_results, merged_elements)
            
            # Step 7: Create final merged content
            merged_content = await self._create_merged_content(
                window_results, merged_elements, discrete_paragraphs
            )
            
            # Step 8: Generate semantic relationships
            semantic_relationships = self._generate_semantic_relationships(semantic_matches)
            
            # Step 9: Identify cross-page elements
            cross_page_elements = self._identify_cross_page_elements(merged_elements)
            
            processing_time = time.time() - start_time
            
            # Update statistics
            self.total_documents_processed += 1
            self.total_processing_time += processing_time
            self.total_pages_processed += total_pages
            self.total_windows_processed += len(windows)
            
            # Create final result
            result = ProcessingResult(
                document_id=document_id,
                total_pages=total_pages,
                processing_windows=[self._convert_window_result(wr) for wr in window_results],
                merged_content=merged_content,
                semantic_relationships=semantic_relationships,
                discrete_paragraphs=[self._convert_to_paragraph(p, idx) for idx, p in enumerate(discrete_paragraphs)],
                cross_page_elements=cross_page_elements,
                processing_time=processing_time,
                window_count=len(windows),
                overlap_detections=len(semantic_matches),
                merge_operations=len(merged_elements),
                confidence_scores=self._calculate_confidence_scores(window_results, semantic_matches),
                error_count=sum(1 for wr in window_results if not wr.success),
                warnings=self._generate_warnings(window_results, window_processor)
            )
            
            logger.info(f"Successfully processed document {document_id} in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error processing document {document_id}: {str(e)}")
            
            # Return error result
            return ProcessingResult(
                document_id=document_id,
                total_pages=0,
                processing_windows=[],
                merged_content=MergedContent(
                    text_content=f"Error processing document: {str(e)}",
                    structured_data={},
                    tables=[],
                    formulas=[],
                    images=[],
                    metadata={"error": str(e)}
                ),
                semantic_relationships=[],
                discrete_paragraphs=[],
                cross_page_elements=[],
                processing_time=processing_time,
                window_count=0,
                overlap_detections=0,
                merge_operations=0,
                confidence_scores={},
                error_count=1,
                warnings=[f"Processing failed: {str(e)}"]
            )
    
    async def process_pdf_pages(
        self,
        file_content: bytes,
        start_page: int = 1,
        end_page: Optional[int] = None,
        prompt: str = "Parse these pages",
        window_size: int = 2,
        overlap: int = 1
    ) -> ProcessingResult:
        """
        Process specific page ranges from a PDF
        
        Args:
            file_content: PDF file content
            start_page: Starting page (1-indexed)
            end_page: Ending page (1-indexed, None for last page)
            prompt: Processing prompt
            window_size: Window size
            overlap: Window overlap
            
        Returns:
            ProcessingResult for the specified pages
        """
        # Convert PDF to images
        pdf_images = await self.dolphin_model.convert_pdf_to_images_async(file_content)
        
        if not pdf_images:
            raise ValueError("Failed to convert PDF to images")
        
        # Extract specified page range
        start_idx = start_page - 1  # Convert to 0-indexed
        end_idx = (end_page or len(pdf_images))  # Default to last page
        
        if start_idx < 0 or start_idx >= len(pdf_images):
            raise ValueError(f"Start page {start_page} is out of range")
        
        if end_idx > len(pdf_images):
            end_idx = len(pdf_images)
        
        selected_pages = pdf_images[start_idx:end_idx]
        
        # Process selected pages
        return await self.process_pdf_with_prompt(
            file_content=b'',  # Not needed since we have images
            prompt=prompt,
            window_size=window_size,
            overlap=overlap
        )
    
    async def _analyze_window_overlaps(self, window_results: List[WindowResult]) -> List[SemanticMatch]:
        """Analyze overlaps between consecutive windows"""
        all_matches = []
        
        for i in range(len(window_results) - 1):
            current_result = window_results[i]
            next_result = window_results[i + 1]
            
            if current_result.success and next_result.success:
                matches = await self.semantic_analyzer.detect_paragraph_overlap(
                    window1_results=current_result.elements,
                    window2_results=next_result.elements,
                    window1_id=current_result.window.window_id,
                    window2_id=next_result.window.window_id
                )
                all_matches.extend(matches)
        
        logger.info(f"Found {len(all_matches)} semantic matches across windows")
        return all_matches
    
    async def _merge_overlapping_content(
        self,
        semantic_matches: List[SemanticMatch],
        window_results: List[WindowResult]
    ) -> List[MergedElement]:
        """Merge overlapping content from semantic matches"""
        all_elements = []
        for wr in window_results:
            if wr.success:
                all_elements.extend(wr.elements)
        
        merged_elements = await self.semantic_analyzer.merge_overlapping_content(
            matches=semantic_matches,
            all_elements=all_elements
        )
        
        logger.info(f"Created {len(merged_elements)} merged elements")
        return merged_elements
    
    async def _compile_discrete_paragraphs(
        self,
        window_results: List[WindowResult],
        merged_elements: List[MergedElement]
    ) -> List[Dict[str, Any]]:
        """Compile discrete paragraphs that weren't merged"""
        all_window_elements = []
        for wr in window_results:
            if wr.success:
                all_window_elements.append(wr.elements)
        
        discrete_paragraphs = await self.semantic_analyzer.compile_discrete_paragraphs(
            all_window_results=all_window_elements,
            merged_elements=merged_elements
        )
        
        return discrete_paragraphs
    
    async def _create_merged_content(
        self,
        window_results: List[WindowResult],
        merged_elements: List[MergedElement],
        discrete_paragraphs: List[Dict[str, Any]]
    ) -> MergedContent:
        """Create the final merged content structure"""
        
        # Combine all text content
        all_text_parts = []
        
        # Add merged elements
        for merged in merged_elements:
            all_text_parts.append(merged.merged_content)
        
        # Add discrete paragraphs
        for paragraph in discrete_paragraphs:
            if 'text' in paragraph:
                all_text_parts.append(str(paragraph['text']))
        
        # Extract structured data
        tables = []
        formulas = []
        images = []
        
        for wr in window_results:
            if not wr.success:
                continue
                
            for element in wr.elements:
                element_type = element.get('label', '')
                
                if element_type in ['table', 'tab']:
                    tables.append({
                        'content': element.get('text', ''),
                        'page': element.get('page_number'),
                        'window': element.get('window_id'),
                        'confidence': element.get('confidence', 0.0)
                    })
                elif element_type in ['formula', 'equation']:
                    formulas.append({
                        'content': element.get('text', ''),
                        'page': element.get('page_number'),
                        'window': element.get('window_id'),
                        'type': 'mathematical'
                    })
                elif element_type in ['figure', 'image']:
                    images.append({
                        'description': element.get('text', ''),
                        'page': element.get('page_number'),
                        'window': element.get('window_id'),
                        'type': 'figure'
                    })
        
        return MergedContent(
            text_content='\n\n'.join(all_text_parts),
            structured_data={
                'total_elements': len(merged_elements) + len(discrete_paragraphs),
                'merged_elements': len(merged_elements),
                'discrete_paragraphs': len(discrete_paragraphs),
                'processing_method': 'sliding_window_semantic_analysis'
            },
            tables=tables,
            formulas=formulas,
            images=images,
            metadata={
                'total_windows': len(window_results),
                'successful_windows': sum(1 for wr in window_results if wr.success),
                'semantic_analyzer_stats': self.semantic_analyzer.get_statistics()
            }
        )
    
    def _generate_semantic_relationships(self, semantic_matches: List[SemanticMatch]) -> List[SemanticRelation]:
        """Generate semantic relationship objects from matches"""
        relationships = []
        
        for match in semantic_matches:
            source_window = match.source_element.get('window_id', 0)
            target_window = match.target_element.get('window_id', 0)
            
            relationship = SemanticRelation(
                source_window=source_window,
                target_window=target_window,
                similarity_score=match.similarity_score,
                relation_type=match.match_type,
                source_element_id=f"elem_{id(match.source_element)}",
                target_element_id=f"elem_{id(match.target_element)}"
            )
            relationships.append(relationship)
        
        return relationships
    
    def _identify_cross_page_elements(self, merged_elements: List[MergedElement]) -> List[CrossPageElement]:
        """Identify elements that span multiple pages"""
        cross_page_elements = []
        
        for merged in merged_elements:
            if merged.page_span[0] != merged.page_span[1]:  # Spans multiple pages
                cross_page_element = CrossPageElement(
                    element_id=merged.element_id,
                    element_type=merged.element_type,
                    start_page=merged.page_span[0],
                    end_page=merged.page_span[1],
                    merged_content=merged.merged_content,
                    confidence_score=merged.confidence_score
                )
                cross_page_elements.append(cross_page_element)
        
        return cross_page_elements
    
    def _convert_window_result(self, window_result: WindowResult) -> WindowResultSchema:
        """Convert WindowResult to schema format"""
        return WindowResultSchema(
            window_id=window_result.window.window_id,
            start_page=window_result.window.start_page,
            end_page=window_result.window.end_page,
            elements=window_result.elements,
            processing_time=window_result.processing_time,
            element_count=window_result.element_count
        )
    
    def _convert_to_paragraph(self, element: Dict[str, Any], index: int) -> Paragraph:
        """Convert element to Paragraph schema"""
        return Paragraph(
            paragraph_id=f"para_{index}_{id(element)}",
            content=str(element.get('text', '')),
            page_number=element.get('page_number', 0),
            position=element.get('position', {}),
            reading_order=element.get('reading_order', index),
            element_type=element.get('label', 'text')
        )
    
    def _calculate_confidence_scores(
        self,
        window_results: List[WindowResult],
        semantic_matches: List[SemanticMatch]
    ) -> Dict[str, float]:
        """Calculate various confidence scores"""
        total_windows = len(window_results)
        successful_windows = sum(1 for wr in window_results if wr.success)
        
        processing_success_rate = successful_windows / max(total_windows, 1)
        
        if semantic_matches:
            avg_similarity = sum(match.similarity_score for match in semantic_matches) / len(semantic_matches)
            high_confidence_matches = sum(1 for match in semantic_matches if match.confidence > 0.8)
            semantic_confidence = high_confidence_matches / len(semantic_matches)
        else:
            avg_similarity = 0.0
            semantic_confidence = 0.0
        
        return {
            'processing_success_rate': processing_success_rate,
            'average_similarity_score': avg_similarity,
            'semantic_analysis_confidence': semantic_confidence,
            'overall_confidence': (processing_success_rate + semantic_confidence) / 2
        }
    
    def _generate_warnings(
        self,
        window_results: List[WindowResult],
        window_processor: SlidingWindowProcessor
    ) -> List[str]:
        """Generate warnings based on processing results"""
        warnings = []
        
        # Check for failed windows
        failed_windows = [wr for wr in window_results if not wr.success]
        if failed_windows:
            warnings.append(f"{len(failed_windows)} windows failed to process")
        
        # Check window processor warnings
        processor_warnings = window_processor.validate_windows()
        warnings.extend(processor_warnings)
        
        # Check for low element counts
        low_element_windows = [wr for wr in window_results if wr.success and wr.element_count < 3]
        if low_element_windows:
            warnings.append(f"{len(low_element_windows)} windows have very few elements")
        
        return warnings
    
    def get_service_statistics(self) -> Dict[str, Any]:
        """Get service processing statistics"""
        avg_processing_time = self.total_processing_time / max(self.total_documents_processed, 1)
        avg_pages_per_doc = self.total_pages_processed / max(self.total_documents_processed, 1)
        avg_windows_per_doc = self.total_windows_processed / max(self.total_documents_processed, 1)
        
        return {
            'documents_processed': self.total_documents_processed,
            'total_processing_time': self.total_processing_time,
            'total_pages_processed': self.total_pages_processed,
            'total_windows_processed': self.total_windows_processed,
            'average_processing_time': avg_processing_time,
            'average_pages_per_document': avg_pages_per_doc,
            'average_windows_per_document': avg_windows_per_doc,
            'dolphin_model_stats': self.dolphin_model.get_stats(),
            'semantic_analyzer_stats': self.semantic_analyzer.get_statistics()
        }
    
    def reset_statistics(self):
        """Reset all service statistics"""
        self.total_documents_processed = 0
        self.total_processing_time = 0.0
        self.total_pages_processed = 0
        self.total_windows_processed = 0
        
        self.dolphin_model.reset_stats()
        self.semantic_analyzer.reset_statistics()
        
        logger.info("PDF processing service statistics reset") 