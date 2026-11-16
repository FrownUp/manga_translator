# bulk_processor.py
import os
import zipfile
import asyncio
from typing import List, Dict, Any
import logging
from datetime import datetime
from pathlib import Path

class BulkProcessor:
    """Process multiple images in batch"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.processing_queue = asyncio.Queue()
        self.results = {}
        
    async def process_batch(self, 
                          image_paths: List[str],
                          output_dir: str,
                          options: Dict[str, Any] = None) -> Dict:
        """
        Process multiple images in batch
        
        Args:
            image_paths: List of image file paths
            output_dir: Directory to save processed images
            options: Processing options
        """
        options = options or {}
        
        total_files = len(image_paths)
        processed_count = 0
        results = {
            'successful': [],
            'failed': [],
            'total_processed': 0,
            'output_zip': None
        }
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        for image_path in image_paths:
            try:
                self.logger.info(f"Processing {image_path} ({processed_count + 1}/{total_files})")
                
                # Process single image using your existing pipeline
                result = await self._process_single_image(image_path, output_dir, options)
                
                if result['success']:
                    results['successful'].append({
                        'input_path': image_path,
                        'output_path': result['output_path'],
                        'transcript': result.get('transcript', ''),
                        'processing_time': result.get('processing_time', 0)
                    })
                else:
                    results['failed'].append({
                        'input_path': image_path,
                        'error': result.get('error', 'Unknown error')
                    })
                
                processed_count += 1
                results['total_processed'] = processed_count
                
                # Progress callback
                if options.get('progress_callback'):
                    progress = (processed_count / total_files) * 100
                    options['progress_callback'](progress, processed_count, total_files)
                    
            except Exception as e:
                self.logger.error(f"Failed to process {image_path}: {e}")
                results['failed'].append({
                    'input_path': image_path,
                    'error': str(e)
                })
        
        # Create zip file if requested
        if options.get('create_zip', True) and results['successful']:
            zip_path = await self._create_output_zip(results['successful'], output_dir)
            results['output_zip'] = zip_path
        
        return results
    
    async def _process_single_image(self, 
                                  image_path: str, 
                                  output_dir: str,
                                  options: Dict) -> Dict:
        """Process single image using your existing pipeline"""
        start_time = datetime.now()
        
        try:
            # Use your existing manga_pipeline
            from manga_pipeline import run_manga_pipeline
            from character_mapper import CharacterMapper
            
            # Set up pipeline components
            character_mapper = CharacterMapper()
            
            # Run pipeline
            transcript = run_manga_pipeline(
                image_path=image_path,
                yolo_model=YOLO_BUBBLE_MODEL,  # Your existing models
                embed_model=EMBED_MODEL,
                classifier_model=RELATION_CLASSIFIER,
                ocr_model=ocr_fusion_model,
                output_dir=output_dir,
                reading_order=options.get('reading_order', 'manga'),
                character_mapper=character_mapper
            )
            
            # Generate output filename
            input_name = Path(image_path).stem
            output_path = os.path.join(output_dir, f"{input_name}_translated.png")
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'success': True,
                'output_path': output_path,
                'transcript': transcript,
                'processing_time': processing_time
            }
            
        except Exception as e:
            self.logger.error(f"Pipeline failed for {image_path}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _create_output_zip(self, 
                               successful_files: List[Dict], 
                               output_dir: str) -> str:
        """Create zip file of processed images"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_path = os.path.join(output_dir, f"manga_translation_{timestamp}.zip")
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add processed images
            for result in successful_files:
                output_path = result['output_path']
                if os.path.exists(output_path):
                    arcname = os.path.basename(output_path)
                    zipf.write(output_path, arcname)
            
            # Add transcripts
            transcript_content = self._generate_batch_transcript(successful_files)
            zipf.writestr("transcripts.txt", transcript_content)
            
            # Add processing summary
            summary = self._generate_processing_summary(successful_files)
            zipf.writestr("processing_summary.json", json.dumps(summary, indent=2))
        
        return zip_path
    
    def _generate_batch_transcript(self, successful_files: List[Dict]) -> str:
        """Generate combined transcript for all processed files"""
        transcript_lines = ["BATCH MANGA TRANSCRIPT", "=" * 50, ""]
        
        for result in successful_files:
            filename = os.path.basename(result['input_path'])
            transcript_lines.extend([
                f"File: {filename}",
                "-" * 30,
                result.get('transcript', 'No transcript available'),
                "\n"
            ])
        
        return "\n".join(transcript_lines)
    
    def _generate_processing_summary(self, successful_files: List[Dict]) -> Dict:
        """Generate processing summary statistics"""
        total_time = sum(result.get('processing_time', 0) for result in successful_files)
        avg_time = total_time / len(successful_files) if successful_files else 0
        
        return {
            'total_files_processed': len(successful_files),
            'total_processing_time_seconds': total_time,
            'average_time_per_file': avg_time,
            'processed_files': [
                {
                    'input_file': os.path.basename(result['input_path']),
                    'output_file': os.path.basename(result['output_path']),
                    'processing_time': result.get('processing_time', 0)
                }
                for result in successful_files
            ],
            'timestamp': datetime.now().isoformat()
        }
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported image formats"""
        return ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff']
    
    def validate_input_files(self, file_paths: List[str]) -> Dict:
        """Validate input files before processing"""
        valid_files = []
        invalid_files = []
        
        for file_path in file_paths:
            if not os.path.exists(file_path):
                invalid_files.append({'path': file_path, 'error': 'File not found'})
                continue
            
            file_ext = Path(file_path).suffix.lower()
            if file_ext not in self.get_supported_formats():
                invalid_files.append({'path': file_path, 'error': f'Unsupported format: {file_ext}'})
                continue
            
            # Check file size (max 50MB)
            file_size = os.path.getsize(file_path)
            if file_size > 50 * 1024 * 1024:
                invalid_files.append({'path': file_path, 'error': 'File too large (max 50MB)'})
                continue
            
            valid_files.append(file_path)
        
        return {
            'valid_files': valid_files,
            'invalid_files': invalid_files,
            'total_valid': len(valid_files),
            'total_invalid': len(invalid_files)
        }

# Global instance
bulk_processor = BulkProcessor()