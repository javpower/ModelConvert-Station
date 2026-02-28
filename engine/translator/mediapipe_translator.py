#!/usr/bin/env python3
"""
MediaPipe Task to ONNX Translator

Handles conversion of MediaPipe .task files to ONNX format.
MediaPipe .task files are ZIP archives containing multiple TFLite models
and metadata. This translator extracts all sub-models and converts them.
"""

import json
import logging
import zipfile
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

import onnx

from .tflite_translator import TFLiteTranslator

logger = logging.getLogger('ModelConvert.MediaPipe')


@dataclass
class MediaPipeConfig:
    """Configuration for MediaPipe task conversion."""
    convert_all_models: bool = True
    opset_version: int = 17
    dequantize: bool = True
    extract_metadata: bool = True


class MediaPipeTranslator:
    """Translator for MediaPipe .task files to ONNX."""
    
    def __init__(self):
        self.supported_extensions = ['.task']
        self.tflite_translator = TFLiteTranslator()
    
    async def convert(
        self,
        input_path: Path,
        output_path: Path,
        custom_args: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Convert MediaPipe .task file to ONNX.
        
        Args:
            input_path: Path to MediaPipe .task file
            output_path: Base path for output ONNX files
            custom_args: Optional configuration overrides
        
        Returns:
            Dictionary with conversion results including all sub-models
        """
        
        custom_args = custom_args or {}
        config = MediaPipeConfig(
            convert_all_models=custom_args.get('convert_all_models', True),
            opset_version=custom_args.get('opset_version', 17),
            dequantize=custom_args.get('dequantize', True),
            extract_metadata=custom_args.get('extract_metadata', True),
        )
        
        logger.info(f"🔄 Converting MediaPipe task: {input_path.name}")
        
        # Extract task archive
        with tempfile.TemporaryDirectory() as extract_dir:
            extract_path = Path(extract_dir)
            
            # Extract ZIP contents
            with zipfile.ZipFile(input_path, 'r') as zf:
                zf.extractall(extract_path)
                extracted_files = zf.namelist()
            
            logger.info(f"📦 Extracted {len(extracted_files)} files")
            
            # Find metadata
            metadata = await self._extract_metadata(extract_path)
            if metadata:
                logger.info(f"📋 Task type: {metadata.get('task_name', 'Unknown')}")
            
            # Find all TFLite models
            tflite_files = list(extract_path.rglob('*.tflite'))
            logger.info(f"🔍 Found {len(tflite_files)} TFLite model(s)")
            
            if not tflite_files:
                raise ValueError("No TFLite models found in .task file")
            
            # Create output directory
            output_dir = output_path.parent / output_path.stem
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Convert each TFLite model
            converted_models = []
            
            for i, tflite_path in enumerate(tflite_files):
                model_name = tflite_path.stem
                model_output_path = output_dir / f"{model_name}.onnx"
                
                logger.info(f"[{i+1}/{len(tflite_files)}] Converting: {model_name}")
                
                try:
                    result = await self.tflite_translator.convert(
                        tflite_path,
                        model_output_path,
                        custom_args={
                            'opset_version': config.opset_version,
                            'dequantize': config.dequantize,
                        }
                    )
                    
                    converted_models.append({
                        'name': model_name,
                        'status': 'success',
                        'output_path': str(model_output_path),
                        'details': result,
                    })
                    
                except Exception as e:
                    logger.error(f"❌ Failed to convert {model_name}: {e}")
                    converted_models.append({
                        'name': model_name,
                        'status': 'failed',
                        'error': str(e),
                    })
            
            # Save metadata
            if config.extract_metadata:
                metadata_path = output_dir / 'task_metadata.json'
                with open(metadata_path, 'w') as f:
                    json.dump({
                        'source_file': input_path.name,
                        'extracted_files': extracted_files,
                        'tflite_models': [str(p.relative_to(extract_path)) for p in tflite_files],
                        'task_metadata': metadata,
                        'converted_models': converted_models,
                    }, f, indent=2)
            
            logger.info(f"✅ MediaPipe conversion complete: {len([m for m in converted_models if m['status'] == 'success'])} succeeded")
            
            return {
                'status': 'success',
                'output_directory': str(output_dir),
                'num_models': len(tflite_files),
                'converted_models': converted_models,
                'metadata': metadata,
            }
    
    async def _extract_metadata(self, extract_path: Path) -> Optional[Dict[str, Any]]:
        """Extract MediaPipe task metadata."""
        
        # Look for metadata files
        metadata_files = list(extract_path.rglob('*.json'))
        
        for meta_file in metadata_files:
            try:
                with open(meta_file, 'r') as f:
                    data = json.load(f)
                    
                    # Check if this looks like MediaPipe metadata
                    if any(key in data for key in ['task_name', 'model_type', 'delegate']):
                        return data
                        
            except (json.JSONDecodeError, IOError):
                continue
        
        # Try to infer from directory structure
        return self._infer_task_type(extract_path)
    
    def _infer_task_type(self, extract_path: Path) -> Optional[Dict[str, Any]]:
        """Infer MediaPipe task type from directory structure and file names."""
        
        all_files = [f.name.lower() for f in extract_path.rglob('*')]
        
        # Common MediaPipe task patterns
        task_patterns = {
            'pose_detection': ['pose', 'pose_detection', 'pose_landmark'],
            'face_detection': ['face', 'face_detection', 'face_landmark'],
            'hand_tracking': ['hand', 'hand_landmark', 'palm_detection'],
            'object_detection': ['object', 'detection'],
            'image_classification': ['classification', 'classifier'],
            'text_classification': ['text', 'bert', 'nlp'],
            'gesture_recognition': ['gesture', 'gesture_recognizer'],
            'image_segmentation': ['segmentation', 'segmenter'],
        }
        
        for task_type, keywords in task_patterns.items():
            if any(keyword in ' '.join(all_files) for keyword in keywords):
                return {'task_name': task_type, 'inferred': True}
        
        return None
    
    async def extract_and_inspect(
        self,
        input_path: Path
    ) -> Dict[str, Any]:
        """
        Extract and inspect MediaPipe task without converting.
        """
        
        logger.info(f"🔍 Inspecting MediaPipe task: {input_path.name}")
        
        with tempfile.TemporaryDirectory() as extract_dir:
            extract_path = Path(extract_dir)
            
            # Extract
            with zipfile.ZipFile(input_path, 'r') as zf:
                zf.extractall(extract_path)
                extracted_files = zf.namelist()
            
            # Get metadata
            metadata = await self._extract_metadata(extract_path)
            
            # Find TFLite models
            tflite_files = list(extract_path.rglob('*.tflite'))
            
            # Inspect each model
            model_inspections = []
            for tflite_path in tflite_files:
                inspection = self.tflite_translator.inspect_model(tflite_path)
                model_inspections.append({
                    'path': str(tflite_path.relative_to(extract_path)),
                    'inspection': inspection,
                })
            
            return {
                'task_file': input_path.name,
                'task_type': metadata.get('task_name', 'Unknown') if metadata else 'Unknown',
                'extracted_files': len(extracted_files),
                'tflite_models': model_inspections,
                'metadata': metadata,
            }
    
    async def convert_single_model(
        self,
        input_path: Path,
        output_path: Path,
        model_name: str,
        custom_args: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Convert a single model from a MediaPipe task file.
        """
        
        custom_args = custom_args or {}
        
        with tempfile.TemporaryDirectory() as extract_dir:
            extract_path = Path(extract_dir)
            
            # Extract
            with zipfile.ZipFile(input_path, 'r') as zf:
                zf.extractall(extract_path)
            
            # Find specific model
            tflite_files = list(extract_path.rglob(f'{model_name}*.tflite'))
            
            if not tflite_files:
                raise ValueError(f"Model '{model_name}' not found in task file")
            
            # Convert first match
            return await self.tflite_translator.convert(
                tflite_files[0],
                output_path,
                custom_args
            )
    
    def list_models(self, input_path: Path) -> List[str]:
        """List all TFLite models in a MediaPipe task file."""
        
        with zipfile.ZipFile(input_path, 'r') as zf:
            return [name for name in zf.namelist() if name.endswith('.tflite')]
