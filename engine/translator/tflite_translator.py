#!/usr/bin/env python3
"""
TensorFlow Lite to ONNX Translator

Handles conversion of TFLite models (.tflite) to ONNX format.
Supports both regular TFLite models and quantized models.
"""

import logging
import struct
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

logger = logging.getLogger('ModelConvert.TFLite')


@dataclass
class TFLiteConfig:
    """Configuration for TFLite to ONNX conversion."""
    input_names: Optional[List[str]] = None
    output_names: Optional[List[str]] = None
    opset_version: int = 17
    dequantize: bool = True  # Dequantize quantized models
    keep_channel_last: bool = False  # Keep NHWC format (vs convert to NCHW)


class TFLiteTranslator:
    """Translator for TensorFlow Lite models to ONNX."""
    
    def __init__(self):
        self.supported_extensions = ['.tflite', '.lite']
    
    async def convert(
        self,
        input_path: Path,
        output_path: Path,
        custom_args: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Convert TFLite model to ONNX.
        
        Args:
            input_path: Path to TFLite model file
            output_path: Path for output ONNX file
            custom_args: Optional configuration overrides
        
        Returns:
            Dictionary with conversion results
        """
        
        custom_args = custom_args or {}
        config = TFLiteConfig(
            input_names=custom_args.get('input_names'),
            output_names=custom_args.get('output_names'),
            opset_version=custom_args.get('opset_version', 17),
            dequantize=custom_args.get('dequantize', True),
            keep_channel_last=custom_args.get('keep_channel_last', False),
        )
        
        logger.info(f"🔄 Converting TFLite model: {input_path.name}")
        
        try:
            # Try using tf2onnx for conversion (most reliable)
            return await self._convert_with_tf2onnx(input_path, output_path, config)
        except Exception as e:
            logger.warning(f"tf2onnx conversion failed: {e}, trying manual conversion...")
            return await self._convert_manual(input_path, output_path, config)
    
    async def _convert_with_tf2onnx(
        self,
        input_path: Path,
        output_path: Path,
        config: TFLiteConfig
    ) -> Dict[str, Any]:
        """Convert using tf2onnx (recommended method)."""
        
        try:
            import tf2onnx
            import tensorflow as tf
            
            # Load TFLite model
            interpreter = tf.lite.Interpreter(model_path=str(input_path))
            interpreter.allocate_tensors()
            
            # Get input/output details
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            logger.info(f"   Inputs: {len(input_details)}")
            for inp in input_details:
                logger.info(f"      - {inp['name']}: {inp['shape']} ({inp['dtype'].__name__})")
            
            logger.info(f"   Outputs: {len(output_details)}")
            for out in output_details:
                logger.info(f"      - {out['name']}: {out['shape']} ({out['dtype'].__name__})")
            
            # Convert using tf2onnx
            model_proto, _ = tf2onnx.convert.from_tflite(
                str(input_path),
                opset=config.opset_version,
                input_names=config.input_names,
                output_names=config.output_names,
            )
            
            # Save model
            onnx.save(model_proto, str(output_path))
            
            logger.info(f"✅ TFLite conversion successful: {output_path.name}")
            
            return {
                'status': 'success',
                'method': 'tf2onnx',
                'num_inputs': len(input_details),
                'num_outputs': len(output_details),
                'opset_version': config.opset_version,
                'output_path': str(output_path),
            }
            
        except ImportError:
            raise ImportError("tf2onnx not available for TFLite conversion")
    
    async def _convert_manual(
        self,
        input_path: Path,
        output_path: Path,
        config: TFLiteConfig
    ) -> Dict[str, Any]:
        """
        Manual conversion as fallback.
        This is a simplified implementation for basic models.
        """
        
        logger.info("🔄 Performing manual TFLite conversion...")
        
        try:
            from tflite import Model as TFLiteModel
        except ImportError:
            raise ImportError("tflite runtime not available for manual conversion")
        
        # Load TFLite model
        with open(input_path, 'rb') as f:
            model_data = f.read()
        
        tflite_model = TFLiteModel.GetRootAsModel(model_data, 0)
        
        # This is a simplified placeholder - real implementation would
        # traverse the TFLite graph and build equivalent ONNX nodes
        
        raise NotImplementedError(
            "Manual TFLite conversion not fully implemented. "
            "Please install tf2onnx for reliable conversion."
        )
    
    async def extract_and_convert(
        self,
        input_path: Path,
        output_dir: Path,
        config: TFLiteConfig
    ) -> List[Dict[str, Any]]:
        """
        Extract and convert multiple models from a TFLite container.
        
        Some TFLite files contain multiple subgraphs.
        """
        
        import tensorflow as tf
        
        interpreter = tf.lite.Interpreter(model_path=str(input_path))
        
        # Get subgraph information
        num_subgraphs = len(interpreter._get_ops_details())
        logger.info(f"📊 Found {num_subgraphs} subgraph(s)")
        
        results = []
        
        # For now, just convert the main model
        # Full implementation would iterate through subgraphs
        output_path = output_dir / f"{input_path.stem}.onnx"
        result = await self.convert(input_path, output_path, config)
        results.append(result)
        
        return results
    
    def inspect_model(self, input_path: Path) -> Dict[str, Any]:
        """Inspect TFLite model structure without converting."""
        
        try:
            import tensorflow as tf
            
            interpreter = tf.lite.Interpreter(model_path=str(input_path))
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Get tensor details
            tensor_details = interpreter.get_tensor_details()
            
            # Get operation details
            op_details = interpreter._get_ops_details()
            
            return {
                'version': interpreter._model_version,
                'num_tensors': len(tensor_details),
                'num_ops': len(op_details),
                'inputs': [
                    {
                        'name': inp['name'],
                        'shape': list(inp['shape']),
                        'dtype': inp['dtype'].__name__,
                        'quantization': inp.get('quantization'),
                    }
                    for inp in input_details
                ],
                'outputs': [
                    {
                        'name': out['name'],
                        'shape': list(out['shape']),
                        'dtype': out['dtype'].__name__,
                        'quantization': out.get('quantization'),
                    }
                    for out in output_details
                ],
                'operators': [op['op_name'] for op in op_details],
            }
            
        except Exception as e:
            return {
                'error': str(e),
            }
    
    async def convert_quantized(
        self,
        input_path: Path,
        output_path: Path,
        config: TFLiteConfig
    ) -> Dict[str, Any]:
        """
        Convert quantized TFLite model with dequantization.
        """
        
        if not config.dequantize:
            # Keep as quantized
            return await self.convert(input_path, output_path, config)
        
        logger.info("🔄 Converting quantized TFLite model with dequantization...")
        
        # First convert normally
        result = await self._convert_with_tf2onnx(input_path, output_path, config)
        
        # Then add dequantization nodes if needed
        # This would involve modifying the ONNX graph
        
        return result
    
    def _parse_tflite_schema(self, model_data: bytes) -> Dict[str, Any]:
        """Parse TFLite flatbuffer schema for detailed inspection."""
        
        # This would use the tflite schema to parse the model
        # For now, return basic info
        
        offset = 4
        if model_data[:4] == b'TFL3':
            version = 3
        else:
            version = 0
        
        return {
            'version': version,
            'size': len(model_data),
        }
