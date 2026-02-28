#!/usr/bin/env python3
"""
TensorFlow to ONNX Translator

Handles conversion of TensorFlow models (SavedModel, Keras .h5, frozen .pb)
to ONNX format using tf2onnx.
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

import tensorflow as tf
import onnx

logger = logging.getLogger('ModelConvert.TensorFlow')


@dataclass
class TensorFlowConfig:
    """Configuration for TensorFlow to ONNX conversion."""
    input_names: Optional[List[str]] = None
    output_names: Optional[List[str]] = None
    opset_version: int = 17
    inputs_as_nchw: Optional[List[str]] = None
    outputs_as_nchw: Optional[List[str]] = None
    large_model: bool = False  # Use external data format for large models


class TensorFlowTranslator:
    """Translator for TensorFlow models to ONNX."""
    
    def __init__(self):
        self.supported_formats = ['saved_model', 'keras', 'frozen_graph']
    
    async def convert(
        self,
        input_path: Path,
        output_path: Path,
        custom_args: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Convert TensorFlow model to ONNX.
        
        Args:
            input_path: Path to TensorFlow model (SavedModel dir, .h5, or .pb)
            output_path: Path for output ONNX file
            custom_args: Optional configuration overrides
        
        Returns:
            Dictionary with conversion results
        """
        
        custom_args = custom_args or {}
        config = TensorFlowConfig(
            input_names=custom_args.get('input_names'),
            output_names=custom_args.get('output_names'),
            opset_version=custom_args.get('opset_version', 17),
            inputs_as_nchw=custom_args.get('inputs_as_nchw'),
            outputs_as_nchw=custom_args.get('outputs_as_nchw'),
            large_model=custom_args.get('large_model', False),
        )
        
        logger.info(f"🔄 Converting TensorFlow model: {input_path.name}")
        
        # Detect model format
        model_format = self._detect_format(input_path)
        logger.info(f"📋 Detected format: {model_format}")
        
        if model_format == 'saved_model':
            return await self._convert_saved_model(input_path, output_path, config)
        elif model_format == 'keras':
            return await self._convert_keras(input_path, output_path, config)
        elif model_format == 'frozen_graph':
            return await self._convert_frozen_graph(input_path, output_path, config)
        else:
            raise ValueError(f"Unsupported TensorFlow format: {model_format}")
    
    def _detect_format(self, input_path: Path) -> str:
        """Detect TensorFlow model format from path."""
        
        if input_path.is_dir():
            # Check for SavedModel
            if (input_path / 'saved_model.pb').exists():
                return 'saved_model'
            # Check for checkpoint
            if list(input_path.glob('*.ckpt*')):
                return 'checkpoint'
        
        if input_path.is_file():
            ext = input_path.suffix.lower()
            if ext == '.h5' or ext == '.keras':
                return 'keras'
            if ext == '.pb':
                return 'frozen_graph'
            # Check for SavedModel in zip
            if ext == '.zip':
                import zipfile
                with zipfile.ZipFile(input_path, 'r') as zf:
                    if 'saved_model.pb' in zf.namelist():
                        return 'saved_model'
        
        # Try to inspect content
        if input_path.is_file():
            with open(input_path, 'rb') as f:
                header = f.read(8)
                if header.startswith(b'\x89HDF'):
                    return 'keras'  # HDF5 format
        
        raise ValueError(f"Cannot detect TensorFlow format for: {input_path}")
    
    async def _convert_saved_model(
        self,
        input_path: Path,
        output_path: Path,
        config: TensorFlowConfig
    ) -> Dict[str, Any]:
        """Convert TensorFlow SavedModel to ONNX."""
        
        logger.info("🔄 Converting SavedModel...")
        
        try:
            import tf2onnx
            
            # Load SavedModel
            model = tf.saved_model.load(str(input_path))
            
            # Get concrete function
            concrete_func = model.signatures.get('serving_default')
            if not concrete_func:
                # Fallback to any signature
                concrete_func = list(model.signatures.values())[0]
            
            # Infer input/output names if not provided
            if not config.input_names:
                config.input_names = [tensor.name.split(':')[0] for tensor in concrete_func.inputs if tensor.dtype != tf.resource]
            if not config.output_names:
                config.output_names = [tensor.name.split(':')[0] for tensor in concrete_func.outputs]
            
            logger.info(f"   Inputs: {config.input_names}")
            logger.info(f"   Outputs: {config.output_names}")
            
            # Convert
            model_proto, external_tensor_storage = tf2onnx.convert.from_saved_model(
                str(input_path),
                input=config.input_names,
                output=config.output_names,
                opset=config.opset_version,
                inputs_as_nchw=config.inputs_as_nchw,
                outputs_as_nchw=config.outputs_as_nchw,
                large_model=config.large_model,
            )
            
            # Save model
            onnx.save(model_proto, str(output_path))
            
            logger.info(f"✅ SavedModel conversion successful: {output_path.name}")
            
            return {
                'status': 'success',
                'format': 'saved_model',
                'input_names': config.input_names,
                'output_names': config.output_names,
                'opset_version': config.opset_version,
                'output_path': str(output_path),
            }
            
        except ImportError:
            logger.error("❌ tf2onnx not available")
            raise
        except Exception as e:
            logger.error(f"❌ SavedModel conversion failed: {e}")
            raise
    
    async def _convert_keras(
        self,
        input_path: Path,
        output_path: Path,
        config: TensorFlowConfig
    ) -> Dict[str, Any]:
        """Convert Keras model to ONNX."""
        
        logger.info("🔄 Converting Keras model...")
        
        try:
            import tf2onnx
            
            # Load Keras model
            model = tf.keras.models.load_model(str(input_path), compile=False)
            
            # Get model info
            logger.info(f"   Model: {model.name}")
            logger.info(f"   Layers: {len(model.layers)}")
            
            # Convert using tf2onnx
            input_signature = [tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype, name='input')]
            
            model_proto, _ = tf2onnx.convert.from_keras(
                model,
                input_signature=input_signature,
                opset=config.opset_version,
                inputs_as_nchw=config.inputs_as_nchw,
                outputs_as_nchw=config.outputs_as_nchw,
            )
            
            # Save model
            onnx.save(model_proto, str(output_path))
            
            logger.info(f"✅ Keras conversion successful: {output_path.name}")
            
            return {
                'status': 'success',
                'format': 'keras',
                'model_name': model.name,
                'num_layers': len(model.layers),
                'opset_version': config.opset_version,
                'output_path': str(output_path),
            }
            
        except Exception as e:
            logger.error(f"❌ Keras conversion failed: {e}")
            raise
    
    async def _convert_frozen_graph(
        self,
        input_path: Path,
        output_path: Path,
        config: TensorFlowConfig
    ) -> Dict[str, Any]:
        """Convert TensorFlow frozen graph to ONNX."""
        
        logger.info("🔄 Converting frozen graph...")
        
        try:
            import tf2onnx
            
            # For frozen graphs, we need input/output names
            if not config.input_names or not config.output_names:
                # Try to infer from graph
                graph = self._load_frozen_graph(input_path)
                
                if not config.input_names:
                    # Find placeholder nodes
                    config.input_names = [
                        node.name for node in graph.node 
                        if node.op == 'Placeholder'
                    ]
                
                if not config.output_names:
                    # Find output nodes (nodes with no outgoing edges in main graph)
                    all_outputs = set()
                    for node in graph.node:
                        for inp in node.input:
                            all_outputs.add(inp.split(':')[0])
                    
                    output_nodes = []
                    for node in graph.node:
                        if node.name not in all_outputs:
                            output_nodes.append(node.name)
                    
                    config.output_names = output_nodes[:5] if len(output_nodes) > 5 else output_nodes
            
            logger.info(f"   Inputs: {config.input_names}")
            logger.info(f"   Outputs: {config.output_names}")
            
            # Convert
            model_proto, _ = tf2onnx.convert.from_graphdef(
                str(input_path),
                input_names=config.input_names,
                output_names=config.output_names,
                opset=config.opset_version,
                inputs_as_nchw=config.inputs_as_nchw,
                outputs_as_nchw=config.outputs_as_nchw,
            )
            
            # Save model
            onnx.save(model_proto, str(output_path))
            
            logger.info(f"✅ Frozen graph conversion successful: {output_path.name}")
            
            return {
                'status': 'success',
                'format': 'frozen_graph',
                'input_names': config.input_names,
                'output_names': config.output_names,
                'opset_version': config.opset_version,
                'output_path': str(output_path),
            }
            
        except Exception as e:
            logger.error(f"❌ Frozen graph conversion failed: {e}")
            raise
    
    def _load_frozen_graph(self, pb_path: Path) -> tf.compat.v1.GraphDef:
        """Load frozen graph from .pb file."""
        
        with tf.io.gfile.GFile(str(pb_path), 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        
        return graph_def
    
    async def convert_checkpoint(
        self,
        checkpoint_dir: Path,
        output_path: Path,
        config: TensorFlowConfig,
        model_builder: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Convert TensorFlow checkpoint to ONNX.
        
        Requires model_builder function to reconstruct model architecture.
        """
        
        if not model_builder:
            raise ValueError(
                "Checkpoint conversion requires model_builder function. "
                "Please provide a function that returns the model architecture."
            )
        
        logger.info("🔄 Converting checkpoint...")
        
        # Build model
        model = model_builder()
        
        # Restore checkpoint
        checkpoint = tf.train.Checkpoint(model=model)
        checkpoint.restore(tf.train.latest_checkpoint(str(checkpoint_dir)))
        
        # Save as SavedModel temporarily
        with tempfile.TemporaryDirectory() as tmpdir:
            saved_model_path = Path(tmpdir) / 'saved_model'
            tf.saved_model.save(model, str(saved_model_path))
            
            # Convert SavedModel
            return await self._convert_saved_model(saved_model_path, output_path, config)
    
    async def validate_onnx(self, onnx_path: Path) -> Dict[str, Any]:
        """Validate the generated ONNX model."""
        
        try:
            model = onnx.load(str(onnx_path))
            onnx.checker.check_model(model)
            
            return {
                'valid': True,
                'opset_version': model.opset_import[0].version if model.opset_import else None,
            }
            
        except Exception as e:
            return {
                'valid': False,
                'error': str(e),
            }
