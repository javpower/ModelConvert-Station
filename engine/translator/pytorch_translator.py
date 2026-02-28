#!/usr/bin/env python3
"""
PyTorch to ONNX Translator

Handles conversion of PyTorch models (.pt, .pth) to ONNX format.
Supports both scripted and traced models, with automatic shape inference.
"""

import logging
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

import torch
import torch.onnx
import onnx

logger = logging.getLogger('ModelConvert.PyTorch')


@dataclass
class PyTorchConfig:
    """Configuration for PyTorch to ONNX conversion."""
    input_shape: Optional[List[int]] = None
    input_names: List[str] = None
    output_names: List[str] = None
    opset_version: int = 17
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None
    do_constant_folding: bool = True
    export_params: bool = True
    
    def __post_init__(self):
        if self.input_names is None:
            self.input_names = ['input']
        if self.output_names is None:
            self.output_names = ['output']


class PyTorchTranslator:
    """Translator for PyTorch models to ONNX."""
    
    def __init__(self):
        self.supported_extensions = ['.pt', '.pth', '.torch']
    
    async def convert(
        self,
        input_path: Path,
        output_path: Path,
        custom_args: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Convert PyTorch model to ONNX.
        
        Args:
            input_path: Path to PyTorch model file
            output_path: Path for output ONNX file
            custom_args: Optional configuration overrides
        
        Returns:
            Dictionary with conversion results
        """
        
        custom_args = custom_args or {}
        config = PyTorchConfig(
            input_shape=custom_args.get('input_shape'),
            input_names=custom_args.get('input_names', ['input']),
            output_names=custom_args.get('output_names', ['output']),
            opset_version=custom_args.get('opset_version', 17),
            dynamic_axes=custom_args.get('dynamic_axes'),
            do_constant_folding=custom_args.get('do_constant_folding', True),
            export_params=custom_args.get('export_params', True),
        )
        
        logger.info(f"🔄 Converting PyTorch model: {input_path.name}")
        
        # Load model
        model = self._load_model(input_path)
        
        # Determine input shape
        if not config.input_shape:
            config.input_shape = self._infer_input_shape(model, custom_args)
        
        logger.info(f"📐 Input shape: {config.input_shape}")
        
        # Create dummy input
        dummy_input = torch.randn(config.input_shape)
        
        # Handle GPU models on CPU environment
        if torch.cuda.is_available():
            dummy_input = dummy_input.cuda()
            model = model.cuda()
        
        model.eval()
        
        # Export to ONNX
        try:
            torch.onnx.export(
                model,
                dummy_input,
                str(output_path),
                input_names=config.input_names,
                output_names=config.output_names,
                opset_version=config.opset_version,
                dynamic_axes=config.dynamic_axes,
                do_constant_folding=config.do_constant_folding,
                export_params=config.export_params,
            )
            
            logger.info(f"✅ PyTorch conversion successful: {output_path.name}")
            
            return {
                'status': 'success',
                'output_path': str(output_path),
                'input_shape': config.input_shape,
                'opset_version': config.opset_version,
            }
            
        except Exception as e:
            logger.error(f"❌ PyTorch export failed: {e}")
            raise
    
    def _load_model(self, input_path: Path) -> torch.nn.Module:
        """Load PyTorch model from file."""
        
        try:
            # Try loading as a complete model
            model = torch.load(input_path, map_location='cpu', weights_only=False)
            
            # If it's a dict with 'model' key (common checkpoint format)
            if isinstance(model, dict):
                if 'model' in model:
                    model = model['model']
                elif 'state_dict' in model:
                    model = model['state_dict']
                elif 'model_state_dict' in model:
                    model = model['model_state_dict']
            
            # If it's still a dict, it's likely a state_dict
            if isinstance(model, dict):
                logger.info("📋 Detected state_dict, attempting to reconstruct model...")
                # We need model architecture - try common patterns
                raise ValueError(
                    "Checkpoint contains only state_dict. "
                    "Please provide the complete model or specify architecture in custom_args."
                )
            
            if not isinstance(model, torch.nn.Module):
                raise ValueError(f"Loaded object is not a torch.nn.Module: {type(model)}")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load PyTorch model: {e}")
            raise
    
    def _infer_input_shape(
        self,
        model: torch.nn.Module,
        custom_args: Dict[str, Any]
    ) -> List[int]:
        """Infer input shape from model structure or custom args."""
        
        # Check if shape is explicitly provided
        if 'input_shape' in custom_args:
            return custom_args['input_shape']
        
        # Try to infer from first layer
        for module in model.modules():
            if isinstance(module, torch.nn.Conv2d):
                # Assume standard image input [N, C, H, W]
                return [1, module.in_channels, 224, 224]
            elif isinstance(module, torch.nn.Linear):
                # Assume flattened input
                return [1, module.in_features]
            elif isinstance(module, torch.nn.Conv1d):
                return [1, module.in_channels, 224]
            elif isinstance(module, torch.nn.Conv3d):
                return [1, module.in_channels, 16, 224, 224]
        
        # Default shapes for common model types
        model_name = model.__class__.__name__.lower()
        if 'resnet' in model_name or 'vgg' in model_name or 'efficientnet' in model_name:
            return [1, 3, 224, 224]
        elif 'transformer' in model_name or 'bert' in model_name:
            return [1, 128]  # [batch, seq_len]
        elif 'lstm' in model_name or 'gru' in model_name or 'rnn' in model_name:
            return [1, 10, 128]  # [batch, seq_len, features]
        
        # Ultimate fallback
        logger.warning("⚠️ Could not infer input shape, using default [1, 3, 224, 224]")
        return [1, 3, 224, 224]
    
    async def convert_with_tracing(
        self,
        model: torch.nn.Module,
        example_inputs: Tuple[torch.Tensor, ...],
        output_path: Path,
        config: PyTorchConfig
    ) -> Dict[str, Any]:
        """
        Convert model using torch.jit.trace (more reliable for some models).
        """
        
        logger.info("🔄 Using torch.jit.trace for conversion...")
        
        traced_model = torch.jit.trace(model, example_inputs)
        
        torch.onnx.export(
            traced_model,
            example_inputs,
            str(output_path),
            input_names=config.input_names,
            output_names=config.output_names,
            opset_version=config.opset_version,
            dynamic_axes=config.dynamic_axes,
        )
        
        return {
            'status': 'success',
            'method': 'trace',
            'output_path': str(output_path),
        }
    
    async def convert_with_scripting(
        self,
        model: torch.nn.Module,
        output_path: Path,
        config: PyTorchConfig
    ) -> Dict[str, Any]:
        """
        Convert model using torch.jit.script (for control flow).
        """
        
        logger.info("🔄 Using torch.jit.script for conversion...")
        
        scripted_model = torch.jit.script(model)
        
        # For scripted models, we need example inputs
        dummy_input = torch.randn(config.input_shape or [1, 3, 224, 224])
        
        torch.onnx.export(
            scripted_model,
            dummy_input,
            str(output_path),
            input_names=config.input_names,
            output_names=config.output_names,
            opset_version=config.opset_version,
            dynamic_axes=config.dynamic_axes,
        )
        
        return {
            'status': 'success',
            'method': 'script',
            'output_path': str(output_path),
        }
    
    async def validate_onnx(self, onnx_path: Path) -> Dict[str, Any]:
        """Validate the generated ONNX model."""
        
        try:
            model = onnx.load(str(onnx_path))
            onnx.checker.check_model(model)
            
            # Try running with onnxruntime
            try:
                import onnxruntime as ort
                session = ort.InferenceSession(str(onnx_path))
                
                input_info = []
                for inp in session.get_inputs():
                    input_info.append({
                        'name': inp.name,
                        'shape': inp.shape,
                        'type': inp.type,
                    })
                
                output_info = []
                for out in session.get_outputs():
                    output_info.append({
                        'name': out.name,
                        'shape': out.shape,
                        'type': out.type,
                    })
                
                return {
                    'valid': True,
                    'onnxruntime_compatible': True,
                    'inputs': input_info,
                    'outputs': output_info,
                }
                
            except ImportError:
                return {
                    'valid': True,
                    'onnxruntime_compatible': None,
                    'message': 'onnxruntime not available for validation',
                }
                
        except Exception as e:
            return {
                'valid': False,
                'error': str(e),
            }
