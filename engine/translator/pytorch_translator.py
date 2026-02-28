#!/usr/bin/env python3
"""
PyTorch to ONNX Translator

Handles conversion of PyTorch models (.pt, .pth) to ONNX format.
Supports both scripted and traced models, with automatic shape inference.
"""

import logging
import tempfile
import copy
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

import torch
import torch.onnx
import onnx

logger = logging.getLogger('ModelConvert.PyTorch')


class LoFTRWrapper(torch.nn.Module):
    """
    Wrapper for LoFTR model to make it ONNX-exportable.
    Based on https://github.com/oooooha/loftr2onnx
    
    The original LoFTR uses a dict-based input/output which is not compatible
    with ONNX export. This wrapper converts it to tensor-based input/output.
    """
    
    def __init__(self, loftr_model):
        super().__init__()
        self.model = loftr_model
        
    def forward(self, image0: torch.Tensor, image1: torch.Tensor):
        """
        Forward pass compatible with ONNX export.
        
        Args:
            image0: First image tensor [B, 1, H, W]
            image1: Second image tensor [B, 1, H, W]
            
        Returns:
            Tuple of (keypoints0, keypoints1, confidence)
        """
        from einops.einops import rearrange
        
        data = {
            "image0": image0,
            "image1": image1,
            "bs": image0.size(0),
            "hw0_i": image0.shape[2:],
            "hw1_i": image1.shape[2:],
        }
        
        # Backbone feature extraction
        if data["hw0_i"] == data["hw1_i"]:
            feats_c, feats_f = self.model.backbone(
                torch.cat([data["image0"], data["image1"]], dim=0)
            )
            (feat_c0, feat_c1), (feat_f0, feat_f1) = feats_c.split(data["bs"]), feats_f.split(data["bs"])
        else:
            (feat_c0, feat_f0), (feat_c1, feat_f1) = self.model.backbone(data["image0"]), self.model.backbone(data["image1"])
        
        data.update({
            "hw0_c": feat_c0.shape[2:],
            "hw1_c": feat_c1.shape[2:],
            "hw0_f": feat_f0.shape[2:],
            "hw1_f": feat_f1.shape[2:],
        })
        
        # Coarse-level LoFTR module
        feat_c0 = rearrange(self.model.pos_encoding(feat_c0), "n c h w -> n (h w) c")
        feat_c1 = rearrange(self.model.pos_encoding(feat_c1), "n c h w -> n (h w) c")
        
        feat_c0, feat_c1 = self.model.loftr_coarse(feat_c0, feat_c1, None, None)
        
        # Match coarse-level
        self.model.coarse_matching(feat_c0, feat_c1, data, mask_c0=None, mask_c1=None)
        
        # Fine-level refinement
        feat_f0_unfold, feat_f1_unfold = self.model.fine_preprocess(feat_f0, feat_f1, feat_c0, feat_c1, data)
        
        if feat_f0_unfold.size(0) != 0:
            feat_f0_unfold, feat_f1_unfold = self.model.loftr_fine(feat_f0_unfold, feat_f1_unfold)
        
        # Match fine-level
        self.model.fine_matching(feat_f0_unfold, feat_f1_unfold, data)
        
        # Return as tuple for ONNX compatibility
        return data["mkpts0_f"], data["mkpts1_f"], data["mconf"]


@dataclass
class PyTorchConfig:
    """Configuration for PyTorch to ONNX conversion."""
    input_shape: Optional[List[int]] = None
    input_shapes: Optional[List[List[int]]] = None  # Support multiple inputs
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
        # Convert string keys to int keys in dynamic_axes (from JSON)
        if self.dynamic_axes is not None:
            converted_axes = {}
            for input_name, axes in self.dynamic_axes.items():
                converted_axes[input_name] = {int(k): v for k, v in axes.items()}
            self.dynamic_axes = converted_axes
    
    def get_input_shapes(self) -> List[List[int]]:
        """Get list of input shapes, supporting both single and multiple inputs."""
        if self.input_shapes is not None:
            return self.input_shapes
        if self.input_shape is not None:
            return [self.input_shape]
        return [[1, 3, 224, 224]]  # default fallback


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
            input_shapes=custom_args.get('input_shapes'),
            input_names=custom_args.get('input_names', ['input']),
            output_names=custom_args.get('output_names', ['output']),
            opset_version=custom_args.get('opset_version', 17),
            dynamic_axes=custom_args.get('dynamic_axes'),
            do_constant_folding=custom_args.get('do_constant_folding', True),
            export_params=custom_args.get('export_params', True),
        )
        
        logger.info(f"🔄 Converting PyTorch model: {input_path.name}")
        
        # Load model
        model = self._load_model(input_path, custom_args)
        
        # Determine input shapes (support multiple inputs)
        input_shapes = config.get_input_shapes()
        if not input_shapes or input_shapes == [[1, 3, 224, 224]]:
            # Try to infer from model
            inferred_shape = self._infer_input_shape(model, custom_args)
            input_shapes = [inferred_shape]
        
        logger.info(f"📐 Input shapes: {input_shapes}")
        logger.info(f"📐 Input names: {config.input_names}")
        
        # Create dummy inputs (support multiple inputs)
        dummy_inputs = tuple(torch.randn(shape) for shape in input_shapes)
        if len(dummy_inputs) == 1:
            dummy_input = dummy_inputs[0]
        else:
            dummy_input = dummy_inputs
        
        # Handle GPU models on CPU environment
        if torch.cuda.is_available():
            if isinstance(dummy_input, tuple):
                dummy_input = tuple(d.cuda() for d in dummy_input)
            else:
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
                'input_shape': input_shapes[0] if len(input_shapes) == 1 else input_shapes,
                'input_shapes': input_shapes,
                'opset_version': config.opset_version,
            }
            
        except Exception as e:
            logger.error(f"❌ PyTorch export failed: {e}")
            raise
    
    def _load_model(self, input_path: Path, custom_args: Optional[Dict[str, Any]] = None) -> torch.nn.Module:
        """Load PyTorch model from file."""
        
        custom_args = custom_args or {}
        
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
                # Try to reconstruct model from torchvision
                # First check if model_architecture is specified in custom_args
                model_arch = custom_args.get('model_architecture')
                reconstructed_model = self._reconstruct_model_from_state_dict(model, input_path, model_arch)
                if reconstructed_model is not None:
                    return reconstructed_model
                raise ValueError(
                    "Checkpoint contains only state_dict. "
                    "Could not auto-detect model architecture. "
                    "Please provide the complete model or specify 'model_architecture' in custom_args."
                )
            
            if not isinstance(model, torch.nn.Module):
                raise ValueError(f"Loaded object is not a torch.nn.Module: {type(model)}")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load PyTorch model: {e}")
            raise
    
    def _reconstruct_model_from_state_dict(
        self,
        state_dict: dict,
        input_path: Path,
        model_arch: Optional[str] = None
    ) -> Optional[torch.nn.Module]:
        """Try to reconstruct model from state_dict using torchvision or custom architectures."""
        
        # Get model name from custom_args or file path
        if model_arch:
            model_name = model_arch.lower()
            logger.info(f"📋 Using specified model architecture: {model_arch}")
        else:
            model_name = input_path.stem.lower()
        
        # Try LOFTR first (special handling for feature matching models)
        if 'loftr' in model_name:
            return self._reconstruct_loftr_model(state_dict)
        
        # Try torchvision models
        try:
            import torchvision.models as models
            
            # Map common model names to torchvision models
            model_mapping = {
                'resnet18': models.resnet18,
                'resnet34': models.resnet34,
                'resnet50': models.resnet50,
                'resnet101': models.resnet101,
                'resnet152': models.resnet152,
                'alexnet': models.alexnet,
                'vgg11': models.vgg11,
                'vgg13': models.vgg13,
                'vgg16': models.vgg16,
                'vgg19': models.vgg19,
                'densenet121': models.densenet121,
                'densenet161': models.densenet161,
                'densenet169': models.densenet169,
                'densenet201': models.densenet201,
                'mobilenet_v2': models.mobilenet_v2,
                'mobilenet_v3_small': models.mobilenet_v3_small,
                'mobilenet_v3_large': models.mobilenet_v3_large,
                'efficientnet_b0': models.efficientnet_b0,
                'efficientnet_b1': models.efficientnet_b1,
                'efficientnet_b2': models.efficientnet_b2,
                'efficientnet_b3': models.efficientnet_b3,
                'efficientnet_b4': models.efficientnet_b4,
                'efficientnet_b5': models.efficientnet_b5,
                'efficientnet_b6': models.efficientnet_b6,
                'efficientnet_b7': models.efficientnet_b7,
                'squeezenet1_0': models.squeezenet1_0,
                'squeezenet1_1': models.squeezenet1_1,
                'shufflenet_v2_x0_5': models.shufflenet_v2_x0_5,
                'shufflenet_v2_x1_0': models.shufflenet_v2_x1_0,
                'shufflenet_v2_x1_5': models.shufflenet_v2_x1_5,
                'shufflenet_v2_x2_0': models.shufflenet_v2_x2_0,
            }
            
            # Find matching model
            matched_model = None
            for name, model_fn in model_mapping.items():
                if name in model_name:
                    logger.info(f"🔍 Auto-detected model architecture: {name}")
                    matched_model = model_fn(weights=None)
                    break
            
            if matched_model is None:
                logger.warning(f"⚠️ Could not auto-detect model architecture from '{model_name}'")
                return None
            
            # Load state_dict into model
            matched_model.load_state_dict(state_dict, strict=False)
            logger.info(f"✅ Successfully loaded state_dict into {matched_model.__class__.__name__}")
            
            return matched_model
            
        except ImportError:
            logger.warning("⚠️ torchvision not available for model reconstruction")
            return None
        except Exception as e:
            logger.warning(f"⚠️ Failed to reconstruct model: {e}")
            return None
    
    def _reconstruct_loftr_model(self, state_dict: dict) -> Optional[torch.nn.Module]:
        """Reconstruct LOFTR model from state_dict with ONNX-compatible wrapper."""
        
        try:
            logger.info("🔍 Attempting to reconstruct LOFTR model...")
            
            # Try importing from kornia (recommended way)
            try:
                from kornia.feature import LoFTR
                logger.info("📦 Using kornia.feature.LoFTR")
                
                # Determine model type from state_dict keys
                # LOFTR-DS (Dual Softmax) vs LOFTR-OT (Optimal Transport)
                is_ot = any('fsrc' in k or 'fref' in k for k in state_dict.keys())
                
                if is_ot:
                    logger.info("📋 Detected LOFTR-OT (Optimal Transport) variant")
                    model = LoFTR(pretrained=None, config={'match_coarse': {'thr': 0.2}})
                else:
                    logger.info("📋 Detected LOFTR-DS (Dual Softmax) variant")
                    model = LoFTR(pretrained=None)
                
                # Load state_dict (handle both direct and nested formats)
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                
                model.load_state_dict(state_dict, strict=False)
                logger.info("✅ Successfully loaded LOFTR state_dict")
                
                # Wrap with ONNX-compatible wrapper
                logger.info("🔄 Wrapping LOFTR with ONNX-compatible wrapper...")
                wrapped_model = LoFTRWrapper(model)
                return wrapped_model
                
            except ImportError:
                logger.warning("⚠️ kornia not available, trying local LoFTR implementation...")
            
            # Fallback: Try importing from local LoFTR package
            try:
                # Try the official LoFTR repo structure
                import sys
                sys.path.insert(0, str(Path.cwd()))
                
                from src.loftr import LoFTR as LocalLoFTR, default_cfg
                logger.info("📦 Using local src.loftr.LoFTR")
                
                # Fix temp_bug_fix for coarse matching
                _cfg = copy.deepcopy(default_cfg)
                _cfg["coarse"]["temp_bug_fix"] = True
                model = LocalLoFTR(config=_cfg)
                
                # Handle checkpoint format
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                elif 'model' in state_dict:
                    state_dict = state_dict['model']
                
                model.load_state_dict(state_dict, strict=False)
                logger.info("✅ Successfully loaded LOFTR state_dict from local implementation")
                
                # Wrap with ONNX-compatible wrapper
                logger.info("🔄 Wrapping LOFTR with ONNX-compatible wrapper...")
                wrapped_model = LoFTRWrapper(model)
                return wrapped_model
                
            except ImportError:
                logger.warning("⚠️ Local LoFTR implementation not found")
                pass
            
            logger.error("❌ Could not reconstruct LOFTR model. Please install kornia: pip install kornia")
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to reconstruct LOFTR model: {e}")
            return None
    
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
            do_constant_folding=config.do_constant_folding,
        )
        
        input_shapes = [list(t.shape) for t in example_inputs] if isinstance(example_inputs, tuple) else [list(example_inputs.shape)]
        
        return {
            'status': 'success',
            'method': 'trace',
            'output_path': str(output_path),
            'input_shapes': input_shapes,
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
        
        # For scripted models, we need example inputs (support multiple inputs)
        input_shapes = config.get_input_shapes()
        if len(input_shapes) == 1:
            dummy_input = torch.randn(input_shapes[0])
        else:
            dummy_input = tuple(torch.randn(shape) for shape in input_shapes)
        
        torch.onnx.export(
            scripted_model,
            dummy_input,
            str(output_path),
            input_names=config.input_names,
            output_names=config.output_names,
            opset_version=config.opset_version,
            dynamic_axes=config.dynamic_axes,
            do_constant_folding=config.do_constant_folding,
        )
        
        return {
            'status': 'success',
            'method': 'script',
            'output_path': str(output_path),
            'input_shapes': input_shapes,
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
