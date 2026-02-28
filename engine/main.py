#!/usr/bin/env python3
"""
ModelConvert-Station Core Engine
URL-Driven Model Conversion Pipeline

This module serves as the central task scheduler and orchestrator for the
model conversion workflow. It handles:
- Task ingestion and validation
- Stream downloading from URLs
- Framework detection and routing
- Conversion execution with exception handling
- Artifact generation and packaging
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from urllib.parse import urlparse

import aiohttp
import aiofiles

# Import translators
from translator.pytorch_translator import PyTorchTranslator
from translator.tensorflow_translator import TensorFlowTranslator
from translator.tflite_translator import TFLiteTranslator
from translator.mediapipe_translator import MediaPipeTranslator
from optimizer import ONNXOptimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('ModelConvert')


@dataclass
class ConversionTask:
    """Represents a single model conversion task."""
    id: str
    source_url: str
    source_framework: Optional[str] = None  # Auto-detect if None
    target_format: str = 'onnx'
    optimization_level: int = 2
    generate_java_template: bool = True
    custom_args: Dict[str, Any] = field(default_factory=dict)
    status: str = 'pending'
    error_message: Optional[str] = None
    output_files: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class StreamDownloader:
    """Async stream downloader supporting various URL types."""
    
    def __init__(self, chunk_size: int = 8192):
        self.chunk_size = chunk_size
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=3600, connect=30)
        self.session = aiohttp.ClientSession(timeout=timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def download(self, url: str, output_path: Path, progress_callback: Optional[Callable] = None) -> Path:
        """Download file from URL with progress tracking."""
        
        # Handle Google Drive URLs
        if 'drive.google.com' in url or 'drive.usercontent.google.com' in url:
            return await self._download_gdrive(url, output_path, progress_callback)
        
        logger.info(f"📥 Downloading: {url}")
        
        async with self.session.get(url, allow_redirects=True) as response:
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            async with aiofiles.open(output_path, 'wb') as f:
                async for chunk in response.content.iter_chunked(self.chunk_size):
                    await f.write(chunk)
                    downloaded += len(chunk)
                    
                    if progress_callback and total_size > 0:
                        progress = (downloaded / total_size) * 100
                        progress_callback(downloaded, total_size, progress)
        
        logger.info(f"✅ Downloaded: {output_path.name} ({downloaded / 1024 / 1024:.2f} MB)")
        return output_path
    
    async def _download_gdrive(self, url: str, output_path: Path, progress_callback: Optional[Callable] = None) -> Path:
        """Handle Google Drive download with confirmation token."""
        import re
        
        # Extract file ID
        file_id_match = re.search(r'(?:id=|/d/|/file/d/)([\w-]+)', url)
        if not file_id_match:
            raise ValueError(f"Cannot extract Google Drive file ID from: {url}")
        
        file_id = file_id_match.group(1)
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        
        logger.info(f"📥 Downloading from Google Drive: {file_id}")
        
        # First request to get confirmation token
        async with self.session.get(download_url) as response:
            if 'confirm' in str(response.url):
                # Large file confirmation needed
                confirm_token = None
                for key, value in response.cookies.items():
                    if key.startswith('download_warning'):
                        confirm_token = value.value
                        break
                
                if confirm_token:
                    download_url = f"{download_url}&confirm={confirm_token}"
        
        # Actual download
        async with self.session.get(download_url) as response:
            response.raise_for_status()
            
            async with aiofiles.open(output_path, 'wb') as f:
                async for chunk in response.content.iter_chunked(self.chunk_size):
                    await f.write(chunk)
        
        logger.info(f"✅ Downloaded from Google Drive: {output_path.name}")
        return output_path
    
    async def sniff_framework(self, file_path: Path) -> Optional[str]:
        """Detect framework from file magic bytes and headers."""
        
        # Read first 8KB for magic byte detection
        async with aiofiles.open(file_path, 'rb') as f:
            header = await f.read(8192)
        
        # Magic byte signatures
        signatures = {
            b'PK\x03\x04': 'zip',  # ZIP-based formats (PyTorch, MediaPipe)
            b'\x89HDF': 'hdf5',    # Keras/TF HDF5
            b'TFL3': 'tflite',     # TFLite flatbuffer
            b'\x08\x00\x00\x00': 'tflite_alt',  # Alternative TFLite
        }
        
        detected_format = None
        for magic, fmt in signatures.items():
            if header.startswith(magic):
                detected_format = fmt
                break
        
        # Deep inspection for ZIP-based formats
        if detected_format == 'zip':
            return await self._inspect_zip_contents(file_path)
        
        # Check for SavedModel directory indicator
        if b'saved_model.pb' in header or (file_path.suffix == '.pb' and b'node' in header):
            return 'tensorflow'
        
        if detected_format in ['tflite', 'tflite_alt']:
            return 'tflite'
        
        # Fallback to extension-based detection
        ext = file_path.suffix.lower()
        ext_map = {
            '.pt': 'pytorch',
            '.pth': 'pytorch',
            '.torch': 'pytorch',
            '.h5': 'keras',
            '.keras': 'keras',
            '.tflite': 'tflite',
            '.lite': 'tflite',
            '.task': 'mediapipe',
            '.pb': 'tensorflow',
        }
        
        return ext_map.get(ext)
    
    async def _inspect_zip_contents(self, file_path: Path) -> Optional[str]:
        """Inspect ZIP archive to determine framework type."""
        import zipfile
        
        try:
            with zipfile.ZipFile(file_path, 'r') as zf:
                namelist = zf.namelist()
                
                # PyTorch model files
                if 'data.pkl' in namelist or any('torch' in name for name in namelist):
                    return 'pytorch'
                
                # MediaPipe task files
                if any(name.endswith('.tflite') for name in namelist) and 'model_metadata.json' in namelist:
                    return 'mediapipe'
                
                # General ZIP with TFLite
                if any(name.endswith('.tflite') for name in namelist):
                    return 'tflite'
                
        except zipfile.BadZipFile:
            pass
        
        return None


class ConversionEngine:
    """Core conversion engine managing task execution."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize translators
        self.translators = {
            'pytorch': PyTorchTranslator(),
            'tensorflow': TensorFlowTranslator(),
            'keras': TensorFlowTranslator(),
            'tflite': TFLiteTranslator(),
            'mediapipe': MediaPipeTranslator(),
        }
        
        # Initialize optimizer
        self.optimizer = ONNXOptimizer()
        
        # Statistics
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'skipped': 0,
        }
    
    async def process_tasks(self, tasks: List[ConversionTask]) -> List[ConversionTask]:
        """Process multiple conversion tasks with resource-aware scheduling."""
        
        self.stats['total'] = len(tasks)
        logger.info(f"🎯 Processing {len(tasks)} conversion task(s)")
        
        # Separate TensorFlow-based tasks (TFLite, MediaPipe) from others
        # to avoid memory pressure and segfaults from concurrent tf2onnx conversions
        tf_tasks = [t for t in tasks if t.source_framework in ('tflite', 'mediapipe')]
        other_tasks = [t for t in tasks if t.source_framework not in ('tflite', 'mediapipe')]
        
        results = []
        
        # Process non-TF tasks concurrently with limited parallelism
        if other_tasks:
            semaphore = asyncio.Semaphore(2)
            async def process_with_limit(task: ConversionTask) -> ConversionTask:
                async with semaphore:
                    return await self._process_single_task(task)
            
            other_results = await asyncio.gather(*[process_with_limit(t) for t in other_tasks])
            results.extend(other_results)
        
        # Process TF-based tasks sequentially to prevent segfaults
        # tf2onnx can crash when multiple conversions run simultaneously
        if tf_tasks:
            logger.info(f"🔄 Processing {len(tf_tasks)} TensorFlow-based task(s) sequentially")
            for task in tf_tasks:
                result = await self._process_single_task(task)
                results.append(result)
                # Small delay to allow memory cleanup between conversions
                await asyncio.sleep(0.5)
        
        # Print summary
        logger.info("=" * 50)
        logger.info("📊 Conversion Summary:")
        logger.info(f"   Total:   {self.stats['total']}")
        logger.info(f"   Success: {self.stats['success']} ✅")
        logger.info(f"   Failed:  {self.stats['failed']} ❌")
        logger.info(f"   Skipped: {self.stats['skipped']} ⏭️")
        logger.info("=" * 50)
        
        return results
    
    async def _process_single_task(self, task: ConversionTask) -> ConversionTask:
        """Process a single conversion task end-to-end."""
        
        task_id = task.id or hashlib.md5(task.source_url.encode()).hexdigest()[:8]
        logger.info(f"[{task_id}] 🚀 Starting conversion task")
        
        temp_dir = Path(tempfile.mkdtemp(prefix=f"convert_{task_id}_"))
        
        try:
            async with StreamDownloader() as downloader:
                # Step 1: Download model
                download_path = temp_dir / 'model_download'
                await downloader.download(task.source_url, download_path)
                
                # Step 2: Auto-detect framework if not specified
                if not task.source_framework:
                    task.source_framework = await downloader.sniff_framework(download_path)
                    logger.info(f"[{task_id}] 🔍 Auto-detected framework: {task.source_framework}")
                
                if not task.source_framework:
                    raise ValueError(f"Cannot detect framework for: {task.source_url}")
                
                if task.source_framework not in self.translators:
                    raise ValueError(f"Unsupported framework: {task.source_framework}")
                
                # Step 3: Create task output directory
                task_output_dir = self.output_dir / task_id
                task_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Step 4: Execute conversion
                translator = self.translators[task.source_framework]
                
                onnx_path = task_output_dir / f"{task_id}.onnx"
                
                conversion_result = await translator.convert(
                    input_path=download_path,
                    output_path=onnx_path,
                    custom_args=task.custom_args
                )
                
                # Step 5: Optimize ONNX (if enabled)
                if task.optimization_level > 0:
                    logger.info(f"[{task_id}] ⚡ Optimizing ONNX (level {task.optimization_level})")
                    
                    # Handle MediaPipe multi-model output
                    if task.source_framework == 'mediapipe':
                        final_onnx_paths = []
                        for model_info in conversion_result.get('converted_models', []):
                            if model_info['status'] == 'success':
                                model_path = Path(model_info['output_path'])
                                optimized_path = model_path.parent / f"{model_path.stem}_optimized.onnx"
                                
                                await self.optimizer.simplify(
                                    input_path=model_path,
                                    output_path=optimized_path,
                                    level=task.optimization_level
                                )
                                final_onnx_paths.append(optimized_path)
                        
                        final_onnx_path = final_onnx_paths[0] if final_onnx_paths else None
                    else:
                        optimized_path = task_output_dir / f"{task_id}_optimized.onnx"
                        
                        await self.optimizer.simplify(
                            input_path=Path(conversion_result['output_path']),
                            output_path=optimized_path,
                            level=task.optimization_level
                        )
                        
                        final_onnx_path = optimized_path
                else:
                    if task.source_framework == 'mediapipe':
                        # Get first successful model's path
                        converted_models = conversion_result.get('converted_models', [])
                        successful_models = [m for m in converted_models if m['status'] == 'success']
                        final_onnx_path = Path(successful_models[0]['output_path']) if successful_models else None
                    else:
                        final_onnx_path = conversion_result['output_path']
                
                # Step 6: Extract metadata
                metadata = await self._extract_metadata(final_onnx_path)
                metadata.update({
                    'task_id': task_id,
                    'source_url': task.source_url,
                    'source_framework': task.source_framework,
                    'conversion_time': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                })
                
                # Save metadata
                metadata_path = task_output_dir / 'metadata.json'
                async with aiofiles.open(metadata_path, 'w') as f:
                    await f.write(json.dumps(metadata, indent=2))
                
                # Step 7: Generate Java template (if enabled)
                if task.generate_java_template:
                    java_path = task_output_dir / 'Inference.java'
                    await self._generate_java_template(metadata, java_path)
                
                # Update task status
                task.status = 'success'
                task.output_files = [str(f) for f in task_output_dir.glob('*')]
                task.metadata = metadata
                self.stats['success'] += 1
                
                logger.info(f"[{task_id}] ✅ Conversion completed successfully")
                
        except Exception as e:
            task.status = 'failed'
            task.error_message = str(e)
            self.stats['failed'] += 1
            logger.error(f"[{task_id}] ❌ Conversion failed: {e}")
            
        finally:
            # Cleanup temp directory
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        return task
    
    async def _extract_metadata(self, onnx_path: Path) -> Dict[str, Any]:
        """Extract metadata from ONNX model."""
        import onnx
        
        model = onnx.load(str(onnx_path))
        
        inputs = []
        for inp in model.graph.input:
            shape = []
            for d in inp.type.tensor_type.shape.dim:
                shape.append(d.dim_value if d.dim_value else d.dim_param)
            
            inputs.append({
                'name': inp.name,
                'shape': shape,
                'dtype': onnx.TensorProto.DataType.Name(inp.type.tensor_type.elem_type)
            })
        
        outputs = []
        for out in model.graph.output:
            shape = []
            for d in out.type.tensor_type.shape.dim:
                shape.append(d.dim_value if d.dim_value else d.dim_param)
            
            outputs.append({
                'name': out.name,
                'shape': shape,
                'dtype': onnx.TensorProto.DataType.Name(out.type.tensor_type.elem_type)
            })
        
        # Count operators
        op_types = set()
        for node in model.graph.node:
            op_types.add(node.op_type)
        
        return {
            'inputs': inputs,
            'outputs': outputs,
            'op_types': sorted(list(op_types)),
            'op_count': len(model.graph.node),
            'ir_version': model.ir_version,
            'producer_name': model.producer_name or 'unknown',
            'file_size_bytes': onnx_path.stat().st_size,
        }
    
    async def _generate_java_template(self, metadata: Dict[str, Any], output_path: Path) -> None:
        """Generate Java inference template using Jinja2."""
        from jinja2 import Template
        
        template_str = '''// Auto-generated by ModelConvert-Station
// Generated at: {{ metadata.conversion_time }}
// Source: {{ metadata.source_url }}

import ai.onnxruntime.*;
import java.util.*;

public class Inference {
    
    private final OrtEnvironment environment;
    private final OrtSession session;
    
    public Inference(String modelPath) throws OrtException {
        this.environment = OrtEnvironment.getEnvironment();
        this.session = environment.createSession(modelPath, new OrtSession.SessionOptions());
    }
    
    /**
     * Run inference on the model.
     {% for input in metadata.inputs %}
     * @param input_{{ loop.index }} Input tensor with shape {{ input.shape }} ({{ input.dtype }})
     {%- endfor %}
     {% for output in metadata.outputs %}
     * @return Output tensor with shape {{ output.shape }} ({{ output.dtype }})
     {%- endfor %}
     */
    public Map<String, OnnxTensor> runInference(
        {%- for input in metadata.inputs %}
        float[] input_{{ loop.index }}{% if not loop.last %},{% endif %}
        {%- endfor %}
    ) throws OrtException {
        
        // Prepare input tensors
        Map<String, OnnxTensor> inputs = new HashMap<>();
        {%- for input in metadata.inputs %}
        long[] shape_{{ loop.index }} = {{ '{' }}{% for dim in input.shape %}{{ dim }}{% if not loop.last %}, {% endif %}{% endfor %}{{ '}' }};
        OnnxTensor tensor_{{ loop.index }} = OnnxTensor.createTensor(environment, 
            new float[][][]{input_{{ loop.index }}});
        inputs.put("{{ input.name }}", tensor_{{ loop.index }});
        {%- endfor %}
        
        // Run inference
        OrtSession.Result results = session.run(inputs);
        
        // Extract outputs
        Map<String, OnnxTensor> outputs = new HashMap<>();
        {%- for output in metadata.outputs %}
        outputs.put("{{ output.name }}", (OnnxTensor) results.get("{{ output.name }}"));
        {%- endfor %}
        
        return outputs;
    }
    
    /**
     * Get model metadata.
     */
    public void printModelInfo() {
        System.out.println("=== Model Information ===");
        System.out.println("Inputs:");
        {%- for input in metadata.inputs %}
        System.out.println("  - {{ input.name }}: {{ input.shape }} ({{ input.dtype }})");
        {%- endfor %}
        System.out.println("Outputs:");
        {%- for output in metadata.outputs %}
        System.out.println("  - {{ output.name }}: {{ output.shape }} ({{ output.dtype }})");
        {%- endfor %}
    }
    
    public void close() throws OrtException {
        session.close();
        environment.close();
    }
    
    public static void main(String[] args) {
        try {
            Inference inference = new Inference("model.onnx");
            inference.printModelInfo();
            
            // Example usage:
            // float[] inputData = new float[...]; // Fill with your data
            // Map<String, OnnxTensor> results = inference.runInference(inputData);
            
            inference.close();
        } catch (OrtException e) {
            e.printStackTrace();
        }
    }
}
'''
        
        template = Template(template_str)
        java_code = template.render(metadata=metadata)
        
        async with aiofiles.open(output_path, 'w') as f:
            await f.write(java_code)
        
        logger.info(f"📝 Generated Java template: {output_path.name}")


def load_tasks(tasks_path: Path) -> List[ConversionTask]:
    """Load tasks from JSON file."""
    with open(tasks_path, 'r') as f:
        data = json.load(f)
    
    tasks = []
    for task_data in data.get('tasks', []):
        tasks.append(ConversionTask(
            id=task_data.get('id'),
            source_url=task_data['source_url'],
            source_framework=task_data.get('source_framework'),
            target_format=task_data.get('target_format', 'onnx'),
            optimization_level=task_data.get('optimization_level', 2),
            generate_java_template=task_data.get('generate_java_template', True),
            custom_args=task_data.get('custom_args', {})
        ))
    
    return tasks


async def main():
    parser = argparse.ArgumentParser(description='ModelConvert-Station Core Engine')
    parser.add_argument('--tasks', required=True, help='Path to tasks.json')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--task-index', help='Specific task index to process')
    
    args = parser.parse_args()
    
    tasks_path = Path(args.tasks)
    output_dir = Path(args.output)
    
    if not tasks_path.exists():
        logger.error(f"Tasks file not found: {tasks_path}")
        sys.exit(1)
    
    # Load tasks
    tasks = load_tasks(tasks_path)
    
    # Filter by index if specified
    if args.task_index:
        idx = int(args.task_index)
        if 0 <= idx < len(tasks):
            tasks = [tasks[idx]]
        else:
            logger.error(f"Invalid task index: {idx}")
            sys.exit(1)
    
    # Process tasks
    engine = ConversionEngine(output_dir)
    results = await engine.process_tasks(tasks)
    
    # Write results summary
    summary_path = output_dir / 'conversion_summary.json'
    with open(summary_path, 'w') as f:
        json.dump({
            'tasks': [
                {
                    'id': t.id,
                    'status': t.status,
                    'source_url': t.source_url,
                    'source_framework': t.source_framework,
                    'error_message': t.error_message,
                    'output_files': t.output_files,
                }
                for t in results
            ]
        }, f, indent=2)
    
    logger.info(f"📄 Summary written to: {summary_path}")
    
    # Exit with error code if any task failed
    if any(t.status == 'failed' for t in results):
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())
