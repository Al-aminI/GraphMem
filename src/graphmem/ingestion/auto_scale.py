"""
Auto-Scaling Worker Configuration
==================================

Automatically determines optimal worker counts based on:
- CPU cores (for ThreadPool concurrent I/O)
- Available system memory (for batch sizes)
- GPU availability (only matters for LOCAL embeddings/LLMs)

Rate Limit Handling:
- Rate limits are NOT enforced by limiting workers
- Instead, use aggressive parallelism and handle rate limits with retry logic
- The ingest_batch() method has infinite retry with exponential backoff

IMPORTANT: For API-based usage (Azure, OpenAI, OpenRouter):
- GPU does NOT help - LLM/embedding calls go to cloud servers
- Workers are set high, rate limit errors are handled by retry

GPU only matters when using:
- embedding_provider="local" (sentence-transformers with CUDA)
- llm_provider="ollama" (local LLM with GPU)

Usage:
    from graphmem.ingestion.auto_scale import AutoScaler
    
    scaler = AutoScaler()
    config = scaler.get_optimal_config(aggressive=True)
    
    # Or detect and print hardware info
    scaler.print_hardware_info()
"""

from __future__ import annotations
import os
import logging
import platform
from dataclasses import dataclass
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class HardwareInfo:
    """Detected hardware information."""
    cpu_cores: int
    cpu_threads: int
    cpu_model: str
    ram_gb: float
    ram_available_gb: float
    
    # GPU info
    gpu_available: bool
    gpu_count: int
    gpu_name: Optional[str]
    gpu_vram_gb: Optional[float]
    gpu_cuda_available: bool
    
    # Platform
    os_name: str
    python_version: str


@dataclass 
class OptimalConfig:
    """Optimal configuration for high-performance ingestion."""
    # Extraction (LLM calls)
    extraction_workers: int
    extraction_rate_limit: int  # per minute
    
    # Embedding
    embedding_workers: int
    embedding_batch_size: int
    embedding_rate_limit: int  # per minute
    
    # General
    chunk_size: int
    max_concurrent_docs: int
    
    # Memory management
    max_memory_mb: int
    
    # Recommendations
    use_gpu_embeddings: bool
    use_local_llm: bool
    
    def __str__(self) -> str:
        return f"""
Optimal Configuration:
  Extraction Workers: {self.extraction_workers}
  Embedding Workers: {self.embedding_workers}
  Embedding Batch Size: {self.embedding_batch_size}
  Max Concurrent Docs: {self.max_concurrent_docs}
  Chunk Size: {self.chunk_size}
  Max Memory: {self.max_memory_mb}MB
  Use GPU Embeddings: {self.use_gpu_embeddings}
  Use Local LLM: {self.use_local_llm}
  Rate Limits: Handled by retry logic (not capped)
"""


class AutoScaler:
    """
    Automatically detects hardware and recommends optimal worker configuration.
    
    Considers:
    - CPU: More cores = more extraction workers (up to API limits)
    - GPU: If available, can use for local embeddings (much faster)
    - RAM: Determines batch sizes and concurrent documents
    - API Limits: Cloud APIs have rate limits that cap parallelism
    
    Example:
        scaler = AutoScaler()
        scaler.print_hardware_info()
        
        config = scaler.get_optimal_config(
            provider="azure",  # or "openai", "local"
            embedding_type="api",  # or "local"
        )
        
        pipeline = HighPerformancePipeline(
            llm, embeddings,
            max_extraction_workers=config.extraction_workers,
            max_embedding_workers=config.embedding_workers,
        )
    """
    
    def __init__(self):
        self.hardware = self._detect_hardware()
    
    def _detect_hardware(self) -> HardwareInfo:
        """Detect available hardware resources."""
        
        # CPU info
        cpu_cores = os.cpu_count() or 4
        cpu_threads = cpu_cores  # May be doubled with hyperthreading
        
        try:
            import multiprocessing
            cpu_threads = multiprocessing.cpu_count()
        except:
            pass
        
        # Try to get CPU model
        cpu_model = "Unknown"
        try:
            if platform.system() == "Darwin":  # macOS
                import subprocess
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True, text=True
                )
                cpu_model = result.stdout.strip()
            elif platform.system() == "Linux":
                with open("/proc/cpuinfo") as f:
                    for line in f:
                        if "model name" in line:
                            cpu_model = line.split(":")[1].strip()
                            break
        except:
            pass
        
        # RAM info
        ram_gb = 8.0  # Default
        ram_available_gb = 4.0
        
        try:
            import psutil
            mem = psutil.virtual_memory()
            ram_gb = mem.total / (1024**3)
            ram_available_gb = mem.available / (1024**3)
        except ImportError:
            # Try alternative methods
            try:
                if platform.system() == "Darwin":
                    import subprocess
                    result = subprocess.run(
                        ["sysctl", "-n", "hw.memsize"],
                        capture_output=True, text=True
                    )
                    ram_gb = int(result.stdout.strip()) / (1024**3)
                    ram_available_gb = ram_gb * 0.5  # Estimate
            except:
                pass
        
        # GPU detection
        gpu_available = False
        gpu_count = 0
        gpu_name = None
        gpu_vram_gb = None
        cuda_available = False
        
        # Try CUDA (NVIDIA)
        try:
            import torch
            if torch.cuda.is_available():
                cuda_available = True
                gpu_available = True
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                gpu_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except ImportError:
            pass
        
        # Try MPS (Apple Silicon)
        if not gpu_available:
            try:
                import torch
                if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    gpu_available = True
                    gpu_count = 1
                    gpu_name = "Apple Silicon (MPS)"
                    # Apple Silicon shares RAM
                    gpu_vram_gb = ram_gb * 0.75  # Can use up to 75% of unified memory
            except:
                pass
        
        return HardwareInfo(
            cpu_cores=cpu_cores,
            cpu_threads=cpu_threads,
            cpu_model=cpu_model,
            ram_gb=ram_gb,
            ram_available_gb=ram_available_gb,
            gpu_available=gpu_available,
            gpu_count=gpu_count,
            gpu_name=gpu_name,
            gpu_vram_gb=gpu_vram_gb,
            gpu_cuda_available=cuda_available,
            os_name=platform.system(),
            python_version=platform.python_version(),
        )
    
    def get_optimal_config(
        self,
        provider: str = "azure",
        embedding_type: str = "api",
        aggressive: bool = False,
    ) -> OptimalConfig:
        """
        Get optimal configuration based on hardware and provider.
        
        Args:
            provider: LLM provider ("azure", "openai", "openrouter", "local")
            embedding_type: "api" (cloud) or "local" (sentence-transformers)
            aggressive: If True, use more aggressive parallelism (may hit rate limits)
        
        Returns:
            OptimalConfig with recommended settings
        """
        hw = self.hardware
        
        # Base calculations
        cpu_factor = hw.cpu_threads
        ram_factor = hw.ram_available_gb
        
        # Calculate extraction workers - no rate limit consideration
        # Use CPU threads as the primary factor for concurrent API calls
        if provider == "local":
            # Local LLM: CPU/GPU bound
            if hw.gpu_available:
                extraction_workers = min(hw.gpu_count * 4, 32)
            else:
                extraction_workers = min(cpu_factor, 16)
        else:
            # API: Use hardware capacity, no rate limit throttling
            # The caller is responsible for handling rate limits with retry logic
            max_by_cpu = cpu_factor * 4  # Network I/O bound, can do much more than CPU count
            extraction_workers = min(max_by_cpu, 40 if aggressive else 20)
        
        # Calculate embedding workers - no rate limit consideration
        if embedding_type == "local":
            # Local embeddings: GPU or CPU bound
            if hw.gpu_available:
                embedding_workers = 1  # GPU does batching internally
                embedding_batch_size = min(int(hw.gpu_vram_gb * 100), 500)  # ~100 texts per GB VRAM
            else:
                embedding_workers = min(cpu_factor, 8)
                embedding_batch_size = 32
        else:
            # API embeddings: Use hardware capacity
            embedding_workers = min(cpu_factor * 2, 32 if aggressive else 16)
            embedding_batch_size = 100  # Most APIs support batches of 100
        
        # Memory-based limits
        # Estimate: each document in processing uses ~10MB
        max_concurrent = int(ram_factor * 1024 / 10)  # 10MB per doc
        max_concurrent = min(max_concurrent, extraction_workers * 10)
        
        # Chunk size based on available memory
        if ram_factor < 4:
            chunk_size = 1000
        elif ram_factor < 8:
            chunk_size = 2000
        else:
            chunk_size = 4000
        
        return OptimalConfig(
            extraction_workers=max(1, extraction_workers),
            extraction_rate_limit=0,  # No rate limit - handled by retry logic
            embedding_workers=max(1, embedding_workers),
            embedding_batch_size=embedding_batch_size,
            embedding_rate_limit=0,  # No rate limit - handled by retry logic
            chunk_size=chunk_size,
            max_concurrent_docs=max(10, max_concurrent),
            max_memory_mb=int(ram_factor * 1024 * 0.8),  # Use 80% of available
            use_gpu_embeddings=hw.gpu_available and embedding_type == "local",
            use_local_llm=provider == "local",
        )
    
    def print_hardware_info(self):
        """Print detected hardware information."""
        hw = self.hardware
        
        print("\n" + "=" * 60)
        print("🖥️  HARDWARE DETECTION")
        print("=" * 60)
        
        print(f"\n📊 CPU:")
        print(f"   Model: {hw.cpu_model}")
        print(f"   Cores: {hw.cpu_cores}")
        print(f"   Threads: {hw.cpu_threads}")
        
        print(f"\n💾 Memory:")
        print(f"   Total RAM: {hw.ram_gb:.1f} GB")
        print(f"   Available: {hw.ram_available_gb:.1f} GB")
        
        print(f"\n🎮 GPU (only used for LOCAL models):")
        if hw.gpu_available:
            print(f"   Available: ✅ Yes")
            print(f"   Count: {hw.gpu_count}")
            print(f"   Name: {hw.gpu_name}")
            print(f"   VRAM: {hw.gpu_vram_gb:.1f} GB" if hw.gpu_vram_gb else "   VRAM: Shared")
            print(f"   CUDA: {'✅ Yes' if hw.gpu_cuda_available else '❌ No'}")
            print(f"   ⚠️  Note: GPU not used for API calls (Azure/OpenAI)")
        else:
            print(f"   Available: ❌ No")
            print(f"   ℹ️  Not needed for API-based providers")
        
        print(f"\n🖥️  System:")
        print(f"   OS: {hw.os_name}")
        print(f"   Python: {hw.python_version}")
        
        print("=" * 60)
    
    def print_recommendations(self, provider: str = "azure", embedding_type: str = "api"):
        """Print optimal configuration recommendations."""
        config = self.get_optimal_config(provider, embedding_type)
        
        print("\n" + "=" * 60)
        print(f"⚡ OPTIMAL CONFIGURATION for {provider.upper()}")
        print("=" * 60)
        print(config)
        
        # Explain the approach
        if provider != "local":
            print(f"✅ AGGRESSIVE PARALLELISM: {config.extraction_workers} workers")
            print(f"   Rate limits handled by retry logic (infinite backoff)")
        else:
            if self.hardware.gpu_available:
                print(f"✅ BOTTLENECK: GPU compute (using {self.hardware.gpu_name})")
            else:
                print(f"⚠️  BOTTLENECK: CPU compute (no GPU detected)")
        
        print("\n📝 Usage:")
        print(f"""
from graphmem import GraphMem, MemoryConfig

memory = GraphMem(config)
result = memory.ingest_batch(
    documents,
    max_workers={config.extraction_workers},
    show_progress=True,
)
""")
        print("=" * 60)


def get_optimal_workers(provider: str = "azure") -> Dict[str, int]:
    """
    Quick helper to get optimal worker counts.
    
    Returns:
        Dict with 'extraction_workers', 'embedding_workers', 'embedding_batch_size'
    """
    scaler = AutoScaler()
    config = scaler.get_optimal_config(provider=provider)
    
    return {
        "extraction_workers": config.extraction_workers,
        "embedding_workers": config.embedding_workers,
        "embedding_batch_size": config.embedding_batch_size,
        "max_concurrent_docs": config.max_concurrent_docs,
    }


def detect_and_configure() -> OptimalConfig:
    """
    Auto-detect hardware and return optimal config.
    
    Convenience function for quick setup.
    """
    scaler = AutoScaler()
    scaler.print_hardware_info()
    
    # Auto-detect provider preference
    provider = "azure"  # Default
    
    # Check for local GPU that could run LLMs
    if scaler.hardware.gpu_available and scaler.hardware.gpu_vram_gb and scaler.hardware.gpu_vram_gb >= 8:
        print("\n💡 Detected GPU with sufficient VRAM for local LLM")
        print("   Consider using provider='local' for maximum speed")
    
    config = scaler.get_optimal_config(provider=provider)
    scaler.print_recommendations(provider=provider)
    
    return config


# GPU-specific optimizations
class GPUOptimizer:
    """
    GPU-specific optimizations for embedding generation.
    
    Uses GPU batching for maximum throughput when available.
    """
    
    def __init__(self):
        self.device = self._detect_best_device()
        self.batch_size = self._optimal_batch_size()
    
    def _detect_best_device(self) -> str:
        """Detect best available device."""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
        except:
            pass
        return "cpu"
    
    def _optimal_batch_size(self) -> int:
        """Calculate optimal batch size for device."""
        if self.device == "cuda":
            try:
                import torch
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                # Rough estimate: 100 texts per GB VRAM for embeddings
                return min(int(vram_gb * 100), 512)
            except:
                return 64
        elif self.device == "mps":
            return 128  # Apple Silicon is efficient
        else:
            return 32  # CPU
    
    def get_config(self) -> Dict[str, Any]:
        """Get GPU-optimized configuration."""
        return {
            "device": self.device,
            "batch_size": self.batch_size,
            "use_fp16": self.device in ("cuda", "mps"),  # Half precision for speed
            "max_length": 512,  # Token limit
        }

