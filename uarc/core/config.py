"""
UARC Configuration System
==========================
Nested dataclass configs for all 7 modules plus runtime settings.
Supports environment variable overrides, presets, and serialization.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict


@dataclass
class TDEConfig:
    """Module 1: Token Difficulty Estimator config."""
    context_dim: int = 128
    hidden_dim: int = 256
    n_hidden: int = 3
    dropout_rate: float = 0.1
    lr: float = 1e-3
    tau_easy: float = 2.5
    tau_hard: float = 8.0
    ema_alpha: float = 0.95


@dataclass
class AIVMConfig:
    """Module 2: AI Virtual Memory Manager config."""
    vram_mb: float = 8192.0
    ram_mb: float = 32768.0
    nvme_mb: float = 524288.0


@dataclass
class DPEConfig:
    """Module 3: Dynamic Precision Engine config."""
    default_budget_gb: float = 8.0


@dataclass
class PLLConfig:
    """Module 4: Predictive Layer Loader config."""
    lookahead_k: int = 4
    slack_ms: float = 20.0


@dataclass
class ACSConfig:
    """Module 5: Adaptive Compute Scheduler config."""
    max_batch_size: int = 32


@dataclass
class NSCConfig:
    """Module 6: Neural Semantic Cache config."""
    embedding_dim: int = 256
    similarity_threshold: float = 0.92
    max_entries: int = 10_000
    ttl_seconds: float = 3600.0


@dataclass
class ModelConfig:
    """Model configuration."""
    n_layers: int = 32
    vocab_size: int = 32000
    hidden_dim: int = 4096
    n_params: int = 7_000_000_000


@dataclass
class UARCConfig:
    """
    Master configuration container for the entire UARC runtime.
    All sub-configs are nested dataclasses with sensible defaults.
    """
    tde: TDEConfig = field(default_factory=TDEConfig)
    aivm: AIVMConfig = field(default_factory=AIVMConfig)
    dpe: DPEConfig = field(default_factory=DPEConfig)
    pll: PLLConfig = field(default_factory=PLLConfig)
    acs: ACSConfig = field(default_factory=ACSConfig)
    nsc: NSCConfig = field(default_factory=NSCConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    # Module enable flags
    enable_tde: bool = True
    enable_nsc: bool = True
    enable_dpe: bool = True
    enable_pll: bool = True

    # Backend selection: "auto", "ollama", "llama_cpp", "simulated"
    backend: str = "auto"
    model_name: str = "llama3.2:1b"   # Ollama model name
    ollama_url: str = "http://localhost:11434"
    model_path: str = ""               # Path to GGUF file (for llama_cpp)

    # Server settings
    port: int = 8000

    @classmethod
    def from_env(cls) -> UARCConfig:
        """Read configuration from environment variables."""
        cfg = cls()
        if v := os.environ.get("UARC_VRAM_GB"):
            cfg.aivm.vram_mb = float(v) * 1024
        if v := os.environ.get("UARC_RAM_GB"):
            cfg.aivm.ram_mb = float(v) * 1024
        if v := os.environ.get("UARC_MAX_BATCH"):
            cfg.acs.max_batch_size = int(v)
        if v := os.environ.get("UARC_PORT"):
            cfg.port = int(v)
        if v := os.environ.get("UARC_NSC_THRESHOLD"):
            cfg.nsc.similarity_threshold = float(v)
        if v := os.environ.get("UARC_LAYERS"):
            cfg.model.n_layers = int(v)
        if v := os.environ.get("UARC_BACKEND"):
            cfg.backend = v
        if v := os.environ.get("UARC_MODEL"):
            cfg.model_name = v
        if v := os.environ.get("UARC_OLLAMA_URL"):
            cfg.ollama_url = v
        if v := os.environ.get("UARC_MODEL_PATH"):
            cfg.model_path = v
        return cfg

    @classmethod
    def for_edge(cls) -> UARCConfig:
        """Preset for edge deployment: 4GB RAM, no GPU."""
        cfg = cls()
        cfg.aivm.vram_mb = 0
        cfg.aivm.ram_mb = 4096
        cfg.aivm.nvme_mb = 65536
        cfg.acs.max_batch_size = 4
        cfg.model.n_layers = 24
        cfg.enable_pll = True
        return cfg

    @classmethod
    def for_gpu(cls, vram_gb: float = 24) -> UARCConfig:
        """Preset for GPU deployment."""
        cfg = cls()
        cfg.aivm.vram_mb = vram_gb * 1024
        cfg.aivm.ram_mb = 65536
        cfg.dpe.default_budget_gb = vram_gb
        cfg.acs.max_batch_size = 32
        return cfg

    def to_dict(self) -> dict:
        """Serialize entire config to dictionary."""
        return asdict(self)
