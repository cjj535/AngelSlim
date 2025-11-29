# Copyright 2025 Tencent Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import sys
from typing import Any, Dict, Optional

import torch

from .compressor.speculative.benchmark import vllm as vllm_benchmark


class SpecEngine:
    """
    High-level interface for speculative decoding benchmarks
    Integrates BenchmarkEngine with additional workflow management
    """

    def __init__(self, config=None, deploy_backend: str = "pytorch"):
        """
        Initialize SpecEngine

        Args:
            config: BenchmarkConfig instance (optional)
            deploy_backend: Backend to use ('pytorch' or 'vllm')
        """
        self.config = config
        self.benchmark_engine = None
        self.results = {}
        self.deploy_backend = deploy_backend.lower()

        # if self.deploy_backend == "pytorch":
        #     self.BenchmarkConfig = pytorch_benchmark.BenchmarkConfig
        #     self.BenchmarkEngine = pytorch_benchmark.BenchmarkEngine
        #     self.BenchmarkMode = pytorch_benchmark.BenchmarkMode
        if self.deploy_backend == "vllm":
            self.BenchmarkConfig = vllm_benchmark.BenchmarkConfig
            self.BenchmarkEngine = vllm_benchmark.BenchmarkEngine
            self.BenchmarkMode = vllm_benchmark.BenchmarkMode
        else:
            raise ValueError(f"Unsupported deploy_backend: {deploy_backend}")

    def setup_benchmark(
        self,
        base_model_path: str,
        eagle_model_path: str,
        model_id: str,
        bench_name: str = "mt_bench",
        output_dir: Optional[str] = None,
        **kwargs,
    ):
        """
        Setup benchmark configuration

        Args:
            base_model_path: Path to base model
            eagle_model_path: Path to Eagle model
            model_id: Model identifier
            bench_name: Benchmark dataset name
            output_dir: Output directory for results
            **kwargs: Additional configuration parameters

        Returns:
            BenchmarkConfig instance
        """
        config_dict = {
            "base_model_path": base_model_path,
            "eagle_model_path": eagle_model_path,
            "model_id": model_id,
            "bench_name": bench_name,
            "output_dir": output_dir,
        }
        config_dict.update(kwargs)

        self.config = self.BenchmarkConfig(**config_dict)
        self.benchmark_engine = self.BenchmarkEngine(self.config)

        return self.config

    def run_eagle_benchmark(self) -> Dict[str, Any]:
        """Run Eagle speculative decoding benchmark only"""
        if not self.benchmark_engine:
            raise RuntimeError(
                "Benchmark not configured. Call setup_benchmark() first."
            )

        self.results = self.benchmark_engine.run_benchmark(self.BenchmarkMode.EAGLE)
        return self.results

    def run_baseline_benchmark(self) -> Dict[str, Any]:
        """Run baseline benchmark only"""
        if not self.benchmark_engine:
            raise RuntimeError(
                "Benchmark not configured. Call setup_benchmark() first."
            )

        self.results = self.benchmark_engine.run_benchmark(self.BenchmarkMode.BASELINE)
        return self.results

    def run_full_benchmark(self) -> Dict[str, Any]:
        """
        Run complete benchmark (both Eagle and baseline) with automatic analysis

        Returns:
            Dictionary containing all results and metrics
        """
        if not self.benchmark_engine:
            raise RuntimeError(
                "Benchmark not configured. Call setup_benchmark() first."
            )

        self.results = self.benchmark_engine.run_benchmark(self.BenchmarkMode.BOTH)
        return self.results

    def calculate_acceptance_length(self, eagle_file: Optional[str] = None) -> float:
        """
        Calculate acceptance length from Eagle benchmark results

        Args:
            eagle_file: Path to Eagle results file
                (optional, uses default if not provided)

        Returns:
            Average acceptance length
        """
        if not self.benchmark_engine:
            raise RuntimeError(
                "Benchmark not configured. Call setup_benchmark() first."
            )

        if eagle_file is None:
            eagle_file = self.benchmark_engine.eagle_file

        return self.benchmark_engine._calculate_acceptance_length(eagle_file)

    def calculate_speedup_ratio(
        self,
        baseline_file: Optional[str] = None,
        eagle_file: Optional[str] = None,
        model_path: Optional[str] = None,
    ) -> float:
        """
        Calculate speedup ratio between baseline and Eagle

        Args:
            baseline_file: Path to baseline results file
            eagle_file: Path to Eagle results file
            model_path: Path to model for tokenization

        Returns:
            Speedup ratio
        """
        if not self.benchmark_engine:
            raise RuntimeError(
                "Benchmark not configured. Call setup_benchmark() first."
            )

        if baseline_file is None:
            baseline_file = self.benchmark_engine.baseline_file
        if eagle_file is None:
            eagle_file = self.benchmark_engine.eagle_file
        if model_path is None:
            model_path = self.config.base_model_path

        return self.benchmark_engine._calculate_speedup_ratio(
            model_path, baseline_file, eagle_file
        )

    def get_performance_report(self) -> str:
        """Generate comprehensive performance report"""
        if not self.benchmark_engine:
            return "Benchmark not configured."

        return self.benchmark_engine.get_performance_summary()

    def cleanup_results(self):
        """Clean up temporary result files"""
        if self.benchmark_engine:
            for file_path in [
                self.benchmark_engine.eagle_file,
                self.benchmark_engine.baseline_file,
                self.benchmark_engine.analysis_file,
            ]:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"Removed: {file_path}")
