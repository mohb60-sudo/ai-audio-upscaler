import torch
import logging
import time
import threading
import gc
import contextlib
import json
import os
import datetime

# Try to import pynvml, but don't crash if missing
try:
    import pynvml
    HAS_NVML = True
except ImportError:
    HAS_NVML = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

logger = logging.getLogger(__name__)

class GPUGuard:
    """
    Safety layer for GPU training and inference.
    
    Protects against:
    - CUDA OOM (Out of Memory) errors by monitoring VRAM.
    - TDR (Timeout Detection and Recovery) crashes on Windows by timing kernels.
    - Thermal throttling / Overheating by monitoring GPU temperature (via NVML).
    - System RAM exhaustion by monitoring virtual memory.
    
    Attributes:
        device (str): Target device ('cuda' or 'cpu').
        max_kernel_ms (int): Maximum allowed kernel duration in ms before warning.
        watchdog_enabled (bool): Whether to run the background monitoring thread.
    """
    def __init__(self, device: str = "cuda", max_kernel_ms: int = 1200, watchdog: bool = False):
        self.device = device
        self.max_kernel_ms = max_kernel_ms
        self.watchdog_enabled = watchdog
        self.stop_event = threading.Event()
        self.backpressure_active = False
        self.backpressure_factor = 1.0 # Multiplier for batch size (0.0 to 1.0)
        
        self.nvml_handle = None
        self.gpu_index = 0
        
        # Metrics history
        self.snapshots = []
        
    def preflight(self):
        """Checks GPU status and initializes NVML."""
        global HAS_NVML  # Allow modification of module-level variable
        
        if not torch.cuda.is_available():
            logger.warning("GPUGuard: No CUDA device found. Safety features disabled.")
            return

        # Set device
        if self.device == "cuda":
            self.device = f"cuda:{torch.cuda.current_device()}"
            
        logger.info(f"GPUGuard: Active on {self.device}")
        
        # NVML Init
        if HAS_NVML:
            try:
                pynvml.nvmlInit()
                self.gpu_index = torch.cuda.current_device()
                self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
                name = pynvml.nvmlDeviceGetName(self.nvml_handle)
                logger.info(f"GPUGuard: NVML initialized for {name}")
                
                # Start watchdog if enabled
                if self.watchdog_enabled:
                    self._start_watchdog()
            except Exception as e:
                logger.warning(f"GPUGuard: NVML init failed: {e}")
                HAS_NVML = False
        else:
            if self.watchdog_enabled:
                logger.warning("GPUGuard: Watchdog requested but pynvml not installed. Install with 'pip install pynvml'.")

    def _start_watchdog(self):
        """Starts the background monitoring thread."""
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self.thread.start()
        logger.info("GPUGuard: Watchdog thread started.")

    def _watchdog_loop(self):
        """Monitor Temp/Power/RAM every 2s."""
        while not self.stop_event.is_set():
            try:
                # GPU Metrics
                temp = 0
                power = 0
                limit = 1
                if HAS_NVML and self.nvml_handle:
                    temp = pynvml.nvmlDeviceGetTemperature(self.nvml_handle, pynvml.NVML_TEMPERATURE_GPU)
                    power = pynvml.nvmlDeviceGetPowerUsage(self.nvml_handle) / 1000.0 # mW -> W
                    limit = pynvml.nvmlDeviceGetEnforcedPowerLimit(self.nvml_handle) / 1000.0
                
                # System RAM Metrics
                ram_percent = 0
                if HAS_PSUTIL:
                    ram_percent = psutil.virtual_memory().percent

                # Snapshot
                snapshot = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "temp_c": temp,
                    "power_w": power,
                    "power_limit_w": limit,
                    "ram_percent": ram_percent,
                    "backpressure": self.backpressure_active
                }
                self.snapshots.append(snapshot)
                if len(self.snapshots) > 100:
                    self.snapshots.pop(0)
                
                # Safety Logic
                # Throttle if Temp > 87C, Power > 95% limit, or RAM > 90%
                gpu_throttle = (temp > 87 or power > (limit * 0.95))
                ram_throttle = (ram_percent > 90)
                
                if gpu_throttle or ram_throttle:
                    if not self.backpressure_active:
                        reason = []
                        if gpu_throttle: reason.append(f"GPU (T={temp}C, P={power:.1f}W)")
                        if ram_throttle: reason.append(f"RAM ({ram_percent}%)")
                        
                        logger.warning(f"GPUGuard: High Load Detected [{', '.join(reason)}]. Throttling...")
                        self.backpressure_active = True
                        self.backpressure_factor = 0.8 # Reduce batch size by 20%
                else:
                    if self.backpressure_active and temp < 80 and ram_percent < 85:
                        logger.info("GPUGuard: Load normalized. Releasing throttle.")
                        self.backpressure_active = False
                        self.backpressure_factor = 1.0
                        
            except Exception as e:
                logger.error(f"GPUGuard Watchdog Error: {e}")
            
            time.sleep(2.0)

    def stop(self):
        """Stops watchdog."""
        self.stop_event.set()
        if HAS_NVML:
            try:
                pynvml.nvmlShutdown()
            except:
                pass

    @contextlib.contextmanager
    def kernel_timer(self, name="step"):
        """Times a CUDA operation and warns if it exceeds TDR limits."""
        if not torch.cuda.is_available():
            yield
            return
            
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        yield
        end.record()
        torch.cuda.synchronize()
        
        elapsed_ms = start.elapsed_time(end)
        if elapsed_ms > self.max_kernel_ms:
            logger.warning(f"GPUGuard: Kernel '{name}' took {elapsed_ms:.1f}ms! (Limit: {self.max_kernel_ms}ms). Risk of TDR crash.")

    def auto_batch_size(self, try_fn, max_bs=None, min_bs=1, target_vram_frac=0.85):
        """
        Binary search for optimal batch size.
        try_fn(batch_size) should run a dummy forward/backward pass and return True if successful.
        """
        if not torch.cuda.is_available():
            return max_bs or 4
            
        logger.info("GPUGuard: Auto-tuning batch size...")
        
        # Get VRAM info
        torch.cuda.empty_cache()
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        free = t - r - a # Rough estimate
        
        # Heuristic start
        # If max_bs is provided, start there.
        # If not, try to guess based on free memory (very rough).
        
        low = min_bs
        high = max_bs if max_bs else 64
        optimal = min_bs
        
        # Binary Search
        while low <= high:
            mid = (low + high) // 2
            logger.info(f"GPUGuard: Testing batch size {mid}...")
            try:
                success = try_fn(mid)
                if success:
                    optimal = mid
                    low = mid + 1
                else:
                    high = mid - 1
            except RuntimeError as e:
                if "out of memory" in str(e):
                    high = mid - 1
                    torch.cuda.empty_cache()
                else:
                    raise e
                    
        logger.info(f"GPUGuard: Optimal batch size found: {optimal}")
        return optimal

    def safe_step(self, step_fn, retries=3):
        """
        Executes a training step with OOM protection and retry logic.
        """
        # Apply backpressure if active
        if self.backpressure_active:
            time.sleep(0.5) # Cool off
            
        for attempt in range(retries + 1):
            try:
                return step_fn()
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.warning(f"GPUGuard: OOM caught on attempt {attempt+1}/{retries+1}.")
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                    if attempt < retries:
                        logger.info("GPUGuard: Retrying...")
                        time.sleep(1.0 * (attempt + 1)) # Exponential backoff
                        continue
                    else:
                        logger.error("GPUGuard: Max retries exceeded. Giving up.")
                        self.log_crash_report(e)
                        raise e
                else:
                    raise e
            except Exception as e:
                 # Handle other CUDA errors (Launch Timeout etc)
                 if "launch failed" in str(e) or "device-side assert" in str(e):
                     logger.critical(f"GPUGuard: Critical CUDA Error: {e}")
                     self.log_crash_report(e)
                     raise e
                 raise e

    def check_system_ram(self, threshold_percent=90):
        """Checks if System RAM is critically low."""
        if not HAS_PSUTIL:
            return False
        return psutil.virtual_memory().percent > threshold_percent

    def optimize_workers(self, requested_workers, avg_file_size_mb=50):
        """
        Calculates safe number of workers based on available RAM.
        Assumes each worker needs ~2x file size + overhead.
        """
        if not HAS_PSUTIL:
            return requested_workers
            
        mem = psutil.virtual_memory()
        available_mb = mem.available / (1024 * 1024)
        
        # Reserve 2GB for OS/System
        safe_available = max(0, available_mb - 2048)
        
        # Estimate per-worker cost (conservative)
        # 100MB overhead + 2x file size (loading + processing)
        worker_cost = 100 + (avg_file_size_mb * 2)
        
        max_workers = int(safe_available / worker_cost)
        
        # Clamp
        safe_workers = min(requested_workers, max_workers)
        safe_workers = max(0, safe_workers) # At least 0 (main thread only)
        
        if safe_workers < requested_workers:
            logger.warning(f"GPUGuard: Reduced workers from {requested_workers} to {safe_workers} to prevent System OOM (Available RAM: {available_mb:.0f}MB).")
            
        return safe_workers

    def log_crash_report(self, exception):
        """Writes a crash report to disk."""
        report = {
            "timestamp": datetime.datetime.now().isoformat(),
            "error": str(exception),
            "snapshots": self.snapshots[-10:] if self.snapshots else []
        }
        try:
            with open("crash_report.json", "w") as f:
                json.dump(report, f, indent=2)
            logger.info("GPUGuard: Crash report saved to crash_report.json")
        except:
            pass
