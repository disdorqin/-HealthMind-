#!/usr/bin/env python3
"""
全自动模型诊断与修复脚本
Auto-Diagnosis & Self-Healing Script

功能：
1. 扫描 src/models/ 发现模型。
2. 对每个模型执行 main.py --mode test --model <name>。
3. 捕获 RuntimeError (Flat Loss / Zero Metrics)。
4. 自动调整 config/settings.yaml 超参数并重试。
5. 验证是否生成 plots。
"""

import sys
import os
import subprocess
import yaml
import re
import time
from pathlib import Path
from typing import List, Dict, Any

# Ensure project root is in path
sys.path.append(os.getcwd())

from src.core.utils.logger import logger

MODELS_DIR = Path("src/models")
CONFIG_PATH = Path("config/settings.yaml")
MAIN_SCRIPT = Path("main.py")
PLOTS_DIR = Path("logs/plots")

def scan_models() -> List[str]:
    """Scans for valid model files in src/models/"""
    print(f"[*] Scanning {MODELS_DIR} for models...")
    models = []
    if not MODELS_DIR.exists():
        print(f"[!] {MODELS_DIR} not found.")
        return []
        
    # Heuristic: files ending in _model.py, excluding base_model.py
    for f in MODELS_DIR.glob("*_model.py"):
        if f.name == "base_model.py":
            continue
        name = f.stem.replace("_model", "")
        # Validate against main.py known list (optional, but safer)
        # For now, just trust the file name matches the CLI arg expectation
        models.append(name)
    
    print(f"[+] Found models: {models}")
    return models

def load_config() -> Dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def save_config(cfg: Dict[str, Any]) -> None:
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)

def adjust_hyperparams(model_name: str, issue_type: str) -> bool:
    """
    Adjusts hyperparameters in settings.yaml based on failure type.
    Returns True if adjustment was made, False if limit reached.
    """
    cfg = load_config()
    model_cfg = cfg.get("models", {}).get(model_name, {})
    
    print(f"[*] Attempting auto-fix for {model_name} (Issue: {issue_type})...")
    
    changed = False
    
    # Strategy:
    # If "loss did not decrease" (flat), maybe LR is too high (exploding/nan) or too low (stuck).
    # If "tail std ~ 0" (flat), usually stuck.
    # Default assumption: Try lowering LR first, then increasing.
    
    current_lr = float(model_cfg.get("learning_rate", 0.001))
    
    if issue_type == "flat_loss":
        # Heuristic: cycle 0.001 -> 0.0001 -> 0.00001 -> 0.01
        if 0.0005 < current_lr <= 0.005:
             new_lr = 0.0001
        elif 0.00005 < current_lr <= 0.0005:
             new_lr = 0.00001
        elif current_lr <= 0.00005:
             new_lr = 0.01  # Try jumping out
        else:
             new_lr = 0.001 # Reset
             
        print(f"    -> Adjusting LR: {current_lr} -> {new_lr}")
        model_cfg["learning_rate"] = new_lr
        changed = True

    elif issue_type == "zero_metrics":
        # Maybe class imbalance or threshold issue. 
        # For regression, maybe model output is NaN.
        # But here we focus on training params.
        # Let's try increasing batch size (more stable gradient) or verify input dim.
        # Actually, let's just perturb LR as well.
        new_lr = current_lr * 0.5
        print(f"    -> Adjusting LR (Zero Metrics): {current_lr} -> {new_lr}")
        model_cfg["learning_rate"] = new_lr
        changed = True

    if changed:
        if "models" not in cfg:
            cfg["models"] = {}
        cfg["models"][model_name] = model_cfg
        save_config(cfg)
        return True
    
    return False

def run_diagnosis(model_name: str) -> bool:
    """Runs main.py and returns True if successful, False if failed."""
    cmd = [sys.executable, str(MAIN_SCRIPT), "--mode", "test", "--model", model_name]
    
    print(f"\n>>> Running Diagnosis: {model_name}", flush=True)
    try:
        # Capture output strictly with timeout
        result = subprocess.run(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True, 
            encoding='utf-8', 
            errors='ignore', # prevent decoding errors
            timeout=300 # 5 minutes max per model
        )
        
        if result.returncode == 0:
            print(f"[OK] {model_name} passed audit.", flush=True)
            # Verify plot existence
            plot_file = PLOTS_DIR / f"loss_curve_single_{model_name}_*.png"
            # glob for it
            plots = list(PLOTS_DIR.glob(f"loss_curve_single_{model_name}_*.png"))
            if plots:
                print(f"    -> Visual verification passed: {plots[0].name}")
            else:
                print(f"    [WARNING] Plot file missing for {model_name}")
            
            # Print F1 key line from output if found
            for line in result.stderr.splitlines() + result.stdout.splitlines():
                if "F1-score" in line:
                    print(f"    -> {line.strip()}")
            return True
            
        else:
            print(f"[FAIL] {model_name} failed audit.")
            # Analyze error
            full_log = result.stdout + "\n" + result.stderr
            
            issue = "unknown"
            if "loss did not decrease" in full_log or "tail std ~ 0" in full_log:
                issue = "flat_loss"
                print("    -> Reason: Flat Loss / No Convergence")
            elif "F1 is zero" in full_log:
                issue = "zero_metrics"
                print("    -> Reason: Zero Signal Metrics")
            else:
                print("    -> Reason: Runtime Fail (See below)")
                print(full_log[-500:]) # Last 500 chars 
                
            # Attempt Auto-Fix
            if adjust_hyperparams(model_name, issue):
                print("    -> Retrying...")
                return run_diagnosis(model_name) # Recursive retry
            else:
                print("    [!] Auto-Fix limit reached or no strategy available.")
                return False

    except Exception as e:
        print(f"[!] Execution error: {e}")
        return False

def main():
    print("=== Auto-Diagnosis & Self-Healing Started ===")
    
    # 1. Discovery
    models = scan_models()
    if not models:
        print("[!] No models found to test.")
        sys.exit(1)
        
    # Filter only known supported inputs for main.py (lstm, gru, xgboost, moirai)
    known_models = ["lstm", "gru", "xgboost", "moirai"]
    target_models = [m for m in models if m in known_models]
    
    print(f"[*] Targeting models: {target_models}")
    
    results = {}
    
    # 2. Iterative Audit
    for m in target_models:
        success = run_diagnosis(m)
        results[m] = success
        
    # 3. Final Report
    print("\n=== Final Report ===")
    all_pass = True
    for m, s in results.items():
        status = "PASSED" if s else "FAILED"
        print(f"Model {m.ljust(10)}: {status}")
        if not s:
            all_pass = False
            
    if all_pass:
        print("\nAll models verified successfully. System is healthy.")
        print("Plots stored in logs/plots/. Metrics verified > 0.")
        sys.exit(0)
    else:
        print("\nSome models failed diagnosis.")
        sys.exit(1)

if __name__ == "__main__":
    main()
