#!/data/data/com.termux/files/usr/bin/python3

import os
import sys
import json
import time
import shlex
import re
import subprocess
from datetime import datetime, timezone
import resource
from threading import Thread, Event

LLAMA_BIN = "/data/data/com.termux/files/home/llama.cpp/build/bin/llama-cli"
MODEL = "/data/data/com.termux/files/home/models/llama-3b/Llama-3.2-3B-Instruct-Q8_0.gguf"
PROMPT_DIR = "/data/data/com.termux/files/home/models/llama-3b/prompts"
LOG_DIR = "/data/data/com.termux/files/home/models/llama-3b/logs"

LLAMA_FLAGS = [
    "-c", "2096",
    "--n-predict", "2048",
    #"--threads", "2",
    "--temp", "0.8",
    "--single-turn",
    "--no-display-prompt",
]

BATTERY_POLL_HZ = 1.0
BATTERY_CMD = ["termux-battery-status"]

# CPU monitoring frequency (same as battery)
CPU_POLL_HZ = 1.0

# Warm-up prompts to run before main loop
WARMUP_PROMPTS = [
    "hello",
    "what is the capital of Italy?"
]

# Expanded regex coverage
# FOR GENERATED TOKENS (output only):
RE_GENERATED_TOKENS = re.compile(
    r'eval\s+time\s*=.*?/\s*(\d+)\s+runs',
    re.IGNORECASE
)

# FOR GENERATION TPS:
RE_GENERATION_TPS = re.compile(
    r'eval\s+time\s*=.*?runs\s*\(.*?([\d.]+)\s+tokens\s+per\s+second',
    re.IGNORECASE
)

# Keep your broader fallbacks too, if desired
RE_TOKENS_KEYED = re.compile(r"(?:evaluated|prompt|output)?\s*tokens?:?\s*(\d+)", re.I)
RE_TPS_GENERIC = re.compile(r"(?:speed:\s*)?([0-9]+(?:\.[0-9]+)?)\s*(?:t/s|token/s|tokens/s|tokens per second)", re.I)

def utc_iso():
    return datetime.now(timezone.utc).isoformat()

def ensure_dirs():
    os.makedirs(LOG_DIR, exist_ok=True)

def list_prompt_files():
    return sorted(
        os.path.join(PROMPT_DIR, n)
        for n in os.listdir(PROMPT_DIR)
        if n.lower().endswith(".txt")
    )

def run_id_from_filename(path):
    return os.path.splitext(os.path.basename(path))[0]

def read_text_file(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()

def battery_snapshot():
    try:
        out = subprocess.check_output(BATTERY_CMD, stderr=subprocess.STDOUT, text=True, timeout=5)
        data = json.loads(out)
        return {
            "ts": utc_iso(),
            "voltage_mV": data.get("voltage"),
            "current_uA": data.get("current") if "current" in data else data.get("current_average"),
            "percentage": data.get("percentage", data.get("level")),
            "plugged": data.get("plugged"),
            "status": data.get("status"),
            "temperature_C": data.get("temperature"),
        }
    except Exception:
        return {"ts": utc_iso(), "voltage_mV": None, "current_uA": None}

def battery_sampler(stop_evt, samples):
    period = 1.0 / BATTERY_POLL_HZ
    while not stop_evt.is_set():
        samples.append(battery_snapshot())
        t0 = time.time()
        while True:
            if stop_evt.is_set():
                break
            remaining = period - (time.time() - t0)
            if remaining <= 0:
                break
            time.sleep(min(0.05, remaining))

def get_cpu_count():
    """
    Get the number of CPU cores.
    """
    try:
        # Try to read from /proc/cpuinfo
        with open('/proc/cpuinfo', 'r') as f:
            return sum(1 for line in f if line.strip().startswith('processor'))
    except Exception:
        # Fallback: try using nproc command
        try:
            out = subprocess.check_output(['nproc'], text=True, timeout=2)
            return int(out.strip())
        except Exception:
            # Default to 8 cores (Pixel 9 has Tensor G4 with 8 cores)
            return 8

def get_process_tree_pids(parent_pid):
    """
    Get the parent PID and all its children PIDs using ps.
    Returns a list of PIDs.
    """
    try:
        # Get all processes with their parent PIDs
        out = subprocess.check_output(
            ["ps", "-eo", "pid,ppid"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2
        )
        
        # Parse the output
        lines = out.strip().split('\n')[1:]  # Skip header
        processes = {}
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    pid = int(parts[0])
                    ppid = int(parts[1])
                    processes[pid] = ppid
                except ValueError:
                    continue
        
        # Find all children recursively
        pids = {parent_pid}
        to_check = [parent_pid]
        while to_check:
            current = to_check.pop()
            for pid, ppid in processes.items():
                if ppid == current and pid not in pids:
                    pids.add(pid)
                    to_check.append(pid)
        
        return list(pids)
    except Exception:
        return [parent_pid]  # Fallback to just parent

def cpu_snapshot(process_pid, num_cores):
    """
    Get CPU usage % normalized to 0-100% (across all cores).
    Uses 'ps' command as workaround for /proc/stat access.
    Returns dict with cpu_percent normalized to total system capacity.
    """
    try:
        # Get all PIDs in the process tree
        pids = get_process_tree_pids(process_pid)
        
        if not pids:
            return {"ts": utc_iso(), "cpu_percent": None}
        
        # Get CPU% for all PIDs
        # ps output format: PID %CPU
        out = subprocess.check_output(
            ["ps", "-p", ",".join(map(str, pids)), "-o", "pid,%cpu"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2
        )
        
        lines = out.strip().split('\n')[1:]  # Skip header
        total_cpu_percent = 0.0
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    cpu_pct = float(parts[1])
                    total_cpu_percent += cpu_pct
                except (ValueError, IndexError):
                    continue
        
        # Normalize to 0-100% by dividing by number of cores
        normalized_cpu_percent = total_cpu_percent / num_cores if num_cores > 0 else total_cpu_percent
        
        return {
            "ts": utc_iso(),
            "cpu_percent": normalized_cpu_percent if normalized_cpu_percent > 0 else None
        }
    except Exception:
        return {"ts": utc_iso(), "cpu_percent": None}

def cpu_sampler(stop_evt, samples, process_pid, num_cores):
    """
    Thread function that periodically samples CPU usage.
    """
    period = 1.0 / CPU_POLL_HZ
    while not stop_evt.is_set():
        samples.append(cpu_snapshot(process_pid, num_cores))
        t0 = time.time()
        while True:
            if stop_evt.is_set():
                break
            remaining = period - (time.time() - t0)
            if remaining <= 0:
                break
            time.sleep(min(0.05, remaining))

def compute_cpu_metrics(samples):
    """
    Compute average and max CPU % from samples (already normalized to 0-100%).
    """
    if not samples:
        return {"avg_cpu_percent": None, "max_cpu_percent": None}
    
    cpu_percents = []
    
    for s in samples:
        cpu_pct = s.get("cpu_percent")
        
        if isinstance(cpu_pct, (int, float)) and cpu_pct is not None:
            cpu_percents.append(float(cpu_pct))
    
    avg_cpu_percent = sum(cpu_percents) / len(cpu_percents) if cpu_percents else None
    max_cpu_percent = max(cpu_percents) if cpu_percents else None
    
    return {
        "avg_cpu_percent": avg_cpu_percent,
        "max_cpu_percent": max_cpu_percent
    }

def compute_energy_and_power(samples, start_ts, end_ts):
    if not samples or end_ts <= start_ts:
        return {"samples": len(samples), "avg_voltage_v": None, "avg_current_a_abs": None, "energy_j": None, "avg_power_w": None}
    
    volts, currents_abs = [], []
    for s in samples:
        v_mV = s.get("voltage_mV")
        i_uA = s.get("current_uA")
        if isinstance(v_mV, (int, float)) and isinstance(i_uA, (int, float)):
            volts.append(float(v_mV) / 1000.0)
            currents_abs.append(abs(float(i_uA)) / 1_000_000.0)
    
    if not volts or not currents_abs:
        return {"samples": len(samples), "avg_voltage_v": None, "avg_current_a_abs": None, "energy_j": None, "avg_power_w": None}
    
    V_avg = sum(volts) / len(volts)
    I_avg = sum(currents_abs) / len(currents_abs)
    dt = end_ts - start_ts
    energy_j = V_avg * I_avg * dt
    
    return {
        "samples": len(samples),
        "avg_voltage_v": V_avg,
        "avg_current_a_abs": I_avg,
        "energy_j": energy_j,
        "avg_power_w": energy_j / dt if dt > 0 else None,
    }

def parse_tokens_and_tps(stdout_text, stderr_text):
    """
    Extract:
    - tokens: from 'eval time = ... / N runs' (generated tokens only)
    - tokens_per_sec: from 'eval time = ... ( ... tokens per second )'
    Ignore sampler 'sampling time' and 'prompt eval time' TPS.
    Scan stderr first, from bottom, then stdout.
    """
    tokens = None
    tps = None
    
    def scan(text):
        nonlocal tokens, tps
        if not text:
            return
        
        lines = text.splitlines()
        # Reverse scan to prefer final totals
        for line in reversed(lines):
            # CRITICAL: Skip 'prompt eval' lines to avoid wrong matches
            if 'prompt eval' in line.lower():
                continue
            
            # Extract generated tokens (from 'eval time' line with 'runs')
            if tokens is None:
                m_gen = RE_GENERATED_TOKENS.search(line)
                if m_gen:
                    try:
                        tokens = int(m_gen.group(1))
                    except Exception:
                        pass
            
            # Extract generation TPS (from 'eval time' line)
            if tps is None:
                m_tps = RE_GENERATION_TPS.search(line)
                if m_tps:
                    try:
                        tps = float(m_tps.group(1))
                    except Exception:
                        pass
            
            if tokens is not None and tps is not None:
                return
        
        # Fallbacks: if one is still None, try broader patterns but filter out sampler & prompt lines
        for line in reversed(lines):
            low = line.lower()
            # Skip lines we don't want
            if "sampling time" in low or "prompt eval" in low:
                continue
            
            if tps is None:
                m_tps = RE_TPS_GENERIC.search(line)
                if m_tps:
                    try:
                        tps = float(m_tps.group(1))
                    except Exception:
                        pass
            
            if tokens is None:
                # Try to find "runs" pattern first (for eval time)
                m_runs = re.search(r'/\s*(\d+)\s+runs', line, re.I)
                if m_runs:
                    try:
                        tokens = int(m_runs.group(1))
                    except Exception:
                        pass
            
            if tokens is not None and tps is not None:
                return
    
    # Prefer stderr, then stdout
    scan(stderr_text)
    if tokens is None or tps is None:
        scan(stdout_text)
    
    return tokens, tps

def get_max_rss_kb():
    child = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    selfr = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "ru_maxrss_children_kb": int(child) if child is not None else None,
        "ru_maxrss_self_kb": int(selfr) if selfr is not None else None,
    }

def build_cmd(prompt_text):
    return [LLAMA_BIN, "-m", MODEL, "-p", prompt_text] + LLAMA_FLAGS

def run_warmup(prompt_text, warmup_index, total_warmups):
    """
    Run a single warm-up inference without logging.
    Raises exception if warm-up fails.
    """
    print(f"[WARMUP] Running warm-up {warmup_index}/{total_warmups}...", flush=True)
    
    cmd = build_cmd(prompt_text)
    
    try:
        # Run the command and discard output
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True
        )
        
        # Wait for completion
        ret = proc.wait(timeout=120)  # 2 minute timeout for warm-up
        
        if ret != 0:
            raise Exception(f"Warm-up {warmup_index} failed with return code {ret}")
        
        print(f"[WARMUP] Warm-up {warmup_index}/{total_warmups} completed", flush=True)
        
        # Always wait 30 seconds after warm-up
        time.sleep(30.0)
        
    except subprocess.TimeoutExpired:
        proc.kill()
        raise Exception(f"Warm-up {warmup_index} timed out after 120 seconds")
    except Exception as e:
        raise Exception(f"Warm-up {warmup_index} failed: {str(e)}")

def run_all_warmups():
    """
    Run all warm-up prompts before starting main loop.
    Exits with error if any warm-up fails.
    """
    total = len(WARMUP_PROMPTS)
    
    if total == 0:
        return
    
    print(f"[WARMUP] Starting {total} warm-up run(s)...", flush=True)
    
    try:
        for i, prompt in enumerate(WARMUP_PROMPTS, start=1):
            run_warmup(prompt, i, total)
        
        print(f"[WARMUP] All warm-ups completed successfully", flush=True)
    
    except Exception as e:
        print(f"[ERROR] Warm-up failed: {str(e)}", file=sys.stderr, flush=True)
        sys.exit(1)

def run_single_prompt(prompt_path):
    run_id = run_id_from_filename(prompt_path)
    log_path = os.path.join(LOG_DIR, f"{run_id}.log")
    
    if os.path.exists(log_path):
        return None
    
    prompt_text = read_text_file(prompt_path)
    print(f"[RUNNING] {run_id} started", flush=True)
    
    # Get CPU count once at the start
    num_cores = get_cpu_count()
    
    battery_samples = []
    cpu_samples = []
    stop_evt = Event()
    
    cmd = build_cmd(prompt_text)
    
    t0 = time.time()
    t0_iso = utc_iso()
    
    proc = None
    
    try:
        # Start the subprocess
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Start monitoring threads with the process PID
        battery_sampler_thread = Thread(target=battery_sampler, args=(stop_evt, battery_samples), daemon=True)
        cpu_sampler_thread = Thread(target=cpu_sampler, args=(stop_evt, cpu_samples, proc.pid, num_cores), daemon=True)
        
        battery_sampler_thread.start()
        cpu_sampler_thread.start()
        
        # Wait for process to complete
        stdout_text, stderr_text = proc.communicate(timeout=None)
        ret = proc.returncode
        
        t1 = time.time()
        t1_iso = utc_iso()
        
        # Stop monitoring threads
        stop_evt.set()
        battery_sampler_thread.join(timeout=2)
        cpu_sampler_thread.join(timeout=2)
        
        # Parse outputs
        tokens, tps = parse_tokens_and_tps(stdout_text, stderr_text)
        elapsed_s = t1 - t0
        
        # Compute metrics
        energy = compute_energy_and_power(battery_samples, t0, t1)
        cpu_metrics = compute_cpu_metrics(cpu_samples)
        rss = get_max_rss_kb()
        
        # Build record
        record = {
            "run_id": run_id,
            "ts_start": t0_iso,
            "ts_end": t1_iso,
            "tokens": tokens,
            "tokens_per_sec": tps,
            "inference_seconds": elapsed_s,
            "avg_cpu_percent": cpu_metrics.get("avg_cpu_percent"),
            "max_cpu_percent": cpu_metrics.get("max_cpu_percent"),
            "battery_sample_count": energy.get("samples"),
            "avg_voltage_v": energy.get("avg_voltage_v"),
            "avg_current_a_abs": energy.get("avg_current_a_abs"),
            "energy_j": energy.get("energy_j"),
            "avg_power_w": energy.get("avg_power_w"),
            "max_rss_children_kb": rss.get("ru_maxrss_children_kb"),
            "max_rss_self_kb": rss.get("ru_maxrss_self_kb"),
            "raw_output": stdout_text,
        }
        
        if ret != 0:
            record["error"] = {"message": "llama-cli returned non-zero", "stderr": stderr_text[-4000:] if stderr_text else None}
        
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        print(f"[COMPLETED] {run_id} finished", flush=True)
        time.sleep(30.0)
        return log_path
        
    except Exception as e:
        stop_evt.set()
        
        if proc and proc.poll() is None:
            try:
                proc.terminate()
                time.sleep(1.0)
            except Exception:
                pass
            try:
                proc.kill()
            except Exception:
                pass
        
        error_record = {
            "run_id": run_id,
            "ts_start": t0_iso,
            "ts_end": utc_iso(),
            "cmd": {"bin": LLAMA_BIN, "model": MODEL, "flags": LLAMA_FLAGS},
            "error": {"message": str(e)},
        }
        
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(error_record, ensure_ascii=False) + "\n")
        
        print(f"[COMPLETED] {run_id} finished with error", flush=True)
        time.sleep(30.0)
        return log_path

def main():
    ensure_dirs()
    
    # Run warm-up prompts first
    run_all_warmups()
    
    # Now proceed with main prompt files
    prompts = list_prompt_files()
    
    if not prompts:
        print("No prompt files found.", file=sys.stderr)
        sys.exit(0)
    
    for p in prompts:
        try:
            run_single_prompt(p)
        except KeyboardInterrupt:
            print("Interrupted by user.", file=sys.stderr)
            break
        except Exception as e:
            try:
                run_id = run_id_from_filename(p)
                log_path = os.path.join(LOG_DIR, f"{run_id}.log")
                record = {"run_id": run_id, "ts": utc_iso(), "error": {"message": str(e)}}
                with open(log_path, "w", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception:
                pass
            time.sleep(30.0)
            continue

if __name__ == "__main__":
    main()
