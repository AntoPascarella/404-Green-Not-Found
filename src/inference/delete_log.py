#!/data/data/com.termux/files/usr/bin/python3

import os
import json

LOG_DIR = "/data/data/com.termux/files/home/models/llama-3b/logs"

def delete_error_logs():
    """
    Delete all log files that contain an 'error' field.
    """
    if not os.path.exists(LOG_DIR):
        print(f"Error: Log directory not found: {LOG_DIR}")
        return
    
    log_files = [f for f in os.listdir(LOG_DIR) if f.endswith('.log')]
    
    if not log_files:
        print("No log files found.")
        return
    
    deleted_count = 0
    kept_count = 0
    
    for log_file in log_files:
        log_path = os.path.join(LOG_DIR, log_file)
        
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                
                if not content:
                    print(f"[DELETE] Empty file: {log_file}")
                    os.remove(log_path)
                    deleted_count += 1
                    continue
                
                # Parse JSON
                data = json.loads(content)
                
                # Check if there's an error field
                if 'error' in data:
                    print(f"[DELETE] Error found in: {log_file} - {data['error'].get('message', 'unknown error')}")
                    os.remove(log_path)
                    deleted_count += 1
                else:
                    kept_count += 1
                    
        except json.JSONDecodeError:
            print(f"[DELETE] Invalid JSON in: {log_file}")
            os.remove(log_path)
            deleted_count += 1
        except Exception as e:
            print(f"[SKIP] Could not process {log_file}: {str(e)}")
    
    print(f"\n[SUMMARY]")
    print(f"  Deleted: {deleted_count} error logs")
    print(f"  Kept: {kept_count} successful logs")
    print(f"  Total: {deleted_count + kept_count} logs processed")

if __name__ == "__main__":
    delete_error_logs()
