from datetime import datetime

def log_message(message, is_error=False):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prefix = "ERROR" if is_error else "INFO"
    with open("program_log.txt", "a") as log_file:
        log_file.write(f"[{timestamp}] {prefix}: {message}\n")