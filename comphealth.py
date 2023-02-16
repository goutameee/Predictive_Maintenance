import psutil
import platform

# Get the system's current CPU usage
cpu_percent = psutil.cpu_percent()

# Get the system's current memory usage
mem_stats = psutil.virtual_memory()
mem_percent = mem_stats.percent

# Get the system's uptime
uptime_seconds = psutil.boot_time()
uptime_hours = uptime_seconds / 3600

# Get the system's operating system
os_name = platform.system()

# Print the results
print(f"CPU Usage: {cpu_percent}%")
print(f"Memory Usage: {mem_percent}%")
print(f"Uptime: {uptime_hours:.2f} hours")
print(f"Operating System: {os_name}")
