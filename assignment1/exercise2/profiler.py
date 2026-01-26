import psutil
import time
import matplotlib.pyplot as plt
import sys
import subprocess

# Profiler function with program we want to measure
def profiler(program):
    # Run code: in our case exercise I and II
    subprocess.Popen([sys.executable, program])

    # Start profiling
    timestamps = []
    cpu_usage = []
    t_zero = time.time()
    for i in range (10):
        cpu_usage_t = psutil.cpu_percent(interval=1, percpu=True)
        timestamps.append(time.time() - t_zero)
        cpu_usage.append(cpu_usage_t)
        print("CPU usage per core:", cpu_usage_t)
        print("-" * 80)
        time.sleep(1)
    plot_cpu_usage(timestamps, cpu_usage)  


# Plot using matplotlib
def plot_cpu_usage(timestamps, cpu_usage):
    num_cores = len(cpu_usage[0])
    
    for i in range(num_cores):
        plt.plot(timestamps, cpu_usage, label=f'Core {i}')

    plt.title('CPU usage percentage per core over time')
    plt.xlabel('Time (s)')
    plt.ylabel('CPU usage (%)')

    plt.show()

def main(program):
    profiler(program)

if __name__ == "__main__":
    main(sys.argv[1])