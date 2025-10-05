import os
import time

def write_file(filename, size_mb=10):
    """Write `size_mb` megabytes of data to a file."""
    data = b'x' * 1024 * 1024  # 1 MB
    with open(filename, 'wb') as f:
        for _ in range(size_mb):
            f.write(data)

def read_file(filename):
    """Read the file in chunks."""
    with open(filename, 'rb') as f:
        while f.read(1024 * 1024):  # read 1 MB at a time
            pass

def main():
    test_dir = "/tmp/perf_test"
    os.makedirs(test_dir, exist_ok=True)
    
    filenames = [os.path.join(test_dir, f"file_{i}.bin") for i in range(4)]
    
    # Write files
    for fn in filenames:
        print(f"Writing {fn}...")
        write_file(fn, size_mb=50)
    
    # Read files
    for fn in filenames:
        print(f"Reading {fn}...")
        read_file(fn)
    
    # Simple CPU load to see function profiling
    print("Doing some CPU work...")
    total = 0
    for i in range(10**7):
        total += i*i
    print("Done CPU work, total =", total)

if __name__ == "__main__":
    main()
