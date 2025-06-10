import subprocess
import time

scripts = [
    ["Extractor.py"],
    ["Truth extractor.py"],
    ["Denoising.py"],
    ["Segmentaion.py"],
    ["IOU.py"],
    ["Grow rate and intensity.py", "--pipeline"]  # Add the flag for pipeline mode
]

for script in scripts:
    try:
        print(f"Running {' '.join(script)}...")
        start_time = time.time()

        # Run script and wait for completion
        result = subprocess.run(
            ["python"] + script,  # Combine the command and arguments
            check=True,
            text=True,
            capture_output=True
        )

        # Print output and time taken
        print(result.stdout)
        elapsed = time.time() - start_time
        print(f"{' '.join(script)} completed in {elapsed:.2f} seconds")

        # Add sleep time between scripts
        time.sleep(1.0)  # 1 second pause between scripts

    except subprocess.CalledProcessError as e:
        print(f"Error in {' '.join(script)}:")
        print(e.stderr)
        break

    except FileNotFoundError:
        print(f"{' '.join(script)} not found. Check filename/spelling.")
        break

print("All scripts executed in order.")