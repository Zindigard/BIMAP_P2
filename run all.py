import subprocess
import time

scripts = [
    "Extractor.py",
    "Truth extractor.py",
    "Denoising.py",
    "Segmentaion.py",
    "IOU.py",
    "Grow rate and intensity.py"
]

for script in scripts:
    try:
        print(f"\n🚀 Running {script}...")
        start_time = time.time()

        # Run script and wait for completion
        result = subprocess.run(
            ["python", script],
            check=True,
            text=True,
            capture_output=True
        )

        # Print output and time taken
        print(result.stdout)
        print(f"✅ {script} completed in {time.time() - start_time:.2f}s")

    except subprocess.CalledProcessError as e:
        print(f"❌ Error in {script}:")
        print(e.stderr)
        break  # Stop if any script fails (remove if you want to continue)

    except FileNotFoundError:
        print(f"⚠️ {script} not found. Check filename/spelling.")
        break

print("\n✨ All scripts executed in order!")