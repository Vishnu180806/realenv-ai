import os
import sys

def check_file(path, description):
    if os.path.exists(path):
        print(f"✅ {description} found: {path}")
        return True
    else:
        print(f"❌ {description} MISSING: {path}")
        return False

def main():
    print("\n" + "="*50)
    print("      OPENENV LOCAL VALIDATOR (SIMULATED)")
    print("="*50 + "\n")
    
    checks = [
        ("openenv.yaml", "OpenEnv Specification"),
        ("pyproject.toml", "Project Configuration"),
        ("uv.lock", "Mandatory Lock File"),
        ("Dockerfile", "Deployment Container"),
        ("inference.py", "Agent Entry Point"),
        ("server/app.py", "FastAPI Server Entry Point"),
    ]
    
    all_passed = True
    for file, desc in checks:
        if not check_file(file, desc):
            all_passed = False
            
    print("\n" + "-"*50)
    if all_passed:
        print("🏆 SUCCESS: Your repository is ready for multi-mode deployment!")
        print("Hint: All required files are in the correct root/server locations.")
    else:
        print("🚩 FAIL: Some required files are missing. Please fix the red items above.")
    print("-"*50 + "\n")

if __name__ == "__main__":
    main()
