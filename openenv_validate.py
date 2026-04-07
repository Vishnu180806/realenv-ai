import os
import sys
from pathlib import Path

def validate_multi_mode_deployment(env_path: Path):
    issues = []
    
    # 1. Check pyproject.toml exists
    pyproject_path = env_path / "pyproject.toml"
    if not pyproject_path.exists():
        issues.append("Missing pyproject.toml")
        return False, issues

    # 2. Check uv.lock exists
    lockfile_path = env_path / "uv.lock"
    if not lockfile_path.exists():
        issues.append("Missing uv.lock - run 'uv lock' to generate it")

    # 3. Parse pyproject.toml
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError:
            # Simple fallback parser for dependencies if no toml library is present
            tomllib = None
            
    if tomllib:
        try:
            with open(pyproject_path, "rb") as f:
                pyproject = tomllib.load(f)
            
            # 4. Check [project.scripts] section
            scripts = pyproject.get("project", {}).get("scripts", {})
            if "server" not in scripts:
                issues.append("Missing [project.scripts] server entry point")
            else:
                server_entry = scripts.get("server", "")
                if ":main" not in server_entry:
                    issues.append(f"Server entry point should reference main function, got: {server_entry}")

            # 5. Check required dependencies
            deps = [str(dep).lower() for dep in pyproject.get("project", {}).get("dependencies", [])]
            has_openenv = any(
                (dep.startswith("openenv") and not dep.startswith("openenv-core")) for dep in deps
            )
            has_core = any(dep.startswith("openenv-core") for dep in deps)
            if not (has_openenv or has_core):
                issues.append("Missing required dependency: openenv-core>=0.2.0")
        except Exception as e:
            issues.append(f"Error parsing pyproject.toml: {e}")

    # 6. Check server/app.py exists
    server_app = env_path / "server" / "app.py"
    if not server_app.exists():
        issues.append("Missing server/app.py")
    else:
        app_content = server_app.read_text(encoding="utf-8")
        if "def main(" not in app_content:
            issues.append("server/app.py missing main() function")
        if "__name__" not in app_content or "main()" not in app_content:
            issues.append("server/app.py main() function not callable (missing if __name__ == '__main__')")

    return len(issues) == 0, issues

def main():
    path = Path(".").absolute()
    print(f"\nRunning OpenEnv Validation for: {path.name}")
    print("-" * 50)
    
    success, issues = validate_multi_mode_deployment(path)
    
    if success:
        print("✅ [OK] Ready for multi-mode deployment")
    else:
        print(f"❌ [FAIL] Not ready for multi-mode deployment")
        print("\nIssues found:")
        for issue in issues:
            print(f"  - {issue}")
    print("-" * 50 + "\n")

if __name__ == "__main__":
    main()
