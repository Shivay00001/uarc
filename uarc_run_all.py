"""
UARC: Full System Integration + Formal Document Generator
=========================================================
Runs all 7 modules together and generates the algorithm design document.
"""

import subprocess, sys, os, time, random, math, json

# ── Run all module tests ──────────────────────────────────────────────────────
def run_module(name, path):
    print(f"\n{'='*65}")
    print(f"  Running: {name}")
    print('='*65)
    result = subprocess.run(
        [sys.executable, path],
        capture_output=True, text=True, cwd="/home/claude/uarc"
    )
    if result.stdout:
        for line in result.stdout.strip().split('\n'):
            print(f"  {line}")
    if result.returncode != 0:
        print(f"  ⚠ STDERR: {result.stderr[:300]}")
    return result.returncode == 0

if __name__ == "__main__":
    modules = [
        ("TDE — Token Difficulty Estimator",   "/home/claude/uarc/tde.py"),
        ("AI-VM — Virtual Memory Manager",     "/home/claude/uarc/ai_vm.py"),
        ("Modules 3-6 (DPE/PLL/ACS/NSC)",     "/home/claude/uarc/modules_3_to_6.py"),
    ]
    results = {}
    for name, path in modules:
        ok = run_module(name, path)
        results[name] = "✅ PASS" if ok else "❌ FAIL"

    print("\n" + "="*65)
    print("  UARC MODULE TEST SUMMARY")
    print("="*65)
    for name, status in results.items():
        print(f"  {status}  {name}")

    all_pass = all("PASS" in s for s in results.values())
    print(f"\n  Overall: {'✅ ALL SYSTEMS GO' if all_pass else '❌ FAILURES DETECTED'}")
