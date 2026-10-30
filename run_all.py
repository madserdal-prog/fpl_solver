"""Runs every test_* function without needing pytest (CI uses pytest; this is a fallback)."""
import importlib, os, sys, traceback
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here); sys.path.insert(0, os.path.dirname(here))
failed = passed = 0
for fn in sorted(f for f in os.listdir(here) if f.startswith("test_") and f.endswith(".py")):
    mod = importlib.import_module(fn[:-3])
    for name in sorted(n for n in dir(mod) if n.startswith("test_")):
        try:
            getattr(mod, name)(); passed += 1; print(f"PASS {fn}::{name}")
        except Exception:
            failed += 1; print(f"FAIL {fn}::{name}"); traceback.print_exc()
print(f"\n{passed} passed, {failed} failed"); sys.exit(1 if failed else 0)
