import os
total_lines = 0
total_files = 0
ext = ('.hpp', '.cpp', '.py', '.h', '.swig')
exclude_dirs = {'__pycache__', 'datasets'}
print("=== File line counts (hypervector-main) ===")
for root in ('src', 'pyhypervec', 'test', 'docs', 'cmake', 'scripts'):
    for r, ds, fs in os.walk(root):
        ds[:] = [d for d in ds if d not in exclude_dirs]
        for f in sorted(fs):
            if any(f.endswith(e) for e in ext):
                fp = os.path.join(r, f)
                try:
                    with open(fp, 'r', encoding='utf-8', errors='ignore') as h:
                        lc = sum(1 for _ in h)
                    total_lines += lc
                    total_files += 1
                    print(f"  {os.path.relpath(fp)}: {lc}")
                except Exception:
                    print(f"  {os.path.relpath(fp)}: [error]")
print(f"\n=== Summary ===")
print(f"Total files: {total_files}")
print(f"Total lines: {total_lines}")