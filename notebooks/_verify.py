import json
nb = json.load(open('notebooks/demo_inference.ipynb', 'r', encoding='utf-8'))
print(f"Total cells: {len(nb['cells'])}")
for i, c in enumerate(nb['cells']):
    first_line = c['source'][0].strip()[:80] if c['source'] else '(empty)'
    print(f"  Cell {i:2d}: {c['cell_type']:8s} | {first_line}")
