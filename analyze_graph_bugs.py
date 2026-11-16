"""Analyze graph.py for bugs."""
import ast

# Read and parse graph.py
graph_path = "ml/rag/chatbot/graph.py"

with open(graph_path, "r") as f:
    source = f.read()

tree = None
try:
    tree = ast.parse(source)
    print("[OK] Syntax is valid")
except SyntaxError as e:
    print(f"[ERROR] Syntax error at line {e.lineno}: {e.msg}")

# Find all functions and show their signatures (skip if parse failed)
if tree:
    print("\n--- Function signatures ---")
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            arg_names = [arg.arg for arg in node.args.args]  # Use .arg for AST function arguments (not .id)
            sig = f"{node.name}({', '.join(arg_names)})"
            print(sig)
