
import sys

def add(a, b): # Bug: String concatenation
    print(f"DEBUG: add called with {a=}, {b=}")
    return a + b
def subtract(a, b): return int(a) - int(b) # Correct

def calculate(op, x_str, y_str):
    x = x_str; y = y_str # Bug: No conversion
    if op == 'add': return add(x, y)
    elif op == 'subtract': return subtract(x, y) # Bug: passes strings
    else: raise ValueError(f"Unknown operation: {op}")

if __name__ == "__main__":
    if len(sys.argv) != 4: print("Usage: python calculator.py <add|subtract> <num1> <num2>"); sys.exit(1)
    op, n1, n2 = sys.argv[1], sys.argv[2], sys.argv[3]
    try: print(f"Result: {calculate(op, n1, n2)}")
    except Exception as e: print(f"Error: {e}"); sys.exit(1)
