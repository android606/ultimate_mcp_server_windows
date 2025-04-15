import re

# Test our regex patterns with improved handling of quotes
patterns = [
    r"sheet(?:s)?\s*(?:named|called|:)?\s*((?:'[^']*'|\"[^\"]*\"|[^,\s]+)(?:\s*,?\s*and\s*|\s*,\s*)?)*((?:'[^']*'|\"[^\"]*\"|[^,\s]+))?",
    r"create (?:a |)sheet(?:s)? (?:named|called)?\s*((?:'[^']*'|\"[^\"]*\"|[^,\s]+)(?:\s*,?\s*and\s*|\s*,\s*)?)*((?:'[^']*'|\"[^\"]*\"|[^,\s]+))?",
    r"in (?:the |)(?:sheet|worksheet) (?:'([^']*)'|\"([^\"]*)\"|([^,\s]+))",
    r"(?:in|to) (?:the |)(?:sheet|worksheet)(?:s|) ((?:'[^']*'|\"[^\"]*\"|[^,\s]+)(?:\s*,?\s*and\s*|\s*,\s*)?)*((?:'[^']*'|\"[^\"]*\"|[^,\s]+))?"
]

test_strings = [
    "sheets named 'Revenue' and 'Expenses'",
    "create sheets named Marketing, Sales, and Expenses",
    "in the sheet 'Financial Data'",
    "to sheets Revenue and 'Expenses'"
]

for i, pattern in enumerate(patterns):
    print(f"Pattern {i+1}: {pattern}")
    for j, test_string in enumerate(test_strings):
        matches = re.findall(pattern, test_string)
        print(f"  String {j+1}: '{test_string}' => {matches}")
    print()

# Simpler versions for the actual code
simplified_patterns = [
    r"sheet(?:s)?\s*(?:named|called|:)?\s*(?:'([^']*)'|\"([^\"]*)\"|([A-Za-z0-9_, ]+))",
    r"create (?:a |)sheet(?:s)? (?:named|called)?\s*(?:'([^']*)'|\"([^\"]*)\"|([A-Za-z0-9_, ]+))",
    r"in (?:the |)(?:sheet|worksheet) (?:'([^']*)'|\"([^\"]*)\"|([A-Za-z0-9_]+))",
    r"(?:in|to) (?:the |)(?:sheet|worksheet)(?:s|) (?:'([^']*)'|\"([^\"]*)\"|([A-Za-z0-9_, ]+))"
]

print("Simplified patterns:")
for i, pattern in enumerate(simplified_patterns):
    print(f"Pattern {i+1}: {pattern}")
    for j, test_string in enumerate(test_strings):
        matches = re.findall(pattern, test_string)
        print(f"  String {j+1}: '{test_string}' => {matches}")
    print() 