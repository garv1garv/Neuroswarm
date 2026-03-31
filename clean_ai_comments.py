import os
import re

def clean_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Remove the AI-like header blocks for C++/Python
    # Removes lines like:
    # // Copyright (c) ...
    
    # regex for Python triple-quote docstrings at the very top
    
    # regex for Python triple-quote docstrings containing NeuroSwarm at the top
    content = re.sub(r'(?s)\A\s*""".*?NeuroSwarm.*?MIT License.*?"""\n?', '', content)
    
    # regex for C++/Python header comments
    lines = content.split('\n')
    new_lines = []
    skip = False
    
    for i, line in enumerate(lines):
        # Skip top boilerplate consecutive comments
        if i < 15 and ('NeuroSwarm — ' in line or 'Copyright (c) 2026 NeuroSwarm' in line or 'MIT License' in line):
            continue
            
        # Remove AI section separators (e.g., // ─── or # ─── or // ===)
        if re.match(r'^\s*(//|#)\s*[─=]{5,}', line):
            continue

        # Remove "Alpha", "Omega" from planner/validator setup
        line = line.replace('"Planner"', '"Planner"')
        line = line.replace('"Validator"', '"Validator"')
        
        # Remove overly enthusiastic words
        line = line.replace('', '')
        
        new_lines.append(line)

    # Join lines back
    output = '\n'.join(new_lines)
    
    # Remove multiple consecutive blank lines
    output = re.sub(r'\n{3,}', '\n\n', output)
    
    # Remove completely blank lines at the top of the file
    output = output.lstrip('\n')

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(output)

def main():
    target_dir = r"c:\Users\garvp\Downloads\9\neuroswarm"
    for root, dirs, files in os.walk(target_dir):
        for file in files:
            if file.endswith(('.py', '.cpp', '.cu', '.h', '.hpp')):
                filepath = os.path.join(root, file)
                try:
                    clean_file(filepath)
                    print(f"Cleaned {filepath}")
                except Exception as e:
                    print(f"Failed to clean {filepath}: {e}")

if __name__ == "__main__":
    main()
