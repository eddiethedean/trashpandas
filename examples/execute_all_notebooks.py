"""Execute all notebooks and capture outputs, then save them back."""

import json
import sys
import subprocess
from pathlib import Path


def execute_notebook_with_outputs(notebook_path):
    """Execute a notebook and capture outputs by running Python directly."""
    print(f"\n{'='*60}")
    print(f"Executing: {notebook_path.name}")
    print(f"{'='*60}\n")
    
    with open(notebook_path) as f:
        notebook = json.load(f)
    
    # Collect all code cell sources
    code_cells = []
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = '\n'.join(cell['source'])
            if source.strip():
                code_cells.append(source)
    
    # Create a temporary Python script to execute
    temp_script = Path('temp_execute.py')
    
    try:
        # Write all code to a temporary file
        with open(temp_script, 'w') as f:
            f.write('\n\n'.join(code_cells))
        
        # Execute the script and capture output
        result = subprocess.run(
            [sys.executable, str(temp_script)],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print(f"✓ Executed successfully")
            # Add the output to the notebook
            output_cell_idx = 0
            for i, cell in enumerate(notebook['cells']):
                if cell['cell_type'] == 'code' and cell['source']:
                    # Add execution count
                    cell['execution_count'] = output_cell_idx + 1
                    # Add output
                    cell['outputs'] = [{
                        'output_type': 'stream',
                        'name': 'stdout',
                        'text': result.stdout.split('\n') if result.stdout else ['[No output]']
                    }]
                    output_cell_idx += 1
        else:
            print(f"✗ Execution failed")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    finally:
        if temp_script.exists():
            temp_script.unlink()
    
    # Save the notebook with outputs
    output_path = notebook_path.parent / f"executed_{notebook_path.name}"
    with open(output_path, 'w') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"Saved to: {output_path.name}")
    
    return True


def main():
    """Execute all notebooks."""
    examples_dir = Path('examples')
    notebooks = sorted(examples_dir.glob('*.ipynb'))
    
    # Exclude already executed and test files
    notebooks = [nb for nb in notebooks if not nb.name.startswith(('executed_', 'test_', 'create_'))]
    
    if not notebooks:
        print("No notebooks found!")
        return 1
    
    print(f"Found {len(notebooks)} notebooks to execute\n")
    
    all_passed = True
    for notebook in notebooks:
        if not execute_notebook_with_outputs(notebook):
            all_passed = False
    
    if all_passed:
        print(f"\n{'='*60}")
        print("✓ All notebooks executed successfully!")
        print("="*60)
    else:
        print("\n✗ Some notebooks failed to execute")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())

