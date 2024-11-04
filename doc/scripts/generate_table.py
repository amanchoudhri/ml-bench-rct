#!/usr/bin/env python3
"""
Generate LaTeX table from datasets.csv using pandas Styler.
"""

import pandas as pd
from pathlib import Path
import sys

def find_project_root():
    """Find the project root by looking for src/ directory"""
    current = Path(__file__).resolve().parent
    while current.parent != current:
        if (current / 'src').exists():
            return current
        current = current.parent
    raise FileNotFoundError("Could not find project root (no src/ directory found)")

def format_dataset_name(name, link):
    """Format dataset name with hyperlink"""
    if pd.isna(link) or pd.isna(name):
        return name
    return f"\\href{{{link}}}{{\\texttt{{{name}}}}}"

def format_int_or_missing(x):
    """Convert to integer if possible, otherwise return ---"""
    try:
        if pd.isna(x) or str(x).strip() == '':
            return '---'
        return f"{int(float(x)):,d}"  # Handle both string and float inputs
    except (ValueError, TypeError):
        return '---'

def generate_latex_table(csv_path, output_path):
    """Generate LaTeX table from CSV file using pandas Styler"""
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Define desired column order (using new names)
    column_order = ['Dataset', 'Task', 'Size', 'Num Classes', 'Average Image Size']
    
    # Select original columns in the desired order
    df_selected = df[column_order]
    
    # Apply hyperlinks to dataset names
    df_selected = df_selected.sort_values(by='Dataset')
    df_selected['Dataset'] = df.apply(
        lambda x: format_dataset_name(x['Dataset'], x['Link']), 
        axis=1
    )
    df_selected.set_index('Dataset', inplace=True)
    df_selected.index.name = None  # Remove index name to avoid the extra row
    print(df_selected.head())
    print(df_selected.index)
    
    # Create Styler object
    styler = df_selected.style
    
    # Format the table
    styler.format(na_rep='---')  # Base NA formatting
    styler.format(  # Integer formatting for Size and Classes
        formatter={
            'Size': format_int_or_missing,
            'Classes': format_int_or_missing
        }
    )
    
    # Generate LaTeX
    latex_code = styler.to_latex(
        column_format='ll' + 'r' * (len(df_selected.columns) - 1),  # left align first col, center others
        position='htbp',
        position_float='centering',
        hrules=True,  # Add booktabs rules
        label='tab:datasets',
        caption='Overview of Image Classification Datasets'
    )
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(latex_code)
    
    print(f"Generated LaTeX table at {output_path}")

if __name__ == '__main__':
    try:
        # Find project root and set up paths
        project_root = find_project_root()
        csv_path = project_root / 'src' / 'datasets.csv'
        output_path = project_root / 'doc' / 'generated' / 'dataset_table.tex'
        
        generate_latex_table(csv_path, output_path)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
