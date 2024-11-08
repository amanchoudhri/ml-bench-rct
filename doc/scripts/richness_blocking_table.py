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


def generate_latex_table(csv_path, output_path):
    """Generate LaTeX table from CSV file using pandas Styler"""
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Compute image size
    richness = df['total_samples'] / df['num_classes']

    poor = richness <= 100
    rich = richness > 100

    df['richness'] = 0
    df.loc[rich, 'richness'] = 1

    poor_label = '< 100 examples/class'
    rich_label = '> 100 examples/class'


    grouped = df.groupby('richness').count()
    grouped['Count'] = grouped['dataset_name']

    grouped['Richness Range'] = [poor_label, rich_label]

    grouped = grouped.loc[:, ['Richness Range', 'Count']]

    grouped.index.name = '$R_{i}$'

    # Create Styler object
    styler = grouped.style

    # Generate LaTeX
    latex_code = styler.to_latex(
        position='htbp',
        position_float='centering',
        hrules=True,  # Add booktabs rules
        label='tab:richness_blocking',
        caption='Number of Datasets by Richness'
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
        csv_path = project_root / 'datasets.csv'
        output_path = project_root / 'doc' / 'generated' / 'richness_blocking_table.tex'
        
        generate_latex_table(csv_path, output_path)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
