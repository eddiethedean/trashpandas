"""Script to create example notebooks for TrashPandas features."""

import json
from pathlib import Path


def create_notebook(notebook_name, cells):
    """Create a Jupyter notebook from a list of cells."""
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.0",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }

    # Write to file
    output_path = Path("examples") / notebook_name
    with open(output_path, "w") as f:
        json.dump(notebook, f, indent=1)
    print(f"Created: {output_path}")


def markdown_cell(source):
    """Create a markdown cell."""
    # Use splitlines to preserve newlines properly
    lines = source.splitlines(True)
    return {"cell_type": "markdown", "metadata": {}, "source": lines}


def code_cell(source):
    """Create a code cell."""
    # Use splitlines to preserve newlines properly
    lines = source.splitlines(True)
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines}


# Notebook 1: Basic Usage
notebook1_cells = [
    markdown_cell("# TrashPandas: Basic Usage\n\nThis notebook demonstrates the basic usage of TrashPandas."),
    markdown_cell("## Installation and Imports"),
    code_cell('import pandas as pd\nimport numpy as np\nimport trashpandas as tp\n\nprint(f"TrashPandas version: {tp.__version__}")\nprint(f"Pandas version: {pd.__version__}")'),
    markdown_cell("## Creating Sample Data"),
    code_cell(
        "# Create sample DataFrames\n"
        "users_df = pd.DataFrame({\n"
        "    'id': [1, 2, 3, 4, 5],\n"
        "    'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],\n"
        "    'age': [25, 30, 35, 28, 32],\n"
        "    'email': ['alice@example.com', 'bob@example.com', 'charlie@example.com', \n"
        "              'diana@example.com', 'eve@example.com'],\n"
        "    'active': [True, True, False, True, True]\n"
        "})\n\n"
        'print("Sample DataFrame created:")\n'
        "print(users_df)",
    ),
    markdown_cell("## 1. CSV Storage"),
    code_cell(
        "# Using context manager (recommended)\n"
        "with tp.CsvStorage('./data_csv') as csv_storage:\n"
        "    # Store DataFrame\n"
        "    csv_storage['users'] = users_df\n"
        "    \n"
        "    # Check how many tables we have\n"
        '    print(f"Stored {len(csv_storage)} tables")\n'
        '    print(f"Table names: {list(csv_storage)}")\n'
        "    \n"
        "    # Retrieve DataFrame\n"
        "    loaded_users = csv_storage['users']\n"
        "    \n"
        '    print("\\nLoaded DataFrame:")\n'
        "    print(loaded_users)\n"
        "    \n"
        "    # Verify data integrity\n"
        '    print("\\nDataFrames match:", users_df.equals(loaded_users))',
    ),
    markdown_cell("## 2. SQL Storage"),
    code_cell(
        "# SQLite database\n"
        "with tp.SqlStorage('sqlite:///./data_sql.db') as sql_storage:\n"
        "    # Store DataFrame\n"
        "    sql_storage['users'] = users_df\n"
        "    \n"
        '    print(f"Stored {len(sql_storage)} tables")\n'
        "    \n"
        "    # Load DataFrame\n"
        "    loaded_users = sql_storage['users']\n"
        '    print("\\nLoaded from SQL:")\n'
        "    print(loaded_users)\n"
        "    \n"
        "    # Query with filtering\n"
        "    active_users = sql_storage.query('users', where_clause=\"active = 1\")\n"
        '    print("\\nActive users:")\n'
        "    print(active_users)",
    ),
    markdown_cell("## 3. Pickle Storage"),
    code_cell(
        "with tp.PickleStorage('./data_pickle') as pickle_storage:\n"
        "    # Store DataFrame\n"
        "    pickle_storage['users'] = users_df\n"
        "    \n"
        '    print(f"Stored {len(pickle_storage)} tables")\n'
        "    \n"
        "    # Load DataFrame\n"
        "    loaded_users = pickle_storage['users']\n"
        '    print("\\nLoaded from Pickle:")\n'
        "    print(loaded_users)\n"
        "    \n"
        "    # Data types are perfectly preserved\n"
        '    print("\\nData types:")\n'
        "    print(loaded_users.dtypes)",
    ),
]

create_notebook("01_basic_usage.ipynb", notebook1_cells)

# Notebook 2: Advanced Features
notebook2_cells = [
    markdown_cell("# TrashPandas: Advanced Features\n\nThis notebook demonstrates advanced features of TrashPandas."),
    markdown_cell("## Imports"),
    code_cell("import pandas as pd\nimport trashpandas as tp\n"),
    markdown_cell("## Compression Support"),
    code_cell(
        "# Create sample data\n"
        "large_df = pd.DataFrame({\n"
        "    'id': range(1000),\n"
        "    'value': range(1000),\n"
        "    'text': ['x'] * 1000\n"
        "})\n\n"
        "# Store with gzip compression\n"
        "with tp.CsvStorage('./data_compressed', compression='gzip') as storage:\n"
        "    storage['large_data'] = large_df\n"
        "    \n"
        "    # Load compressed data (automatic detection)\n"
        "    loaded = storage['large_data']\n"
        '    print(f"Successfully loaded {len(loaded)} rows from compressed file")',
    ),
    markdown_cell("## Bulk Operations"),
    code_cell(
        "# Create multiple DataFrames\n"
        "users = pd.DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob']})\n"
        "products = pd.DataFrame({'id': [1, 2], 'name': ['Widget', 'Gadget']})\n"
        "orders = pd.DataFrame({'id': [1, 2], 'user_id': [1, 2]})\n\n"
        "all_data = {'users': users, 'products': products, 'orders': orders}\n\n"
        "with tp.SqlStorage('sqlite:///./bulk.db') as storage:\n"
        "    # Store many at once\n"
        "    storage.store_many(all_data)\n"
        '    print(f"Stored {len(storage)} tables")\n'
        "    \n"
        "    # Load many at once\n"
        "    loaded = storage.load_many(['users', 'products', 'orders'])\n"
        '    print(f"Loaded {len(loaded)} tables")\n'
        "    \n"
        "    # Delete many at once\n"
        "    storage.delete_many(['orders'])\n"
        '    print(f"After deletion: {len(storage)} tables")',
    ),
    markdown_cell("## Data Type Preservation"),
    code_cell(
        "# Create DataFrame with various types\n"
        "df = pd.DataFrame({\n"
        "    'int_col': [1, 2, 3],\n"
        "    'float_col': [1.1, 2.2, 3.3],\n"
        "    'str_col': ['a', 'b', 'c'],\n"
        "    'bool_col': [True, False, True],\n"
        "    'date_col': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])\n"
        "})\n\n"
        'print("Original data types:")\n'
        "print(df.dtypes)\n\n"
        "with tp.PickleStorage('./data_types') as storage:\n"
        "    storage['typed_data'] = df\n"
        "    loaded = storage['typed_data']\n"
        "    \n"
        '    print("\\nLoaded data types:")\n'
        "    print(loaded.dtypes)\n"
        "    \n"
        '    print("\\nData types preserved:")\n'
        "    print(df.dtypes.equals(loaded.dtypes))",
    ),
]

create_notebook("02_advanced_features.ipynb", notebook2_cells)

# Notebook 3: Format Conversion
notebook3_cells = [
    markdown_cell("# TrashPandas: Format Conversion\n\nConvert DataFrames between different storage formats."),
    markdown_cell("## Imports"),
    code_cell("import pandas as pd\nimport trashpandas as tp\n"),
    markdown_cell("## Converting Between Formats"),
    code_cell(
        "# Create sample data\n"
        "df = pd.DataFrame({\n"
        "    'id': [1, 2, 3],\n"
        "    'name': ['Alice', 'Bob', 'Charlie'],\n"
        "    'age': [25, 30, 35]\n"
        "})\n\n"
        "# Store in CSV first\n"
        "with tp.CsvStorage('./data_conversion') as csv_storage:\n"
        "    csv_storage['users'] = df\n"
        "    \n"
        "    # Convert CSV to SQL\n"
        "    tp.csv_to_sql('users', './data_conversion', 'sqlite:///./converted.db')\n"
        '    print("Converted from CSV to SQL")\n'
        "    \n"
        "    # Load from SQL to verify\n"
        "    with tp.SqlStorage('sqlite:///./converted.db') as sql_storage:\n"
        "        loaded = sql_storage['users']\n"
        '        print("\\nLoaded from converted SQL database:")\n'
        "        print(loaded)",
    ),
    markdown_cell("## Batch Conversion"),
    code_cell(
        "# Create multiple DataFrames\n"
        "users = pd.DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob']})\n"
        "products = pd.DataFrame({'id': [1, 2], 'name': ['Widget', 'Gadget']})\n\n"
        "with tp.CsvStorage('./data_batch') as csv_storage:\n"
        "    csv_storage['users'] = users\n"
        "    csv_storage['products'] = products\n"
        "    \n"
        "    # Convert all tables from CSV to SQL\n"
        "    tp.csv_to_sql_all('./data_batch', 'sqlite:///./batch_converted.db')\n"
        '    print("Batch conversion complete")\n'
        "    \n"
        "    # Verify\n"
        "    with tp.SqlStorage('sqlite:///./batch_converted.db') as sql_storage:\n"
        '        print(f"Converted {len(sql_storage)} tables")',
    ),
]

create_notebook("03_format_conversion.ipynb", notebook3_cells)

# Notebook 4: Query Capabilities
notebook4_cells = [
    markdown_cell("# TrashPandas: Query Capabilities\n\nAdvanced querying features for SQL storage."),
    markdown_cell("## Imports"),
    code_cell("import pandas as pd\nimport trashpandas as tp\n"),
    markdown_cell("## Basic Queries"),
    code_cell(
        "# Create comprehensive sample data\n"
        "users = pd.DataFrame({\n"
        "    'id': [1, 2, 3, 4, 5],\n"
        "    'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],\n"
        "    'age': [25, 30, 35, 28, 32],\n"
        "    'city': ['NYC', 'SF', 'NYC', 'LA', 'NYC'],\n"
        "    'active': [True, True, False, True, True]\n"
        "})\n\n"
        "with tp.SqlStorage('sqlite:///./query_demo.db') as storage:\n"
        "    storage['users'] = users\n"
        "    \n"
        "    # Get all data\n"
        "    all_users = storage.query('users')\n"
        '    print(f"All users: {len(all_users)} rows")\n'
        "    print(all_users)",
    ),
    markdown_cell("## Filtering with WHERE Clause"),
    code_cell(
        "with tp.SqlStorage('sqlite:///./query_demo.db') as storage:\n"
        "    # Filter by age\n"
        "    young_users = storage.query('users', where_clause=\"age < 30\")\n"
        '    print("Users under 30:")\n'
        "    print(young_users)\n"
        "    \n"
        "    # Filter by city\n"
        "    nyc_users = storage.query('users', where_clause=\"city = 'NYC'\")\n"
        '    print("\\nUsers in NYC:")\n'
        "    print(nyc_users)\n"
        "    \n"
        "    # Filter by boolean\n"
        "    active_users = storage.query('users', where_clause=\"active = 1\")\n"
        '    print("\\nActive users:")\n'
        "    print(active_users)",
    ),
    markdown_cell("## Selecting Specific Columns"),
    code_cell(
        "with tp.SqlStorage('sqlite:///./query_demo.db') as storage:\n"
        "    # Get only name and email\n"
        "    names = storage.query('users', columns=['name', 'city'])\n"
        '    print("Names and cities:")\n'
        "    print(names)",
    ),
    markdown_cell("## Limiting Results"),
    code_cell(
        "with tp.SqlStorage('sqlite:///./query_demo.db') as storage:\n"
        "    # Get first 3 users\n"
        "    top_users = storage.query('users', limit=3)\n"
        '    print("First 3 users:")\n'
        "    print(top_users)",
    ),
]

create_notebook("04_query_capabilities.ipynb", notebook4_cells)

print("\nAll notebooks created successfully!")

