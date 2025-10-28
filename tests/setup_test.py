import os

from pandas import DataFrame


def create_df() -> DataFrame:
    return DataFrame({"id": [1, 2, 3],
                      "name": ["Joe", "Bob", "John"],
                      "age": [23, 34, 44]})


def create_named_index_df() -> DataFrame:
    df = create_df()
    df.index.name = "x"
    return df


def create_two_named_index_df() -> DataFrame:
    df = create_df()
    return df.set_index(["id", "age"])


def create_two_unnamed_index_df() -> DataFrame:
    df = create_two_named_index_df()
    df.index.names = [None, None]
    return df


def create_two_index_one_named_df() -> DataFrame:
    df = create_two_named_index_df()
    df.index.names = ["id", None]
    return df


def create_df_string() -> DataFrame:
    df = create_df()
    df["name"] = df["name"].astype("string")
    return df


def delete_all_files(path):
    """Delete all files in a directory, handling Windows file locking issues."""
    import shutil
    import time

    # Check if directory exists first
    if not os.path.exists(path):
        return

    for f in os.listdir(path):
        file_path = os.path.join(path, f)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except (PermissionError, OSError) as e:
            # On Windows, files might be locked by SQLite connections
            # Try a few times with small delays
            for attempt in range(3):
                time.sleep(0.1)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                    break
                except (PermissionError, OSError):
                    if attempt == 2:  # Last attempt
                        # Log the error but don't fail the test
                        # Using warnings module instead of print
                        import warnings
                        warnings.warn(
                            f"Could not delete {file_path}: {e}", stacklevel=2,
                        )
                    continue
