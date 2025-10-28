"""Tests for table name validation."""

import pandas as pd
import pytest

from trashpandas.exceptions import ValidationError
from trashpandas.validation import validate_schema_name, validate_table_name


class TestTableNameValidation:
    """Test table name validation functionality."""

    def test_valid_names(self):
        """Test that valid names pass validation."""
        valid_names = [
            "users",
            "user_data",
            "table123",
            "MyTable",
            "Table_2024",
            "a",  # Single character
            "USERS",
            "UsErS",  # Mixed case
        ]
        for name in valid_names:
            validate_table_name(name)  # Should not raise

    def test_valid_names_with_dots(self):
        """Test that names with dots pass when allowed."""
        valid_names = [
            "schema.users",
            "db.schema.customers",
            "my_schema.my_data",
        ]
        for name in valid_names:
            validate_table_name(name, name_format="qualified")  # Should not raise

    def test_empty_names(self):
        """Test that empty names are rejected."""
        invalid_names = ["", "   ", "\t", "\n"]
        for name in invalid_names:
            with pytest.raises(ValidationError) as exc_info:
                validate_table_name(name)
            assert "empty or whitespace-only" in str(exc_info.value).lower()

    def test_none_name(self):
        """Test that None is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            validate_table_name(None)
        assert "cannot be None" in str(exc_info.value)

    def test_too_long_names(self):
        """Test that overly long names are rejected."""
        long_name = "a" * 256  # Default max is 255
        with pytest.raises(ValidationError) as exc_info:
            validate_table_name(long_name)
        assert "too long" in str(exc_info.value).lower()

    def test_path_traversal(self):
        """Test that path traversal sequences are blocked."""
        dangerous_names = [
            "../etc/passwd",
            "..\\windows\\system32",
            "./../../secret",
            "valid/../bad",
            "table..name",
        ]
        for name in dangerous_names:
            with pytest.raises(ValidationError) as exc_info:
                validate_table_name(name)
            assert "path traversal" in str(exc_info.value).lower()

    def test_absolute_paths(self):
        """Test that absolute paths are blocked."""
        dangerous_names = [
            "/etc/passwd",
            "\\windows\\system32",
            "/tmp/test_data",  # noqa: S108
            "C:/Users/test",
            "D:\\data",
        ]
        for name in dangerous_names:
            with pytest.raises(ValidationError) as exc_info:
                validate_table_name(name)
            assert "path" in str(exc_info.value).lower()

    def test_sql_injection_characters(self):
        """Test that SQL injection characters are blocked."""
        dangerous_names = [
            "users; DROP TABLE",
            "'; DROP TABLE users--",
            "admin' OR '1'='1",
            "table--comment",
            "table/*comment*/",
            'table"name',
            "table'name",
            "table;name",
            "table=value",
        ]
        for name in dangerous_names:
            with pytest.raises(ValidationError) as exc_info:
                validate_table_name(name, storage_type="sql")
            # Should raise ValidationError (either from character validation
            # or SQL injection check)
            assert exc_info.value is not None

    def test_sql_reserved_keywords(self):
        """Test that SQL reserved keywords are rejected."""
        keywords = [
            "SELECT", "INSERT", "UPDATE", "DELETE", "DROP",
            "CREATE", "ALTER", "TABLE", "INDEX", "VIEW",
            "FROM", "WHERE", "JOIN", "UNION",
            # Test case insensitivity
            "select", "Select", "SeLeCt",
        ]
        for keyword in keywords:
            with pytest.raises(ValidationError) as exc_info:
                validate_table_name(keyword, storage_type="sql")
            assert "reserved keyword" in str(exc_info.value).lower()

    def test_filesystem_reserved_names(self):
        """Test that filesystem reserved names are rejected."""
        names = [
            "CON", "PRN", "AUX", "NUL",
            "COM1", "COM2", "COM9",
            "LPT1", "LPT9",
            # Test case insensitivity
            "con", "Con", "cOn",
        ]
        for name in names:
            with pytest.raises(ValidationError) as exc_info:
                validate_table_name(name, storage_type="csv")
            assert "reserved name" in str(exc_info.value).lower()

    def test_invalid_characters(self):
        """Test that invalid filesystem characters are rejected."""
        invalid_chars = ["<", ">", ":", '"', "|", "?", "*"]
        for char in invalid_chars:
            name = f"tab{char}le"
            with pytest.raises(ValidationError) as exc_info:
                validate_table_name(name, storage_type="csv")
            error_msg = str(exc_info.value).lower()
            assert "invalid" in error_msg or "character" in error_msg

    def test_invalid_start_characters(self):
        """Test that names starting with numbers are rejected."""
        invalid_names = ["123table", "9users", "0data"]
        for name in invalid_names:
            with pytest.raises(ValidationError) as exc_info:
                validate_table_name(name)
            assert "must start with a letter" in str(exc_info.value).lower()

    def test_dots_not_allowed_by_default(self):
        """Test that dots are rejected by default."""
        with pytest.raises(ValidationError):
            validate_table_name("schema.users", name_format="strict")

    def test_storage_type_specific_validation(self):
        """Test that validation is specific to storage type."""
        # SQL reserved keyword should fail for SQL storage
        with pytest.raises(ValidationError):
            validate_table_name("SELECT", storage_type="sql")

        # Filesystem reserved name should fail for file storage
        with pytest.raises(ValidationError):
            validate_table_name("CON", storage_type="csv")

        # Use a non-reserved name that's valid everywhere
        validate_table_name("my_data", storage_type="sql")
        validate_table_name("my_data", storage_type="csv")


class TestSchemaNameValidation:
    """Test schema name validation functionality."""

    def test_none_schema_allowed(self):
        """Test that None schema is allowed."""
        validate_schema_name(None)  # Should not raise

    def test_valid_schema_names(self):
        """Test that valid schema names pass."""
        valid_names = ["public", "my_schema", "schema123"]
        for name in valid_names:
            validate_schema_name(name)  # Should not raise

    def test_invalid_schema_names(self):
        """Test that invalid schema names are rejected."""
        invalid_names = ["", "schema; DROP", "../etc"]
        for name in invalid_names:
            with pytest.raises(ValidationError):
                validate_schema_name(name)


class TestIntegrationWithStorage:
    """Test validation integration with storage classes."""

    def test_sql_storage_validates_table_names(self):
        """Test that SqlStorage validates table names."""
        import tempfile

        from trashpandas.sql import SqlStorage

        with tempfile.TemporaryDirectory():
            storage = SqlStorage("sqlite:///:memory:")
            df = pd.DataFrame({"a": [1, 2, 3]})

            # Valid name should work
            storage.store(df, "valid_table")

            # Invalid names should raise ValidationError
            with pytest.raises(ValidationError):
                storage.store(df, "DROP")

            with pytest.raises(ValidationError):
                storage.store(df, "../etc/passwd")

    def test_csv_storage_validates_table_names(self):
        """Test that CsvStorage validates table names."""
        import tempfile

        from trashpandas.csv import CsvStorage

        with tempfile.TemporaryDirectory() as tmpdir:
            storage = CsvStorage(tmpdir)
            df = pd.DataFrame({"a": [1, 2, 3]})

            # Valid name should work
            storage.store(df, "valid_table")

            # Invalid names should raise ValidationError
            with pytest.raises(ValidationError):
                storage.store(df, "CON")

            with pytest.raises(ValidationError):
                storage.store(df, "../etc/passwd")

    def test_pickle_storage_validates_table_names(self):
        """Test that PickleStorage validates table names."""
        import tempfile

        from trashpandas.pickle import PickleStorage

        with tempfile.TemporaryDirectory() as tmpdir:
            storage = PickleStorage(tmpdir)
            df = pd.DataFrame({"a": [1, 2, 3]})

            # Valid name should work
            storage.store(df, "valid_table")

            # Invalid names should raise ValidationError
            with pytest.raises(ValidationError):
                storage.store(df, "NUL")

            with pytest.raises(ValidationError):
                storage.store(df, "table;drop")

    def test_load_validates_table_names(self):
        """Test that load operations also validate table names."""
        import tempfile

        from trashpandas.sql import SqlStorage

        with tempfile.TemporaryDirectory():
            storage = SqlStorage("sqlite:///:memory:")

            # Invalid names should raise ValidationError even for load
            with pytest.raises(ValidationError):
                storage.load("../../../etc/passwd")

    def test_delete_validates_table_names(self):
        """Test that delete operations also validate table names."""
        import tempfile

        from trashpandas.csv import CsvStorage

        with tempfile.TemporaryDirectory() as tmpdir:
            storage = CsvStorage(tmpdir)

            # Invalid names should raise ValidationError even for delete
            with pytest.raises(ValidationError):
                storage.delete("CON")

