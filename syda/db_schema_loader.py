import json
import os
from typing import Dict, List, Optional, Union


def _map_sql_type(sql_type) -> str:
    t = str(sql_type).lower()
    if "int" in t:
        return "integer"
    elif "char" in t or "text" in t or "varchar" in t or "clob" in t:
        return "string"
    elif "date" in t or "time" in t:
        return "date"
    elif "decimal" in t or "numeric" in t or "float" in t or "real" in t or "double" in t:
        return "float"
    elif "bool" in t:
        return "boolean"
    else:
        return "string"


class DatabaseSchemaLoader:
    """Load schemas from relational databases (SQLite, MySQL, PostgreSQL) via SQLAlchemy.

    Usage::

        from syda import SyntheticDataGenerator, DatabaseSchemaLoader, ModelConfig

        loader = DatabaseSchemaLoader("sqlite:///mydb.db")

        # Option A — pass schema dicts directly
        schemas = loader.load_schemas()
        results = generator.generate_for_schemas(schemas=schemas)

        # Option B — write schema files first, pass file paths
        schema_files = loader.save_schemas("schemas/")
        results = generator.generate_for_schemas(schemas=schema_files)
    """

    def __init__(self, connection_string_or_engine: Union[str, object]):
        try:
            from sqlalchemy import create_engine, inspect as sa_inspect
        except ImportError:
            raise ImportError("SQLAlchemy is required: pip install sqlalchemy")

        if isinstance(connection_string_or_engine, str):
            self._engine = create_engine(connection_string_or_engine)
        else:
            self._engine = connection_string_or_engine

        from sqlalchemy import inspect as sa_inspect
        self._inspector = sa_inspect(self._engine)

    def load_schemas(
        self,
        table_names: Optional[List[str]] = None,
    ) -> Dict[str, Dict]:
        """Return schema dicts keyed by table name, ready for generate_for_schemas(schemas=...)."""
        return {t: self._build_table_schema(t) for t in self._resolve_tables(table_names)}

    def save_schemas(
        self,
        output_dir: str,
        table_names: Optional[List[str]] = None,
        format: str = "yaml",
    ) -> Dict[str, str]:
        """Save one schema file per table; return {table_name: absolute_file_path}.

        The returned dict can be passed directly to generate_for_schemas(schemas=...).
        """
        if format not in ("yaml", "json"):
            raise ValueError(f"Unsupported format '{format}'. Use 'yaml' or 'json'.")

        os.makedirs(output_dir, exist_ok=True)
        result = {}
        for table_name in self._resolve_tables(table_names):
            schema = self._build_table_schema(table_name)
            file_path = os.path.abspath(os.path.join(output_dir, f"{table_name}.{format}"))
            self._write_schema_file(schema, file_path, format)
            result[table_name] = file_path
        return result

    def _resolve_tables(self, table_names: Optional[List[str]]) -> List[str]:
        all_tables = self._inspector.get_table_names()
        if table_names is None:
            return all_tables
        missing = [t for t in table_names if t not in all_tables]
        if missing:
            raise ValueError(f"Tables not found in database: {missing}")
        return table_names

    def _build_table_schema(self, table_name: str) -> Dict:
        columns = self._inspector.get_columns(table_name)
        pk_cols = set(
            self._inspector.get_pk_constraint(table_name).get("constrained_columns", [])
        )

        fk_map = {}
        for fk in self._inspector.get_foreign_keys(table_name):
            referred_cols = fk["referred_columns"]
            for i, col in enumerate(fk["constrained_columns"]):
                fk_map[col] = {
                    "referred_table": fk["referred_table"],
                    "referred_column": referred_cols[i] if i < len(referred_cols) else referred_cols[0],
                }

        schema = {}
        for col in columns:
            col_name = col["name"]
            is_pk = col_name in pk_cols

            if col_name in fk_map:
                col_def = {
                    "type": "foreign_key",
                    # "schema" is the key SchemaLoader._load_dict_schema() expects
                    "references": {
                        "schema": fk_map[col_name]["referred_table"],
                        "field": fk_map[col_name]["referred_column"],
                    },
                }
            else:
                col_def = {"type": _map_sql_type(col["type"])}

            if is_pk:
                col_def["primary_key"] = True
                col_def["not_null"] = True
            elif not col.get("nullable", True):
                col_def["not_null"] = True

            schema[col_name] = col_def

        return schema

    def _write_schema_file(self, schema: Dict, file_path: str, format: str) -> None:
        if format == "json":
            with open(file_path, "w") as f:
                json.dump(schema, f, indent=2)
            return

        try:
            import yaml
            with open(file_path, "w") as f:
                yaml.dump(schema, f, default_flow_style=False, allow_unicode=True)
        except ImportError:
            lines = []
            for col_name, col_def in schema.items():
                lines.append(f"{col_name}:")
                for key, value in col_def.items():
                    if isinstance(value, dict):
                        lines.append(f"  {key}:")
                        for k, v in value.items():
                            lines.append(f"    {k}: {v}")
                    elif isinstance(value, bool):
                        lines.append(f"  {key}: {'true' if value else 'false'}")
                    else:
                        lines.append(f"  {key}: {value}")
            with open(file_path, "w") as f:
                f.write("\n".join(lines) + "\n")
