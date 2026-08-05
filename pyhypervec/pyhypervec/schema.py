from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class DataType:
    VARCHAR = "VARCHAR"
    FLOAT_VECTOR = "FLOAT_VECTOR"
    BOOL = "BOOL"
    INT64 = "INT64"
    DOUBLE = "DOUBLE"


@dataclass
class CollectionSchema:
    auto_id: bool = False
    enable_dynamic_field: bool = True
    description: str = ""
    fields: list[dict[str, Any]] = field(default_factory=list)

    def add_field(self, field_name: str, datatype: str, **kwargs: Any) -> None:
        row = {"name": field_name, "datatype": datatype}
        row.update(kwargs)
        self.fields.append(row)

    def to_dict(self) -> dict[str, Any]:
        return {
            "auto_id": self.auto_id,
            "enable_dynamic_field": self.enable_dynamic_field,
            "description": self.description,
            "fields": list(self.fields),
        }


@dataclass
class IndexParams:
    indexes: list[dict[str, Any]] = field(default_factory=list)

    def add_index(
        self,
        field_name: str,
        *,
        metric_type: str = "L2",
        index_type: str = "HNSWFlat",
        params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        row = {
            "field_name": field_name,
            "metric_type": metric_type,
            "index_type": index_type,
            "params": dict(params or {}),
        }
        row.update(kwargs)
        self.indexes.append(row)

    def to_dict(self) -> dict[str, Any]:
        return {"indexes": list(self.indexes)}
