"""Small, checked BFBS reflection reader for scalar/vector/table message fields.

Only the bootstrap reflection schema has fixed field ordinals, as specified by
Google's stable reflection.fbs:
https://github.com/google/flatbuffers/blob/master/reflection/reflection.fbs
Actual Foxglove field offsets, scalar types, defaults, and struct layouts are
read from the MCAP's own binary schema. No video or gripper layout is assumed.
Unsupported unions/arrays are rejected when accessed.
"""

from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import Any

from ..models import MediaError


# reflection.BaseType scalar values; formats are little-endian FlatBuffers types.
_SCALAR = {1: "B", 2: "?", 3: "b", 4: "B", 5: "h", 6: "H", 7: "i",
           8: "I", 9: "q", 10: "Q", 11: "f", 12: "d"}
_STRING, _VECTOR, _OBJECT = 13, 14, 15


class _Buffer:
    def __init__(self, data: bytes):
        self.data = data

    def check(self, pos: int, length: int) -> None:
        if pos < 0 or length < 0 or pos + length > len(self.data):
            raise MediaError("Malformed FlatBuffer: field outside message bounds")

    def number(self, fmt: str, pos: int) -> Any:
        self.check(pos, struct.calcsize("<" + fmt))
        return struct.unpack_from("<" + fmt, self.data, pos)[0]

    def indirect(self, pos: int) -> int:
        target = pos + self.number("I", pos)
        self.check(target, 1)
        return target

    def string(self, pos: int) -> str:
        target = self.indirect(pos)
        size = self.number("I", target)
        self.check(target + 4, size)
        try:
            return self.data[target + 4 : target + 4 + size].decode("utf-8")
        except UnicodeError as exc:
            raise MediaError("Malformed FlatBuffer: invalid UTF-8") from exc

    def vector(self, pos: int, item_size: int) -> tuple[int, int]:
        target = self.indirect(pos)
        size = self.number("I", target)
        self.check(target + 4, size * item_size)
        return target + 4, size


class _Table:
    def __init__(self, buffer: _Buffer, pos: int):
        self.buffer, self.pos = buffer, pos
        self.vtable = pos - buffer.number("i", pos)
        self.vlength = buffer.number("H", self.vtable)
        if self.vlength < 4 or self.vlength % 2:
            raise MediaError("Malformed FlatBuffer vtable")
        buffer.check(self.vtable, self.vlength)

    def offset(self, vtable_offset: int) -> int | None:
        if vtable_offset >= self.vlength:
            return None
        offset = self.buffer.number("H", self.vtable + vtable_offset)
        if not offset:
            return None
        self.buffer.check(self.pos + offset, 1)
        return self.pos + offset

    def slot(self, ordinal: int) -> int | None:
        return self.offset(4 + 2 * ordinal)

    def scalar(self, ordinal: int, fmt: str, default: Any = 0) -> Any:
        pos = self.slot(ordinal)
        return default if pos is None else self.buffer.number(fmt, pos)

    def string(self, ordinal: int) -> str:
        pos = self.slot(ordinal)
        return "" if pos is None else self.buffer.string(pos)

    def table(self, ordinal: int) -> _Table:
        pos = self.slot(ordinal)
        if pos is None:
            raise MediaError("BFBS schema is missing a required table")
        return _Table(self.buffer, self.buffer.indirect(pos))

    def tables(self, ordinal: int) -> list[_Table]:
        pos = self.slot(ordinal)
        if pos is None:
            return []
        start, size = self.buffer.vector(pos, 4)
        if size > 4096:
            raise MediaError("Unsupported excessively large BFBS schema")
        return [_Table(self.buffer, self.buffer.indirect(start + index * 4))
                for index in range(size)]


@dataclass(frozen=True)
class _Field:
    offset: int
    base_type: int
    element_type: int
    object_index: int
    default: int | float


@dataclass(frozen=True)
class _Object:
    name: str
    fields: dict[str, _Field]
    is_struct: bool
    byte_size: int


def _object(table: _Table) -> _Object:
    # reflection.Object(name=0, fields=1, is_struct=2, bytesize=4).
    fields = {}
    for field in table.tables(1):
        # reflection.Field(name=0, type=1, offset=3, defaults=4/5).
        kind = field.table(1)
        base = kind.scalar(0, "b")
        if field.string(0) in fields:
            raise MediaError("Duplicate field in BFBS schema")
        fields[field.string(0)] = _Field(
            offset=field.scalar(3, "H"), base_type=base,
            element_type=kind.scalar(1, "b"), object_index=kind.scalar(2, "i", -1),
            default=field.scalar(5, "d") if base in (11, 12) else field.scalar(4, "q"),
        )
    return _Object(table.string(0), fields, table.scalar(2, "?", False),
                   table.scalar(4, "i"))


class FlatbufferSchema:
    def __init__(self, data: bytes, expected_name: str | None = None):
        if len(data) < 8 or data[4:8] != b"BFBS":
            raise MediaError("Unsupported schema: expected binary FlatBuffers BFBS")
        buffer = _Buffer(data)
        schema = _Table(buffer, buffer.indirect(0))
        # reflection.Schema(objects=0, root_table=4).
        self.objects = [_object(item) for item in schema.tables(0)]
        self.root = _object(schema.table(4))
        if expected_name and self.root.name != expected_name:
            raise MediaError("FlatBuffers root type does not match the MCAP schema name")

    def decode(self, data: bytes) -> MessageView:
        buffer = _Buffer(data)
        return MessageView(self, self.root, buffer, buffer.indirect(0))


class MessageView:
    def __init__(self, schema: FlatbufferSchema, obj: _Object, buffer: _Buffer, pos: int):
        self.schema, self.obj, self.buffer, self.pos = schema, obj, buffer, pos
        self.table = None if obj.is_struct else _Table(buffer, pos)

    @property
    def fields(self) -> tuple[str, ...]:
        return tuple(self.obj.fields)

    def get(self, name: str, default: Any = None) -> Any:
        field = self.obj.fields.get(name)
        if field is None:
            return default
        pos = self.pos + field.offset if self.obj.is_struct else self.table.offset(field.offset)
        if pos is None:
            return field.default if field.base_type in _SCALAR else default
        return self._value(field.base_type, pos, field)

    def _object_type(self, field: _Field) -> _Object:
        if not 0 <= field.object_index < len(self.schema.objects):
            raise MediaError("Invalid BFBS object index")
        return self.schema.objects[field.object_index]

    def _value(self, base: int, pos: int, field: _Field) -> Any:
        if base in _SCALAR:
            return self.buffer.number(_SCALAR[base], pos)
        if base == _STRING:
            return self.buffer.string(pos)
        if base == _OBJECT:
            obj = self._object_type(field)
            location = pos if obj.is_struct else self.buffer.indirect(pos)
            return MessageView(self.schema, obj, self.buffer, location)
        if base == _VECTOR:
            element = field.element_type
            if element in _SCALAR:
                fmt = _SCALAR[element]
                start, size = self.buffer.vector(pos, struct.calcsize("<" + fmt))
                if element == 4:  # raw encoded video bytes
                    return self.buffer.data[start : start + size]
                return list(struct.unpack_from("<" + str(size) + fmt, self.buffer.data, start))
            if element in (_STRING, _OBJECT):
                obj = self._object_type(field) if element == _OBJECT else None
                stride = obj.byte_size if obj and obj.is_struct else 4
                start, size = self.buffer.vector(pos, stride)
                return [self._value(element, start + index * stride, field)
                        for index in range(size)]
        raise MediaError(f"Unsupported FlatBuffers field type: {base}")
