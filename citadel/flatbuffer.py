"""Decode CompressedVideo using its embedded BFBS field definitions.

Only reflection.fbs metadata slots are fixed:
https://github.com/google/flatbuffers/blob/master/reflection/reflection.fbs
"""
import struct

from flatbuffers.table import Table


def scalar(table, slot, kind, default=0):
    offset = table.Offset(4 + 2 * slot)
    return struct.unpack_from("<" + kind, table.Bytes, table.Pos + offset)[0] if offset else default


def text(table, slot):
    offset = table.Offset(4 + 2 * slot)
    return table.String(table.Pos + offset).decode() if offset else ""


def child(table, slot):
    offset = table.Offset(4 + 2 * slot)
    if not offset:
        raise ValueError("Missing BFBS table")
    return Table(table.Bytes, table.Indirect(table.Pos + offset))


def children(table, slot):
    offset = table.Offset(4 + 2 * slot)
    if not offset:
        return []
    start = table.Vector(offset)
    return [Table(table.Bytes, table.Indirect(start + 4 * i))
            for i in range(table.VectorLen(offset))]


KINDS = {1: "B", 2: "?", 3: "b", 4: "B", 5: "h", 6: "H",
         7: "i", 8: "I", 9: "q", 10: "Q", 11: "f", 12: "d"}


class VideoSchema:
    def __init__(self, data):
        if data[4:8] != b"BFBS":
            raise ValueError("Expected binary FlatBuffers schema")
        schema = Table(data, struct.unpack_from("<I", data)[0])
        self.objects = children(schema, 0)
        self.root = child(schema, 4)
        if text(self.root, 0) != "foxglove.CompressedVideo":
            raise ValueError("Expected foxglove.CompressedVideo schema")

    def decode(self, data):
        return self._object(self.root, data, struct.unpack_from("<I", data)[0])

    def _object(self, definition, data, position):
        table = Table(data, position)
        is_struct = scalar(definition, 2, "?")
        result = {}
        for field in children(definition, 1):
            name, type_info = text(field, 0), child(field, 1)
            kind = scalar(type_info, 0, "b")
            offset = scalar(field, 3, "H")
            local = offset if is_struct else table.Offset(offset)
            if not is_struct and not local:
                result[name] = (scalar(field, 5, "d") if kind in (11, 12)
                                else scalar(field, 4, "q") if kind in KINDS else None)
                continue
            location = position + local
            if kind in KINDS:
                value = struct.unpack_from("<" + KINDS[kind], data, location)[0]
            elif kind == 13:
                value = table.String(location).decode()
            elif kind == 14 and scalar(type_info, 1, "b") == 4:
                start, size = table.Vector(local), table.VectorLen(local)
                if start < 0 or start + size > len(data):
                    raise ValueError("FlatBuffer byte vector is outside the message")
                value = data[start:start + size]
            elif kind == 15:
                index = scalar(type_info, 2, "i", -1)
                if not 0 <= index < len(self.objects):
                    raise ValueError("Invalid BFBS object index")
                nested = self.objects[index]
                target = location if scalar(nested, 2, "?") else table.Indirect(location)
                value = self._object(nested, data, target)
            else:
                raise ValueError("Unsupported CompressedVideo field type")
            result[name] = value
        return result
