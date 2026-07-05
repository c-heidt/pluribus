# cython: language_level=3
"""Info-set key encoder (Phase 1b) — Cython port of
``environment.poker_env.encode_info_set``.

Byte-identical to the Python reference: unsigned LEB128 ``varint(cluster)``, then
per ``(stage, actions)`` a ``_STAGE_ID`` byte + ``varint(len(actions))`` + one
alphabet byte per token (or the ``0xFF`` raw-token fallback: mark +
``varint(len(utf8))`` + utf8 bytes).

**The alphabet is DUMPED from Python, never hard-coded here.**  ``_ACTION_BYTE``
and ``_STAGE_ID`` are derived in ``poker_env`` from ``RAISE_SIZES_BY_STAGE``; a
hard-coded copy would silently drift when the raise grid changes (→ keys that
hash differently from what the tables were written under = "untrained"
everywhere).  ``configure()`` installs the live tables at wire time; the encoder
refuses to run until it has them.

The bytes are built in a growable C buffer (:class:`_Buf`) — the same buffer +
varint machinery Phase 3's int-action in-core encoder will reuse.  The per-token
``dict.get`` string→code lookup stays (unavoidable while tokens are Python
strings); Phase 2/3 replaces tokens with int action codes and a C lookup table.
"""

from libc.stdlib cimport malloc, realloc, free
from libc.string cimport memcpy
from cpython.bytes cimport PyBytes_FromStringAndSize


# Tables dumped from poker_env via configure() — never hard-coded (see docstring).
cdef object _STAGE_ID = None       # {stage_str: int}
cdef object _ACTION_BYTE = None    # {stage_str: {token_str: int code}}
cdef int _RAW_MARK = 0xFF
cdef bint _configured = False


def configure(stage_id, action_byte, raw_mark=0xFF):
    """Install the encoding tables dumped from ``poker_env`` (call once at wire time)."""
    global _STAGE_ID, _ACTION_BYTE, _RAW_MARK, _configured
    _STAGE_ID = dict(stage_id)
    _ACTION_BYTE = {stage: dict(table) for stage, table in action_byte.items()}
    _RAW_MARK = int(raw_mark)
    _configured = True


cdef class _Buf:
    """Minimal growable byte buffer with C-level append (freed on dealloc)."""

    cdef unsigned char* data
    cdef Py_ssize_t size
    cdef Py_ssize_t cap

    def __cinit__(self, Py_ssize_t cap=64):
        self.data = <unsigned char*>malloc(cap)
        if self.data == NULL:
            raise MemoryError()
        self.size = 0
        self.cap = cap

    def __dealloc__(self):
        if self.data != NULL:
            free(self.data)

    cdef void _ensure(self, Py_ssize_t extra) except *:
        cdef Py_ssize_t need = self.size + extra
        cdef Py_ssize_t newcap
        cdef unsigned char* grown
        if need > self.cap:
            newcap = self.cap * 2
            while newcap < need:
                newcap *= 2
            grown = <unsigned char*>realloc(self.data, newcap)
            if grown == NULL:
                raise MemoryError()
            self.data = grown
            self.cap = newcap

    cdef void put_byte(self, unsigned char b) except *:
        self._ensure(1)
        self.data[self.size] = b
        self.size += 1

    cdef void put_varint(self, unsigned long long value) except *:
        # Unsigned LEB128 — identical to poker_env._put_uvarint.
        cdef unsigned int byte
        while True:
            byte = value & 0x7F
            value >>= 7
            if value:
                self.put_byte(<unsigned char>(byte | 0x80))
            else:
                self.put_byte(<unsigned char>byte)
                return

    cdef void put_raw(self, const unsigned char* src, Py_ssize_t n) except *:
        self._ensure(n)
        memcpy(self.data + self.size, src, n)
        self.size += n


def encode_info_set(cluster, history_items):
    """Return the compact injective ``bytes`` key for ``(cluster, history)``.

    Drop-in for ``poker_env.encode_info_set``; see that function and this
    module's docstring.  ``history_items`` is an iterable of ``(stage, actions)``
    pairs (``self._history.items()`` or a canonicalised history).
    """
    if not _configured:
        raise RuntimeError(
            "poker_ai._core._infoset.encode_info_set used before configure() — "
            "the action alphabet must be installed from poker_env first."
        )
    cdef _Buf buf = _Buf(64)
    buf.put_varint(<unsigned long long>int(cluster))

    stage_id = _STAGE_ID
    action_byte = _ACTION_BYTE
    cdef int raw_mark = _RAW_MARK
    cdef bytes raw

    for stage, actions in history_items:
        buf.put_byte(<unsigned char>(<int>stage_id[stage]))
        buf.put_varint(<unsigned long long>len(actions))
        table = action_byte[stage]
        for token in actions:
            code = table.get(token)
            if code is None:
                raw = str(token).encode("utf-8")
                buf.put_byte(<unsigned char>raw_mark)
                buf.put_varint(<unsigned long long>len(raw))
                buf.put_raw(<const unsigned char*>raw, len(raw))
            else:
                buf.put_byte(<unsigned char>(<int>code))

    return PyBytes_FromStringAndSize(<char*>buf.data, buf.size)
