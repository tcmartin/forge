# File reads

Read returns a numbered window from a local UTF-8 text file. It captures up to 64 KiB of line content and 2,000 lines, with bounded line-number formatting. It scans at most 8 MiB per request and has a five-second deadline after permission checks. Image and PDF reading is currently unavailable.

Use `offset` for the starting line and `limit` for the number of lines. Lines start at 1; an omitted or zero offset starts at the beginning. CRLF and LF line endings are supported.

```json
{"file_path": "src/main.rs", "offset": 40, "limit": 30}
```

A partial result includes its continuation position:

```text
has_more=true next_offset=70 next_byte_offset=0
```

Pass `next_offset` as `offset` and `next_byte_offset` as `byte_offset` in the next request. The byte offset is within the starting line, measured in UTF-8 bytes without its CRLF terminator. Long lines can span several requests. Offsets inside a Unicode character are rejected.

```json
{"file_path": "large.txt", "offset": 1, "byte_offset": 65535, "limit": 1}
```

Read does not scan the rest of the file to report a total line count. A scan-limit error means the requested position was beyond the bounded scan. Captured output remains available through the scoped `EvidenceRead` reference when the display is shortened.

Matching simultaneous reads may share one active observation. Completed observations are evicted. A later request reads again, including after a same-length external edit. Set `refresh` to `true` to request an independent observation while a matching read is active.

Permission checks apply before joining an observation. Cancellation releases waiting readers. Detected file replacement, path retargeting or mutation during capture returns an error. These checks cover observable file identity and metadata changes; they do not provide operating-system isolation from unrestricted external writers.
