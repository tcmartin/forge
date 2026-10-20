# Tool discovery

Forge starts each query with the authorized standard tools and ToolSearch. Advanced tools and connected MCP tools become available after a successful search. The selected schemas appear on the next model request.

ToolSearch accepts keywords or an explicit selection:

```json
{"query":"search project issues","max_results":5}
```

```json
{"query":"select:ToolName,AnotherTool","max_results":2}
```

An explicit selection fails if any name is unknown or denied. Repeated names are deduplicated. Search never adds a tool excluded by the current policy. Selected tools still require their normal execution permissions.

A call to an undisclosed or unavailable tool is rejected before execution and tool hooks. Moving to another workspace or replacing the workspace invalidates the prior discovery scope. Completed calls are not replayed.

Without ToolSearch, Forge exposes the explicitly authorized inventory at query entry, subject to the same active-schema limits.

| Limit | Value |
| --- | ---: |
| Catalog entries | 4,096 |
| Active schemas | 128 |
| Search results | 20, default 5 |
| Query text | 4,096 UTF-8 bytes |
| Query terms | 64 |
| Tool name | 256 UTF-8 bytes |
| Individual input schema | 64 KiB |
| Encoded active definitions | 1 MiB |

Exceeding a limit produces an error. A failed activation retains the prior active set.

Forge applies MCP catalog changes during a query. A change invalidates prior selections. Search again to activate tools from the updated catalog. A call already admitted may finish once; stale selections cannot start another call. A failed refresh stops the next inference request with an explicit error.

Open `/mcp` and press `r` to reconnect. New queries and future background jobs use the refreshed runtime. Idle connections from the old runtime are released. Jobs already running keep their captured connection until they finish.

MCP input framing, catalog pagination, schema sizes, notification queues and initialization are bounded. Malformed or oversized catalogs are unavailable until a successful refresh or reconnect.

Runtime verification covers the tested Unix host and synthetic OpenAI and Anthropic provider paths. Native workspace replacement detection on other platforms remains unverified.
