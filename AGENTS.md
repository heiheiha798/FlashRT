# AGENTS.md instructions for /data/home/tianjianyang/code/FlashRT

<INSTRUCTIONS>
<!-- CODEGRAPH_START -->
## CodeGraph

This project has a CodeGraph MCP server (`codegraph_*` tools) configured. CodeGraph is a tree-sitter-parsed knowledge graph of every symbol, edge, and file. Reads are sub-millisecond and return structural information grep cannot.

### When to prefer codegraph over native search

Use codegraph for **structural** questions -- what calls what, what would break, where is X defined, what is X's signature. Use native grep/read only for **literal text** queries (string contents, comments, log messages) or after you already have a specific file open.

| Question | Tool |
|---|---|
| "Where is X defined?" / "Find symbol named X" | `codegraph_search` |
| "What calls function Y?" | `codegraph_callers` |
| "What does Y call?" | `codegraph_callees` |
| "How does X reach/become Y? / trace the flow from X to Y" | `codegraph_trace` |
| "What would break if I changed Z?" | `codegraph_impact` |
| "Show me Y's signature / source / docstring" | `codegraph_node` |
| "Give me focused context for a task/area" | `codegraph_context` |
| "See several related symbols' source at once" | `codegraph_explore` |
| "What files exist under path/" | `codegraph_files` |
| "Is the index healthy?" | `codegraph_status` |

### Rules of thumb

- **Answer directly -- don't delegate exploration.** For "how does X work" / architecture questions, answer with 2-3 codegraph calls: `codegraph_context` first, then one `codegraph_explore` for the source of the symbols it surfaces. For a specific flow ("how does X reach Y") start with `codegraph_trace` from-to, then one `codegraph_explore` for the bodies.
- **Trust codegraph results.** They come from a full AST parse. Do not re-verify them with grep.
- **Don't grep first** when looking up a symbol by name. Use `codegraph_search`.
- **Don't chain `codegraph_search` + `codegraph_node`** when you just want context. Use `codegraph_context`.
- **Don't loop `codegraph_node` over many symbols.** Use one `codegraph_explore` call for related source.
- **Index lag:** if a codegraph response reports files edited since the last index sync, read those specific files directly. Files not listed there are fresh.

### If `.codegraph/` doesn't exist

The MCP server returns "not initialized." Ask the user: *"I notice this project doesn't have CodeGraph initialized. Want me to run `codegraph init -i` to build the index?"*
<!-- CODEGRAPH_END -->

--- project-doc ---

任何场景下，如果开发过程中使用了兜底或者 fallback 策略，都先向我确认。禁止擅自添加兜底逻辑。

只用一次的功能请直接 inline 实现，不要单独抽一个函数。

严禁直接使用 `git clean`、`git reset --hard`、`git checkout -- .`、`git restore .`
或其他等价命令清除所有未缓存/未跟踪文件。只有在用户明确逐字要求清理这些文件时才允许执行。

在准备提交 PR 之前，必须先对照 `README.md` 中链接出去的 `CONTRIBUTING.md`
逐条检查本次改动是否满足对应要求，尤其是：

- PR 描述中的环境信息、命令、测试结果、precision / latency 证据
- kernel binding / CMake 改动的 build/import 验证要求
- 未覆盖项和限制必须明确写进 PR 描述

</INSTRUCTIONS>
