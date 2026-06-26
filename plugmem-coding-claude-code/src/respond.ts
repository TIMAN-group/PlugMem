// Format core responses into Claude Code hook stdout JSON.
//
// CC reads stdout as JSON when the hook exits 0. The supported shape for
// context injection is:
//   { "hookSpecificOutput": {
//       "hookEventName": "<name>",
//       "additionalContext": "<text>"
//     }
//   }
// For hooks where injection isn't supported (PreCompact, Stop, SessionEnd),
// we just print nothing.

import type { ContextInjection } from "@plugmem/coding-core";

export function injectionToStdout(
  hookEventName: string,
  injection: ContextInjection | null,
): string | null {
  if (!injection || !injection.text) return null;
  return JSON.stringify({
    hookSpecificOutput: {
      hookEventName,
      additionalContext: injection.text,
    },
  });
}
