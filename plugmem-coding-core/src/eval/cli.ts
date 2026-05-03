#!/usr/bin/env node
// CLI for running an eval fixture.
//
// Usage:
//   node dist/eval/cli.js run --fixture path/to/fixture.json [--out report.json]
//   node dist/eval/cli.js summarize --token-log path/to/tokens.jsonl
//
// The runner expects a live PlugMem service at the URL specified in the
// fixture. Set PLUGMEM_TOKEN_USAGE_FILE on the server side to also collect
// internal-token costs; then run `summarize` against that path.

import { readFile, writeFile } from "node:fs/promises";

import { runFixture } from "./runner.js";
import { summarizeTokenLog } from "./metrics.js";
import type { EvalFixture, EvalRunReport } from "./types.js";

function parseArgs(argv: string[]): {
  command: string;
  flags: Record<string, string>;
} {
  const [command, ...rest] = argv;
  const flags: Record<string, string> = {};
  for (let i = 0; i < rest.length; i++) {
    const arg = rest[i]!;
    if (!arg.startsWith("--")) continue;
    const next = rest[i + 1];
    if (next && !next.startsWith("--")) {
      flags[arg.slice(2)] = next;
      i++;
    } else {
      flags[arg.slice(2)] = "true";
    }
  }
  return { command: command ?? "", flags };
}

async function cmdRun(flags: Record<string, string>): Promise<void> {
  const fixturePath = flags.fixture;
  if (!fixturePath) {
    process.stderr.write("missing --fixture <path>\n");
    process.exit(2);
  }

  const fixture = JSON.parse(
    await readFile(fixturePath, "utf8"),
  ) as EvalFixture;

  const noReset = flags["no-reset"] === "true";
  const report = await runFixture(fixture, { resetGraph: !noReset });

  const out = renderReport(report);
  if (flags.out) {
    await writeFile(flags.out, out, "utf8");
    process.stderr.write(`[eval] wrote ${flags.out}\n`);
  } else {
    process.stdout.write(out + "\n");
  }
}

async function cmdSummarize(flags: Record<string, string>): Promise<void> {
  const path = flags["token-log"];
  if (!path) {
    process.stderr.write("missing --token-log <path>\n");
    process.exit(2);
  }
  const summary = await summarizeTokenLog(path);
  process.stdout.write(JSON.stringify(summary, null, 2) + "\n");
}

function renderReport(report: EvalRunReport): string {
  return JSON.stringify(report, null, 2);
}

async function main(): Promise<void> {
  const { command, flags } = parseArgs(process.argv.slice(2));
  switch (command) {
    case "run":
      await cmdRun(flags);
      break;
    case "summarize":
      await cmdSummarize(flags);
      break;
    case "":
    case "help":
    case "--help":
      process.stdout.write(
        [
          "plugmem-eval — eval harness CLI",
          "",
          "Commands:",
          "  run --fixture <path> [--out <report.json>] [--no-reset]",
          "    Run a fixture against a live PlugMem service.",
          "  summarize --token-log <path>",
          "    Aggregate a server-side token_usage_file JSONL by phase.",
        ].join("\n") + "\n",
      );
      break;
    default:
      process.stderr.write(`unknown command: ${command}\n`);
      process.exit(2);
  }
}

const isMain = import.meta.url === `file://${process.argv[1]}`;
if (isMain) {
  main().catch((err) => {
    process.stderr.write(
      `[eval] fatal: ${err instanceof Error ? err.stack ?? err.message : String(err)}\n`,
    );
    process.exit(1);
  });
}
