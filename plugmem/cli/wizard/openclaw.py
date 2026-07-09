"""OpenClaw integration flow for the setup wizard."""
from __future__ import annotations

import json
import os
import re
import urllib.request
import urllib.error
import subprocess
from pathlib import Path
from urllib.parse import urlparse

from plugmem.cli.config import PlugmemConfig
from plugmem.cli.wizard.ui import error, header, info, prompt_action, prompt_choice, prompt_text, success, warn


def run_openclaw_section(cfg: PlugmemConfig) -> None:
    header("OpenClaw Integration")

    integrate = prompt_choice(
        "Would you like to integrate this instance with an OpenClaw project workspace?",
        choices=["yes", "no"],
        default="yes",
    )
    if integrate == "no":
        return

    # 1. Ask for workspace directory
    project_dir = ""
    while True:
        project_dir = prompt_text("Enter the absolute path to your OpenClaw project folder", default="./")
        path = Path(project_dir).resolve()
        if path.is_dir():
            project_dir = str(path)
            break
        error(f"Directory '{project_dir}' does not exist. Please enter a valid path.")

    # Find repo root
    # wizard -> cli -> plugmem -> repo_root
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    plugin_dir = repo_root / "openclaw-plugmem-plugin"

    if not plugin_dir.is_dir():
        error(f"Plugin directory not found at {plugin_dir}. Cannot build plugin.")
        return

    # 2. Verify dependencies and build the plugin
    while True:
        if not plugin_dir.is_dir():
            error(f"Plugin directory not found at {plugin_dir}. Cannot build plugin.")
            action = prompt_action("Plugin directory missing. What now?")
            if action == "skip":
                warn("Skipping OpenClaw integration.")
                return
            continue

        # Check if already built and ask to rebuild
        dist_js = plugin_dir / "dist" / "index.js"
        rebuild = "yes"
        if dist_js.exists():
            rebuild = prompt_choice(
                "OpenClaw plugin is already built. Rebuild it?",
                choices=["yes", "no"],
                default="no",
            )

        if rebuild == "no":
            success("Skipped rebuilding OpenClaw plugin (using existing build).")
            break

        # Verify Node.js and npm
        info("Verifying system dependencies (Node.js, npm, Git)...")
        if not _check_command(["node", "-v"], "Node.js"):
            action = prompt_action("Node.js is missing. What now?")
            if action == "skip":
                warn("Skipping OpenClaw integration.")
                return
            continue
        if not _check_command(["npm", "-v"], "npm"):
            action = prompt_action("npm is missing. What now?")
            if action == "skip":
                warn("Skipping OpenClaw integration.")
                return
            continue
        _check_command(["git", "--version"], "Git")

        # Build plugmem-coding-core first if needed
        core_dir = repo_root / "plugmem-coding-core"
        if core_dir.is_dir() and not (core_dir / "dist" / "index.js").exists():
            info("Building core dependency (@plugmem/coding-core)...")
            try:
                subprocess.run(["npm", "install"], cwd=str(core_dir), shell=True, check=True, capture_output=True)
                subprocess.run(["npm", "run", "build"], cwd=str(core_dir), shell=True, check=True, capture_output=True)
                success("Core dependency built successfully.")
            except subprocess.CalledProcessError as e:
                stderr_msg = e.stderr.decode("utf-8", "replace") if e.stderr else str(e)
                error(f"Failed to build core dependency (@plugmem/coding-core): {stderr_msg}")
                action = prompt_action("Build failed. What now?")
                if action == "skip":
                    warn("Skipping OpenClaw integration.")
                    return
                continue

        # Build the plugin
        info("Building OpenClaw plugin (npm install && npm run build)...")
        try:
            subprocess.run(["npm", "install"], cwd=str(plugin_dir), shell=True, check=True, capture_output=True)
            subprocess.run(["npm", "run", "build"], cwd=str(plugin_dir), shell=True, check=True, capture_output=True)
            success("Plugin built successfully.")
            break
        except subprocess.CalledProcessError as e:
            stderr_msg = e.stderr.decode("utf-8", "replace") if e.stderr else str(e)
            error(f"Failed to build OpenClaw plugin: {stderr_msg}")
            action = prompt_action("Build failed. What now?")
            if action == "skip":
                warn("Skipping OpenClaw integration.")
                return
            continue

    dist_js = plugin_dir / "dist" / "index.js"
    if not dist_js.exists():
        error(f"Plugin compiled but output file not found at {dist_js}.")
        return

    # 3. Derive graph ID
    graph_id = _derive_graph_id(project_dir)
    graph_id = prompt_text("Enter the default Graph ID to use for memory", default=graph_id)

    # 4. Integrate workspace configuration (.env file) safely
    daemon_url = f"http://{cfg.service.host}:{cfg.service.port}"
    env_file_path = Path(project_dir) / ".env"
    env_content = ""
    if env_file_path.exists():
        try:
            env_content = env_file_path.read_text(encoding="utf-8")
        except Exception:
            pass

    updates = {
        "PLUGMEM_URL": f'"{daemon_url}"',
        "PLUGMEM_BASE_URL": f'"{daemon_url}"',
        "PLUGMEM_API_KEY": f'"{cfg.service.api_key}"',
        "PLUGMEM_DEFAULT_GRAPH_ID": f'"{graph_id}"'
    }

    lines = env_content.splitlines()
    new_lines = []
    found_keys = set()
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            new_lines.append(line)
            continue
        if "=" in line:
            k, _ = line.split("=", 1)
            k_stripped = k.strip()
            if k_stripped in updates:
                new_lines.append(f"{k}={updates[k_stripped]}")
                found_keys.add(k_stripped)
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)

    added_any = False
    for k, v in updates.items():
        if k not in found_keys:
            if not added_any:
                if new_lines and new_lines[-1].strip() != "":
                    new_lines.append("")
                new_lines.append("# Added by PlugMem Setup Wizard")
                added_any = True
            new_lines.append(f"{k}={v}")

    try:
        env_file_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        success(f"Successfully integrated environment variables with: {env_file_path}")
    except Exception as e:
        warn(f"Could not write environment file {env_file_path}: {e}")

    # 5. Pre-register graph on running daemon
    info(f"Registering default Graph ID '{graph_id}' with local PlugMem daemon...")
    register_url = f"{daemon_url}/api/v1/graphs"
    try:
        req = urllib.request.Request(
            register_url,
            data=json.dumps({"graph_id": graph_id}).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "X-API-Key": cfg.service.api_key,
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=4) as conn:
            if conn.status == 201 or conn.status == 200:
                success(f"Successfully pre-registered graph ID '{graph_id}' on PlugMem server.")
    except urllib.error.HTTPError as e:
        if e.code == 409:
            success(f"Graph ID '{graph_id}' is already registered on the server.")
        else:
            warn(f"PlugMem server returned error code {e.code}. Graph will be created on start.")
    except Exception:
        warn("PlugMem server is not currently running. Skipping graph pre-registration.")

    success(f"OpenClaw workspace integration complete under Graph ID '{graph_id}'.")
    info("")
    info("To run your OpenCode agent with PlugMem:")
    info("  1. Start the PlugMem daemon: plugmem start")
    info("  2. Open a terminal in your workspace:")
    info(f"     cd \"{project_dir}\"")
    info("  3. Run your OpenCode agent:")
    info("     # (Settings are loaded automatically from .env file!)")
    info("     # Now launch OpenCode")


def _check_command(cmd_args: list[str], name: str) -> bool:
    try:
        res = subprocess.run(cmd_args, capture_output=True, text=True, check=True, shell=True)
        version_str = res.stdout.strip() or res.stderr.strip()
        cleaned_version = version_str.split("\n")[0].strip()
        success(f"{name} is installed: {cleaned_version}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        error(f"{name} is NOT installed or not in PATH. Please install it to continue.")
        return False


def _get_git_remote(cwd: str) -> str | None:
    try:
        res = subprocess.run(["git", "remote", "get-url", "origin"], cwd=cwd, capture_output=True, text=True, check=True, shell=True)
        return res.stdout.strip()
    except Exception:
        return None


def _parse_git_url(raw: str) -> tuple[str, str, str] | None:
    raw = raw.strip()
    if not raw:
        return None
    if raw.endswith(".git"):
        raw = raw[:-4]
    if "://" in raw:
        try:
            u = urlparse(raw)
            host = u.netloc.split("@")[-1].split(":")[0].lower()
            path = u.path.strip("/")
            parts = path.split("/")
            if len(parts) >= 2:
                repo = parts[-1]
                owner = "/".join(parts[:-1])
                return host, owner, repo
        except Exception:
            pass
    else:
        m = re.match(r"^[^@\s]+@([^:\s]+):([^\s]+)$", raw)
        if m:
            host = m.group(1).lower()
            path = m.group(2).strip("/")
            parts = path.split("/")
            if len(parts) >= 2:
                repo = parts[-1]
                owner = "/".join(parts[:-1])
                return host, owner, repo
    return None


def _derive_graph_id(cwd: str) -> str:
    abs_cwd = os.path.abspath(cwd)
    remote = _get_git_remote(abs_cwd)
    if remote:
        ident = _parse_git_url(remote)
        if ident:
            host, owner, repo = ident
            return f"repo_openclaw_{host}_{owner}_{repo}"
    
    safe_path = re.sub(r"[^a-zA-Z0-9._-]", "_", abs_cwd)
    return f"repo_openclaw_local_{safe_path}"
