"""Claude Code CLI integration module.

Provides a client for interacting with the Claude Code CLI via subprocess,
enabling LLM-powered analysis and question answering.
"""

import asyncio
import json
from pathlib import Path

from neev_voice.config import NeevSettings


class ClaudeCodeClient:
    """Client for Claude Code CLI interaction via subprocess.

    Executes `claude -p` commands as async subprocesses and parses
    the JSON output for structured responses.

    Attributes:
        settings: Application settings containing Claude configuration.
    """

    def __init__(self, settings: NeevSettings) -> None:
        """Initialize ClaudeCodeClient with application settings.

        Args:
            settings: Application settings with Claude model and timeout config.
        """
        self.settings = settings

    async def query(self, prompt: str, context: str | None = None) -> str:
        """Send a prompt to Claude Code CLI and return the response.

        Runs `claude -p "<prompt>" --output-format json` as a subprocess
        and extracts the response text from the JSON output.

        Args:
            prompt: The prompt to send to Claude.
            context: Optional context to prepend to the prompt.

        Returns:
            The text response from Claude.

        Raises:
            RuntimeError: If the subprocess fails or times out.
        """
        full_prompt = f"{context}\n\n{prompt}" if context else prompt

        cmd = [
            "claude",
            "-p",
            full_prompt,
            "--output-format",
            "json",
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.settings.claude_timeout,
            )
        except asyncio.TimeoutError:
            process.kill()
            raise RuntimeError(f"Claude CLI timed out after {self.settings.claude_timeout}s")
        except FileNotFoundError:
            raise RuntimeError(
                "Claude CLI not found. Install it from https://docs.anthropic.com/en/docs/claude-code"
            )

        if process.returncode != 0:
            error_msg = stderr.decode().strip() if stderr else "Unknown error"
            raise RuntimeError(f"Claude CLI error (exit code {process.returncode}): {error_msg}")

        output = stdout.decode().strip()
        return self._parse_response(output)

    async def analyze_codebase(self, path: Path, question: str) -> str:
        """Analyze a codebase directory with a specific question.

        Provides codebase context to Claude for code-aware responses.

        Args:
            path: Path to the codebase directory.
            question: The question to ask about the codebase.

        Returns:
            Claude's analysis response.

        Raises:
            FileNotFoundError: If the codebase path does not exist.
            RuntimeError: If the Claude CLI call fails.
        """
        if not path.exists():
            raise FileNotFoundError(f"Codebase path not found: {path}")

        context = f"Analyze the codebase at: {path}"
        return await self.query(question, context=context)

    @staticmethod
    def _parse_response(output: str) -> str:
        """Parse Claude CLI JSON output to extract response text.

        Handles both JSON format (from --output-format json) and
        plain text fallback.

        Args:
            output: Raw output from Claude CLI subprocess.

        Returns:
            Extracted response text.
        """
        if not output:
            return ""

        try:
            data = json.loads(output)
            # Claude CLI JSON format has a "result" field
            if isinstance(data, dict):
                return data.get("result", data.get("text", str(data)))
            if isinstance(data, list) and data:
                # Handle array of content blocks
                texts = []
                for block in data:
                    if isinstance(block, dict) and block.get("type") == "text":
                        texts.append(block.get("text", ""))
                return "\n".join(texts) if texts else str(data)
            return str(data)
        except json.JSONDecodeError:
            # Fallback: return raw output as text
            return output
