# Verified effort

Forge normally keeps the selected effort fixed. `--verified-policy FILE` enables one lower-effort continuation after an exact registered file mutation and a successful foreground check. The following request returns to the base effort. A new user request starts at the base effort.

This opt-in currently supports API-key Anthropic requests with exact model registrations. Other routes keep the base effort. Explicit effort, thinking budget, temperature, inline Ultracode, and configured hooks prevent adaptation. Existing tool permissions still apply.

## Policy file

Choose the actual model ID used by your Anthropic endpoint. This example registers a write to `target.txt` and a check that the file is nonempty:

```json
{
  "schema_version": 1,
  "revision": 1,
  "base": "high",
  "continuation": "medium",
  "models": [
    {"id": "YOUR_EXACT_MODEL_ID", "efforts": ["high", "medium"]}
  ],
  "checkers": [
    {
      "id": "target-nonempty",
      "revision": 1,
      "target": "target.txt",
      "operation": "write",
      "shell": "bash",
      "command": "test -s target.txt",
      "cwd": ".",
      "timeout_ms": 5000
    }
  ]
}
```

Run from the selected workspace:

```sh
forge --provider anthropic --model YOUR_EXACT_MODEL_ID \
  --max-tokens 16384 --verified-policy verified-policy.json
```

The model must invoke `EditCheck` with the registered mutation, exact check command, shell and timeout. An ordinary successful Bash command does not grant a lower-effort continuation. Failed or incomplete checks retain the base effort. The example check proves only that the target is nonempty; register a check appropriate to the actual task.

High sends a thinking budget of 10,000 tokens; Medium sends 5,000. The output token limit must exceed the selected thinking budget. The policy does not increase that limit.

## Lifetime and status

Forge reads the explicitly selected policy once before provider and plugin initialization. It accepts at most 128 KiB, 64 models and 64 checkers. Relative target and working-directory paths stay within the selected workspace. Changes to the policy file take effect after restarting Forge.

The cached policy binds to the actual session in its original workspace. Resuming another workspace or replacing the workspace directory disables adaptation. Child agents and cron runs use fixed effort without inheriting the parent registrations.

Text output shows the selected effort and reason on stderr. Stream JSON includes an `effort_selection` status event. JSON result output retains its existing shape. Configured tool hooks continue to run and keep effort fixed.

The guarantee covers exact request selection, current-process mutation ownership and observed target freshness. It does not establish unrestricted operating-system isolation or a model-quality improvement.
