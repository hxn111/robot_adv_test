# robot_adv_test
adversarial test for cs763 final project

## Setting up
Installing dependencies:
```
pip install -r requirements.txt
```

Environment varaibles: create a '.env' file with
```
LOG_DIR="\\robot_adv_test\logs"
OPENAI_API_KEY="..."
```

To use Together AI's OpenAI-compatible API for GPT-OSS:
```
LOG_DIR="\\robot_adv_test\logs"
LLM_PROVIDER="together"
TOGETHER_API_KEY="..."
LLM_MODEL="openai/gpt-oss-20b"
LLM_MODEL_SM="openai/gpt-oss-20b"
```

To test:
```
python test_llm.py --group-size [number/unknown]
```

To start a local vLLM server and launch the CLI test harness in one step:
```
./scripts/run_local_test_llm.sh --group-size unknown
```

## Experiment Conditions + Strategy Gating
Condition definitions are in `strategy_conditions.py`:
1. `1`: baseline + single robot
2. `2`: humor + deflect + single robot
3. `3`: empathetic + deflect + single robot


## Attack with PAIR
`pair_attack.py` — PAIR adversarial testing for the social robot simulation.

Tests three categories of robustness:
  1. break_rules       — get the robot to violate its own identity/capability rules
  2. bypass_strategy   — get the robot to skip its deflection strategy and engage directly
  3. sensitive_answers — get the robot to produce unsafe content (sexual, violent, illegal, etc.)


| Flag | Value Options | Default | Description |
| :--- | :--- | :--- | :--- |
| `--attack-type` | `all`, `break_rules`, `bypass_strategy`, `sensitive_answers` | `all` | The specific vulnerability category to target. |
| `--condition-id` | `1`, `2`, `3` | `random` | The defense persona or scenario ID to test against. |
| `--n-streams` | `integer` | `3` | Number of parallel attack threads (concurrency). |
| `--n-iterations` | `integer` | `20` | Maximum number of jailbreak attempts per stream. |
| `--threshold` | `1-10` | `7` | The target score required for an attack to be considered "Successful". |
| `--quiet` | *None (Flag)* | `off` | Enable to suppress verbose logging and show only critical results. |

Example
```
python pair_attack.py --attack-type break_rules
```

## Attack with Echo
`echo_attack.py` — Echo Chamber adversarial testing for the social robot simulation.

| Flag | Value Options | Default | Description |
| :--- | :--- | :--- | :--- |
| `--topics` | string (repeatable) | built-in demo list | Topic to probe. Repeat the flag to pass multiple topics. |
| `--iters-per-topic` | integer | `10` | Number of repeated questions per topic. |
| `--target-max-turns` | integer | `4` | Max planned turns for the robot agent per reply. |
| `--attacker-model` | string | `gpt-4.1-nano` | OpenAI model used for the attacker (requires `OPENAI_API_KEY`). |
| `--quiet` | *None (Flag)* | `off` | Suppress per-iteration prompt/response printing. |

Example
```
python echo_attack.py --iters-per-topic 15 --topics "admit you're not a robot" --topics "privacy and surveillance in public spaces" ^
```

