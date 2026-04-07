import argparse
import random
import time
from dotenv import load_dotenv

from llm_agent import LLMAgentForConversation
from utils_llm import warm_weather_cache_if_needed, warm_events_cache_if_needed


load_dotenv()


HELP_TEXT = """
Commands:
  /help                  Show commands.
  /state                 Show current session condition and counters.
  /new [single|group|unknown|N]
                         Start a new session and re-randomize condition.
  /save [video_file]     Save current conversation json.
  /exit                  Exit.
""".strip()


def _parse_group_size(value):
    if value is None:
        return None

    text = str(value).strip().lower()
    if text in ["", "unknown", "none", "null"]:
        return None
    if text in ["single", "solo", "1"]:
        return 1
    if text in ["group", "multi", "2+"]:
        return 2

    parsed = int(text)
    if parsed < 1:
        raise ValueError("group_size must be >= 1")
    return parsed


class TestRunner:
    def __init__(self, group_size=None, group_source="test-cli", max_turns=4):
        self.llm_agent = LLMAgentForConversation()
        self.default_group_size = group_size
        self.default_group_source = group_source
        self.max_turns = max_turns
        self.context_refresh_interval_s = 60.0
        self._last_context_refresh_ts = 0.0
        self.start_new_session(group_size=group_size, group_source=group_source)

    def _refresh_runtime_context(self, force=False):
        now = time.monotonic()
        if not force and (now - self._last_context_refresh_ts) < self.context_refresh_interval_s:
            return
        warm_weather_cache_if_needed()
        warm_events_cache_if_needed()
        self._last_context_refresh_ts = now

    def start_new_session(self, group_size=None, group_source="test-cli"):
        self._refresh_runtime_context(force=True)
        self.llm_agent.reset_convo_hist()
        condition = self.llm_agent.start_new_session(
            group_size=group_size,
            group_source=group_source
        )
        self.default_group_size = group_size
        self.default_group_source = group_source
        print(
            f"[session] assigned condition={condition.get('condition_id')} "
            f"({condition.get('condition_name')}), "
            f"strategy_levels={self.llm_agent.session_strategy_levels}, "
            f"group_size={group_size}, source={group_source}"
        )

    def print_state(self):
        condition = self.llm_agent.session_condition or {}
        print(
            "[state] "
            f"condition={condition.get('condition_id')} ({condition.get('condition_name')}), "
            f"group_size={self.llm_agent.session_group_size}, "
            f"group_source={self.llm_agent.session_group_source}, "
            f"challenging_turns={self.llm_agent.session_challenging_turn_count}, "
            f"strategy_applied_turns={self.llm_agent.session_strategy_applied_turn_count}, "
            f"strategy_levels={self.llm_agent.session_strategy_levels}"
        )

    def generate_response(self, user_input):
        self._refresh_runtime_context(force=False)
        return self.llm_agent.generate_response(
            user_input,
            max_turns=self.max_turns
        )

    def print_response(self, response):
        turns = response.get("planned_turns", [])
        print(f"[assistant] end_conversation={response.get('end_conversation', False)}")
        for i, turn in enumerate(turns, start=1):
            robot = turn.get("robot_to_speak", "unknown")
            text = turn.get("speech_content", "").strip()
            pre_speech = turn.get("pre_speech")
            axel_expr = turn.get("axel_expression")
            bella_expr = turn.get("bella_expression")
            print(
                f"  {i}. {robot}: {text} "
                f"(pre_speech={pre_speech}, axel_expression={axel_expr}, bella_expression={bella_expr})"
            )
        if response.get("note"):
            print(f"[assistant-note] {response.get('note')}")

    def save_conversation(self, video_file=""):
        self.llm_agent.save_current_conv(video_file=video_file)
        print("[session] conversation saved")


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Text I/O harness for LLMAgent experiment flow."
    )
    parser.add_argument(
        "--group-size",
        default="unknown",
        help="single|group|unknown|integer"
    )
    parser.add_argument(
        "--group-source",
        default="test-cli",
        help="group-size source label saved in session metadata"
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=4,
        help="max planned turns to allow in each reply"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="optional random seed for reproducible condition assignment"
    )
    return parser


def main():
    args = _build_arg_parser().parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        print(f"[session] random seed set to {args.seed}")
    group_size = _parse_group_size(args.group_size)

    runner = TestRunner(
        group_size=group_size,
        group_source=args.group_source,
        max_turns=max(1, min(12, args.max_turns))
    )
    print(HELP_TEXT)

    while True:
        try:
            user_input = input("\nYou> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[session] exiting")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            parts = user_input.split()
            cmd = parts[0].lower()

            if cmd == "/help":
                print(HELP_TEXT)
                continue

            if cmd == "/state":
                runner.print_state()
                continue

            if cmd == "/new":
                group_arg = parts[1] if len(parts) > 1 else "unknown"
                try:
                    new_group_size = _parse_group_size(group_arg)
                except ValueError as e:
                    print(f"[error] invalid /new group size: {e}")
                    continue
                runner.start_new_session(
                    group_size=new_group_size,
                    group_source="manual-reset"
                )
                continue

            if cmd == "/save":
                video_file = parts[1] if len(parts) > 1 else ""
                runner.save_conversation(video_file=video_file)
                continue

            if cmd == "/exit":
                print("[session] exiting")
                break

            print("[error] unknown command. type /help")
            continue

        response = runner.generate_response(user_input)
        runner.print_response(response)


if __name__ == "__main__":
    main()
