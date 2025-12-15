import multiprocessing
import queue
import re
import traceback
from typing import Any, Dict, List, Optional, Union

from math_verify import parse, verify

DEFAULT_MATH_VERIFY_TIMEOUT_S = 10.0
DEFAULT_MP_START_METHOD = "spawn"


class MathVerifySubprocessError(RuntimeError):
    """Raised when the math verification helper process fails unexpectedly."""


class MathVerifyTimeoutError(MathVerifySubprocessError):
    """Raised when math verification exceeds the allowed wall-clock time."""


def _math_verify_pipeline(candidate_answer: str, gold_answer: str) -> Dict[str, Any]:
    parsed_candidate_answer = parse(candidate_answer, parsing_timeout=None)
    parsed_gold_answer = parse(gold_answer, parsing_timeout=None)
    evaluation_result = verify(parsed_candidate_answer, parsed_gold_answer, timeout_seconds=None)
    parsed_value = None
    if isinstance(parsed_candidate_answer, list) and len(parsed_candidate_answer) > 1:
        parsed_value = parsed_candidate_answer[1]
    return {"result": evaluation_result, "parsed": parsed_value, "timed_out": False}


def _math_verify_subprocess_main(
    result_queue: "multiprocessing.queues.Queue",
    candidate_answer: str,
    gold_answer: str,
) -> None:
    try:
        result_queue.put(("ok", _math_verify_pipeline(candidate_answer, gold_answer)))
    except Exception:
        result_queue.put(("error", traceback.format_exc()))


def _run_math_verify_in_subprocess(
    candidate_answer: str,
    gold_answer: str,
    timeout_seconds: Optional[float],
    ctx: multiprocessing.context.BaseContext,
) -> Dict[str, Any]:
    result_queue = ctx.Queue()
    process = ctx.Process(
        target=_math_verify_subprocess_main,
        args=(result_queue, candidate_answer, gold_answer),
        daemon=True,
    )
    process.start()
    try:
        if timeout_seconds is None:
            process.join()
        else:
            process.join(timeout_seconds)
            if process.is_alive():
                process.terminate()
                process.join()
                raise MathVerifyTimeoutError(
                    f"math_verify exceeded {timeout_seconds} seconds while evaluating the answer."
                )
            process.join()

        try:
            status, payload = result_queue.get_nowait()
        except queue.Empty as exc:
            raise MathVerifySubprocessError(
                "math_verify subprocess exited without returning a result."
            ) from exc
        if status == "ok":
            return payload
        raise MathVerifySubprocessError(payload)
    finally:
        result_queue.close()
        result_queue.join_thread()


def extract_boxed_expressions(text: str) -> Union[List[str], str]:
    """
    Extracts LaTeX boxed expressions from a string.

    Args:
        text (str): Input string containing LaTeX expressions.

    Returns:
        list[str] or str: Extracted expressions or a message if none found.
    """
    pattern = r"\\boxed\{((?:[^{}]|(?:\{[^{}]*\}))*)\}"
    matches = re.findall(pattern, text)
    return matches[-1] if matches else None


class MathEvaluator:
    def __init__(
        self,
        add_boxed_to_gold_answer: bool = True,
        extract_candidate_answer: bool = True,
        math_verify_timeout_s: Optional[float] = DEFAULT_MATH_VERIFY_TIMEOUT_S,
        mp_start_method: str = DEFAULT_MP_START_METHOD,
    ):
        self.add_boxed_to_gold_answer = add_boxed_to_gold_answer
        self.extract_candidate_answer = extract_candidate_answer
        self.math_verify_timeout_s = math_verify_timeout_s
        self._mp_ctx = multiprocessing.get_context(mp_start_method)

    @staticmethod
    def add_boxed(text):
        return f"\\boxed{{{text}}}"

    def __call__(self, candidate_answer: str, gold_answer: str) -> Dict[str, Any]:
        return self.evaluate_answer(candidate_answer, gold_answer)

    def evaluate_answer(self, candidate_answer: str, gold_answer: str) -> Dict[str, Any]:
        if self.add_boxed_to_gold_answer:
            gold_answer = self.add_boxed(gold_answer)
        if self.extract_candidate_answer:
            candidate_answer = extract_boxed_expressions(candidate_answer)
        if candidate_answer is None:
            return {"result": False, "parsed": None, "timed_out": False}

        try:
            return _run_math_verify_in_subprocess(
                candidate_answer,
                gold_answer,
                self.math_verify_timeout_s,
                self._mp_ctx,
            )
        except MathVerifyTimeoutError:
            return {"result": False, "parsed": None, "timed_out": True}


def compute_score(solution_str: str, ground_truth: str, data_source: str, extra_info: Dict[str, str]):
    evaluator = MathEvaluator()
    result = evaluator.evaluate_answer(solution_str, ground_truth)
    return {
        "score": float(result["result"]),
    }

