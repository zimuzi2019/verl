import re
import string
import re
from pylatexenc import latex2text
import sympy
from sympy.parsing import sympy_parser
from typing import Optional
import json
def maybe_normalize_float(span: str):
    if (
        span
        and (
            re.match(r"^[+-][0-9]+[.]?[0-9]*$", span)
            or (re.match(r"^[0-9]*[.]?[0-9]*$", span))
        )
        and span != "."
    ):
        return str(float(span))
    else:
        return span


def maybe_normalize_number(text: str) -> str:
    units = [
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
    ]
    for index, unit in enumerate(units):
        if text == unit:
            return str(float(index))
    return text


def remove_punc(text: str) -> str:
    exclude = set(string.punctuation)
    return "".join(ch for ch in text if ch not in exclude)


def remove_articles(text: str) -> str:
    return re.sub(r"\b(a|an|the)\b", " ", text)


def check_overlap(str1, str2):
    str1 = remove_punc(str1.replace(" ", ""))
    str2 = remove_punc(str2.replace(" ", ""))
    if str1 in str2 or str2 in str1:
        return True
    count = 0
    for letter in str1:
        if letter != "0":
            if letter in str2:
                count += 1
    if len(str1) == 0 or len(str2) == 0:
        return True
    else:
        return True if count / len(str1) > 0.5 or count / len(str2) > 0.5 else False


def get_answer(pred):
    match = re.search(r"(The|the) answer is ([^\.]+)\.$", pred)
    if match:
        return match.group(2).strip('"'), True
    return pred, False


def eval_ex_match(pred, gold_result):
    pred = pred.lower()
    gold_result = str(gold_result).lower()
    compare_1 = pred.replace(".", "").replace(",", "").replace("%", "")
    if len(compare_1) == 0:
        compare_1 = " "
    compare_2 = gold_result.replace(".", "").replace(",", "").replace("%", "")
    if len(compare_2) == 0:
        compare_2 = " "
    if compare_1[0] == "-":
        compare_1 = compare_1[1:]
    if compare_2[0] == "-":
        compare_2 = compare_2[1:]
    if (
        compare_1.isdigit() == True
        and compare_2.isdigit() == True
        and pred.count(".") < 2
        and gold_result != "-"
    ):
        if pred[-1] == ".":
            pred = pred[0 : len(pred) - 1]
        gold_result = gold_result.replace(",", "").replace("%", "")
        pred = pred.replace(",", "").replace("%", "")
        pred = abs(float(pred))
        gold_result = abs(float(gold_result))
        if abs(pred - gold_result) < 0.01:
            return True, str(pred), str(gold_result)
        else:
            return False, str(pred), str(gold_result)

    if " and " in pred and "|" in gold_result:
        pred = pred.replace(" and ", ", ")

    pred = [span.strip() for span in pred.split(", ")]

    if "|" in gold_result:
        gold_result = [span.strip() for span in gold_result.split("|")]
    else:
        gold_result = [span.strip() for span in gold_result.split(", ")]

    pred = [
        maybe_normalize_number(remove_punc(remove_articles(span.strip())))
        for span in pred
    ]
    gold_result = [
        maybe_normalize_number(remove_punc(remove_articles(span.strip())))
        for span in gold_result
    ]

    clean_float = True
    if clean_float:
        pred = [maybe_normalize_float(span) for span in pred]
        gold_result = [maybe_normalize_float(span) for span in gold_result]
    indicater = False
    for item in pred:
        if item in gold_result:
            indicater = True

    if sorted(pred) == sorted(gold_result):
        indicater = True
    return sorted(pred) == sorted(gold_result), sorted(pred), sorted(gold_result)


def match_all(data, option):
    if len(data["label"]) == len(data[option + " prediction"]):
        flag = True
        for i in range(len(data["label"])):
            if_match, pred, label = eval_ex_match(
                str(data["label"][i]), data[option + " prediction"][i]
            )
            if if_match == False:
                flag = False
        if flag == True:
            return True
    else:
        return False
    

def find_matching_brace(s, start):
    """
    给定字符串 s 和 start 位置的 '{'，返回与之匹配的 '}' 的索引。
    如果没有匹配，返回 -1。
    """
    count = 0
    for i in range(start, len(s)):
        if s[i] == '{':
            count += 1
        elif s[i] == '}':
            count -= 1
            if count == 0:
                return i
    return -1
 
def extract_boxed_content(response_content):
    """
    提取 response_content 中 \boxed{...} 内的完整内容（处理嵌套括号）。
    如果找不到，则返回 None。
    """
    keyword = r"\boxed{"
    idx = response_content.find(keyword)
    if idx == -1:
        return None
    # 定位第一个 '{' 的位置
    start = idx + len(keyword) - 1  # 此时 s[start] 应为 '{'
    if response_content[start] != '{':
        return None
    end = find_matching_brace(response_content, start)
    if end == -1:
        return None
    # 返回内部内容，不包括最外层括号
    return response_content[start+1:end].strip()
 
def remove_latex_text(s):
    """
    移除字符串 s 中所有 \text{...} 结构（支持嵌套括号）。
    """
    keyword = r"\text{"
    while keyword in s:
        idx = s.find(keyword)
        # 找到 \text{ 后面的 '{' 索引
        start = idx + len(keyword) - 1  # 应该为 '{'
        if start >= len(s) or s[start] != '{':
            # 格式不对，跳过
            break
        end = find_matching_brace(s, start)
        if end == -1:
            break
        # 提取 \text{...} 内部内容
        inner = s[start+1:end].strip()
        # 替换整个 \text{...} 结构为内部内容
        s = s[:idx] + inner + s[end+1:]
    return s
 
def extract_boxed_answer(response_content):
    """
    综合使用上面两个函数，先提取 \boxed{...} 内的内容，
    然后移除其中的所有 \text{...} 结构，并返回最终答案。
    """
    content = extract_boxed_content(response_content)
    if content is None:
        return None
    # 移除所有 \text{...} 结构
    content = remove_latex_text(content)
    # 同时将 "\%" 替换为 "%" 等常见 LaTeX 转义符（如有需要）
    content = content.replace(r"\%", "%")
    return content.strip()

# Dan Hendrycks' code
def mathd_normalize_answer(answer: Optional[str]) -> Optional[str]:
    if answer is None:
        return None
    answer = answer.strip()
    try:
        # Remove enclosing `\text{}`.
        m = re.search("^\\\\text\{(?P<text>.+?)\}$", answer)
        if m is not None:
            answer = m.group("text").strip()
        return _strip_string(answer)
    except:
        return answer

def _strip_string(string):
    def _fix_fracs(string):
        substrs = string.split("\\frac")
        new_str = substrs[0]
        if len(substrs) > 1:
            substrs = substrs[1:]
            for substr in substrs:
                new_str += "\\frac"
                if substr[0] == "{":
                    new_str += substr
                else:
                    try:
                        assert len(substr) >= 2
                    except:
                        return string
                    a = substr[0]
                    b = substr[1]
                    if b != "{":
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}{" + b + "}" + post_substr
                        else:
                            new_str += "{" + a + "}{" + b + "}"
                    else:
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}" + b + post_substr
                        else:
                            new_str += "{" + a + "}" + b
        string = new_str
        return string


    def _fix_a_slash_b(string):
        if len(string.split("/")) != 2:
            return string
        a = string.split("/")[0]
        b = string.split("/")[1]
        try:
            a = int(a)
            b = int(b)
            assert string == "{}/{}".format(a, b)
            new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
            return new_string
        except:
            return string


    def _remove_right_units(string):
        # "\\text{ " only ever occurs (at least in the val set) when describing units
        if "\\text{ " in string:
            splits = string.split("\\text{ ")
            assert len(splits) == 2
            return splits[0]
        else:
            return string


    def _fix_sqrt(string):
        if "\\sqrt" not in string:
            return string
        splits = string.split("\\sqrt")
        new_string = splits[0]
        for split in splits[1:]:
            if split[0] != "{":
                a = split[0]
                new_substr = "\\sqrt{" + a + "}" + split[1:]
            else:
                new_substr = "\\sqrt" + split
            new_string += new_substr
        return new_string
    # linebreaks
    string = string.replace("\n", "")
    # print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    # print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    # print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    # print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    # print(string)

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string


# sympy might hang -- we don't care about trying to be lenient in these cases
BAD_SUBSTRINGS = ["^{", "^("]
BAD_REGEXES = ["\^[0-9]+\^", "\^[0-9][0-9]+"]
TUPLE_CHARS = "()[]"


def _sympy_parse(expr: str):
    """Parses an expression with sympy."""
    py_expr = expr.replace("^", "**")
    return sympy_parser.parse_expr(
        py_expr,
        transformations=(
            sympy_parser.standard_transformations
            + (sympy_parser.implicit_multiplication_application,)
        ),
    )


def _parse_latex(expr: str) -> str:
    """Attempts to parse latex to an expression sympy can read."""
    expr = expr.replace("\\tfrac", "\\frac")
    expr = expr.replace("\\dfrac", "\\frac")
    expr = expr.replace("\\frac", " \\frac")  # Play nice with mixed numbers.
    expr = latex2text.LatexNodes2Text().latex_to_text(expr)

    # Replace the specific characters that this parser uses.
    expr = expr.replace("√", "sqrt")
    expr = expr.replace("π", "pi")
    expr = expr.replace("∞", "inf")
    expr = expr.replace("∪", "U")
    expr = expr.replace("·", "*")
    expr = expr.replace("×", "*")

    return expr.strip()


def _is_float(num: str) -> bool:
    try:
        float(num)
        return True
    except ValueError:
        return False


def _is_int(x: float) -> bool:
    try:
        return abs(x - int(round(x))) <= 1e-7
    except:
        return False


def _is_frac(expr: str) -> bool:
    return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expr))


def _str_is_int(x: str) -> bool:
    try:
        x = _strip_properly_formatted_commas(x)
        x = float(x)
        return abs(x - int(round(x))) <= 1e-7
    except:
        return False


def _str_to_int(x: str) -> bool:
    x = x.replace(",", "")
    x = float(x)
    return int(x)


def _inject_implicit_mixed_number(step: str):
    """
    Automatically make a mixed number evalable
    e.g. 7 3/4 => 7+3/4
    """
    p1 = re.compile("([0-9]) +([0-9])")
    step = p1.sub("\\1+\\2", step)  ## implicit mults
    return step


def _strip_properly_formatted_commas(expr: str):
    # We want to be careful because we don't want to strip tuple commas
    p1 = re.compile("(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expr = p1.sub("\\1\\3\\4", expr)
        if next_expr == expr:
            break
        expr = next_expr
    return next_expr


def _normalize(expr: str) -> str:
    """Normalize answer expressions."""
    if expr is None:
        return None

    # Remove enclosing `\text{}`.
    m = re.search("^\\\\text\{(?P<text>.+?)\}$", expr)
    if m is not None:
        expr = m.group("text")

    expr = expr.replace("\\%", "%")
    expr = expr.replace("\\$", "$")
    expr = expr.replace("$", "")
    expr = expr.replace("%", "")
    expr = expr.replace(" or ", " , ")
    expr = expr.replace(" and ", " , ")

    expr = expr.replace("million", "*10^6")
    expr = expr.replace("billion", "*10^9")
    expr = expr.replace("trillion", "*10^12")

    for unit in [
        "degree",
        "cm",
        "centimeter",
        "meter",
        "mile",
        "second",
        "minute",
        "hour",
        "day",
        "week",
        "month",
        "year",
        "foot",
        "feet",
        "inch",
        "yard",
    ]:
        expr = re.sub(f"{unit}(es)?(s)? *(\^[0-9]+)?", "", expr)
    expr = re.sub(f"\^ *\\\\circ", "", expr)

    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]

    expr = re.sub(",\\\\! *", "", expr)
    if _is_float(expr) and _is_int(float(expr)):
        expr = str(int(round(float(expr))))
    if "\\" in expr:
        try:
            expr = _parse_latex(expr)
        except:
            pass

    # edge case with mixed numbers and negative signs
    expr = re.sub("- *", "-", expr)

    expr = _inject_implicit_mixed_number(expr)
    expr = expr.replace(" ", "")

    # if we somehow still have latex braces here, just drop them
    expr = expr.replace("{", "")
    expr = expr.replace("}", "")

    # don't be case sensitive for text answers
    expr = expr.lower()

    if _str_is_int(expr):
        expr = str(_str_to_int(expr))

    return expr


def count_unknown_letters_in_expr(expr: str):
    expr = expr.replace("sqrt", "")
    expr = expr.replace("frac", "")
    letters_in_expr = set([x for x in expr if x.isalpha()])
    return len(letters_in_expr)


def should_allow_eval(expr: str):
    # we don't want to try parsing unknown text or functions of more than two variables
    if count_unknown_letters_in_expr(expr) > 2:
        return False

    for bad_string in BAD_SUBSTRINGS:
        if bad_string in expr:
            return False

    for bad_regex in BAD_REGEXES:
        if re.search(bad_regex, expr) is not None:
            return False

    return True


def are_equal_under_sympy(ground_truth_normalized: str, given_normalized: str):
    are_equal = False
    try:
        expr = f"({ground_truth_normalized})-({given_normalized})"
        if should_allow_eval(expr):
            sympy_diff = _sympy_parse(expr)
            simplified = sympy.simplify(sympy_diff)
            if simplified == 0:
                are_equal = True
    except:
        pass
    return are_equal


def split_tuple(expr: str):
    """
    Split the elements in a tuple/interval, while handling well-formatted commas in large numbers
    """
    expr = _strip_properly_formatted_commas(expr)
    if len(expr) == 0:
        return []
    if (
        len(expr) > 2
        and expr[0] in TUPLE_CHARS
        and expr[-1] in TUPLE_CHARS
        and all([ch not in expr[1:-1] for ch in TUPLE_CHARS])
    ):
        elems = [elem.strip() for elem in expr[1:-1].split(",")]
    else:
        elems = [expr]
    return elems


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    
    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]
    
    return retval

def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None

def find_matching_brace(s, start):
    """
    给定字符串 s 和 start 位置的 '{'，返回与之匹配的 '}' 的索引。
    如果没有匹配，返回 -1。
    """
    count = 0
    for i in range(start, len(s)):
        if s[i] == '{':
            count += 1
        elif s[i] == '}':
            count -= 1
            if count == 0:
                return i
    return -1

def remove_latex_text(s):
    """
    移除字符串 s 中所有 \text{...} 结构（支持嵌套括号）。
    """
    keyword = r"\text{"
    if s is not None:
        while keyword in s:
            idx = s.find(keyword)
            # 找到 \text{ 后面的 '{' 索引
            start = idx + len(keyword) - 1  # 应该为 '{'
            if start >= len(s) or s[start] != '{':
                # 格式不对，跳过
                break
            end = find_matching_brace(s, start)
            if end == -1:
                break
            # 提取 \text{...} 内部内容
            inner = s[start+1:end].strip()
            # 替换整个 \text{...} 结构为内部内容
            s = s[:idx] + inner + s[end+1:]
    return s

def extract_boxed_answer(solution: str) -> str:
    """Extract the answer from inside a LaTeX \\boxed{} command"""
    solution = last_boxed_only_string(solution)
    solution = remove_boxed(solution)
    solution = remove_latex_text(solution)
    return solution

def grade_answer_sympy(given_answer: str, ground_truth: str) -> bool:
    ground_truth_normalized = _normalize(ground_truth)
    given_normalized = _normalize(given_answer)

    if ground_truth_normalized is None:
        return False

    if ground_truth_normalized == given_normalized:
        return True

    if len(given_normalized) == 0:
        return False

    ground_truth_elems = split_tuple(ground_truth_normalized)
    given_elems = split_tuple(given_normalized)

    if len(ground_truth_elems) > 1 and (
        ground_truth_normalized[0] != given_normalized[0]
        or ground_truth_normalized[-1] != given_normalized[-1]
    ):
        is_correct = False
    elif len(ground_truth_elems) != len(given_elems):
        is_correct = False
    else:
        for ground_truth_elem, given_elem in zip(ground_truth_elems, given_elems):
            if _is_frac(ground_truth_elem) and _is_frac(given_elem):
                # if fractions aren't reduced, then shouldn't be marked as correct
                # so, we don't want to allow sympy.simplify in this case
                is_correct = ground_truth_elem == given_elem
            elif _str_is_int(ground_truth_elem) != _str_is_int(given_elem):
                # if the ground truth answer is an integer, we require the given answer to be a strict match (no sympy.simplify)
                is_correct = False
            else:
                is_correct = are_equal_under_sympy(ground_truth_elem, given_elem)
            if not is_correct:
                break

    return is_correct

def grade_answer_mathd(given_answer: str, ground_truth: str) -> bool:
    ground_truth_normalized_mathd = mathd_normalize_answer(ground_truth)
    given_answer_normalized_mathd = mathd_normalize_answer(given_answer)

    # be at least as lenient as mathd
    if ground_truth_normalized_mathd == given_answer_normalized_mathd:
        return True
    return False

def grade_answer_hitab(given_answer: str, ground_truth: str) -> bool:
    if "\\boxed" in ground_truth:
        extracted = extract_boxed_answer(ground_truth)
        if extracted is not None:
            ground_truth = extracted

    if "\\boxed" in given_answer:
        extracted = extract_boxed_answer(given_answer)
        if extracted is not None:
            given_answer = extracted

    is_match, normalized_pred, normalized_gold = eval_ex_match(given_answer, ground_truth)
    return is_match

def extract_answer(passage: str) -> str:
    if "\\boxed" in passage:
        return extract_boxed_answer(passage)
    return None

def grade_answer_verl(solution_str, ground_truth):
    if not ground_truth:
        return False
    if '\\boxed' in ground_truth:
        ground_truth = extract_answer(ground_truth)
    given_answer = extract_answer(solution_str)
    if given_answer is None:
        return False
    return grade_answer_mathd(given_answer, ground_truth) \
        or grade_answer_sympy(given_answer, ground_truth) \
        or grade_answer_hitab(given_answer, ground_truth)

# Delimiter constants
THOUGHT_DELIMITER_START = "<think>"
THOUGHT_DELIMITER_END = "</think>"
ANSWER_DELIMITER_START = "<answer>"
ANSWER_DELIMITER_END = "</answer>"

CORRECT_EXACT_MATCH_SCORE = 1.0
INCORRECT_EXACT_MATCH_SCORE = 0.0

def hitab_exact_match_fn(data_source, solution_str, ground_truth, extra_info):
    """
    Evaluate the Hitab answer for correctness.
    
    Parameters:
        data_source: The problem statement or any related data (used for logging).
        solution_str: The model's output string, expected to contain the <think> and </think> delimiters.
        ground_truth: The correct answer(s), provided as a string or a list of strings.
        extra_info: Additional information (currently unused).
    
    Returns:
        bool: True if the extracted model answer matches any of the correct answers; otherwise, False.
    """
    assert data_source == "hitab"
    
    solution_str = str(solution_str)
    
    if THOUGHT_DELIMITER_END in solution_str:
        model_solution = solution_str.split(THOUGHT_DELIMITER_END)[1]
    else:
        return INCORRECT_EXACT_MATCH_SCORE

    # Extract the answer (e.g., removing any additional formatting)
    model_answer = extract_answer(model_solution)
    if model_answer is None:
        return INCORRECT_EXACT_MATCH_SCORE

    # Ensure ground_truth is a list for uniform processing
    if isinstance(ground_truth, (str, float, int)):
        ground_truth = [ground_truth]

    # Preprocess each correct answer (e.g., handle formats like "\boxed")
    processed_ground_truths = []
    for truth in ground_truth:
        truth_str = str(truth)
        if "\\boxed" in truth_str:
            processed_truth = extract_answer(truth_str)
            if processed_truth is not None:
                processed_ground_truths.append(processed_truth)
        else:
            processed_ground_truths.append(truth_str)
    
    if not processed_ground_truths:
        return INCORRECT_EXACT_MATCH_SCORE

    # Compare the extracted answer with each correct answer
    for truth in processed_ground_truths:
        if grade_answer_hitab(model_answer, truth):
            return CORRECT_EXACT_MATCH_SCORE

    return INCORRECT_EXACT_MATCH_SCORE

import ast
from typing import List

def compute_refucos_score(linked_cell: str, ground_truth: str) -> float:
    """
    计算 linked_cell 中的每个子项，在 ground_truth 字符串中出现的比例。

    要求 linked_cell 必须是一个标准的 Python 列表表示，比如 "['a','b','c']"。

    Args:
        linked_cell: 标准格式的列表字符串。
        ground_truth: 待匹配的字符串。

    Returns:
        float: 匹配比例，0.0 到 1.0 之间。
    """
    # 直接解析，出错就抛给调用者
    items = ast.literal_eval(linked_cell)
    if not isinstance(items, (list, tuple)):
        raise ValueError("linked_cell 必须是列表或元组格式")
    items = [str(x) for x in items]

    if not items:
        return 0.0
    match_count = sum(1 for s in items if s in ground_truth)
    return match_count / len(items)

def compute_score(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """The scoring function for GSM8k.
 
    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.
 
    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    # ground_truth = 格式为：f"""{ground_truth} + ### linked cells: {linked_cells}"""
    #分别提取出ground_truth和linked_cells
    #assert "### linked cells:" in ground_truth
    #ground_truth, linked_cells = ground_truth.split("### linked cells: ")[0]
    #提取出solution_str中的linked_cells
    #linked_cells = solution_str.split("### linked cells: ")[1]
    if hitab_exact_match_fn("hitab", solution_str, ground_truth, None) == INCORRECT_EXACT_MATCH_SCORE:
        score = 0.0
    #refucos_score = compute_refucos_score(linked_cells, ground_truth)
    #return score * refucos_score
    return score