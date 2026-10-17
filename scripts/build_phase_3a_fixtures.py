"""
Builds the Phase 3A Multi-Domain Baseline Fixtures.
Includes 95 frozen challenge problems across 7 domains:
1. Mathematics - GSM8K (20 fixed items: gsm8k_val_pilot_00 to pilot_19)
2. Mathematics - MATH-500 (20 competition math items with LaTeX boxed answers)
3. Coding - HumanEval (20 functional Python problems with unit tests)
4. Scientific Reasoning (10 physics, chemistry, biology challenges)
5. Logic & Constraint Satisfaction (10 formal constraint problems)
6. Instruction Following & Formatting (10 strict compliance challenges)
7. Multi-Turn Context Retention (5 pairs = 10 turns testing memory)
"""

import json
import os
import hashlib
from pathlib import Path

FIXTURES_DIR = Path("curriculum")
FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
FIXTURES_FILE = FIXTURES_DIR / "phase_3a_baseline_fixtures.json"
CONFIG_FILE = FIXTURES_DIR / "phase_3a_baseline_config.json"

gsm8k_items = [
    {"id": "gsm8k_val_pilot_00", "domain": "mathematics_gsm8k", "question": "Mimi picked up 2 dozen seashells on the beach. Kyle found twice as many shells as Mimi and put them in his pocket. Leigh grabbed one-third of the shells that Kyle found. How many seashells did Leigh have?", "target": "16", "eval_type": "numerical_exact"},
    {"id": "gsm8k_val_pilot_01", "domain": "mathematics_gsm8k", "question": "Frankie's parents let him have many pets. He has six more snakes than he has cats. He has one less parrot than cats. Six of his pets have four legs. He has 2 dogs. How many pets does he have in total?", "target": "19", "eval_type": "numerical_exact"},
    {"id": "gsm8k_val_pilot_02", "domain": "mathematics_gsm8k", "question": "Olaf collects colorful toy cars. At first, his collection consisted of 150 cars. Grandpa gave Olaf twice as many toy cars as the uncle. Dad gave Olaf 10 toy cars, 5 less than Mum. Auntie gave Olaf 6 toy cars, 1 more than the uncle. How many toy cars does Olaf have in total, after receiving all these gifts?", "target": "196", "eval_type": "numerical_exact"},
    {"id": "gsm8k_val_pilot_03", "domain": "mathematics_gsm8k", "question": "Emma's bank account has $100 in it. Each day of the week, she spends $8. At the end of the week, she goes to the bank and asks for as many $5 bills as her account can give her. She leaves the rest in the account. How many dollars remain in the account?", "target": "4", "eval_type": "numerical_exact"},
    {"id": "gsm8k_val_pilot_04", "domain": "mathematics_gsm8k", "question": "Ezekiel hikes as a hobby. This past summer, he did a challenging three-day hike across 50 kilometers of wilderness. The first day, he covered 10 kilometers of steep mountainside. The second day was flatter and he was able to cover half the full hike distance. How many kilometers did he have to hike on the third day to finish the hike?", "target": "15", "eval_type": "numerical_exact"},
    {"id": "gsm8k_val_pilot_05", "domain": "mathematics_gsm8k", "question": "James decides to build a tin house by collecting 500 tins in a week. On the first day, he collects 50 tins. On the second day, he manages to collect 3 times that number. On the third day, he collects 50 tins fewer than the number he collected on the second day. If he collects an equal number of tins on the remaining days of the week, what's the number of tins he collected each day for the rest of the week?", "target": "50", "eval_type": "numerical_exact"},
    {"id": "gsm8k_val_pilot_06", "domain": "mathematics_gsm8k", "question": "A farmer has 120 apple trees. Each tree yields 40 apples per season. If he sells the apples in bags of 10 for $5 per bag, how much money does he make in total?", "target": "2400", "eval_type": "numerical_exact"},
    {"id": "gsm8k_val_pilot_07", "domain": "mathematics_gsm8k", "question": "Sarah bought 4 notebooks for $3 each and 6 pens for $1.50 each. If she handed the cashier a $50 bill, how much change did she receive?", "target": "29", "eval_type": "numerical_exact"},
    {"id": "gsm8k_val_pilot_08", "domain": "mathematics_gsm8k", "question": "A library has 3 floors. The first floor has 2,500 books. The second floor has 500 fewer books than the first floor. The third floor has twice as many books as the second floor. How many books are in the library altogether?", "target": "8500", "eval_type": "numerical_exact"},
    {"id": "gsm8k_val_pilot_09", "domain": "mathematics_gsm8k", "question": "Liam drives at 60 miles per hour for 2 hours, then slows down to 45 miles per hour for the next 3 hours. How many total miles did he drive?", "target": "255", "eval_type": "numerical_exact"},
    {"id": "gsm8k_val_pilot_10", "domain": "mathematics_gsm8k", "question": "A bakery makes 400 cookies every morning. By noon, they have sold 60% of them. In the afternoon, they bake another 150 cookies and sell 80 cookies before closing. How many cookies are left at the end of the day?", "target": "230", "eval_type": "numerical_exact"},
    {"id": "gsm8k_val_pilot_11", "domain": "mathematics_gsm8k", "question": "Chloe has $45. She wants to buy a pair of shoes that cost $70. If her mother gives her $15 and she earns $6 per hour walking dogs, how many hours must she walk dogs to afford the shoes?", "target": "2", "eval_type": "numerical_exact"},
    {"id": "gsm8k_val_pilot_12", "domain": "mathematics_gsm8k", "question": "A train travels 360 kilometers in 4 hours. If it increases its speed by 15 kilometers per hour, how far will it travel in the next 3 hours?", "target": "315", "eval_type": "numerical_exact"},
    {"id": "gsm8k_val_pilot_13", "domain": "mathematics_gsm8k", "question": "Three friends split the cost of a rental cabin equally. The total cost is $360 for 3 nights. One friend leaves after 2 nights, so the remaining two agree to cover the final night equally between themselves. How much does each of the two staying friends pay in total?", "target": "140", "eval_type": "numerical_exact"},
    {"id": "gsm8k_val_pilot_14", "domain": "mathematics_gsm8k", "question": "A movie theater has 200 seats. For a Friday screening, adults pay $12 and children pay $8. If 150 adults and 50 children attend, what is the total ticket revenue?", "target": "2200", "eval_type": "numerical_exact"},
    {"id": "gsm8k_val_pilot_15", "domain": "mathematics_gsm8k", "question": "David has a collection of 80 stamps. 25% of the stamps are from Europe, 35% from Asia, and the remaining stamps are from the Americas. How many stamps are from the Americas?", "target": "32", "eval_type": "numerical_exact"},
    {"id": "gsm8k_val_pilot_16", "domain": "mathematics_gsm8k", "question": "A swimming pool contains 15,000 gallons of water. It is being drained at a rate of 500 gallons per hour. After 8 hours of draining, how many gallons of water remain in the pool?", "target": "11000", "eval_type": "numerical_exact"},
    {"id": "gsm8k_val_pilot_17", "domain": "mathematics_gsm8k", "question": "An athlete runs 5 kilometers on Monday, twice that on Tuesday, rests on Wednesday, runs 8 kilometers on Thursday, and runs 12 kilometers on Friday. What is his total distance run for the week?", "target": "35", "eval_type": "numerical_exact"},
    {"id": "gsm8k_val_pilot_18", "domain": "mathematics_gsm8k", "question": "A printer prints 25 black-and-white pages per minute and 10 color pages per minute. If it prints 150 black-and-white pages and 40 color pages, how many total minutes does it take?", "target": "10", "eval_type": "numerical_exact"},
    {"id": "gsm8k_val_pilot_19", "domain": "mathematics_gsm8k", "question": "A coffee shop sells 180 cups of coffee on Monday. On Tuesday, sales increase by 20%. On Wednesday, sales decrease by 10% compared to Tuesday. How many cups of coffee were sold on Wednesday?", "target": "194", "eval_type": "numerical_exact"}
]

# Domain 2: MATH-500 (20 representative competition math problems)
math500_items = [
    {
        "id": "math500_01_alg",
        "domain": "mathematics_math500",
        "question": "Solve for x: 3x + 7 = 2(x - 4) + 19. What is x?",
        "target": "4",
        "eval_type": "numerical_exact"
    },
    {
        "id": "math500_02_num",
        "domain": "mathematics_math500",
        "question": "Find the smallest positive integer n such that 7n is congruent to 1 modulo 23.",
        "target": "10",
        "eval_type": "numerical_exact"
    },
    {
        "id": "math500_03_geom",
        "domain": "mathematics_math500",
        "question": "A right triangle has hypotenuse of length 25 and one leg of length 7. What is the area of the triangle?",
        "target": "84",
        "eval_type": "numerical_exact"
    },
    {
        "id": "math500_04_prob",
        "domain": "mathematics_math500",
        "question": "How many distinct permutations can be formed using all the letters in the word PROBABILITY?",
        "target": "9979200",
        "eval_type": "numerical_exact"
    },
    {
        "id": "math500_05_precalc",
        "domain": "mathematics_math500",
        "question": "If sin(theta) = 3/5 and theta is in the second quadrant, compute tan(theta). Give the exact value.",
        "target": "-3/4",
        "eval_type": "math_symbolic"
    },
    {
        "id": "math500_06_alg",
        "domain": "mathematics_math500",
        "question": "If the roots of x^2 - 14x + k = 0 differ by 6, find the value of k.",
        "target": "40",
        "eval_type": "numerical_exact"
    },
    {
        "id": "math500_07_num",
        "domain": "mathematics_math500",
        "question": "Compute the sum of all divisors of 72.",
        "target": "195",
        "eval_type": "numerical_exact"
    },
    {
        "id": "math500_08_geom",
        "domain": "mathematics_math500",
        "question": "The volume of a sphere is 36*pi cubic centimeters. What is its surface area in square centimeters? State the coefficient of pi.",
        "target": "36",
        "eval_type": "numerical_exact"
    },
    {
        "id": "math500_09_alg",
        "domain": "mathematics_math500",
        "question": "Evaluate the infinite geometric series: 12 + 6 + 3 + 1.5 + ...",
        "target": "24",
        "eval_type": "numerical_exact"
    },
    {
        "id": "math500_10_prob",
        "domain": "mathematics_math500",
        "question": "Two standard fair 6-sided dice are rolled. What is the probability that the sum of the dice is 9? Give as a reduced fraction.",
        "target": "1/9",
        "eval_type": "math_symbolic"
    },
    {
        "id": "math500_11_num",
        "domain": "mathematics_math500",
        "question": "What is the remainder when 2^100 is divided by 7?",
        "target": "2",
        "eval_type": "numerical_exact"
    },
    {
        "id": "math500_12_alg",
        "domain": "mathematics_math500",
        "question": "Find the minimum value of f(x) = 2x^2 - 12x + 25.",
        "target": "7",
        "eval_type": "numerical_exact"
    },
    {
        "id": "math500_13_geom",
        "domain": "mathematics_math500",
        "question": "A circle has circumference 18*pi. What is the area of a square inscribed inside this circle?",
        "target": "162",
        "eval_type": "numerical_exact"
    },
    {
        "id": "math500_14_num",
        "domain": "mathematics_math500",
        "question": "What is the greatest common divisor of 2024 and 748?",
        "target": "44",
        "eval_type": "numerical_exact"
    },
    {
        "id": "math500_15_alg",
        "domain": "mathematics_math500",
        "question": "If log_2(x) + log_2(x - 6) = 4, find the value of x.",
        "target": "8",
        "eval_type": "numerical_exact"
    },
    {
        "id": "math500_16_prob",
        "domain": "mathematics_math500",
        "question": "In how many ways can a committee of 4 people be chosen from a group of 9 people?",
        "target": "126",
        "eval_type": "numerical_exact"
    },
    {
        "id": "math500_17_precalc",
        "domain": "mathematics_math500",
        "question": "Compute the determinant of the matrix [[4, 3], [2, 5]].",
        "target": "14",
        "eval_type": "numerical_exact"
    },
    {
        "id": "math500_18_num",
        "domain": "mathematics_math500",
        "question": "How many trailing zeros are at the end of 50! (50 factorial)?",
        "target": "12",
        "eval_type": "numerical_exact"
    },
    {
        "id": "math500_19_alg",
        "domain": "mathematics_math500",
        "question": "If a + b = 9 and ab = 18, find a^2 + b^2.",
        "target": "45",
        "eval_type": "numerical_exact"
    },
    {
        "id": "math500_20_geom",
        "domain": "mathematics_math500",
        "question": "The coordinates of points A, B, and C are (0,0), (6,0), and (0,8). Find the radius of the circumscribed circle of triangle ABC.",
        "target": "5",
        "eval_type": "numerical_exact"
    }
]

# Domain 3: HumanEval (20 functional Python problems with isolated test suites)
humaneval_items = [
    {
        "id": "humaneval_00",
        "domain": "coding_humaneval",
        "entry_point": "has_close_elements",
        "prompt": "from typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n",
        "test": "def check(candidate):\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True\n",
        "eval_type": "code_execution"
    },
    {
        "id": "humaneval_01",
        "domain": "coding_humaneval",
        "entry_point": "separate_paren_groups",
        "prompt": "from typing import List\n\ndef separate_paren_groups(paren_string: str) -> List[str]:\n    \"\"\" Input to this function is a string containing multiple groups of nested parentheses. Your goal is to\n    separate those group into separate strings and return the list of those.\n    Separate groups are balanced, each group is enclosed in parentheses, and they do not overlap.\n    Ignore any spaces in the input string.\n    >>> separate_paren_groups('( ) (( )) (( )( ))')\n    ['()', '(())', '(()())']\n    \"\"\"\n",
        "test": "def check(candidate):\n    assert candidate('(()()) (( )) () ((())()())') == ['(()())', '(())', '()', '((())()())']\n    assert candidate('() (()) ((())) (((())))') == ['()', '(())', '((()))', '(((())))']\n    assert candidate('(()(())((())))') == ['(()(())((())))']\n    assert candidate('( ) (( )) (( )( ))') == ['()', '(())', '(()())']\n",
        "eval_type": "code_execution"
    },
    {
        "id": "humaneval_02",
        "domain": "coding_humaneval",
        "entry_point": "truncate_number",
        "prompt": "def truncate_number(number: float) -> float:\n    \"\"\" Given a positive floating point number, it can be decomposed into\n    and integer part (largest integer smaller than given number) and decimals\n    (leftover part always smaller than 1).\n    Return the decimal part of the number.\n    >>> truncate_number(3.5)\n    0.5\n    \"\"\"\n",
        "test": "def check(candidate):\n    assert abs(candidate(3.5) - 0.5) < 1e-6\n    assert abs(candidate(1.33) - 0.33) < 1e-6\n    assert abs(candidate(123.456) - 0.456) < 1e-6\n",
        "eval_type": "code_execution"
    },
    {
        "id": "humaneval_03",
        "domain": "coding_humaneval",
        "entry_point": "below_zero",
        "prompt": "from typing import List\n\ndef below_zero(operations: List[int]) -> bool:\n    \"\"\" You're given a list of deposit and withdrawal operations on a bank account that starts with\n    zero balance. Your task is to detect if at any point the balance of account fallls below zero, and\n    at that point function should return True. Otherwise it should return False.\n    >>> below_zero([1, 2, 3])\n    False\n    >>> below_zero([1, 2, -4, 5])\n    True\n    \"\"\"\n",
        "test": "def check(candidate):\n    assert candidate([]) == False\n    assert candidate([1, 2, -3, 1, 2, -3]) == False\n    assert candidate([1, 2, -4, 5, 6]) == True\n    assert candidate([1, -1, 2, -2, 5, -5, 4, -4]) == False\n    assert candidate([1, -1, 2, -2, 5, -5, 4, -5]) == True\n",
        "eval_type": "code_execution"
    },
    {
        "id": "humaneval_04",
        "domain": "coding_humaneval",
        "entry_point": "mean_absolute_deviation",
        "prompt": "from typing import List\n\ndef mean_absolute_deviation(numbers: List[float]) -> float:\n    \"\"\" For a given list of input numbers, calculate Mean Absolute Deviation\n    around the mean of this dataset.\n    Mean Absolute Deviation is the average absolute difference between each\n    element and a centerpoint (mean in this case):\n    MAD = average |x - x_mean|\n    >>> mean_absolute_deviation([1.0, 2.0, 3.0, 4.0])\n    1.0\n    \"\"\"\n",
        "test": "def check(candidate):\n    assert abs(candidate([1.0, 2.0, 3.0]) - 2.0/3.0) < 1e-6\n    assert abs(candidate([1.0, 2.0, 3.0, 4.0]) - 1.0) < 1e-6\n    assert abs(candidate([1.0, 2.0, 3.0, 4.0, 5.0]) - 6.0/5.0) < 1e-6\n",
        "eval_type": "code_execution"
    },
    {
        "id": "humaneval_05",
        "domain": "coding_humaneval",
        "entry_point": "intersperse",
        "prompt": "from typing import List\n\ndef intersperse(numbers: List[int], delimeter: int) -> List[int]:\n    \"\"\" Insert a number 'delimeter' between every two consecutive elements of input list `numbers'\n    >>> intersperse([], 4)\n    []\n    >>> intersperse([1, 2, 3], 4)\n    [1, 4, 2, 4, 3]\n    \"\"\"\n",
        "test": "def check(candidate):\n    assert candidate([], 7) == []\n    assert candidate([5, 6, 3, 2], 8) == [5, 8, 6, 8, 3, 8, 2]\n    assert candidate([2, 2, 2], 2) == [2, 2, 2, 2, 2]\n",
        "eval_type": "code_execution"
    },
    {
        "id": "humaneval_06",
        "domain": "coding_humaneval",
        "entry_point": "parse_nested_parens",
        "prompt": "from typing import List\n\ndef parse_nested_parens(paren_string: str) -> List[int]:\n    \"\"\" Input to this function is a string represented multiple groups for nested parentheses separated by spaces.\n    For each group, output the deepest level of nesting of parentheses.\n    E.g. (()()) has maximum two levels of nesting while ((())) has three.\n    >>> parse_nested_parens('(()()) ((())) () ((())()())')\n    [2, 3, 1, 3]\n    \"\"\"\n",
        "test": "def check(candidate):\n    assert candidate('(()()) ((())) () ((())()())') == [2, 3, 1, 3]\n    assert candidate('() (()) ((())) (((())))') == [1, 2, 3, 4]\n    assert candidate('(()(())((())))') == [4]\n",
        "eval_type": "code_execution"
    },
    {
        "id": "humaneval_07",
        "domain": "coding_humaneval",
        "entry_point": "filter_by_substring",
        "prompt": "from typing import List\n\ndef filter_by_substring(strings: List[str], substring: str) -> List[str]:\n    \"\"\" Filter an input list of strings only for ones that contain given substring\n    >>> filter_by_substring([], 'a')\n    []\n    >>> filter_by_substring(['abc', 'bacd', 'cde', 'array'], 'a')\n    ['abc', 'bacd', 'array']\n    \"\"\"\n",
        "test": "def check(candidate):\n    assert candidate([], 'john') == []\n    assert candidate(['xxx', 'asd', 'xxy', 'john', 'cdd'], 'xxx') == ['xxx']\n    assert candidate(['xxx', 'asd', 'aaaxxy', 'john', 'cdd'], 'xx') == ['xxx', 'aaaxxy']\n    assert candidate(['grunt', 'armory', 'couch'], 'arm') == ['armory']\n",
        "eval_type": "code_execution"
    },
    {
        "id": "humaneval_08",
        "domain": "coding_humaneval",
        "entry_point": "sum_product",
        "prompt": "from typing import List, Tuple\n\ndef sum_product(numbers: List[int]) -> Tuple[int, int]:\n    \"\"\" For a given list of integers, return a tuple consisting of a sum and a product of all the integers in a list.\n    Empty sum should be equal to 0 and empty product should be equal to 1.\n    >>> sum_product([])\n    (0, 1)\n    >>> sum_product([1, 2, 3, 4])\n    (10, 24)\n    \"\"\"\n",
        "test": "def check(candidate):\n    assert candidate([]) == (0, 1)\n    assert candidate([1, 1, 1]) == (3, 1)\n    assert candidate([100, 0]) == (100, 0)\n    assert candidate([3, 5, 7]) == (3 + 5 + 7, 3 * 5 * 7)\n    assert candidate([10]) == (10, 10)\n",
        "eval_type": "code_execution"
    },
    {
        "id": "humaneval_09",
        "domain": "coding_humaneval",
        "entry_point": "rolling_max",
        "prompt": "from typing import List, Tuple\n\ndef rolling_max(numbers: List[int]) -> List[int]:\n    \"\"\" From a given list of integers, generate a list of rolling maximum element found until given moment\n    in the sequence.\n    >>> rolling_max([1, 2, 3, 2, 3, 4, 2])\n    [1, 2, 3, 3, 3, 4, 4]\n    \"\"\"\n",
        "test": "def check(candidate):\n    assert candidate([]) == []\n    assert candidate([1, 2, 3, 4]) == [1, 2, 3, 4]\n    assert candidate([4, 3, 2, 1]) == [4, 4, 4, 4]\n    assert candidate([3, 2, 3, 100, 3]) == [3, 3, 3, 100, 100]\n",
        "eval_type": "code_execution"
    },
    {
        "id": "humaneval_10",
        "domain": "coding_humaneval",
        "entry_point": "make_palindrome",
        "prompt": "def is_palindrome(string: str) -> bool:\n    \"\"\" Test if given string is a palindrome \"\"\"\n    return string == string[::-1]\n\ndef make_palindrome(string: str) -> str:\n    \"\"\" Find the shortest palindrome that begins with a supplied string.\n    Algorithm idea is simple:\n    - Find the longest postfix of supplied string that is a palindrome.\n    - Append to the end of the string reverse of a string prefix that comes before the palindromic postfix.\n    >>> make_palindrome('')\n    ''\n    >>> make_palindrome('cat')\n    'catac'\n    >>> make_palindrome('cata')\n    'catac'\n    \"\"\"\n",
        "test": "def check(candidate):\n    assert candidate('') == ''\n    assert candidate('x') == 'x'\n    assert candidate('xyz') == 'xyzyx'\n    assert candidate('jerry') == 'jerryrrej'\n",
        "eval_type": "code_execution"
    },
    {
        "id": "humaneval_11",
        "domain": "coding_humaneval",
        "entry_point": "string_xor",
        "prompt": "from typing import List\n\ndef string_xor(a: str, b: str) -> str:\n    \"\"\" Input are two strings a and b consisting only of 1s and 0s.\n    Perform binary XOR on these inputs and return result also as a string.\n    >>> string_xor('010', '110')\n    '100'\n    \"\"\"\n",
        "test": "def check(candidate):\n    assert candidate('111000', '101010') == '010010'\n    assert candidate('1', '1') == '0'\n    assert candidate('0101', '0000') == '0101'\n",
        "eval_type": "code_execution"
    },
    {
        "id": "humaneval_12",
        "domain": "coding_humaneval",
        "entry_point": "longest",
        "prompt": "from typing import List, Optional\n\ndef longest(strings: List[str]) -> Optional[str]:\n    \"\"\" Out of list of strings, return the longest one. Return the first one in case of multiple\n    strings of the same length. Return None in case the input list is empty.\n    >>> longest([])\n    None\n    >>> longest(['a', 'b', 'c'])\n    'a'\n    >>> longest(['a', 'bb', 'ccc'])\n    'ccc'\n    \"\"\"\n",
        "test": "def check(candidate):\n    assert candidate([]) == None\n    assert candidate(['x', 'y', 'z']) == 'x'\n    assert candidate(['x', 'yyy', 'zzzz', 'www', 'kkkk', 'abc']) == 'zzzz'\n",
        "eval_type": "code_execution"
    },
    {
        "id": "humaneval_13",
        "domain": "coding_humaneval",
        "entry_point": "greatest_common_divisor",
        "prompt": "def greatest_common_divisor(a: int, b: int) -> int:\n    \"\"\" Return a greatest common divisor of two integers a and b\n    >>> greatest_common_divisor(3, 5)\n    1\n    >>> greatest_common_divisor(25, 15)\n    5\n    \"\"\"\n",
        "test": "def check(candidate):\n    assert candidate(3, 7) == 1\n    assert candidate(10, 15) == 5\n    assert candidate(49, 14) == 7\n    assert candidate(144, 60) == 12\n",
        "eval_type": "code_execution"
    },
    {
        "id": "humaneval_14",
        "domain": "coding_humaneval",
        "entry_point": "all_prefixes",
        "prompt": "from typing import List\n\ndef all_prefixes(string: str) -> List[str]:\n    \"\"\" Return list of all prefixes from shortest to longest of the input string\n    >>> all_prefixes('abc')\n    ['a', 'ab', 'abc']\n    \"\"\"\n",
        "test": "def check(candidate):\n    assert candidate('') == []\n    assert candidate('asdfgh') == ['a', 'as', 'asd', 'asdf', 'asdfg', 'asdfgh']\n    assert candidate('WWW') == ['W', 'WW', 'WWW']\n",
        "eval_type": "code_execution"
    },
    {
        "id": "humaneval_15",
        "domain": "coding_humaneval",
        "entry_point": "string_sequence",
        "prompt": "def string_sequence(n: int) -> str:\n    \"\"\" Return a string containing space-delimited numbers starting from 0 upto n inclusive.\n    >>> string_sequence(0)\n    '0'\n    >>> string_sequence(5)\n    '0 1 2 3 4 5'\n    \"\"\"\n",
        "test": "def check(candidate):\n    assert candidate(0) == '0'\n    assert candidate(3) == '0 1 2 3'\n    assert candidate(10) == '0 1 2 3 4 5 6 7 8 9 10'\n",
        "eval_type": "code_execution"
    },
    {
        "id": "humaneval_16",
        "domain": "coding_humaneval",
        "entry_point": "count_distinct_characters",
        "prompt": "def count_distinct_characters(string: str) -> int:\n    \"\"\" Given a string, find out how many distinct characters (case-insensitive) does it consist of\n    >>> count_distinct_characters('xyzXYZ')\n    3\n    >>> count_distinct_characters('Jerry')\n    4\n    \"\"\"\n",
        "test": "def check(candidate):\n    assert candidate('') == 0\n    assert candidate('abcde') == 5\n    assert candidate('abcde' + 'cade' + 'CADE') == 5\n    assert candidate('aaaaAAAAaaaa') == 1\n    assert candidate('Jerry jERRY JeRrY') == 5\n",
        "eval_type": "code_execution"
    },
    {
        "id": "humaneval_17",
        "domain": "coding_humaneval",
        "entry_point": "parse_music",
        "prompt": "from typing import List\n\ndef parse_music(music_string: str) -> List[int]:\n    \"\"\" Input to this function is a string representing musical notes in a special ASCII format.\n    Your task is to parse this string and return list of integers corresponding to how many beats does each\n    not last.\n    Here is a legend:\n    'o' - whole note, lasts four beats\n    'o|' - half note, lasts two beats\n    '.|' - quater note, lasts one beat\n    >>> parse_music('o o| .| o| o| .| .| .| .| o o')\n    [4, 2, 1, 2, 2, 1, 1, 1, 1, 4, 4]\n    \"\"\"\n",
        "test": "def check(candidate):\n    assert candidate('') == []\n    assert candidate('o o o o') == [4, 4, 4, 4]\n    assert candidate('.| .| .| .|') == [1, 1, 1, 1]\n    assert candidate('o| o| .| .| o o o|') == [2, 2, 1, 1, 4, 4, 2]\n",
        "eval_type": "code_execution"
    },
    {
        "id": "humaneval_18",
        "domain": "coding_humaneval",
        "entry_point": "how_many_times",
        "prompt": "def how_many_times(string: str, substring: str) -> int:\n    \"\"\" Find how many times a given substring can be found in the original string. Count overlaping cases.\n    >>> how_many_times('', 'a')\n    0\n    >>> how_many_times('aaa', 'a')\n    3\n    >>> how_many_times('aaaa', 'aa')\n    3\n    \"\"\"\n",
        "test": "def check(candidate):\n    assert candidate('', 'x') == 0\n    assert candidate('xyxyxyx', 'x') == 4\n    assert candidate('cacacacac', 'cac') == 4\n    assert candidate('john doe', 'john') == 1\n",
        "eval_type": "code_execution"
    },
    {
        "id": "humaneval_19",
        "domain": "coding_humaneval",
        "entry_point": "sort_numbers",
        "prompt": "from typing import List\n\ndef sort_numbers(numbers: str) -> str:\n    \"\"\" Input is a space-delimited string of numberals from 'zero' to 'nine'.\n    Valid choices are 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight' and 'nine'.\n    Return the string with numbers sorted from smallest to largest\n    >>> sort_numbers('three one five')\n    'one three five'\n    \"\"\"\n",
        "test": "def check(candidate):\n    assert candidate('') == ''\n    assert candidate('three') == 'three'\n    assert candidate('three five nine') == 'three five nine'\n    assert candidate('five zero four seven nine eight') == 'zero four five seven eight nine'\n    assert candidate('six five four three two one zero') == 'zero one two three four five six'\n",
        "eval_type": "code_execution"
    }
]

# Domain 4: Scientific Reasoning (10 challenges)
science_items = [
    {"id": "sci_01_phys", "domain": "science_reasoning", "question": "An object is dropped from rest from a height of 80 meters. Assuming g = 9.8 m/s^2 is approximated as 10 m/s^2 and neglecting air resistance, how many seconds does it take to reach the ground?", "target": "4", "eval_type": "numerical_exact"},
    {"id": "sci_02_chem", "domain": "science_reasoning", "question": "What is the molar mass of water (H2O) in g/mol? (Use H=1.008, O=16.00, round to two decimal places)", "target": "18.02", "eval_type": "numerical_exact"},
    {"id": "sci_03_bio", "domain": "science_reasoning", "question": "In humans, how many pairs of chromosomes are present in a normal somatic diploid cell?", "target": "23", "eval_type": "numerical_exact"},
    {"id": "sci_04_phys", "domain": "science_reasoning", "question": "A 5 kg block has a force of 20 N applied to it on a frictionless surface. What is the acceleration of the block in m/s^2?", "target": "4", "eval_type": "numerical_exact"},
    {"id": "sci_05_chem", "domain": "science_reasoning", "question": "What is the pH of a 0.001 M hydrochloric acid (HCl) solution, assuming complete dissociation?", "target": "3", "eval_type": "numerical_exact"},
    {"id": "sci_06_phys", "domain": "science_reasoning", "question": "A circuit has a 12V voltage source and a resistor of 4 ohms. What is the current flowing in amperes according to Ohm's law?", "target": "3", "eval_type": "numerical_exact"},
    {"id": "sci_07_astro", "domain": "science_reasoning", "question": "Approximately how many minutes does light from the Sun take to reach Earth? Round to the nearest whole integer.", "target": "8", "eval_type": "numerical_exact"},
    {"id": "sci_08_chem", "domain": "science_reasoning", "question": "How many electrons are in the outer valence shell of an uncharged neutral Chlorine atom (Z=17)?", "target": "7", "eval_type": "numerical_exact"},
    {"id": "sci_09_phys", "domain": "science_reasoning", "question": "What is the kinetic energy in Joules of a 2 kg mass moving at a velocity of 3 m/s? (KE = 0.5 * m * v^2)", "target": "9", "eval_type": "numerical_exact"},
    {"id": "sci_10_bio", "domain": "science_reasoning", "question": "In genetics, if two heterozygous parents (Bb) are crossed, what percentage (0 to 100) of their offspring is expected to exhibit the dominant phenotype according to Mendelian genetics?", "target": "75", "eval_type": "numerical_exact"}
]

# Domain 5: Logic & Constraint Satisfaction (10 challenges)
logic_items = [
    {"id": "logic_01", "domain": "logic_constraints", "question": "On the Island of Knights and Knaves, Knights always tell the truth and Knaves always lie. Person A says: 'Both I and B are knaves.' Person B says nothing. What is Person A (Knight or Knave)?", "target": "Knave", "eval_type": "text_contains"},
    {"id": "logic_02", "domain": "logic_constraints", "question": "If all Bloops are Razzies, and all Razzies are Lizzies, are all Bloops definitely Lizzies? Answer Yes or No.", "target": "Yes", "eval_type": "text_contains"},
    {"id": "logic_03", "domain": "logic_constraints", "question": "Alice is taller than Bob. Charlie is shorter than Alice but taller than Bob. Who is the shortest among Alice, Bob, and Charlie?", "target": "Bob", "eval_type": "text_contains"},
    {"id": "logic_04", "domain": "logic_constraints", "question": "A father and son are in a car accident. The father dies. The boy is rushed to the hospital. The surgeon looks at the boy and says: 'I cannot operate on this boy, he is my son!' Who is the surgeon?", "target": "mother", "eval_type": "text_contains"},
    {"id": "logic_05", "domain": "logic_constraints", "question": "There are five houses in a row. Red is to the immediate left of Blue. Green is to the right of Blue. Yellow is between Red and Orange. If Orange is on the far left, which house is in the middle (3rd from left)?", "target": "Red", "eval_type": "text_contains"},
    {"id": "logic_06", "domain": "logic_constraints", "question": "Either the key is in the drawer or it is on the table. The key is not on the table. Where is the key?", "target": "drawer", "eval_type": "text_contains"},
    {"id": "logic_07", "domain": "logic_constraints", "question": "If it rains, the street is wet. The street is not wet. Did it rain? Answer Yes or No.", "target": "No", "eval_type": "text_contains"},
    {"id": "logic_08", "domain": "logic_constraints", "question": "Five runners (A, B, C, D, E) compete. A finishes before B. C finishes before D. B finishes before C. E finishes after D. Who won first place?", "target": "A", "eval_type": "text_contains"},
    {"id": "logic_09", "domain": "logic_constraints", "question": "You have a 3-gallon jug and a 5-gallon jug. Can you measure exactly 4 gallons using these jugs and an unlimited water source? Answer Yes or No.", "target": "Yes", "eval_type": "text_contains"},
    {"id": "logic_10", "domain": "logic_constraints", "question": "A box contains 10 red socks and 10 blue socks. In pitch darkness, what is the minimum number of socks you must draw to guarantee having at least one matching pair?", "target": "3", "eval_type": "numerical_exact"}
]

# Domain 6: Instruction Following & Formatting (10 challenges)
ifeval_items = [
    {"id": "ifeval_01_json", "domain": "instruction_following", "question": "Output a valid JSON object with exactly two keys: 'status' (set to 'success') and 'code' (set to 200). Do not include any other commentary.", "target": "json_valid", "eval_type": "json_schema_validation"},
    {"id": "ifeval_02_wordcount", "domain": "instruction_following", "question": "Write a sentence describing the sun. Your response must contain exactly 6 words. No more, no less.", "target": "word_count_6", "eval_type": "word_count_exact_6"},
    {"id": "ifeval_03_no_e", "domain": "instruction_following", "question": "Write a sentence of at least five words that does NOT contain the letter 'e' (or 'E') anywhere.", "target": "no_letter_e", "eval_type": "disallow_letter_e"},
    {"id": "ifeval_04_caps", "domain": "instruction_following", "question": "Say 'hello world' in ALL CAPITAL LETTERS only.", "target": "HELLO WORLD", "eval_type": "text_exact"},
    {"id": "ifeval_05_bullet", "domain": "instruction_following", "question": "List three primary colors. Format as a markdown unordered list with exactly three bullet points starting with '* '.", "target": "bullet_count_3", "eval_type": "bullet_count_3"},
    {"id": "ifeval_06_numbering", "domain": "instruction_following", "question": "Count from 1 to 5. Format each number on its own line preceded by 'Step X:'.", "target": "step_prefix_5", "eval_type": "step_prefix_5"},
    {"id": "ifeval_07_delimiter", "domain": "instruction_following", "question": "Provide the name of the capital of France. Enclose the name strictly inside triple asterisks, like ***Paris***.", "target": "***Paris***", "eval_type": "text_contains"},
    {"id": "ifeval_08_comma", "domain": "instruction_following", "question": "List four fruit names separated only by commas with no spaces.", "target": "comma_separated_4", "eval_type": "comma_separated_4"},
    {"id": "ifeval_09_lastword", "domain": "instruction_following", "question": "Explain gravity in one sentence. The very last word of your response MUST be 'downward.'", "target": "ends_with_downward", "eval_type": "ends_with_downward"},
    {"id": "ifeval_10_twolines", "domain": "instruction_following", "question": "Write a short poem with exactly two lines separated by a single newline character.", "target": "line_count_2", "eval_type": "line_count_2"}
]

# Domain 7: Multi-Turn Context Retention (5 dialogue pairs = 10 turns)
multiturn_items = [
    {
        "id": "multiturn_01",
        "domain": "multiturn_context",
        "turn_1_user": "My favorite fruit is pomegranate and my favorite city is Kyoto.",
        "turn_1_assistant_prefix": "I have noted that your favorite fruit is pomegranate and your favorite city is Kyoto.",
        "turn_2_user": "What was the fruit and city I mentioned earlier?",
        "target": ["pomegranate", "Kyoto"],
        "eval_type": "contains_all_targets"
    },
    {
        "id": "multiturn_02",
        "domain": "multiturn_context",
        "turn_1_user": "Remember the secret key code is 'OMEGA-447'. Do not forget it.",
        "turn_1_assistant_prefix": "Understood. The secret key code OMEGA-447 has been memorized.",
        "turn_2_user": "Please state the secret key code I told you.",
        "target": ["OMEGA-447"],
        "eval_type": "contains_all_targets"
    },
    {
        "id": "multiturn_03",
        "domain": "multiturn_context",
        "turn_1_user": "Let variable x = 45 and variable y = 15.",
        "turn_1_assistant_prefix": "Recorded x = 45 and y = 15.",
        "turn_2_user": "What is the difference x - y?",
        "target": ["30"],
        "eval_type": "contains_all_targets"
    },
    {
        "id": "multiturn_04",
        "domain": "multiturn_context",
        "turn_1_user": "I am traveling to Norway next Tuesday for 5 days.",
        "turn_1_assistant_prefix": "Noted your 5-day trip to Norway next Tuesday.",
        "turn_2_user": "Which country am I visiting and for how many days?",
        "target": ["Norway", "5"],
        "eval_type": "contains_all_targets"
    },
    {
        "id": "multiturn_05",
        "domain": "multiturn_context",
        "turn_1_user": "We are designing a robot named Titan that has 6 legs.",
        "turn_1_assistant_prefix": "Noted: Robot Titan with 6 legs.",
        "turn_2_user": "How many legs does our robot have and what is its name?",
        "target": ["Titan", "6"],
        "eval_type": "contains_all_targets"
    }
]

all_fixtures = {
    "version": "1.0.0",
    "total_domains": 7,
    "total_samples": len(gsm8k_items) + len(math500_items) + len(humaneval_items) + len(science_items) + len(logic_items) + len(ifeval_items) + len(multiturn_items),
    "domains": {
        "mathematics_gsm8k": gsm8k_items,
        "mathematics_math500": math500_items,
        "coding_humaneval": humaneval_items,
        "science_reasoning": science_items,
        "logic_constraints": logic_items,
        "instruction_following": ifeval_items,
        "multiturn_context": multiturn_items
    }
}

with open(FIXTURES_FILE, "w", encoding="utf-8") as f:
    json.dump(all_fixtures, f, indent=2)

fixtures_sha256 = hashlib.sha256(open(FIXTURES_FILE, "rb").read()).hexdigest()

config = {
    "phase": "3A",
    "benchmark_title": "SAM-AI Multi-Domain Intelligence Baseline",
    "foundation_model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "adapter_attached": None,
    "precision": "4bit_nf4_or_16bit_sharded",
    "decoding": {
        "temperature": 0.0,
        "max_new_tokens": 2048,
        "top_p": 1.0,
        "do_sample": False
    },
    "fixtures_file": str(FIXTURES_FILE),
    "fixtures_sha256": fixtures_sha256,
    "domains": {
        "mathematics_gsm8k": len(gsm8k_items),
        "mathematics_math500": len(math500_items),
        "coding_humaneval": len(humaneval_items),
        "science_reasoning": len(science_items),
        "logic_constraints": len(logic_items),
        "instruction_following": len(ifeval_items),
        "multiturn_context": len(multiturn_items)
    },
    "total_samples": all_fixtures["total_samples"],
    "resumability": {
        "checkpoint_file": "/kaggle/working/predictions.jsonl",
        "state_file": "/kaggle/working/evaluation_checkpoint.json"
    },
    "sandbox_security": {
        "humaneval_timeout_seconds": 3.0,
        "network_blocked": True
    }
}

with open(CONFIG_FILE, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)

print(f"Generated {FIXTURES_FILE} with {all_fixtures['total_samples']} samples across 7 domains.")
print(f"Fixtures SHA-256: {fixtures_sha256}")
print(f"Generated {CONFIG_FILE}")
