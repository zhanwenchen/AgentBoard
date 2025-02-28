from typing import List, Dict
from dataclasses import dataclass, field, asdict
import textwrap
import re
from fractions import Fraction


@dataclass
class Path:
    '''
    _summary_

    Args:
        a (int): _description_
        c (list, optional): _description_. Defaults to [1,2].

    Raises:
        AssertionError: _description_

    Attributes:
        action (str): 'look around'
        observation (str): 'This room is called the hallway. In it, you see: \n\ta picture\n\ta substance called air\n\tthe agent\nYou also see:\n\tA door to the green house (that is open)\n\tA door to the living room (that is open)\n\tA door to the art studio (that is open)\n\tA door to the kitchen (that is open)\n\tA door to the bedroom (that is open)\n\tA door to the workshop (that is open)'
        score (float): 0.0
        is_completed (bool): False
        freelook (str): 'This room is called the hallway. In it, you see: \n\ta picture\n\ta substance called air\n\tthe agent\nYou also see:\n\tA door to the green house (that is open)\n\tA door to the living room (that is open)\n\tA door to the art studio (that is open)\n\tA door to the kitchen (that is open)\n\tA door to the bedroom (that is open)\n\tA door to the workshop (that is open)\n'
        inventory (str): 'In your inventory, you see:\n\tan orange\n'
    Returns:
        _type_: _description_
    '''
    action: str
    observation: str
    score: float
    is_completed: str
    freelook: str
    inventory: str
    valid_actions: List[str] = field(default_factory=list)

    def __post_init__(self):
        # Convert fields that need type conversion
        self.score = float(self.score)
        self.is_completed = self.is_completed.lower() == 'true'


@dataclass
class Sequence:
    variationIdx: int
    fold: str
    taskDescription: str
    paths: List[Path] = field(default_factory=list)


def decimal_with_repeating(val, max_denominator=1000):
    """
    Converts a float (val) to a fraction, and then prints it
    in a repeating-decimal form if there's a repeating cycle.

    :param val: The float you want to display.
    :param max_denominator: Upper bound for fraction approximation.
    :return: A string like '0.(3)' or '0.03(3)', etc.
    """
    # 1) Convert float -> Fraction
    frac = Fraction(val).limit_denominator(max_denominator)
    numerator, denominator = frac.numerator, frac.denominator

    # If it’s an integer (denominator=1), just return it
    if denominator == 1:
        return str(numerator)

    # Determine the sign
    negative = (numerator * denominator) < 0
    numerator, denominator = abs(numerator), abs(denominator)

    # 2) Integer part
    integer_part = numerator // denominator
    remainder = numerator % denominator

    # Start building result
    result = "-" if negative else ""
    result += str(integer_part)

    # If there's no remainder, it's just an integer
    if remainder == 0:
        return result

    result += "."

    # 3) Long division to find repeating cycle
    # We'll track each remainder in a dictionary: remainder -> index in the decimal result
    remainders_seen = {}
    decimal_digits = ""

    idx = 0  # position in the decimal expansion
    repeating_index = -1  # where the cycle starts, if any

    while remainder != 0:
        if remainder in remainders_seen:
            # We found a repeating remainder => start of cycle
            repeating_index = remainders_seen[remainder]
            break

        # Record the index of this remainder
        remainders_seen[remainder] = idx

        remainder *= 10
        digit = remainder // denominator
        remainder = remainder % denominator
        decimal_digits += str(digit)
        idx += 1

    if remainder == 0:
        # No repeating cycle; just append digits
        result += decimal_digits
    else:
        # We have a repeating cycle
        non_repeating = decimal_digits[:repeating_index]
        repeating = decimal_digits[repeating_index:]
        result += f"{non_repeating}({repeating})"

    return result


def print_chat_dialogue(
    sequence: Sequence,
    act_width: int = 25,
    obs_width: int = 60,
    score_width: int = 6,
    delta_width: int = 8,
    completed_col_width: int = 12,  # Increased width for alignment
    freelook_col_width: int = 30,
    inventory_col_width: int = 30,
    include_freelook: bool = True,
    include_inventory: bool = True
):
    """
    Prints sequence-level info (variationIdx, fold, taskDescription),
    then prints each Path in a table with columns:
      - ACTION
      - OBSERVATION
      - SCORE
      - DELTA
      - COMPLETED? (checkmark or X)
      - FREELOOK (optional)
      - INVENTORY (optional)

    Parameters:
        sequence (Sequence): The sequence object containing metadata and paths.
        act_width (int): Column width for ACTION.
        obs_width (int): Column width for OBSERVATION.
        score_width (int): Column width for SCORE.
        delta_width (int): Column width for DELTA.
        completed_col_width (int): Column width for COMPLETED?.
        freelook_col_width (int): Column width for FREELOOK.
        inventory_col_width (int): Column width for INVENTORY.
        include_freelook (bool): Whether to include the FREELOOK column.
        include_inventory (bool): Whether to include the INVENTORY column.
    """

    def flatten_whitespace(txt: str) -> str:
        """Convert tabs/newlines/etc. into single spaces."""
        return re.sub(r"\s+", " ", txt.strip())

    def wrap_text(txt: str, width: int) -> List[str]:
        """Wrap text to a given width."""
        return textwrap.wrap(flatten_whitespace(txt), width=width) or [""]

    # ANSI color codes
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"

    # Sequence-level metadata
    print(f"{'Sequence Variation:':<20} {sequence.variationIdx}")
    print(f"{'Fold:':<20} {sequence.fold}")
    print(f"{'Task Description:':<20}")
    for line in textwrap.wrap(sequence.taskDescription, width=100):
        print(f"  {line}")
    print("\n")  # Add space before the table

    #
    # Table Headers
    #
    headers = ["ACTION", "OBSERVATION", "SCORE", "DELTA", "COMPLETED?"]
    if include_freelook:
        headers.append("FREELOOK")
    if include_inventory:
        headers.append("INVENTORY")

    # Adjust header widths based on included columns
    column_widths = {
        "ACTION": act_width,
        "OBSERVATION": obs_width,
        "SCORE": score_width,
        "DELTA": delta_width,
        "COMPLETED?": completed_col_width,
        "FREELOOK": freelook_col_width,
        "INVENTORY": inventory_col_width
    }

    # Build header line
    header_line = " | ".join(f"{header:<{column_widths[header]}}" for header in headers)
    print(header_line)

    # Calculate total width (number of separators = number of columns - 1)
    num_separators = len(headers) - 1
    total_width = sum(column_widths[header] for header in headers) + num_separators * 3
    print("-" * total_width)

    #
    # Iterate through each path
    #
    last_path = None
    for path in sequence.paths:
        # Calculate delta
        if last_path is None:
            delta = 0
        else:
            delta = path.score - last_path.score
        last_path = path

        # Wrap multiline columns
        act_lines = wrap_text(path.action, act_width)
        obs_lines = wrap_text(path.observation, obs_width)
        freelook_lines = wrap_text(path.freelook, freelook_col_width) if include_freelook else [""]
        inventory_lines = wrap_text(path.inventory, inventory_col_width) if include_inventory else [""]

        # Determine the number of lines needed
        max_lines = max(
            len(act_lines),
            len(obs_lines),
            len(freelook_lines),
            len(inventory_lines)
        )

        for i in range(max_lines):
            # ACTION, OBSERVATION, FREELOOK, INVENTORY line i (or blank)
            act_text = act_lines[i] if i < len(act_lines) else ""
            obs_text = obs_lines[i] if i < len(obs_lines) else ""
            flk_text = freelook_lines[i] if i < len(freelook_lines) else "" if include_freelook else ""
            inv_text = inventory_lines[i] if i < len(inventory_lines) else "" if include_inventory else ""

            if i == 0:
                sco_text = f"{path.score:.2f}"
                dlt_text = f"{delta:.2f}"
                # Append a space to the checkmark/X for alignment
                raw_symbol = "✓ " if path.is_completed else "✗ "
            else:
                sco_text = ""
                dlt_text = ""
                raw_symbol = ""

            # Pad columns without color
            row = []
            row.append(f"{act_text:<{act_width}}")
            row.append(f"{obs_text:<{obs_width}}")
            row.append(f"{sco_text:<{score_width}}")
            row.append(f"{dlt_text:<{delta_width}}")
            row.append(f"{raw_symbol:<{completed_col_width}}")
            if include_freelook:
                row.append(f"{flk_text:<{freelook_col_width}}")
            if include_inventory:
                row.append(f"{inv_text:<{inventory_col_width}}")

            # Apply colors
            colored_row = []
            colored_row.append(f"{GREEN}{row[0]}{RESET}")  # ACTION in GREEN
            colored_row.append(f"{YELLOW}{row[1]}{RESET}")  # OBSERVATION in YELLOW
            colored_row.append(f"{BLUE}{row[2]}{RESET}")   # SCORE in BLUE

            # Delta coloring
            if dlt_text:
                numeric_delta = float(dlt_text)
                if numeric_delta > 0:
                    delta_colored = f"{GREEN}{row[3]:<{delta_width}}{RESET}"
                elif numeric_delta == 0:
                    delta_colored = f"{RED}{row[3]:<{delta_width}}{RESET}"
                else:
                    delta_colored = f"{BLUE}{row[3]:<{delta_width}}{RESET}"
            else:
                delta_colored = f"{row[3]:<{delta_width}}"
            colored_row.append(delta_colored)

            # COMPLETED? column
            if raw_symbol.startswith("✓"):
                cmp_colored = f"{GREEN}{row[4]:<{completed_col_width}}{RESET}"
            elif raw_symbol.startswith("✗"):
                cmp_colored = f"{RED}{row[4]:<{completed_col_width}}{RESET}"
            else:
                cmp_colored = f"{row[4]:<{completed_col_width}}"

            colored_row.append(cmp_colored)

            # FREELOOK / INVENTORY in YELLOW
            if include_freelook:
                colored_row.append(f"{YELLOW}{row[5]:<{freelook_col_width}}{RESET}")
            if include_inventory:
                colored_row.append(f"{YELLOW}{row[6]:<{inventory_col_width}}{RESET}")

            # Assemble the final row string
            final_row = " | ".join(colored_row)
            print(final_row)

        # Separator after each path
        print("-" * total_width)
