import typing as t

from lark import Token


class EnglishNumberParser():
    """A class that can translate strings of common English words that
    describe a number into the number described
    """
    def __init__(self) -> None:
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

        tens = [
            "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty",
            "ninety"
        ]

        scales = ["hundred", "thousand", "million", "billion", "trillion"]

        fractions = ["third", "fourth", "fifth"
                     ] + [unit.strip("e") + "th" for unit in units]
        fractions[9] = "twelfth"

        fractions_tens = [unit.strip("y") + "ieth" for unit in tens]

        self.numwords: t.Dict[str, t.Tuple[t.Union[float, int], int]] = {}
        self.numwords["and"] = (1, 0)
        self.numwords["dozen"] = (12, 0)
        self.numwords["score"] = (20, 0)
        self.numwords["gross"] = (144, 0)
        for idx, word in enumerate(units):
            self.numwords[word] = (1, idx)
        for idx, word in enumerate(tens):
            idx += 2
            self.numwords[word] = (1, idx * 10)
        for idx, word in enumerate(scales):
            self.numwords[word] = (10**(idx * 3 or 2), 0)
        for idx, word in enumerate(fractions):
            idx += 3
            self.numwords[word] = (1.0 / idx, 0)
            self.numwords[word + "s"] = (1.0 / idx, 0)
        for idx, word in enumerate(fractions_tens):
            idx += 2
            self.numwords[word] = (1.0 / (idx * 10), 0)
            self.numwords[word + "s"] = (1.0 / (idx * 10), 0)

    def parse(self, tokens: t.List[Token]) -> float:
        current: t.Union[int, float] = 0
        result = current

        c = set(str(token).strip() for token in tokens)
        if len(c | set(("and", ""))) == 2:
            raise ValueError("No full numbers")
        if not tokens:
            raise ValueError("No input tokens")
        for token in tokens:
            if str(token).lower() not in self.numwords:
                raise ValueError("{} is not a number part".format(token))

            scale, increment = self.numwords[str(token).lower()]
            current = current * scale + increment
            if scale > 100:
                result += current
                current = 0
        return result + current
