"""Question 3: Longest Substring Without Repeating Characters (Internship Assessment).

Finds the length of the longest contiguous substring without duplicate characters.
"""

def longest_unique_substring_length(s: str) -> int:
    """Calculate the length of the longest substring with all unique characters."""
    max_length = 0

    for i in range(len(s)):
        current = ""

        for j in range(i, len(s)):
            if s[j] in current:
                break
            current += s[j]

        max_length = max(max_length, len(current))

    return max_length


if __name__ == "__main__":
    test_cases = [
        ("abcdefggggas", 7),  # "abcdefg"
        ("ssssssssss", 1),    # "s"
        ("sdfscsdfgtg", 5),   # "csdfg" or "sdfgt"
    ]

    for string_val, expected in test_cases:
        result = longest_unique_substring_length(string_val)
        print(f"Input: '{string_val}' -> Longest Length: {result} (Expected: {expected})")
        assert result == expected
