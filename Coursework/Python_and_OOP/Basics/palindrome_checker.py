"""Palindrome Checker.

Demonstrates a two-pointer approach to determine whether a given string
or sequence reads the same backwards as forward.
"""

def is_palindrome(text: str) -> bool:
    """Check if the given text is a palindrome using two pointers."""
    i = 0
    j = len(text) - 1

    while i < j:
        if text[i] == text[j]:
            i += 1
            j -= 1
        else:
            return False
    return True


if __name__ == "__main__":
    user_input = input("Enter a string or number to check if it's a palindrome: ")
    if is_palindrome(user_input):
        print(f"'{user_input}' is a palindrome.")
    else:
        print(f"'{user_input}' is not a palindrome.")
