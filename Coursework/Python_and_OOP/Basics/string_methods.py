"""Common Python String Methods and Operations.

Strings in Python are immutable sequences of Unicode characters.
This script demonstrates standard string manipulation methods.
"""

sample_str = "Abu bakar!!!!"

print("--- Case Conversions ---")
print("lower():", sample_str.lower())
print("upper():", sample_str.upper())
print("title():", sample_str.title())
print("swapcase():", sample_str.swapcase())

print("\n--- Stripping Whitespace & Characters ---")
print("rstrip('!'):", sample_str.rstrip("!"))
print("rstrip():", sample_str.rstrip())

print("\n--- Search & Replace ---")
print("replace('bakar', 'Bakar'):", sample_str.replace("bakar", "Bakar"))
print("find('bakar'):", sample_str.find("bakar"))
print("endswith('!!!!'):", sample_str.endswith("!!!!"))

print("\n--- Splitting & Alignment ---")
print("split(' '):", sample_str.split(" "))
print("center(50):", sample_str.center(50))

print("\n--- Character Type Checks ---")
print("'abc12'.isalnum():", "abc12".isalnum())
print("'abc'.isalpha():", "abc".isalpha())
print("'   '.isspace():", "   ".isspace())
