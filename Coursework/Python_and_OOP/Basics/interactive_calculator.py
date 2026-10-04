"""Interactive Terminal Calculator.

Demonstrates modular arithmetic functions, Python 3.10+ match-case statement,
continuous loop handling, and robust exception management (ZeroDivisionError, ValueError).
"""

def add(a: float, b: float) -> float:
    return a + b

def subtract(a: float, b: float) -> float:
    return a - b

def multiply(a: float, b: float) -> float:
    return a * b

def divide(a: float, b: float) -> float:
    return a / b

def floor_divide(a: float, b: float) -> float:
    return a // b

def modulus(a: float, b: float) -> float:
    return a % b

def exponentiate(a: float, b: float) -> float:
    return a ** b


def main():
    print("=== Interactive Calculator ===")
    print("Type 'exit' at the first number prompt to quit.\n")

    while True:
        try:
            raw_input = input("Enter first number (or 'exit'): ").strip().lower()
            if raw_input == "exit":
                print("Shutting down. Goodbye!")
                break

            a = float(raw_input)
            operator = input("Operator (+, -, *, /, //, %, **): ").strip()
            b = float(input("Enter second number: "))

            match operator:
                case "+":
                    print(f"Result: {add(a, b)}")
                case "-":
                    print(f"Result: {subtract(a, b)}")
                case "*":
                    print(f"Result: {multiply(a, b)}")
                case "/":
                    print(f"Result: {divide(a, b)}")
                case "//":
                    print(f"Result: {floor_divide(a, b)}")
                case "%":
                    print(f"Result: {modulus(a, b)}")
                case "**":
                    print(f"Result: {exponentiate(a, b)}")
                case _:
                    print(f"Invalid operator: '{operator}'")

        except ValueError:
            print("Error: Please enter valid numeric values.")
        except ZeroDivisionError:
            print("Error: Cannot divide by zero.")
        print("-" * 35)


if __name__ == "__main__":
    main()
