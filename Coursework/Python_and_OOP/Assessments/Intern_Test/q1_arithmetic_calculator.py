"""Question 1: Arithmetic Calculator (Internship Assessment).

A CLI calculator that performs addition, subtraction, multiplication,
and division with user inputs and error handling.
"""

def add(x: float, y: float) -> float:
    return x + y


def sub(x: float, y: float) -> float:
    return x - y


def mul(x: float, y: float) -> float:
    return x * y


def div(x: float, y: float) -> float | str:
    if y == 0:
        return "Error: Division by zero"
    return x / y


def main():
    print("Calculator")
    print("Operators: +  -  *  /")

    try:
        a = float(input("Enter first number: "))
        op = input("Enter operator (+, -, *, /): ").strip()
        b = float(input("Enter second number: "))

        if op == "+":
            print("Result:", add(a, b))
        elif op == "-":
            print("Result:", sub(a, b))
        elif op == "*":
            print("Result:", mul(a, b))
        elif op == "/":
            print("Result:", div(a, b))
        else:
            print("Invalid operator")

    except ValueError:
        print("Invalid input. Please enter numbers only.")


if __name__ == "__main__":
    main()
