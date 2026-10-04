"""Encapsulation with Property Getters and Setters.

Demonstrates Python's @property decorator to encapsulate private/protected
attributes with managed getter and setter methods.
"""

class Teacher:
    def __init__(self, name: str, dept: str, subject: str, salary: int | float | str):
        self.name = name
        self.dept = dept
        self.subject = subject
        self._salary = int(salary)

    def print_name(self):
        print(f"Teacher Name: {self.name} | Department: {self.dept} | Subject: {self.subject}")

    @property
    def salary(self) -> int:
        """Getter for the teacher's salary."""
        print("Fetching salary...")
        return self._salary

    @salary.setter
    def salary(self, value: int | float):
        """Setter to validate and update salary."""
        if value < 0:
            raise ValueError("Salary cannot be negative.")
        print(f"Updating salary to: {value}")
        self._salary = int(value)


if __name__ == "__main__":
    t1 = Teacher(name="Abubakar", dept="Computer Science", subject="OOP", salary=0)
    t1.print_name()

    # Set salary using property setter syntax
    t1.salary = 2500

    # Access salary using property getter
    print(f"Current Salary: {t1.salary}")
