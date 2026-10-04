"""The Diamond Problem & Method Resolution Order (MRO).

Demonstrates multiple inheritance in Python, cooperative calls using super(),
and how Python resolves method calls using the C3 Linearization algorithm.
"""

class A:
    def process(self):
        print("A process")


class B(A):
    def process(self):
        print("B process")
        super().process()


class C(A):
    def process(self):
        print("C process")
        super().process()


class D(B, C):
    def process(self):
        print("D process")
        super().process()


if __name__ == "__main__":
    print("--- Calling D().process() ---")
    d = D()
    d.process()

    print("\n--- Method Resolution Order (MRO) for D ---")
    for idx, cls in enumerate(D.mro(), 1):
        print(f"{idx}. {cls.__name__}")
