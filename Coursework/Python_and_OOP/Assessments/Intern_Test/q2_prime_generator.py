"""Question 2: Prime Number Generator (Internship Assessment).

Generates all prime numbers up to n using trial division up to sqrt(num).
Time complexity per number: O(sqrt(num)).
"""

def generate_primes(n: int) -> list[int]:
    """Return a list of prime numbers strictly less than n."""
    primes = []

    if n <= 2:
        return primes

    for num in range(2, n):
        is_prime = True

        for i in range(2, int(num ** 0.5) + 1):
            if num % i == 0:
                is_prime = False
                break

        if is_prime:
            primes.append(num)

    return primes


if __name__ == "__main__":
    limit = 15
    primes = generate_primes(limit)
    print(f"Primes up to {limit}: {primes}")

    # Additional test verification
    assert primes == [2, 3, 5, 7, 11, 13]
