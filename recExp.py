
# Question 2
def integerDivision(n, x):
    if n - x <= 0:
        return 1
    else:
        n = n - x
        return 1 + integerDivision(n, x)

print integerDivision(36, 6)


#Question 3
def recExp(x, n):
    if n == 0:
        return 1
    else:
        n -= 1
        return x * recExp(x, n)

print recExp(2, 8)


# Question 4
# foo(25) = 4
