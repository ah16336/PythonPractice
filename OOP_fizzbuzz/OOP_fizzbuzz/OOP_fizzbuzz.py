class FizzBuzz():
    def __init__(self, low, high):
        self.low=low
        self.high=high

    def divisible_by(self, num, divisibleby):
        if num % divisibleby == 0:
            return True
        else:
            return False

    # finding fizzbuzz for a given value
    def find_one(self):
        five = self.divisible_by(self.high, 5)
        three = self.divisible_by(self.high, 3)
        if five and three:
            print("FizzBuzz")
        elif five:
            print("Buzz")
        elif three:
            print("Fizz")
        else:
            print(self.high)

    # finding fizzbuzz for values that fall in defined interval
    def find_all(self):
        for i in range(self.low, self.high + 1):
            ob = FizzBuzz(0, i)
            ob.find_one()

# creating objects
first = FizzBuzz(10, 30)
second = FizzBuzz(20, 55)
third = FizzBuzz(0, 42)

# testing first method
first.find_one()
second.find_one()
third.find_one()

print("\n")

# testing second method
third.find_all()

print("\n")

second.find_all()
