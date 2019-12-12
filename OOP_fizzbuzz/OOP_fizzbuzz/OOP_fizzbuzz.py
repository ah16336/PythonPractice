class FizzBuzz():
    def __init__(self, low, high):
        self.low=low
        self.high=high
    
    # finding fizzbuzz for a given value
    def find_one(self):
        if self.high % 5 == 0 and self.high % 3 == 0:
            print("FizzBuzz")
        elif self.high % 5 == 0:
            print("Buzz")
        elif self.high % 3 == 0:
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
