class FizzBuzz():
    def __init__(self, number):
        self.number=number
    
    # finding fizzbuzz for a given value
    def find_one(self):
        if self.number % 5 == 0 and self.number % 3 == 0:
            print("FizzBuzz")
        elif self.number % 5 == 0:
            print("Buzz")
        elif self.number % 3 == 0:
            print("Fizz")
        else:
            print(self.number)

    # finding fizzbuzz for values smaller than or equal to given value
    def find_all(self):
        for i in range(self.number + 1):
            ob = FizzBuzz(i)
            print(ob.find_one())

# creating objects
first = FizzBuzz(30)
second = FizzBuzz(55)
third = FizzBuzz(42)

# testing first method
first.find_one()
second.find_one()
third.find_one()

print("\n")

# testing second method
third.find_all()
