# solving problems from https://projecteuler.net/archives

# Problem 1
class EulerFizzBuzz():
    def __init__(self, num):
        self.num = num
    
    # divisible by method
    def divisible_by(self, numb, divisibleby):
        if numb % divisibleby == 0:
            return True
        else:
            return False

    # finding sum of natural numbers divisble by 5 and 3 up to given value
    def sum(self):
        s = 0
        for i in range(1, self.num):
            ob = EulerFizzBuzz(i)
            five = ob.divisible_by(i, 5)
            three = ob.divisible_by(i, 3)
            if five or three:
                s += i
        return s

validation = EulerFizzBuzz(10)
print(validation.sum()) # should equal 23

answer = EulerFizzBuzz(1000)
print(answer.sum())


# Problem 4
# finds the largest palindrome that is the multiple of 2 three-digit numbers.
class Palindrome():
    
    def find_products(self):
        j = 1
        for i in range(100, 1000):
            for j in range(100, 1000):
                num = i * j
                string_num = str(num)
                if string_num[0] == string_num[-1] and string_num[1] == string_num[-2] and string_num[2] == string_num[-3]:
                #if [string_num[x] == string_num[x * (-1) - 1] for x in range(3)]:
                    print([i, j, num])
        return 0;


# Testinf Palindrome class
test = Palindrome()
#print(test.find_products())


# Problem 5
class SmallMultiple():

    def test(self):
        x = 21
        i = 1
        while i < 21:
            if x % i == 0:
                i += 1
            else:
                x += 1
                i = 1
        return x


new = SmallMultiple()
#print(new.test()) # 232792560 (takes a long time to run)

# Problem 6
class SumSquareDiff():
    def __init__(self, num):
        self.num = num

    def find_diff(self):
        s = 0
        s_square = 0
        for i in range(1, self.num + 1):
            s += i
            s_square += i ** 2
        s_2 = s ** 2
        diff = s_2 - s_square
        return diff


validation_ssd = SumSquareDiff(10)
#print(validation_ssd.find_diff()) # should be 2640
ssd = SumSquareDiff(100)
#print(ssd.find_diff()) # 25164150 (doesn't take so long)

# Problem 8
class LargestProduct():
    
    def find(self):
        thousand = 7316717653133062491922511967442657474235534919493496983520312774506326239578318016984801869478851843858615607891129494954595017379583319528532088055111254069874715852386305071569329096329522744304355766896648950445244523161731856403098711121722383113622298934233803081353362766142828064444866452387493035890729629049156044077239071381051585930796086670172427121883998797908792274921901699720888093776657273330010533678812202354218097512545405947522435258490771167055601360483958644670632441572215539753697817977846174064955149290862569321978468622482839722413756570560574902614079729686524145351004748216637048440319989000889524345065854122758866688116427171479924442928230863465674813919123162824586178664583591245665294765456828489128831426076900422421902267105562632111110937054421750694165896040807198403850962455444362981230987879927244284909188845801561660979191338754992005240636899125607176060588611646710940507754100225698315520005593572972571636269561882670428252483600823257530420752963450
        str_thousand = str(thousand)
        print("Length:", len(str_thousand))
        s = 1
        max = 1
        position_zero = 0
        i = 0
        while i < 987:
            for j in range(i, i+13):
                s *= int(str_thousand[j])
            if s > max:
                max = s
                position_zero = i
            i += 1
            s = 1
        return [position_zero, max]

validation_lp = LargestProduct() # test with i + 4 in the method defintion
#print(validation_lp.find()) #  should return 5832 when using 4 spaces
# answer returns [197, 23514624000] when using 13 spaces.

