def validate_input(string):
    if not string.isnumeric():
        print("Please enter a valid whole number. Try again.")
    return string.isnumeric()


def fizzbuzz(number):
    if number % 15 == 0:
        return "fizzbuzz"
    if number % 5 == 0:
        return "fizz"
    if number % 3 == 0:
        return "buzz"



user_input = input("Please enter number: ")
while not validate_input(user_input):
    user_input = input("Please enter number: ")

number = int(user_input)
print(fizzbuzz(number))


