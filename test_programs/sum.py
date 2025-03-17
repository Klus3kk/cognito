def calculate_sum(numbers):
    total = 0
    for num in numbers:
        total = total + num
    return total

def find_max(numbers):
    if len(numbers) == 0:
        return None
    max_val = numbers[0]
    for i in range(len(numbers)):
        if numbers[i] > max_val:
            max_val = numbers[i]
    return max_val

if __name__ == "__main__":
    data = [1, 5, 7, 10, 13, 20]
    print("Sum:", calculate_sum(data))
    print("Max:", find_max(data))