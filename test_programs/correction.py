def open_and_read_file(file_path):
    file = open(file_path, 'r')
    content = file.read()
    return content

def calculate_average(numbers):
    total = 0
    count = 0
    for num in numbers:
        total = total + num
        count = count + 1
    return total / count

def repeat_string(s, n):
    result = ""
    for i in range(n):
        result = result + s
    return result

if __name__ == "__main__":
    try:
        content = open_and_read_file("sample.txt")
        print(content)
        
        avg = calculate_average([1, 2, 3, 4, 5])
        print(f"Average: {avg}")
        
        repeated = repeat_string("Hello", 3)
        print(repeated)
    except Exception as e:
        print(f"Error: {e}")