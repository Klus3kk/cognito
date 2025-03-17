def fibonacci(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)

def process_matrix(matrix):
    result = []
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            for k in range(len(matrix)):
                if matrix[i][j] > 0:
                    result.append(matrix[i][j] * k)
    return result

def build_string(items):
    result = ""
    for item in items:
        result = result + str(item) + ","
    return result

if __name__ == "__main__":
    n = 10
    print(f"Fibonacci({n}) = {fibonacci(n)}")
    
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    process_matrix(matrix)
    
    items = list(range(100))
    build_string(items)