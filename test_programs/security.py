import os

def read_file(filename):
    f = open(filename, 'r')
    data = f.readlines()
    return data

def execute_command(command):
    result = os.system(command)
    return result

def process_input(user_input):
    result = eval(user_input)
    return result

def main():
    filename = input("Enter filename: ")
    data = read_file(filename)
    
    command = input("Enter command: ")
    execute_command(command)
    
    expr = input("Enter expression: ")
    result = process_input(expr)
    print("Result:", result)

if __name__ == "__main__":
    main()