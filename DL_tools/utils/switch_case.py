def success(msg):
    print(msg)

def debug(msg):
    print(msg)

def error(msg):
    print(msg)

def warning(msg):
    print(msg)

def other(msg):
    print(msg)

def notify_result(num, msg):
    numbers = {
        '0' : success,
        '1' : debug,
        2 : warning,
        3 : error
    }

    method = numbers.get(num, other)
    if method:
        method(msg)

if __name__ == "__main__":
    notify_result('0', "1")
    notify_result('1', "2")
    notify_result(2, "3")
    notify_result(3, "4")
    notify_result(4, "5")

    