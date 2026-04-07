import datetime

def log_with_timestamp(file, content):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    with open(file, "a") as f:
        f.write(f'[{timestamp}]: {content}\n')