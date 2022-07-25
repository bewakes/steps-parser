
colors = {
    'red': '\u001b[31m',
    'yellow': '\u001b[33m',
    'green': '\u001b[32m',
    'RESET': '\u001b[0m',
}

class Logger:
    @staticmethod
    def log(*args, **kwargs):
        print(colors['green'], end='')
        print(*args, **kwargs)
        print(colors['RESET'], end='')

    @staticmethod
    def warn(*args, **kwargs):
        print(colors['yellow'], end='')
        print(*args, **kwargs)
        print(colors['RESET'], end='')

    @staticmethod
    def err(*args, **kwargs):
        print(colors['red'], end='')
        print(*args, **kwargs)
        print(colors['RESET'], end='')
