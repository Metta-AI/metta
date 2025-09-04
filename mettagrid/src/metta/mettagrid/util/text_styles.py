from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

USE_COLORAMA_COLORS = True


def colorize(text, color):
    if not USE_COLORAMA_COLORS:
        return text
    return f"{color}{text}{Style.RESET_ALL}"


def red(text):
    return colorize(text, Fore.RED)


def green(text):
    return colorize(text, Fore.GREEN)


def yellow(text):
    return colorize(text, Fore.YELLOW)


def cyan(text):
    return colorize(text, Fore.CYAN)


def blue(text):
    return colorize(text, Fore.BLUE)


def bold(text):
    return colorize(text, Style.BRIGHT)


def magenta(text):
    return colorize(text, Fore.MAGENTA)


def use_colors(use_colors: bool):
    global USE_COLORAMA_COLORS
    USE_COLORAMA_COLORS = use_colors
