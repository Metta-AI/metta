import colorama

# Initialize colorama
colorama.init(autoreset=True)

USE_COLORAMA_COLORS = True


def colorize(text, color):
    if not USE_COLORAMA_COLORS:
        return text
    return f"{color}{text}{colorama.Style.RESET_ALL}"


def red(text):
    return colorize(text, colorama.Fore.RED)


def green(text):
    return colorize(text, colorama.Fore.GREEN)


def yellow(text):
    return colorize(text, colorama.Fore.YELLOW)


def cyan(text):
    return colorize(text, colorama.Fore.CYAN)


def blue(text):
    return colorize(text, colorama.Fore.BLUE)


def bold(text):
    return colorize(text, colorama.Style.BRIGHT)


def magenta(text):
    return colorize(text, colorama.Fore.MAGENTA)


def use_colors(use_colors: bool):
    global USE_COLORAMA_COLORS
    USE_COLORAMA_COLORS = use_colors
