from raylib import rl, colors


class FontRenderer:
    def __init__(self, ffi, font_path: str):
        self.ffi = ffi
        self.font_path = font_path
        self.fonts = {}
    
    def get(self, size: int):
        """
        Loads the font at the given size if it is not already loaded.
        """

        if size not in self.fonts:
            self.fonts[size] = rl.LoadFontEx(self.font_path.encode(), size, self.ffi.NULL, 0)
        return self.fonts[size]

    def render_text(self, text: str, x: int, y: int, size: int, color=colors.WHITE):
        font = self.get(size)
        rl.DrawTextEx(font, text.encode(), (x, y), size, 1, color)

    def measure_text(self, text: str, size: int):
        font = self.get(size)
        return rl.MeasureTextEx(font, text.encode(), size, 1)

    def render_text_right_aligned(self, text: str, x: int, y: int, size: int, color=colors.WHITE):
        text_size = self.measure_text(text, size)
        self.render_text(text, x - text_size.x, y, size, color)

    def unload(self):
        for font in self.fonts.values():
            rl.UnloadFont(font)

