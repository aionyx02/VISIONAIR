# src/mouse_controller.py
import pyautogui

class MouseController:
    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        pyautogui.FAILSAFE = False

    def move_mouse(self, x, y):
        pyautogui.moveTo(x, y)

    def left_click(self):
        pyautogui.click()

    def double_click(self):
        pyautogui.doubleClick()

    def right_click(self):
        pyautogui.click(button='right')

    def scroll(self, amount):
        pyautogui.scroll(amount)
