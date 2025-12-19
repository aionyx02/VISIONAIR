import pyautogui

class MouseController:
    def __init__(self, screen_width, screen_height):
        """
        初始化滑鼠控制器
        :param screen_width: 螢幕寬度 (像素，例如 1920)
        :param screen_height: 螢幕高度 (像素，例如 1080)
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        pyautogui.FAILSAFE = False

    def move_mouse(self, x, y):
        """
        移動滑鼠游標到指定座標
        :param x: 螢幕上的 X 座標
        :param y: 螢幕上的 Y 座標
        """
        pyautogui.moveTo(x, y)

    def left_click(self):
        """執行滑鼠左鍵點擊"""
        pyautogui.click()

    def double_click(self):
        """執行滑鼠左鍵雙擊"""
        pyautogui.doubleClick()

    def right_click(self):
        """執行滑鼠右鍵點擊"""
        pyautogui.click(button='right')

    def scroll(self, amount):
        """
        執行滑鼠滾輪滾動
        :param amount: 滾動量 (正數向上，負數向下)
        """
        pyautogui.scroll(amount)