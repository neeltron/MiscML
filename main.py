from kivy.app import App
from kivy.uix.widget import Widget
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen


class MainWindow(Screen):
    pass


class SecondWindow(Screen):
    pass


class WindowManager(ScreenManager):
    pass


kv = Builder.load_file("MiscML.kv")


class MiscMLApp(App):
    def build(self): 
        return kv


if __name__ == "__main__":
    MiscMLApp().run()
