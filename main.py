import kivy
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.textinput import TextInput
from kivy.uix.relativelayout import RelativeLayout 


flag = 0

class textinp(Screen): 
    pass


class MainWindow(Screen):
    pass


class SecondWindow(Screen):
    textinp()
    

class WindowManager(ScreenManager):
    pass


kv = Builder.load_file("MiscML.kv")

class MiscMLApp(App):
    def build(self): 
        return textinp()
    
  
    # Arranging that what you write will be shown to you 
    # in IDLE 
    def process(self): 
        text = self.root.ids.input.text 
        print(text) 


if __name__ == "__main__":
    MiscMLApp().run()
