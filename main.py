from kivy.app import App
from kivy.uix.widget import Widget


class MiscML(Widget):
    pass


class MiscMLApp(App):
    def build(self):
        return MiscML()


MiscMLApp().run()
