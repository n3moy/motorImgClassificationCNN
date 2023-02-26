from flask import Flask

app = Flask(__name__)

from .modules.routes import mod


app.register_blueprint(mod)
