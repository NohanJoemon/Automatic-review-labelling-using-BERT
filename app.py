from flask import Flask, render_template, request
import predictor
import os

app = Flask(__name__)

# set paths to upload folder
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
wpath = os.path.join(APP_ROOT,"static/weight.pth")

@app.route("/", methods=["GET","POST"])
def predict(wpath=wpath):
    pred=None
    selected=None
    if request.method  == "POST":
        text = str(request.form["usertext"])
        pred= predictor.predict(text,wpath)
        selected=text

    return render_template("index.html",pred=pred, selected=selected)

if __name__=="__main__":
    app.run(host='0.0.0.0',port=5000,debug=True)