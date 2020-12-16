from classifier import Classifier
import time
from flask import Flask, render_template, request
app = Flask(__name__)

print("Load classifier")
start_time = time.time()
classifier = Classifier()
print("Classifier is successfully loaded")
print(time.time() - start_time, "seconds")


@app.route("/", methods = ["POST", "GET"])
def index_page(text = "", prediction_message = ""):
    if request.method == "POST":
        text = request.form["text"]

        prediction_message = classifier.get_result_message(text)

    return render_template('simple_page.html', text = text, prediction_message = prediction_message)

if __name__ == "__main__":
    app.run(host = '127.0.0.1', port = 8080, debug = True)
