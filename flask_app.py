from flask import request, Flask , render_template
import pandas as pd
import matplotlib.pyplot as plt
from assignment_functions import compute_accuracy, load_pkl
import os
from flask import render_template

app = Flask(__name__)




@app.route('/')
def home():
    return render_template('deployment.html')
    
@app.route('/',  methods=['POST'])
def uploadFile():   
    if request.method == 'POST':
        f = request.files.get('file')
        df = pd.read_csv(f)
        X_test = df
        y_test = X_test['price']
        X_test = X_test.drop('price', axis = 1)
        xgb_model = load_pkl('models/XGBRegressor.pkl', app)
        model_name = type(xgb_model).__name__
        score = xgb_model.score(X_test,y_test)
        score = round(score, 2)
        y_pred = xgb_model.predict(X_test)
        accuracy = compute_accuracy(y_test, y_pred)
        accuracy = round(accuracy, 2)
        df_y_results = pd.DataFrame({'y_true': y_test, 'y_pred': y_pred.round(2)})
    
        plt.figure(figsize =(10, 8))
        plt.title("Plot")
        plt.xlabel('count of data')
        plt.ylabel('price')
        plt.plot(y_pred[:100], color = 'red',linewidth=0.75)
        plt.plot(y_test[:100], color = 'blue',linewidth=0.75)
        path = os.path.join(app.root_path, 'static/new_plot.png')
        plt.savefig(path)

    return render_template("deployment.html", score = score, accuracy = accuracy, tables=[df_y_results.to_html(classes='data')],
                           titles=df_y_results.columns.values, img_url = '/static/new_plot.png', model_name = model_name)
        
        
if __name__ == '__main__':
    app.run(debug=True)   