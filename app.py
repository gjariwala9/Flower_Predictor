from flask import Flask, render_template, session, url_for, redirect, flash
import numpy as np
from flask_wtf import FlaskForm
from wtforms import TextField, SubmitField
from wtforms.validators import DataRequired
from tensorflow.keras.models import load_model
import tensorflow as tf
import joblib
# physical_devices = tf.config.list_physical_devices('GPU') 
# tf.config.experimental.set_memory_growth(physical_devices[0], True)


def return_prediction(model, scaler, sample_json):
    s_len = sample_json['sepal_length']
    s_wid = sample_json['sepal_width']
    p_len = sample_json['petal_length']
    p_wid = sample_json['petal_width']
    
    flower = [[s_len, s_wid, p_len, p_wid]]
    
    flower = scaler.transform(flower)
    
    classes = np.array(['setosa', 'versicolor', 'virginica'])
    
    class_ind = model.predict_classes(flower)[0]
    
    return classes[class_ind]

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecretkey'

class FlowerForm(FlaskForm):
	sep_len = TextField('Sepal Length', validators=[DataRequired()])
	sep_wid = TextField('Sepal Width', validators=[DataRequired()])
	pet_len = TextField('Petal Length', validators=[DataRequired()])
	pet_wid = TextField('Petal Width', validators=[DataRequired()])

	submit = SubmitField("Predict")

@app.route("/",methods=['GET','POST'])
def index():

	form = FlowerForm()

	if form.validate_on_submit():
		try:

			session['sep_len'] = float(form.sep_len.data)
			session['sep_wid'] = float(form.sep_wid.data)
			session['pet_len'] = float(form.pet_len.data)
			session['pet_wid'] = float(form.pet_wid.data)
			return redirect(url_for("prediction"))

		except ValueError:
			flash('Invalid Input. Input should be a number.', 'danger')

	
	return render_template('home.html',form=form)



flower_model = load_model("final_model.h5")
flower_scaler = joblib.load('scaler.pkl')

@app.route('/prediction')
def prediction():
	content = {}
	try:
		content['sepal_length'] = float(session['sep_len'])
		content['sepal_width'] = float(session['sep_wid'])
		content['petal_length'] = float(session['pet_len'])
		content['petal_width'] = float(session['pet_wid'])

		results = return_prediction(flower_model, flower_scaler, content)
	except KeyError:
		flash('Give some input first.', 'danger')
		return redirect(url_for("index"))

	return render_template('prediction.html', results=results)


@app.errorhandler(404)
def error_404(error):
	return render_template('errors/404.html'), 404

@app.errorhandler(403)
def error_403(error):
	return render_template('errors/403.html'), 403

@app.errorhandler(500)
def error_500(error):
	return render_template('errors/500.html'), 500


if __name__=='__main__':
	app.run(threaded=True, port=5000)	


