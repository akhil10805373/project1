from flask import render_template

def privacy():
    return render_template('privacy.html')

def index():
    return render_template('index.html')
