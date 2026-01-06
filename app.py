from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os

app = Flask(__name__)

# تدريب نموذج بسيط
def create_model():
    np.random.seed(42)
    n = 1000
    math_scores = np.random.randint(0, 101, n)
    reading_scores = np.random.randint(0, 101, n)
    writing_scores = np.random.randint(0, 101, n)
    
    X = np.column_stack([math_scores, reading_scores, writing_scores])
    y = (math_scores + reading_scores + writing_scores) / 3
    
    model = LinearRegression()
    model.fit(X, y)
    return model

model = create_model()

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html lang="ar" dir="rtl">
    <head>
        <meta charset="UTF-8">
        <title>نظام توقع أداء الطلاب</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                margin: 0; 
                padding: 20px; 
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .container { 
                max-width: 600px; 
                background: white; 
                padding: 40px; 
                border-radius: 15px; 
                text-align: center;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            }
            .btn { 
                display: inline-block; 
                padding: 15px 30px; 
                margin: 10px; 
                background: #007bff; 
                color: white; 
                text-decoration: none; 
                border-radius: 25px;
                transition: all 0.3s;
            }
            .btn:hover { 
                background: #0056b3; 
                transform: translateY(-2px);
            }
            .feature {
                background: #f8f9fa;
                padding: 15px;
                margin: 10px 0;
                border-radius: 8px;
                text-align: right;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎓 نظام توقع أداء الطلاب</h1>
            <p>نظام ذكي لتوقع أداء الطلاب باستخدام الذكاء الاصطناعي</p>
            
            <div class="feature">📊 تحليل البيانات التعليمية</div>
            <div class="feature">🤖 توقعات دقيقة بالذكاء الاصطناعي</div>
            <div class="feature">📈 تقارير مفصلة وإحصائيات</div>
            
            <a href="/predict" class="btn">🚀 بدء التوقع</a>
            <a href="/about" class="btn">ℹ️ حول النظام</a>
        </div>
    </body>
    </html>
    '''

@app.route('/predict')
def predict_page():
    return '''
    <!DOCTYPE html>
    <html lang="ar" dir="rtl">
    <head>
        <meta charset="UTF-8">
        <title>توقع الأداء</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                background: #f0f2f5; 
                margin: 0; 
                padding: 20px; 
            }
            .container { 
                max-width: 600px; 
                margin: 0 auto; 
                background: white; 
                padding: 40px; 
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            .form-group {
                margin: 20px 0;
            }
            label {
                display: block;
                margin-bottom: 8px;
                font-weight: bold;
                color: #333;
            }
            input { 
                width: 100%; 
                padding: 12px; 
                border: 2px solid #ddd; 
                border-radius: 8px;
                font-size: 16px;
                box-sizing: border-box;
            }
            input:focus {
                border-color: #007bff;
                outline: none;
            }
            button { 
                width: 100%; 
                padding: 15px; 
                background: #28a745; 
                color: white; 
                border: none; 
                border-radius: 8px; 
                cursor: pointer;
                font-size: 18px;
                font-weight: bold;
            }
            button:hover {
                background: #218838;
            }
            .result {
