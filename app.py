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
                margin-top: 30px; 
                padding: 25px; 
                background: linear-gradient(135deg, #e7f3ff 0%, #f0f8ff 100%); 
                border-radius: 10px; 
                display: none;
                border-left: 5px solid #007bff;
            }
            .back-btn {
                display: inline-block;
                margin-top: 20px;
                padding: 10px 20px;
                background: #6c757d;
                color: white;
                text-decoration: none;
                border-radius: 5px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 توقع أداء الطالب</h1>
            <p>أدخل درجات الطالب في المواد الثلاث للحصول على توقع الأداء العام</p>
            
            <form id="predictForm">
                <div class="form-group">
                    <label>📐 درجة الرياضيات (0-100):</label>
                    <input type="number" id="math" min="0" max="100" required placeholder="مثال: 85">
                </div>
                
                <div class="form-group">
                    <label>📚 درجة القراءة (0-100):</label>
                    <input type="number" id="reading" min="0" max="100" required placeholder="مثال: 78">
                </div>
                
                <div class="form-group">
                    <label>✍️ درجة الكتابة (0-100):</label>
                    <input type="number" id="writing" min="0" max="100" required placeholder="مثال: 82">
                </div>
                
                <button type="submit">🔮 توقع الأداء</button>
            </form>
            
            <div id="result" class="result">
                <h3>🎯 نتيجة التوقع:</h3>
                <p id="prediction"></p>
            </div>
            
            <a href="/" class="back-btn">← العودة للرئيسية</a>
        </div>
        
        <script>
            document.getElementById('predictForm').onsubmit = function(e) {
                e.preventDefault();
                
                const math = document.getElementById('math').value;
                const reading = document.getElementById('reading').value;
                const writing = document.getElementById('writing').value;
                
                fetch('/api/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        math: parseFloat(math), 
                        reading: parseFloat(reading), 
                        writing: parseFloat(writing)
                    })
                })
                .then(response => response.json())
                .then(data => {
                    document.getElementById('prediction').innerHTML = 
                        `<strong>الدرجة المتوقعة:</strong> ${data.score.toFixed(1)}/100<br>
                         <strong>التقييم:</strong> ${data.grade}<br>
                         <strong>النصيحة:</strong> ${data.advice}`;
                    document.getElementById('result').style.display = 'block';
                })
                .catch(error => {
                    alert('حدث خطأ في التوقع. حاول مرة أخرى.');
                    console.error('Error:', error);
                });
            }
        </script>
    </body>
    </html>
    '''

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.json
        prediction = model.predict([[data['math'], data['reading'], data['writing']]])[0]
        
        if prediction >= 90: 
            grade = "ممتاز 🌟"
            advice = "أداء رائع! استمر في التفوق"
        elif prediction >= 80: 
            grade = "جيد جداً 👍"
            advice = "أداء جيد، يمكن تحسينه أكثر"
        elif prediction >= 70: 
            grade = "جيد ✅"
            advice = "تحتاج إلى مزيد من التركيز"
        elif prediction >= 60: 
            grade = "مقبول ⚠️"
            advice = "يحتاج إلى تحسين واضح"
        else: 
            grade = "يحتاج تحسين 📚"
            advice = "يحتاج إلى مساعدة إضافية ودعم"
        
        return jsonify({
            'score': prediction, 
            'grade': grade,
            'advice': advice
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/about')
def about():
    return '''
    <!DOCTYPE html>
    <html lang="ar" dir="rtl">
    <head>
        <meta charset="UTF-8">
        <title>حول النظام</title>
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
                text-align: center;
            }
            .feature {
                background: #e7f3ff;
                padding: 20px;
                margin: 15px 0;
                border-radius: 8px;
                text-align: right;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>ℹ️ حول النظام</h1>
            <p>نظام ذكي لتوقع أداء الطلاب باستخدام تعلم الآلة والذكاء الاصطناعي</p>
            
            <div class="feature">
                <h3>🤖 التقنية المستخدمة</h3>
                <p>يستخدم النظام نموذج Linear Regression لتحليل البيانات وتوقع الأداء</p>
            </div>
            
            <div class="feature">
                <h3>📊 كيف يعمل</h3>
                <p>يحلل النظام درجات الطالب في الرياضيات والقراءة والكتابة لتوقع الأداء العام</p>
            </div>
            
            <div class="feature">
                <h3>🎯 الهدف</h3>
                <p>مساعدة المعلمين وأولياء الأمور في تحديد نقاط القوة والضعف لدى الطلاب</p>
            </div>
            
            <a href="/" style="display: inline-block; margin-top: 20px; padding: 10px 20px; background: #007bff; color: white; text-decoration: none; border-radius: 5px;">العودة للرئيسية</a>
        </div>
    </body>
    </html>
    '''

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 تشغيل نظام توقع أداء الطلاب على المنفذ {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
