from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os
import sqlite3
import hashlib
from datetime import datetime

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'student-ai-2026-default-key')

# إعداد قاعدة البيانات
def init_db():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            math_score REAL,
            reading_score REAL,
            writing_score REAL,
            predicted_score REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    conn.commit()
    conn.close()

# تحميل البيانات وتدريب النموذج
def load_and_train_model():
    try:
        # إنشاء بيانات تجريبية إذا لم يكن الملف موجود
        if not os.path.exists('StudentPerformance.csv'):
            # إنشاء بيانات تجريبية
            np.random.seed(42)
            n_samples = 1000
            
            data = {
                'math score': np.random.randint(0, 101, n_samples),
                'reading score': np.random.randint(0, 101, n_samples),
                'writing score': np.random.randint(0, 101, n_samples),
                'gender': np.random.choice(['male', 'female'], n_samples),
                'race/ethnicity': np.random.choice(['group A', 'group B', 'group C', 'group D', 'group E'], n_samples),
                'parental level of education': np.random.choice(['some high school', 'high school', 'some college', 'associate\'s degree', 'bachelor\'s degree', 'master\'s degree'], n_samples),
                'lunch': np.random.choice(['standard', 'free/reduced'], n_samples),
                'test preparation course': np.random.choice(['none', 'completed'], n_samples)
            }
            
            df = pd.DataFrame(data)
            df['total_score'] = (df['math score'] + df['reading score'] + df['writing score']) / 3
        else:
            df = pd.read_csv('StudentPerformance.csv')
            df['total_score'] = (df['math score'] + df['reading score'] + df['writing score']) / 3
        
        # تحضير البيانات للتدريب
        X = df[['math score', 'reading score', 'writing score']]
        y = df['total_score']
        
        # تقسيم البيانات
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # تدريب النموذج
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # تقييم النموذج
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"تم تدريب النموذج بنجاح!")
        print(f"R² Score: {r2:.3f}")
        print(f"MSE: {mse:.3f}")
        
        return model, df, {'r2': r2, 'mse': mse}
    
    except Exception as e:
        print(f"خطأ في تحميل البيانات: {e}")
        return None, None, None

# تشفير كلمة المرور
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# تحقق من كلمة المرور
def verify_password(password, hashed):
    return hash_password(password) == hashed

# تحميل النموذج عند بدء التطبيق
model, data, model_stats = load_and_train_model()

@app.route('/')
def index():
    """الصفحة الرئيسية"""
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    
    return '''
    <!DOCTYPE html>
    <html lang="ar" dir="rtl">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>نظام توقع أداء الطلاب</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                margin: 0;
                padding: 0;
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .container {
                background: white;
                padding: 40px;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                text-align: center;
                max-width: 500px;
                width: 90%;
            }
            h1 {
                color: #333;
                margin-bottom: 20px;
            }
            .btn {
                display: inline-block;
                padding: 12px 30px;
                margin: 10px;
                background: #667eea;
                color: white;
                text-decoration: none;
                border-radius: 25px;
                transition: all 0.3s;
            }
            .btn:hover {
                background: #764ba2;
                transform: translateY(-2px);
            }
            .features {
                margin: 30px 0;
                text-align: right;
            }
            .feature {
                margin: 10px 0;
                padding: 10px;
                background: #f8f9fa;
                border-radius: 8px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎓 نظام توقع أداء الطلاب</h1>
            <p>نظام ذكي لتوقع أداء الطلاب باستخدام الذكاء الاصطناعي</p>
            
            <div class="features">
                <div class="feature">📊 تحليل البيانات التعليمية</div>
                <div class="feature">🤖 توقعات دقيقة بالذكاء الاصطناعي</div>
                <div class="feature">📈 تقارير مفصلة وإحصائيات</div>
                <div class="feature">🎯 مساعدة في تحسين الأداء</div>
            </div>
            
            <a href="/dashboard" class="btn">لوحة التحكم</a>
            <a href="/login" class="btn">تسجيل الدخول</a>
            <a href="/register" class="btn">إنشاء حساب</a>
        </div>
    </body>
    </html>
    '''

@app.route('/dashboard')
def dashboard():
    """لوحة التحكم الرئيسية"""
    return '''
    <!DOCTYPE html>
    <html lang="ar" dir="rtl">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>لوحة التحكم - نظام توقع أداء الطلاب</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background: #f5f5f5;
            }
            .container {
                max-width: 800px;
                margin: 0 auto;
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            .form-group {
                margin: 15px 0;
            }
            label {
                display: block;
                margin-bottom: 5px;
                font-weight: bold;
            }
            input[type="number"] {
                width: 100%;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 5px;
                font-size: 16px;
            }
            .btn {
                background: #007bff;
                color: white;
                padding: 12px 30px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
            }
            .btn:hover {
                background: #0056b3;
            }
            .result {
                margin-top: 20px;
                padding: 20px;
                background: #e7f3ff;
                border-radius: 5px;
                display: none;
            }
            .nav {
                margin-bottom: 20px;
            }
            .nav a {
                margin-right: 15px;
                color: #007bff;
                text-decoration: none;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="nav">
                <a href="/">الرئيسية</a>
                <a href="/dashboard">لوحة التحكم</a>
                <a href="/statistics">الإحصائيات</a>
            </div>
            
            <h1>🎓 لوحة التحكم - توقع أداء الطلاب</h1>
            
            <form id="predictionForm">
                <div class="form-group">
                    <label for="math_score">درجة الرياضيات (0-100):</label>
                    <input type="number" id="math_score" name="math_score" min="0" max="100" required>
                </div>
                
                <div class="form-group">
                    <label for="reading_score">درجة القراءة (0-100):</label>
                    <input type="number" id="reading_score" name="reading_score" min="0" max="100" required>
                </div>
                
                <div class="form-group">
                    <label for="writing_score">درجة الكتابة (0-100):</label>
                    <input type="number" id="writing_score" name="writing_score" min="0" max="100" required>
                </div>
                
                <button type="submit" class="btn">توقع الأداء</button>
            </form>
            
            <div id="result" class="result">
                <h3>نتيجة التوقع:</h3>
                <p id="prediction-text"></p>
            </div>
        </div>
        
        <script>
            document.getElementById('predictionForm').addEventListener('submit', function(e) {
                e.preventDefault();
                
                const mathScore = document.getElementById('math_score').value;
                const readingScore = document.getElementById('reading_score').value;
                const writingScore = document.getElementById('writing_score').value;
                
                fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        math_score: parseFloat(mathScore),
                        reading_score: parseFloat(readingScore),
                        writing_score: parseFloat(writingScore)
                    })
                })
                .then(response => response.json())
                .then(data => {
                    document.getElementById('prediction-text').innerHTML = 
                        `الدرجة المتوقعة: <strong>${data.predicted_score.toFixed(2)}</strong><br>
                         مستوى الأداء: <strong>${data.performance_level}</strong><br>
                         التوصيات: ${data.recommendations}`;
                    document.getElementById('result').style.display = 'block';
                })
                .catch(error => {
                    console.error('Error:', error);
                    alert('حدث خطأ في التوقع');
                });
            });
        </script>
    </body>
    </html>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    """API للتوقع"""
    try:
        data = request.get_json()
        
        if not model:
            return jsonify({'error': 'النموذج غير متاح'}), 500
        
        # استخراج البيانات
        math_score = float(data['math_score'])
        reading_score = float(data['reading_score'])
        writing_score = float(data['writing_score'])
        
        # التحقق من صحة البيانات
        if not all(0 <= score <= 100 for score in [math_score, reading_score, writing_score]):
            return jsonify({'error': 'الدرجات يجب أن تكون بين 0 و 100'}), 400
        
        # التوقع
        prediction = model.predict([[math_score, reading_score, writing_score]])[0]
        
        # تحديد مستوى الأداء
        if prediction >= 90:
            performance_level = "ممتاز"
            recommendations = "استمر في الأداء الرائع!"
        elif prediction >= 80:
            performance_level = "جيد جداً"
            recommendations = "أداء جيد، يمكن تحسينه أكثر"
        elif prediction >= 70:
            performance_level = "جيد"
            recommendations = "تحتاج إلى مزيد من التركيز"
        elif prediction >= 60:
            performance_level = "مقبول"
            recommendations = "يحتاج إلى تحسين كبير"
        else:
            performance_level = "ضعيف"
            recommendations = "يحتاج إلى مساعدة إضافية"
        
        return jsonify({
            'predicted_score': prediction,
            'performance_level': performance_level,
            'recommendations': recommendations,
            'input_scores': {
                'math': math_score,
                'reading': reading_score,
                'writing': writing_score
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'خطأ في التوقع: {str(e)}'}), 500

@app.route('/statistics')
def statistics():
    """صفحة الإحصائيات"""
    if not model or data is None:
        return "البيانات غير متاحة"
    
    # حساب إحصائيات أساسية
    stats = {
        'total_students': len(data),
        'avg_math': data['math score'].mean(),
        'avg_reading': data['reading score'].mean(),
        'avg_writing': data['writing score'].mean(),
        'avg_total': data['total_score'].mean()
    }
    
    return f'''
    <!DOCTYPE html>
    <html lang="ar" dir="rtl">
    <head>
        <meta charset="UTF-8">
        <title>الإحصائيات</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .container {{ max-width: 800px; margin: 0 auto; }}
            .stat-card {{ 
                background: #f8f9fa; 
                padding: 20px; 
                margin: 10px 0; 
                border-radius: 8px; 
                border-left: 4px solid #007bff;
            }}
            .nav {{ margin-bottom: 20px; }}
            .nav a {{ margin-right: 15px; color: #007bff; text-decoration: none; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="nav">
                <a href="/">الرئيسية</a>
                <a href="/dashboard">لوحة التحكم</a>
                <a href="/statistics">الإحصائيات</a>
            </div>
            
            <h1>📊 إحصائيات النظام</h1>
            
            <div class="stat-card">
                <h3>📈 إحصائيات عامة</h3>
                <p><strong>عدد الطلاب:</strong> {stats['total_students']:,}</p>
                <p><strong>دقة النموذج (R²):</strong> {model_stats['r2']:.3f}</p>
            </div>
            
            <div class="stat-card">
                <h3>📊 متوسط الدرجات</h3>
                <p><strong>الرياضيات:</strong> {stats['avg_math']:.1f}</p>
                <p><strong>القراءة:</strong> {stats['avg_reading']:.1f}</p>
                <p><strong>الكتابة:</strong> {stats['avg_writing']:.1f}</p>
                <p><strong>المجموع:</strong> {stats['avg_total']:.1f}</p>
            </div>
            
            <div class="stat-card">
                <h3>🎯 معلومات النموذج</h3>
                <p><strong>نوع النموذج:</strong> Linear Regression</p>
                <p><strong>حالة التدريب:</strong> مكتمل ✅</p>
                <p><strong>آخر تحديث:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
            </div>
        </div>
    </body>
    </html>
    '''

@app.route('/login', methods=['GET', 'POST'])
def login():
    """تسجيل الدخول"""
    if request.method == 'POST':
        # معالجة تسجيل الدخول
        return redirect(url_for('dashboard'))
    
    return '''
    <html>
    <head><title>تسجيل الدخول</title></head>
    <body style="font-family: Arial; padding: 50px; text-align: center;">
        <h2>تسجيل الدخول</h2>
        <form method="post">
            <p><input type="text" name="username" placeholder="اسم المستخدم" required></p>
            <p><input type="password" name="password" placeholder="كلمة المرور" required></p>
            <p><button type="submit">دخول</button></p>
        </form>
        <a href="/">العودة للرئيسية</a>
    </body>
    </html>
    '''

@app.route('/register', methods=['GET', 'POST'])
def register():
    """إنشاء حساب جديد"""
    if request.method == 'POST':
        # معالجة التسجيل
        return redirect(url_for('login'))
    
    return '''
    <html>
    <head><title>إنشاء حساب</title></head>
    <body style="font-family: Arial; padding: 50px; text-align: center;">
        <h2>إنشاء حساب جديد</h2>
        <form method="post">
            <p><input type="text" name="username" placeholder="اسم المستخدم" required></p>
            <p><input type="password" name="password" placeholder="كلمة المرور" required></p>
            <p><button type="submit">إنشاء حساب</button></p>
        </form>
        <a href="/">العودة للرئيسية</a>
    </body>
    </html>
    '''

if __name__ == '__main__':
    # إنشاء قاعدة البيانات
    init_db()
    
    # تحديد المنفذ للنشر
    port = int(os.environ.get('PORT', 5000))
    
    print("🚀 بدء تشغيل نظام توقع أداء الطلاب...")
    print(f"📊 حالة النموذج: {'جاهز ✅' if model else 'غير متاح ❌'}")
    print(f"🌐 المنفذ: {port}")
    
    # تشغيل التطبيق
    app.run(
        host='0.0.0.0',
        port=port,
        debug=os.environ.get('FLASK_ENV') == 'development'
    )