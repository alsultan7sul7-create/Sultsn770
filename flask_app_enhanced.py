# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, make_response
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import sqlite3
import hashlib
import os
import os
from datetime import datetime
import json
import io
import csv

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'student-performance-prediction-2026-enhanced')

# Global variables for the model
model = None
label_encoder = None
model_metrics = {}

def init_db():
    """إنشاء قاعدة البيانات وجداولها"""
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    # إنشاء جدول المستخدمين
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            full_name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # إنشاء جدول التنبؤات
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            hours_studied INTEGER,
            previous_scores INTEGER,
            extracurricular TEXT,
            sleep_hours INTEGER,
            sample_papers INTEGER,
            predicted_performance REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    conn.close()

def load_and_train_model():
    """تحميل وتدريب النموذج من dataset"""
    global model, label_encoder, model_metrics
    
    try:
        # البحث عن ملف البيانات
        possible_paths = [
            'extracted1/StudentPerformance.csv',
            'StudentPerformance.csv',
            'الذكاء2/extracted1/StudentPerformance.csv'
        ]
        
        dataset_path = None
        for path in possible_paths:
            if os.path.exists(path):
                dataset_path = path
                break
        
        if not dataset_path:
            raise FileNotFoundError("لم يتم العثور على ملف البيانات")
        
        print(f"تحميل البيانات من: {dataset_path}")
        df = pd.read_csv(dataset_path)
        
        # معالجة البيانات
        le = LabelEncoder()
        df['Extracurricular_Activities_Encoded'] = le.fit_transform(df['Extracurricular Activities'])
        
        features = ['Hours Studied', 'Previous Scores', 'Extracurricular_Activities_Encoded', 
                   'Sleep Hours', 'Sample Question Papers Practiced']
        target = 'Performance Index'
        
        X = df[features]
        y = df[target]
        
        # تقسيم البيانات
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # تدريب النموذج
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # حفظ label encoder
        label_encoder = le
        
        # حساب المقاييس
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        model_metrics = {
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'intercept': model.intercept_,
            'coefficients': model.coef_.tolist(),
            'feature_names': features,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'total_samples': len(df)
        }
        
        print(f"تم تدريب النموذج بنجاح!")
        print(f"دقة الاختبار (R²): {model_metrics['test_r2']:.4f}")
        print(f"خطأ RMSE: {model_metrics['test_rmse']:.4f}")
        print(f"عدد عينات التدريب: {model_metrics['training_samples']:,}")
        
        return True
        
    except Exception as e:
        print(f"خطأ في تحميل النموذج: {e}")
        return False

def hash_password(password):
    """تشفير كلمة المرور"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password, hashed):
    """التحقق من كلمة المرور"""
    return hash_password(password) == hashed

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return '''
    <html>
    <head><title>نظام توقع أداء الطلاب</title></head>
    <body style="font-family: Arial; text-align: center; padding: 50px;">
        <h1>🎓 نظام توقع أداء الطلاب</h1>
        <p>نظام ذكي لتوقع أداء الطلاب باستخدام الذكاء الاصطناعي</p>
        <a href="/dashboard" style="padding: 10px 20px; background: #007bff; color: white; text-decoration: none; border-radius: 5px;">لوحة التحكم</a>
    </body>
    </html>
    '''


@app.route('/login', methods=['GET', 'POST'])
def login():
    """تسجيل الدخول"""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute('SELECT id, password, full_name FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        conn.close()
        
        if user and verify_password(password, user[1]):
            session['user_id'] = user[0]
            session['username'] = username
            session['full_name'] = user[2]
            flash('تم تسجيل الدخول بنجاح!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('اسم المستخدم أو كلمة المرور غير صحيحة', 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """إنشاء حساب جديد"""
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        full_name = request.form['full_name']
        
        # التحقق من صحة البيانات
        if password != confirm_password:
            flash('كلمات المرور غير متطابقة', 'error')
            return render_template('register.html')
        
        if len(password) < 6:
            flash('كلمة المرور يجب أن تكون 6 أحرف على الأقل', 'error')
            return render_template('register.html')
        
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        
        try:
            hashed_password = hash_password(password)
            cursor.execute('''
                INSERT INTO users (username, email, password, full_name)
                VALUES (?, ?, ?, ?)
            ''', (username, email, hashed_password, full_name))
            conn.commit()
            
            # تسجيل الدخول التلقائي
            user_id = cursor.lastrowid
            session['user_id'] = user_id
            session['username'] = username
            session['full_name'] = full_name
            
            flash('تم إنشاء الحساب بنجاح!', 'success')
            return redirect(url_for('dashboard'))
            
        except sqlite3.IntegrityError:
            flash('اسم المستخدم أو البريد الإلكتروني مستخدم بالفعل', 'error')
        finally:
            conn.close()
    
    return render_template('register.html')

@app.route('/dashboard')
def dashboard():
    """لوحة التحكم الرئيسية"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # جلب آخر التنبؤات للمستخدم
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT hours_studied, previous_scores, extracurricular, sleep_hours, 
               sample_papers, predicted_performance, created_at
        FROM predictions 
        WHERE user_id = ? 
        ORDER BY created_at DESC 
        LIMIT 5
    ''', (session['user_id'],))
    recent_predictions = cursor.fetchall()
    conn.close()
    
    return render_template('dashboard.html', 
                         predictions=recent_predictions, 
                         model_metrics=model_metrics)

@app.route('/predict', methods=['POST'])
def predict():
    """توقع الأداء"""
    if 'user_id' not in session:
        return jsonify({'error': 'غير مسموح'}), 401
    
    if not model:
        return jsonify({'error': 'النموذج غير متاح'}), 500
    
    try:
        # استلام البيانات
        hours_studied = int(request.form['hours_studied'])
        previous_scores = int(request.form['previous_scores'])
        extracurricular = request.form['extracurricular']
        sleep_hours = int(request.form['sleep_hours'])
        sample_papers = int(request.form['sample_papers'])
        
        # تحويل الأنشطة اللامنهجية
        extracurricular_encoded = 1 if extracurricular == 'Yes' else 0
        
        # إعداد البيانات للتنبؤ
        input_data = np.array([[hours_studied, previous_scores, extracurricular_encoded, 
                               sleep_hours, sample_papers]])
        
        # التنبؤ
        prediction = model.predict(input_data)[0]
        prediction = max(0, min(100, prediction))  # تحديد النطاق بين 0-100
        
        # حفظ التنبؤ في قاعدة البيانات
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO predictions 
            (user_id, hours_studied, previous_scores, extracurricular, 
             sleep_hours, sample_papers, predicted_performance)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (session['user_id'], hours_studied, previous_scores, extracurricular,
              sleep_hours, sample_papers, prediction))
        conn.commit()
        conn.close()
        
        # تحديد مستوى الأداء
        if prediction >= 90:
            performance_level = "ممتاز"
            performance_color = "#28a745"
        elif prediction >= 80:
            performance_level = "جيد جداً"
            performance_color = "#17a2b8"
        elif prediction >= 70:
            performance_level = "جيد"
            performance_color = "#ffc107"
        elif prediction >= 60:
            performance_level = "مقبول"
            performance_color = "#fd7e14"
        else:
            performance_level = "ضعيف"
            performance_color = "#dc3545"
        
        # توليد التوصيات
        recommendations = generate_recommendations(hours_studied, previous_scores, 
                                                 extracurricular == 'Yes', sleep_hours, sample_papers)
        
        return jsonify({
            'success': True,
            'prediction': round(prediction, 2),
            'performance_level': performance_level,
            'performance_color': performance_color,
            'recommendations': recommendations
        })
        
    except Exception as e:
        return jsonify({'error': f'خطأ في التنبؤ: {str(e)}'}), 500

def generate_recommendations(hours_studied, previous_scores, has_extracurricular, sleep_hours, sample_papers):
    """توليد التوصيات الشخصية"""
    recommendations = []
    
    if hours_studied < 6:
        recommendations.append("زيادة ساعات الدراسة - كل ساعة إضافية تحسن الأداء بـ 2.85 نقطة")
    
    if previous_scores < 70:
        recommendations.append("التركيز على تحسين الدرجات الحالية - أساس مهم للأداء المستقبلي")
    
    if not has_extracurricular:
        recommendations.append("المشاركة في الأنشطة اللامنهجية - تحسن الأداء بـ 0.61 نقطة")
    
    if sleep_hours < 7:
        recommendations.append("زيادة ساعات النوم - كل ساعة إضافية تحسن الأداء بـ 0.48 نقطة")
    
    if sample_papers < 5:
        recommendations.append("ممارسة المزيد من أوراق الأسئلة النموذجية")
    
    if not recommendations:
        recommendations.append("أداؤك ممتاز! استمر على هذا المنوال")
    
    return recommendations

@app.route('/history')
def history():
    """سجل التنبؤات"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT hours_studied, previous_scores, extracurricular, sleep_hours, 
               sample_papers, predicted_performance, created_at
        FROM predictions 
        WHERE user_id = ? 
        ORDER BY created_at DESC
    ''', (session['user_id'],))
    all_predictions = cursor.fetchall()
    conn.close()
    
    return render_template('history.html', predictions=all_predictions)

@app.route('/model-info')
def model_info_page():
    """صفحة معلومات النموذج"""
    return render_template('model_info.html', model_metrics=model_metrics)

@app.route('/statistics')
def statistics():
    """صفحة الإحصائيات العامة"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # جلب الإحصائيات من قاعدة البيانات
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    # إجمالي المستخدمين
    cursor.execute('SELECT COUNT(*) FROM users')
    total_users = cursor.fetchone()[0]
    
    # إجمالي التنبؤات
    cursor.execute('SELECT COUNT(*) FROM predictions')
    total_predictions = cursor.fetchone()[0]
    
    # تنبؤات اليوم
    cursor.execute('SELECT COUNT(*) FROM predictions WHERE DATE(created_at) = DATE("now")')
    today_predictions = cursor.fetchone()[0]
    
    # متوسط الأداء
    cursor.execute('SELECT AVG(predicted_performance) FROM predictions')
    avg_performance = cursor.fetchone()[0] or 0
    
    # أفضل النتائج
    cursor.execute('SELECT predicted_performance FROM predictions ORDER BY predicted_performance DESC LIMIT 5')
    top_performances = [row[0] for row in cursor.fetchall()]
    
    # توزيع مستويات الأداء
    cursor.execute('''
        SELECT 
            SUM(CASE WHEN predicted_performance >= 90 THEN 1 ELSE 0 END) as excellent,
            SUM(CASE WHEN predicted_performance >= 80 AND predicted_performance < 90 THEN 1 ELSE 0 END) as very_good,
            SUM(CASE WHEN predicted_performance >= 70 AND predicted_performance < 80 THEN 1 ELSE 0 END) as good,
            SUM(CASE WHEN predicted_performance >= 60 AND predicted_performance < 70 THEN 1 ELSE 0 END) as acceptable,
            SUM(CASE WHEN predicted_performance < 60 THEN 1 ELSE 0 END) as poor
        FROM predictions
    ''')
    distribution = cursor.fetchone()
    performance_distribution = {
        'excellent': distribution[0] or 0,
        'very_good': distribution[1] or 0,
        'good': distribution[2] or 0,
        'acceptable': distribution[3] or 0,
        'poor': distribution[4] or 0
    }
    
    # اتجاهات الاستخدام (آخر 7 أيام)
    cursor.execute('''
        SELECT DATE(created_at) as date, COUNT(*) as count
        FROM predictions 
        WHERE created_at >= DATE('now', '-7 days')
        GROUP BY DATE(created_at)
        ORDER BY date
    ''')
    usage_data_raw = cursor.fetchall()
    
    # إعداد بيانات الرسم البياني
    from datetime import datetime, timedelta
    today = datetime.now()
    usage_labels = [(today - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(6, -1, -1)]
    usage_data = [0] * 7
    
    for date_str, count in usage_data_raw:
        if date_str in usage_labels:
            index = usage_labels.index(date_str)
            usage_data[index] = count
    
    conn.close()
    
    return render_template('statistics.html',
                         total_users=total_users,
                         total_predictions=total_predictions,
                         today_predictions=today_predictions,
                         avg_performance=avg_performance,
                         top_performances=top_performances,
                         performance_distribution=performance_distribution,
                         usage_labels=usage_labels,
                         usage_data=usage_data)

@app.route('/compare')
def compare():
    """صفحة مقارنة الأداء"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    # متوسط أداء المستخدم
    cursor.execute('SELECT AVG(predicted_performance) FROM predictions WHERE user_id = ?', (session['user_id'],))
    user_avg = cursor.fetchone()[0] or 0
    
    # أفضل نتيجة للمستخدم
    cursor.execute('SELECT MAX(predicted_performance) FROM predictions WHERE user_id = ?', (session['user_id'],))
    user_best = cursor.fetchone()[0] or 0
    
    # المتوسط العام
    cursor.execute('SELECT AVG(predicted_performance) FROM predictions')
    global_avg = cursor.fetchone()[0] or 0
    
    # ترتيب المستخدم
    cursor.execute('''
        SELECT COUNT(DISTINCT user_id) + 1 as rank
        FROM predictions p1
        WHERE (SELECT AVG(predicted_performance) FROM predictions p2 WHERE p2.user_id = p1.user_id) > ?
    ''', (user_avg,))
    user_rank = cursor.fetchone()[0]
    
    # إجمالي المستخدمين
    cursor.execute('SELECT COUNT(DISTINCT user_id) FROM predictions')
    total_users = cursor.fetchone()[0]
    
    conn.close()
    
    return render_template('compare.html',
                         user_avg=user_avg,
                         user_best=user_best,
                         global_avg=global_avg,
                         user_rank=user_rank,
                         total_users=total_users,
                         trend_labels=['الأسبوع 1', 'الأسبوع 2', 'الأسبوع 3', 'الأسبوع 4'],
                         user_trend=[user_avg-5, user_avg-2, user_avg+1, user_avg],
                         user_factors={'hours_studied': 6, 'previous_scores': 75, 'sleep_hours': 7, 'sample_papers': 4},
                         global_factors={'hours_studied': 5, 'previous_scores': 70, 'sleep_hours': 6, 'sample_papers': 3})

@app.route('/help')
def help_page():
    """صفحة المساعدة والأسئلة الشائعة"""
    return render_template('help.html')

@app.route('/export-data')
def export_data():
    """تصدير بيانات المستخدم"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    # جلب جميع تنبؤات المستخدم
    cursor.execute('''
        SELECT hours_studied, previous_scores, extracurricular, sleep_hours, 
               sample_papers, predicted_performance, created_at
        FROM predictions 
        WHERE user_id = ? 
        ORDER BY created_at DESC
    ''', (session['user_id'],))
    
    predictions = cursor.fetchall()
    conn.close()
    
    # إنشاء CSV
    output = io.StringIO()
    writer = csv.writer(output)
    
    # كتابة العناوين
    writer.writerow(['ساعات الدراسة', 'الدرجات السابقة', 'الأنشطة اللامنهجية', 
                     'ساعات النوم', 'أوراق الأسئلة', 'الأداء المتوقع', 'التاريخ'])
    
    # كتابة البيانات
    for prediction in predictions:
        writer.writerow(prediction)
    
    # إنشاء الاستجابة
    response = make_response(output.getvalue())
    response.headers['Content-Type'] = 'text/csv; charset=utf-8'
    response.headers['Content-Disposition'] = f'attachment; filename=predictions_{session["username"]}.csv'
    
    return response

@app.route('/api/stats')
def api_stats():
    """API للإحصائيات المباشرة"""
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    cursor.execute('SELECT COUNT(*) FROM users')
    total_users = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM predictions')
    total_predictions = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM predictions WHERE DATE(created_at) = DATE("now")')
    today_predictions = cursor.fetchone()[0]
    
    cursor.execute('SELECT AVG(predicted_performance) FROM predictions')
    avg_performance = cursor.fetchone()[0] or 0
    
    conn.close()
    
    return jsonify({
        'total_users': total_users,
        'total_predictions': total_predictions,
        'today_predictions': today_predictions,
        'avg_performance': avg_performance
    })

@app.route('/api/model-info')
def api_model_info():
    """معلومات النموذج API"""
    if not model:
        return jsonify({'error': 'النموذج غير متاح'}), 500
    
    return jsonify(model_metrics)

@app.route('/logout')
def logout():
    """تسجيل الخروج"""
    session.clear()
    flash('تم تسجيل الخروج بنجاح', 'info')
    return redirect(url_for('index'))

if __name__ == '__main__':
    # إنشاء قاعدة البيانات
    init_db()
    
    # تحميل وتدريب النموذج
    if load_and_train_model():
        print("النموذج جاهز للاستخدام!")
    else:
        print("تحذير: فشل في تحميل النموذج")
    
    # تحديد المنفذ للنشر
    port = int(os.environ.get('PORT', 5000))
    
    # تشغيل التطبيق
    app.run(
        host='0.0.0.0',
        port=port,
        debug=os.environ.get('FLASK_ENV') == 'development'

    )
