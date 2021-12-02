from flask import Flask,redirect,render_template,Response,request,session
from keras import utils
from camera import Video
import utils
from flask_session import Session


app = Flask(__name__)

app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register',methods=['GET','POST'])
def register():
    if request.method == "POST":
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm = request.form.get('confirm')
        response = utils.register(name,email,password,confirm)
        session['token'] = response['access_token']
        return redirect('/dashboard')
    else:
        return render_template('register.html')


@app.route('/login',methods=['POST','GET'])
def login():
    if request.method == "POST":
        email = request.form.get('email')
        password = request.form.get('password')
        response=utils.login(email,password)

        if response != None:
            session['token'] = response['access_token']
            return redirect('/dashboard')
        else:
            return redirect('/')
    else:
        return render_template('login.html')

@app.route('/camera')
def camera():
    token = session.get('token')
    if token == None:
        return redirect('/')
    else:
        return render_template('camera.html',token = token)

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def stop(camera):
    camera.__del__()

@app.route('/video')
def video():
    token = session.get('token')
    return Response(gen(Video(token)),
    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop',methods = ["GET"])
def stopCamera():
    token = session.get('token')
    stop(Video(token))
    return redirect('/dashboard')

@app.route('/dashboard')
def dashboard():
    token = session.get('token')
    if token == None:
        return redirect('/')
    else:
        signals = utils.signals(token)
        user = utils.profile(token)
        return render_template('dashboard.html',signals=signals,user=user)

@app.route('/profile',methods=['GET'])
def profile():
    token = session.get('token')
    if token == None:
        return redirect('/login')
    else:
        response = utils.profile(token)
        return render_template('profile.html',user = response)



@app.route('/logout',methods = ['GET'])
def logout():
    token = session.get('token')
    if token != None:
        utils.logout(token)
        session.clear()
        return redirect('/')
    else:
        return redirect('/login')


if __name__ == '__main__':
    app.run(debug=True)