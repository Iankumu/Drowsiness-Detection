from flask import Flask,redirect,render_template,Response,request,session,make_response
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

@app.route('/login',methods=['POST','GET'])
def login():
    if request.method == "POST":
        email = request.form.get('email')
        password = request.form.get('password')
        response=utils.login(email,password)
        session['token'] = response['access_token']
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

@app.route('/video')
def video():
    return Response(gen(Video()),
    mimetype='multipart/x-mixed-replace; boundary=frame')

# @login_required
@app.route('/logout',methods = ['GET'])
def logout():
    token = session.get('token')
    if token != None:
        utils.logout(token)
        session.clear()
        return redirect('/')
    else:
        return redirect('/login')

if  __name__ == '__main__':
    app.run(debug=True)