# Drowsiness Detection System

This is a simple application that detects if a driver is sleepy and alerts them. The project uses [Convolution neural networks](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53) at its core and is trained to detect if the eye is open or closed.

## Technologies Used

- [Python 3.8](https://www.python.org/)
- [Flask](https://flask.palletsprojects.com/)
- [Keras](https://keras.io/)
- [Tensorflow](https://www.tensorflow.org/)
- [Numpy](https://numpy.org/)
- [Mediapipe](https://mediapipe.dev/)
- [OpenCV](https://opencv.org/)
- [Scipy](https://docs.scipy.org)
- A simple Laravel backend

## Setup

To install the project locally, clone the repository

```bash
$ git clone https://github.com/Iankumu/Drowsiness-Detection.git
$ cd Drowsiness-Detection
$ git submodule --update init

$ pip install -r requirements.txt
```

If you are having trouble with `opencv`,you can install the [headless version of opencv](https://pypi.org/project/opencv-python-headless/)

You can also run the Laravel Backend on a separate terminal. Before you do so, ensure you have [composer](https://getcomposer.org/) and php installed in your system. To run the backend just run the following commands.

```bash
$ cd DrowsinessAPI
$ cp.env.example .env
$ composer install
$ php artisan key:generate
$ php artisan migrate
$ php artisan passport:install
$ php artisan serve
```

Also ensure you have a database created(preferably mysql) as Laravel will try and connect to your database and create the migrations.

## Application Rundown

The Application mainly uses [Mediapipe](https://mediapipe.dev/) and [OpenCV](https://opencv.org/) to try and extract various landmarks on the face. It uses coordinates to located certain features. These features include the `face`,`eyes` and `eyebrows`.

It then passes the coordinates to a function that calculates the [eucleadian distance](https://en.wikipedia.org/wiki/Euclidean_distance) between the eyes and this is then used to determine if the person blinked.The blinks are stored in the database and will be used to provide a clear result when combined with other factors.

We also extract the eyes from the frames and pass to the model which tries to predict if the eyes are closed or open.

The last two factors measured are the [eye aspect ratio](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9044337/) and the [percentage of eye closed](https://iopscience.iop.org/article/10.1088/1742-6596/1090/1/012037/pdf) which are also stored in the database.

When all these factors are combined, we are able to come up with a conclusive decision as to whether the driver is sleepy or not.

If the driver is sleepy, an alarm is triggered and alerts the driver in form of sound.
