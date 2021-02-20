import dash
import dash_core_components as dcc
import dash_html_components as html
import dash.dependencies  as dd
import plotly.express as px

import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img, load_img

from PIL import Image as pilImage
import io

from base64 import decodebytes

import datetime

import pandas as pd

import numpy as np

import os

from flask import Flask

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css', './assests/app.css']

server = Flask('mod4-dash')

app = dash.Dash(__name__, external_stylesheets=external_stylesheets, server=server)

#load in the classification model from the source files
modelPath = './final-model'  #os.Path('final-model')
model = keras.models.load_model(modelPath)

app.layout = html.Div(
  children=[
    html.Div(
      children = [
        html.H1(
          children = [
            '''
            Classify an Image
            '''],
          style={
            'width': '60%',
            'lineHeight': 'auto',
            'textAlign': 'center',
            'margin': '2.5% auto',
            'fontSize' : '3em',
          },
        ),
        html.P(
          children = [
            '''
            This is some template text. Go ahead and give whatever instructions you would like!
            '''],
          style={
            'width': '60%',
            'lineHeight': 'auto',
            'textAlign': 'center',
            'margin': 'auto auto 2.5% auto',
            'fontSize' : '1.2em',
          },
        ),

        # This is the upload widget that productionized the model and auto predicts the class of the image uploaded.  
        dcc.Upload(
          id='upload-image',
          children=[
            html.Div([
              'Drag and Drop or ',
              html.A('Select Files')
            ]),
            html.Br()
          ],
          style={
              'width': '20%',
              'height': '60px',
              'lineHeight': '60px',
              'borderWidth': '1px',
              'borderStyle': 'dashed',
              'borderRadius': '5px',
              'textAlign': 'center',
              'margin': 'auto',
              'font-size': '20px' 
          },
          # Allow multiple files to be uploaded
          multiple=True
        ),
      ]
    ),
    html.Div(id = 'prediction-output'),
    html.Div(id='output-image-upload'),
    dcc.Store(
      id = 'user-session',
    )
  ],
  className='app')

def parse_contents(content, filename):
  try:
    imageBytes = decodebytes(content.split(',')[1].encode('utf-8'))
    image = pilImage.open(io.BytesIO(imageBytes))
    image = image.convert('RGB')
    image = imageToDisplay = image.resize((256, 256), pilImage.NEAREST)
    image = img_to_array(image).reshape((1,256,256,3))

    print('fail 2')

    generator = ImageDataGenerator(
      rescale = 1./255)

    print('fail 5')
    pred = model.predict(image)
    label = np.where(model.predict(image) > .5, 'Pneumonia','Normal')
    print(pred)

    print('fail 6')
  except:
    print('The file image uploaded is not supported')
    preds = 'The file type you have uploaded is not supported for this model. Plese use: jpeg, png'

  return html.Div(
  children = [
    html.H4('File Name: '+filename),
    html.H5('The prediction for this image is: '+ str(label).replace('[', '').replace(']', '').replace("'", '')),
    html.H6('The calculated probability of having Pneunonia was: '+ str(pred).replace('[', '').replace(']', '').replace("'", '')),
    html.Hr(),
    html.Br(),

    # HTML images accept base64 encoded strings in the same format
    # that is supplied by the upload
    html.Img(src=imageToDisplay, id = filename),
    html.Hr(),],
  style={
        'width': '60%',
        'textAlign': 'center',
        'margin': 'auto'
    })

# callback to save the users image into the session as JSON
@app.callback(dd.Output('user-session', 'data'),
              dd.Output('output-image-upload', 'children'),
              dd.Input('upload-image', 'contents'),
              dd.State('upload-image', 'filename'))
def update_user_session(list_of_contents, list_of_names):
  # create an empty list to contin our dictonaries
  children = []

  # loop through the uploaded images and save the image to the users session in a dcc.Store
  children = []
  data = []
  if list_of_contents is not None:
    for content,name in zip(list_of_contents, list_of_names):

      # save each of the uploaded images and their file names into a dictonary (JSON)
      data.append({'content':content, 'name':name})
      children.append(parse_contents(content, name))

    return data, children
  else:
    return data, children

if __name__ == '__main__':
    app.run_server(debug=True)