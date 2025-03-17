from __future__ import print_function

# Interactive visualization command in Jupyter Lab
# !pip install ipywidgets

from io import BytesIO
from PIL import Image

import cv2
import os

# from ipywidgets import interact, interactive, fixed, interact_manual, FloatSlider
# import ipywidgets as widgets

import numpy as np              # import NumPy library

import numbers

# Plot the function:
import matplotlib.pyplot as plt

# import math # for floor and ceiling
import sympy as sp
x = sp.symbols('x')

import pygame
import sys
import time

# Import plotly
# !pip install kaleido

# import plotly
import plotly.graph_objects as go
import plotly.express as px

# Data frames using pandas
import pandas as pd

from IPython.display import HTML
from base64 import b64encode

# Import functions and library
import moviepy.editor
from moviepy.editor import VideoFileClip, vfx

class table():
  def __init__(self, 
               x = None, y = None, 
               column_labels=None,
               img_path = None, img_loc = None, 
               img_speed = None):
    
    self.fig = go.Figure()

    # Input arugments validation      
    if x is not None and y is not None:
      if len(x) != len(y):
        raise ValueError('Number of columns should be equal to the number of column labels')
        
      if not isinstance(x, list) or not isinstance(y, list):
        raise ValueError('Table inputs should be formatted as lists')
      
      for column_label in column_labels:
        if not isinstance(column_label, str):
          raise ValueError('Column labels should be strings')
      
      for data_values in x:
        if not isinstance(data_values, numbers.Number):
          raise ValueError('Data values should be numbers')
      
      for data_values in y:
        if not isinstance(data_values, numbers.Number):
          raise ValueError('Data values should be numbers')
        
      self.column_labels = column_labels
      self.data_values   = [x, y]
    else:
      self.column_labels = []
      self.data_values = []
    
    if img_path is not None:
      if not isinstance(img_path, str):
        raise ValueError('path to image should be string')
      
      self.img_path = img_path
    else:
      self.img_path = []
    
    if img_loc is not None:
      if not isinstance(img_loc[0], numbers.Number) and  not isinstance(img_loc[1], numbers.Number):
        raise ValueError('Image location should be a number')
      
      if img_loc[0] > 0 and img_loc[1] > 0:
        raise ValueError('Image location should be non negative')
      
      self.img_loc = img_loc
    else:
      self.img_loc = (0, 0)
    
    if img_speed is not None:
      if img_speed < 1 or not isinstance(x, numbers.Number):
        raise ValueError('Speed should be number more than 1')
      
      self.img_speed = img_speed
    else:
      self.img_speed = 0
  
  def showTable(self, fun, domain, img_name=None):
    """ creates a Table from a function and its domain, and it saves it to img_name.
    """
    # Generate the x-values and y-values
    x = sp.symbols('x')
    y_values = []
    x_values = np.linspace(*domain) # Unpack the list of elements.
    for val in x_values:
      y_values.append(float(fun.subs(x, val)))

    # Create the labels:
    self.column_labels = ["x", "y="+str(fun)]
    
    # Data
    self.data_values = [x_values.tolist(), y_values]
    
    # Plot the table
    self.plotTable()

    # Save the image
    if (img_name is not None):
      self.saveImage(img_name)


  def saveImage(self, filename):
      """ saveImage(filename) saves filename.ext where ext=png, jpeg, svg, pdf
      """
      self.fig.write_image(filename, engine="kaleido", scale=1.0)
      print("Wrote ", filename)
      
  def plotTable(self):
      """ plots a table from given columns
      """
      # Clear figure
      self.fig.data = []

      # Clear layout
      self.fig.layout = {}

      # Add the table plot
      print(self.column_labels)
      print(self.data_values)

      print(len(self.column_labels)) 
      print(len(self.data_values))

      # Generate the format string.
      format_vals = []
      for vals in self.data_values:
        if isinstance(vals[0], str):
          format_vals.append("text")
        else:
          format_vals.append(".5f")
      print(format_vals)

      self.fig.add_trace(go.Table(header=dict(values = self.column_labels), 
                                  cells = dict(values = self.data_values,
                                               format=format_vals)))
      
      # Update the layout
      self.fig.update_layout(autosize = False)

      # Update the figure:
      self.fig.show()
  
def plotTablesLines(tables = None, 
                    fig_title = None, x_label = None, y_label = None, 
                    legend_title = None, legend_labels = None,
                    equation_labels = None,
                    img_name = None):
    """ plots a table as plot from given columns
        Optional:
          Can setup the step, minX, maxX, minY, maxY
    """
  
    if fig_title is not None:
      if not isinstance(fig_title, str):
        raise ValueError('Figure title should be a string')
    
    if x_label is not None:
      if not isinstance(x_label, str):
        raise ValueError('X label should be a string')
    
    if y_label is not None:
      if not isinstance(y_label, str):
        raise ValueError('Y label should be a string')
    
    if legend_title is not None:
      if not isinstance(legend_title, str):
        raise ValueError('Legened label should be a string')
    
    if legend_labels is not None:
      for equation in legend_labels:
        if not isinstance(equation, str):
          raise ValueError('Equations should be a string')
      
      if len(legend_labels) != len(tables):
        raise ValueError('Numbers of labels must be equal to number of tables')
    
    # Clear figure data
    tmp = table()
    tmp.fig.data = []
    
    
    # Overlay multiple plots in a figure
    if tables is not None:

      for tbl_idx, tbl in enumerate(tables):
        
        # Line equations
        if legend_labels is not None:
          tbl_name = legend_labels[tbl_idx]
        else:
          tbl_name = []
        
        # Adding plots onto existing figure
        point_size = 10
        line_width = 3
        tmp.fig.add_trace( go.Scatter(x = tbl.data_values[0], 
                                          y = tbl.data_values[1], 
                                          mode='lines+markers',
                                          marker=dict(
                                            size=point_size,     # Point size
                                            line=dict(width=2)), # Width of the border around markers
                                          line=dict(
                                            width=line_width),   # Line width
                                      name = tbl_name))
      
      # Update the figure layout with titles
      tmp.fig.update_layout(title=
                                {'text': fig_title,
                                    'y':0.9,
                                    'x':0.4,
                                    'xanchor': 'center',
                                    'yanchor': 'top'
                                },
                                xaxis_title= x_label,
                                yaxis_title= y_label,
                                legend_title= legend_title,
                                font=
                                dict(
                                  family="Courier New, monospace",
                                  size=13,
                                  color="RebeccaPurple"
                                  )
                                )
      
    # Update the layout
    tmp.fig.update_layout(autosize = False)

    # Update the figure
    tmp.fig.show()

    # Verify data type and save if possible:
    if img_name is not None:
      if not isinstance(img_name, str):
          raise ValueError('Equations should be a string')

      # Convert the image to OpenCV format:
      fig_bytes = tmp.fig.to_image(format="png")
      buf = BytesIO(fig_bytes)
      img = Image.open(buf)
      img_rgb = np.asarray(img)
      bgr_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
      cv2.imwrite(img_name, bgr_img) 
      # tmp.saveImage(img_name)


def check_lists(names, images, speeds):
  """ Dynamic data types checking.
  """
  # print("names=", names)
  # print("images=", images)
  # print("speeds=", speeds)
  if not isinstance(names, (list, tuple, np.ndarray)):  # Check if it's an array-like object
    raise ValueError("names should be an array of strings!")
  
  if not all(isinstance(item, str) for item in names): 
    raise ValueError("names should be an array of strings!")

  if not isinstance(images, (list, tuple, np.ndarray)):  # Check if it's an array-like object
    raise ValueError("images should be an array of filenames!")
  
  if not all(isinstance(item, str) for item in images): 
    raise ValueError("images should be an array of filenames!")

  if not isinstance(speeds, (list, tuple, np.ndarray)):  # Check if it's an array-like object
    raise ValueError("speeds should be an array of numbers!")
  
  if not all(isinstance(item, (int, float)) for item in speeds): 
    raise ValueError("speeds should be an array of numbers!")
  
  if (len(names) != len(images) or
      len(names) != len(speeds)
      ):
    print(names, images, speeds)
    print("They are not the same length")
    raise ValueError("All lists must be of the same length!")

  if not names:
        raise ValueError("names list must not be empty!")


class simulationVideo:
  """
  The race class is used for preparing a race video simulation.
  The users can specify characters for the race. 
  Each character has its own race lane and image.

  Refer to GraphSpeeds_lesson3.ipynb for usage.

  Game characters (required)
  --------------------------
  data: This is a list of names, images, speeds.

  target_width:  the number of pixels for each character. Default=100 pixels.


  Simulation parameters (required)
  --------------------------------
  duration:         This is the amount of time in seconds to simulate.
  race_distance:    This is the physical distance being simulated.
                    It can be in meters, feet, or any unit of distance.
  vid_title:        The name for the video. Default="Race".
   
  
  Units (optional)
  ----------------
  distance_string: string describing distance (e.g., "m", "miles")
  time_string:     string describing time (e.g., "hours", "seconds")
  speed_string:    speed string (e.g., "miles/hour")

  Video parameters (optional)
  ---------------------------
  video_name: the file to prepare.
  fps:        frames per second.
  vid_width:  The number of columns in the video.
  vid_height: The number of rows in the video.
  max_frames: The maximum number of frames allowed for the video.
  target_width:     the target width for the downloaded character images.
  simulation_speed: simulation_speed gives the skip number of frames.
                    Default=1.0. If simulation_speed=2, the frame rate is halved.
  """

  def __init__(self, data, duration, race_distance, vid_title):
    """ Setup the simulation basic parameters.
    """
    # Set the target_size according to image size in the future!
    self.names  = data[0]
    self.images = data[1]
    self.speeds = data[2]
    check_lists(self.names, self.images, self.speeds)

    # Construct the location array: 2 elements each one!!!
    target_size = 50
    self.locs = [(0, target_size)]
    for i in range(len(self.names)-1):
      loc = (0, (i+1)*3*target_size)
      self.locs.append(loc)


    self.duration = duration 
    self.race_distance = race_distance
    self.vid_title     = vid_title 

    # Call functions to set default values:
    self.set_units()
    self.set_video()


  def resize_characters(self, py_imgs, py_rects, py_rect_speed, orig_speeds, coords, img_names):
    """ helper function for resizing character images and storing them locally. """

    for name, image, speed, loc in zip(self.names, self.images, self.speeds, self.locs):
        # load the given character image:
        py_img = pygame.image.load(image)

        # Get the size of the image
        image_size = py_img.get_size()
        width  = image_size[0]
        height = image_size[1]

        # Transform the width to 100 pixels
        # height/new = width/target_width
        img_name = name
        height_size = self.target_width * (height/width)
        img_size   = (self.target_width, height_size)
        py_img = pygame.transform.scale(py_img, img_size)

        # Check if it fits or not.
        loc0_ok = (loc[0] >= 0) and (loc[0] + py_img.get_width() <= self.vid_width)
        loc1_ok = (loc[1] + py_img.get_height() <= self.vid_height) and (loc[1] >= 0)
        if (loc0_ok and loc1_ok):
          py_rect = pygame.Rect(loc[0], loc[1], py_img.get_width(), py_img.get_height())
        else:
          print("ERROR in placing an image in the video!")
          print("Table loc [0] = ", loc[0], ", Table loc [1] = ", loc[1])
          print("Resized image width = ", py_img.get_width(), ", Resized image height = ", py_img.get_height())
          print("Number of rows in the video = ", self.vid_height)
          print("Number of columns in the video = ", self.vid_width)
          raise ValueError("Exiting the code: too many characters!")
         
        py_imgs.append(py_img)
        py_rects.append(py_rect)
        py_rect_speed.append(speed)
        orig_speeds.append(speed)
        coords.append([loc[0], loc[1]])
        img_names.append(img_name)
  
  

  def set_units(self, distance_string="miles", time_string="hours", speed_string="miles/hour"):
    """ Setup the units for the simulation. Default is miles/hour.
    """
    self.distance_string = distance_string
    self.time_string     = time_string
    self.speed_string    = speed_string


  def set_video(self, video_name="race.mp4", fps=30, vid_width=800, vid_height=600, 
                max_frames=10000, target_width=100, simulation_speed=1.0):
    """ Setup the video simulation parameters. Default values are provided.
    """
    # Video dimension check
    if vid_width <= 0 or not isinstance(vid_width, numbers.Number):
      raise ValueError('video width should be a postive number')

    if vid_height <= 0 or not isinstance(vid_height, numbers.Number):
      raise ValueError('video height should be a postive number')
    
    if fps <= 0 or not isinstance(fps, numbers.Number):
      raise ValueError('Video fps should be a postive number')
    
    # pygame.display.set_caption('Race to Bumpers') # Enable this to run locally    
    # Initialize pygame 
    pygame.init()
    
    # Setup the video  
    self.video_name = video_name
    self.fps = fps
    self.vid_width  = vid_width
    self.vid_height = vid_height
    self.max_frames = max_frames

    self.target_width  = target_width
    self.simulation_speed = simulation_speed
    
    # Setup some default scales:
    self.end_line_scale = 0.3         # Between 0 and 1.
    self.end_line_color = (255, 0, 0) # red
    self.disp_font_sz = 25            # font size

    # Stop line 
    self.end_line = self.vid_width - self.end_line_scale * self.vid_width
    self.end_line_start = (self.end_line, 0)
    self.end_line_end = (self.end_line, self.vid_height)
    
    # Axis units
    self.axis_unit = 10
    self.axis_line = self.vid_height - 0.1*self.vid_height
    self.dist_axis = (0, self.axis_line)
    self.axis_font_sz = 15

    # Black color
    self.black = (0, 0, 0)

    # Video character font size
    self.vid_disp_font = pygame.font.Font(None, self.disp_font_sz)
    
    # Extract table objects attributes
    self.py_imgs   = []
    self.py_rects  = []
    self.py_rect_speed = []
    self.orig_speeds = []
    self.coords    = []
    self.img_names = []
    self.resize_characters(self.py_imgs, self.py_rects, self.py_rect_speed, 
                           self.orig_speeds, self.coords, self.img_names)
          
    # Calculate pixel motions:
    self.time_between_frames = 1/fps*self.simulation_speed 
    self.num_of_frames  = self.duration*fps/self.simulation_speed

    self.pixel_distance = self.race_distance / (self.end_line - self.target_width)
    # print("number_of_frames=", num_of_frames)

    # Rescale py_rect_speed:
    # x m/s -> y pixels/frame
    # y * fps * pixel_distance  is distance in one second
    # x / (fps * pixel_distance)  
    self.py_rect_speed = [(1/self.fps)* float(x)*self.simulation_speed / self.pixel_distance 
                          for x in self.py_rect_speed]


    if (self.num_of_frames > self.max_frames):
        print("Too many frames!")
        print("num_of_frames = ", self.num_of_frames)
        print("duration = ", self.duration, " fps = ", self.fps, " simulation_speed = ", self.simulation_speed)
        print("You should reduce duration, reduce fps, or increase simulation_speed")
        print("You should run set_video() to do this!")
        print("Then run create_video() again.")
        return 

    
  def __repr__(self) -> str:
    str_rep = "race_video class parameters.\n"
    str_rep += "vid_title = "+str(self.vid_title)+"\n"
    str_rep += "Tables:"
    str_rep += str(self.tables)
    str_rep += "Initialization parameters\n"
    str_rep += "target_width  = "+str(self.target_width)+"\n"
    str_rep += "duration      = "+str(self.duration)+"\n" 
    str_rep += "race_distance = "+str(self.race_distance)+"\n"
    str_rep += "simulation_speed = "+str(self.simulation_speed)+"\n\n"

    str_rep += "Units\n"
    str_rep += "distance_string = "+str(self.distance_string)+"\n"
    str_rep += "time_string     = "+str(self.time_string)+"\n"
    str_rep += "speed_string    = "+str(self.speed_string)+"\n"

    str_rep += "Video\n"
    str_rep += "video_name = "+str(self.video_name)+"\n"
    str_rep += "vid_width  = "+str(self.vid_width)+"\n"
    str_rep += "vid_height = "+str(self.vid_height)+"\n"
    str_rep += "max_frames = "+str(str.max_frames)+"\n"  
    return(str_rep)
  
  
  def open_video(self):
    """ Helper function to open up video file. """
    
    # Video to store pygame
    if os.path.exists(self.video_name):
      # Remove the file
      os.remove(self.video_name)
      print(f"File '{self.video_name}' has been removed.")
    else:
      print(f"The file '{self.video_name}' will be created.")
      
    # Open up the video file:
    self.out_vid = cv2.VideoWriter(self.video_name, 
                    cv2.VideoWriter_fourcc(*'MJPG'), 
                    self.fps, (self.vid_width, self.vid_height))
    
    # Check if VideoWriter was successfully initialized
    if not self.out_vid.isOpened():
      raise RuntimeError("Error: Failed to initialize video writer.")
    else:
      print("VideoWriter initialized successfully.")
      
      
  def create_video(self):
    """ Creates the video simulation stores it in a video file. """
    self.open_video()
    
    # Set pygame display with video dimensions
    vid_disp = pygame.display.set_mode((self.vid_width, self.vid_height))
    
    # Simulation loop:
    current_duration = 0.0 
    race_clock = 0.0 
    frame_num = 0.0

    # Set the offset off the end line:
    offset = 20

    # Update pygame display for each table
    stop = np.full(len(self.py_rects), False) 
    stop_distances = np.full(len(self.py_rects), 0.0)
    stop_times     = np.full(len(self.py_rects), 0.0)
    while True:
      # Fill display with white color
      vid_disp.fill((255, 255, 255))
      
      for py_idx, py_rect in enumerate(self.py_rects):
        # Zero image speed once we reach reach the stop line
        # Set up the stop_distance and stop_time if we reached the end for the first time
        if ((not stop[py_idx]) and (race_clock*self.orig_speeds[py_idx] >= self.race_distance)):
          stop[py_idx] = True 
          stop_times[py_idx]     = race_clock
          stop_distances[py_idx] = race_clock*self.orig_speeds[py_idx]

        # Place the character image
        vid_disp.blit(self.py_imgs[py_idx], py_rect)
        
        # Move the character
        if (not stop[py_idx]): 
          x, y = self.coords[py_idx]
          dx   = self.py_rect_speed[py_idx]*frame_num
          
          py_img  = self.py_imgs[py_idx]
          self.py_rects[py_idx] = pygame.Rect(round(x+dx), int(y), 
                                    py_img.get_width(), py_img.get_height())
            
        # Overlay image name on video display
        img_name = self.vid_disp_font.render(f"{self.img_names[py_idx]}", True, (0, 0, 0))
        img_name_rect = img_name.get_rect()
        img_name_rect.topleft = (self.end_line + offset, py_rect.y)
        vid_disp.blit(img_name, img_name_rect)
        
        # Overlay image distance on video display
        # distance = (py_rect.right - self.target_width)*self.pixel_distance
        if (stop[py_idx]):
          distance = stop_distances[py_idx]
        else:
          distance = race_clock * self.orig_speeds[py_idx]

        dist = self.vid_disp_font.render(f"Distance: {distance:.2f} {self.distance_string}", 
                                         True, (0, 0, 0))
        dist_rect = dist.get_rect()
        dist_rect.topleft = (self.end_line + offset, py_rect.y + 20)
        vid_disp.blit(dist, dist_rect)
        
        # Overlay time on video display
        # the_time = distance/self.orig_speeds[py_idx]
        if (stop[py_idx]):
          the_time = stop_times[py_idx]
        else:
          the_time = race_clock

        dist = self.vid_disp_font.render(f"Time: {the_time:.2f} {self.time_string}", True, (0, 0, 0))
        dist_rect = dist.get_rect()
        dist_rect.topleft = (self.end_line + offset, py_rect.y + 40)
        vid_disp.blit(dist, dist_rect)
        
        # Overlay image speed on video display 
        spd = self.vid_disp_font.render(f"Speed: {self.orig_speeds[py_idx]} {self.speed_string}", True, (0, 0, 0))
        spd_rect = spd.get_rect()
        spd_rect.topleft = (self.end_line + offset, py_rect.y + 60)
        vid_disp.blit(spd, spd_rect)
        
        # Update rectangle at image speed
        # py_rect.move_ip([py_rect_speed[py_idx], 0])
        
      # Overlay clock time on video display
      race_timer = self.vid_disp_font.render(f"Time: {race_clock:.2f} {self.time_string}", True, (0, 0, 0))
      race_timer_rect = race_timer.get_rect()
      race_timer_rect.topleft = (self.end_line + offset, self.py_rects[-1].y + 100)
      vid_disp.blit(race_timer, race_timer_rect)
      
      # Overlay video title
      vid_title_ = self.vid_disp_font.render(f"{self.vid_title}", True, (0, 0, 0))
      vid_title_rect = vid_title_.get_rect()
      vid_title_rect.topleft = (self.end_line + offset, 10)
      vid_disp.blit(vid_title_, vid_title_rect)
      
      # Draw start line, stop line and bottom line
      pygame.draw.line(vid_disp, self.end_line_color, 
                       (self.target_width, 0), (self.target_width, self.vid_height)) # start line
      pygame.draw.line(vid_disp, self.end_line_color, 
                     self.end_line_start, self.end_line_end, 1)  
      pygame.draw.line(vid_disp, self.black, 
                     (0, self.axis_line), 
                     (self.end_line, self.axis_line), 1)  
      
      # Update entire pygame display
      pygame.display.flip()
      
      # Save pygame display as video 
      cv2_img = pygame.surfarray.array3d(vid_disp)
      cv2_img = cv2_img.transpose([1, 0, 2])
      cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR)
      self.out_vid.write(cv2_img)

      frame_num += 1.0
      current_duration = frame_num / self.fps  
      race_clock = current_duration  * self.simulation_speed
      
      
      # Terminate based on duration or all reached the end.
      stop_cond = current_duration > self.duration 
      stop_cond = stop_cond or all(stop) 
      
      if (frame_num > self.max_frames):
        print("Too many frames!")
        print("frame_num = ", frame_num)
        stop_cond = True 
        
      # Wait and break after reaching stop line        
      if stop_cond:
        # Fill up with the same frame.
        frames_left = int((self.duration - current_duration)*self.fps)
        if (frames_left > 0):
            for i in range(frames_left):
              self.out_vid.write(cv2_img)
              
        self.out_vid.release()
        pygame.quit()
        print("video file = ", self.video_name," closed.")
        break 
      
    race_video = moviepy.editor.VideoFileClip(self.video_name)
    return(race_video)



def CreateVideo(video_name, file_list, fps, durations):
  # Check array lengths:
  if (len(durations) != len(file_list)):
    print("The lists are of different lengths!")
    print("The number of elements in file_list = ", len(file_list))
    print("The number of elements in durations = ", len(durations))
    print("I cannot create the video until this is fixed!")
    return


  #self.plot_to_frame()
  height_list  = []
  width_list   = []

  # Find the biggest sizes
  for filename in file_list:
    # Check if we are working with a video file:
    if (filename.lower().endswith('.mp4')):
      # Open the file:
      print("Opening ", filename)
      cap = cv2.VideoCapture(filename)
      
      # Check if video opened successfully
      if not cap.isOpened():
        print("Error: Could not open video: file name = ", filename)
        raise ValueError("file name = "+filename+" cannot be opened!")
      
      # Get video properties
      width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
      height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
      frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

      # Append them:
      height_list.append(height)
      width_list.append(width)

      cap.release()
    else:
      print("Opening ", filename)
      img = cv2.imread(filename)

      if img is None:
        print("Error: Could not open image: file name = ", filename)
        raise ValueError("file name = "+filename+" cannot be opened!")
      
      height, width, channels = img.shape
      height_list.append(height)
      width_list.append(width)

  #print(height_list)
  h_video=np.max(height_list)
  w_video=np.max(width_list)
  print("video: height = ", h_video, " width = ", w_video)

  # Remove the file if it is already here:
  if os.path.exists(video_name):
    # Remove the file
    os.remove(video_name)
    print(f"File '{video_name}' has been removed.")
  else:
    print(f"The file '{video_name}' will be created.")

  #fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
  video = cv2.VideoWriter(video_name,
            cv2.VideoWriter_fourcc(*'MJPG'),
            fps,
            (w_video, h_video))
  
  # Check if VideoWriter was successfully initialized
  if not video.isOpened():
    raise RuntimeError("Error: Failed to initialize video writer.")
  else:
    print("VideoWriter initialized successfully.")

  # Save all of the frames after padding
  for filename, duration in zip(file_list, durations):
    # Check if we are working with a video file:
    if (filename.lower().endswith('.mp4')):
      # Open the file:
      cap = cv2.VideoCapture(filename)
      
      # Check if video opened successfully
      if not cap.isOpened():
        print("Error: Could not open video: file name = ", filename)
        return None
      
      # Read all the frames:
      while True:
        # Read frame
        ret, frame = cap.read()

        # If frame is read correctly ret is True
        if not ret:
            break
        
        # pad the frame and write to the file:
        video = padding(frame, video, h_video, w_video)

      # When everything done, release the video capture object
      cap.release()
    else:
      # pad the frame and write to the file:
      image = cv2.imread(filename)
      if image is None:
        print("I cannot open filename = ", filename)
        break

      if image.shape[2] == 4:
        # Split the channels
        b, g, r, a = cv2.split(image)
        
        # Merge the BGR channels back
        frame = cv2.merge([b, g, r])
      else:
        frame = image

      num_of_frames = int(duration*fps) 
      for i in range(num_of_frames):
        video = padding(frame, video, h_video, w_video)

  # Close the video
  video.release()
  final_video = moviepy.editor.VideoFileClip(video_name)
  return(final_video)
  
def padding(frame, video, h_video, w_video):
  old_h, old_w, channels = frame.shape
  if divmod(h_video - old_h, 2)[1] != 0:
    pad_h_t = int((h_video - 1 - old_h) /2)
    pad_h_b = int((h_video + 1 - old_h) /2)

  else:
    pad_h_t = int((h_video - old_h) /2)
    pad_h_b = int((h_video - old_h) /2)

  if divmod(w_video - old_w, 2)[1] != 0:
    pad_w_l = int((w_video - 1 - old_w) /2)
    pad_w_r = int((w_video + 1 - old_w) /2)

  else:
    pad_w_l = int((w_video - old_w) /2)
    pad_w_r = int((w_video - old_w) /2)

  padding_image = cv2.copyMakeBorder(frame,
    pad_h_t, pad_h_b,
    pad_w_l, pad_w_r,
    cv2.BORDER_CONSTANT,
    None,
    value = [255, 255, 255])

  new_h, new_w, channels = padding_image.shape
  video.write(padding_image)
  return video

# Original method for displaying video in Jupyter Notebook.
class MakeVideo:
  def __init__(self, mp4_fname, width, height):
    """ compresses the video file and displays in Jupyter notebook.
        mp4_fname represents the filename of the mp4 video.
    """
    self.mp4_fname = mp4_fname
    self.width  = width
    self.height = height

    # Compress file
    temp_file = "temp_video.mp4"
    self.compress(temp_file)

    # Create HTML for video display
    self.HTML_vid(temp_file)

  def compress(self, compressed_fname):
    """ compresses the given video file to compressed_finame
        If the filename is found, it replaces it with the current filename.
    """
    # Remove the compressed file name:
    if os.path.exists(compressed_fname):
      os.remove(compressed_fname)

    # Use ffmpeg to compress
    mp4_str = f"ffmpeg -i {self.mp4_fname} -vcodec libx264 {compressed_fname}"
    os.system(mp4_str)
    print("Compressed "+ self.mp4_fname + " into " + compressed_fname)

  def HTML_vid(self, compressed_fname):
    """ displays the compressed file in Juyter notebook.
    """
    mp4 = open(compressed_fname,'rb').read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    self.HTML_str = """
                 <video width="%s" height="%s" controls loop autoplay>
                    <source src="%s" type="video/mp4">
                 </video> """ % (self.width, self.height, data_url)

# Function to change video speed.
# Example usage: Double the speed
# change_playback_speed("all.mp4", "all2.mp4", 0.5)
def changeVideoSpeed(input_video, output_video, speed_factor):
    """ changeVideoSpeed() can be used to create a video at a different speed. """
    video = VideoFileClip(input_video)
    # Speed up or slow down the video
    new_video = video.fx(vfx.speedx, speed_factor)
    new_video.write_videofile(output_video)


# Reduce or enlarge to fit the image size.
def fit_scale(lines, width, height, font, start_row, start_col, line_space=1.5):
  """ computes the best font scale for a given
   lines:  message to fit. 
   width:  number of pixels for each row of text.
   height: number of pixels for each column of text.
   font:   font name.
   start_row:  starting row percentage for text.
   start_col:  starting column percentage for text.
   line_space: number of pixels between lines.

   returns scale
  """
  for scale in reversed(range(0, 60, 1)):
    y = start_col*height
    for line in lines:
      (text_width, text_height), _ = cv2.getTextSize(line, fontFace=font, 
                                                     fontScale=scale/10, 
                                                     thickness=1)
      y = y + text_height*line_space
      size_ok = (text_width <= width*(1 - start_row)) and (y <= height)
      if not size_ok:
        break
    
    # If the size worked, return it!
    if (size_ok):
      return (scale/10)

  return 0 # way too small!

# Create and save an image with text.
def textImage(output_path, 
              multiline_string,
              start_row = 0.1,
              start_col = 0.1,
              height = 600,
              width  = 800, 
              font_color=(0, 0, 0), 
              line_space=1.5):
    """ creates a text image
    
    output_path: image file name.
    multiline_string: a string that can be use multiple lines.
    
    Bottom left placement based on:
      start_row:  Starting row (from 0 to 1).
      start_col:  Starting column (from 0 to 1).       
        
    line_space: Line spacing 1, 2, etc.
    
    height: the number of row pixels.
    width:  the number of column pixels.
    font_color: the color of the text.
    """
    
    if (line_space < 1.0):
      raise ValueError(f"line_space={line_space} needs to be >1.")
  
    # Create a blank white image
    image = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Splitting the multiline string into a list of lines
    lines = multiline_string.splitlines()

    # Set font type
    font = cv2.FONT_HERSHEY_SIMPLEX 
    font_thickness = 1

    # Setup the parameters:
    y = start_col*height
    font_scale = fit_scale(lines=lines, width=width, height=height, font=font, 
                           start_row=start_row, start_col=start_col, line_space=line_space)
    
    # Keep font_scale if okay
    if (font_scale == 0):
      print("Your string is too long!")
      print(multiline_string)
      raise ValueError("Reduce your text!")
    
    # Avoid enlarging:
    if (font_scale > 1):
      font_scale = 1 

    # Add each line of text to the image
    y = int(start_col*height)
    for line in lines:
      (text_width, text_height), _ = cv2.getTextSize(line, fontFace=font, 
                                                     fontScale=font_scale, 
                                                     thickness=1)
      x = int(start_row*width)
      # print(f"x={x}, y={y}, text_width={text_width}, text_height={text_height}")
      cv2.putText(image, line, (x, y), 
                  font, font_scale, font_color, font_thickness, cv2.LINE_AA)
      delta_y = float(text_height)*line_space
      # print(f"delta_x={delta_y}")
      y = y + int(delta_y)
    
    # Save the result
    cv2.imwrite(output_path, image)
