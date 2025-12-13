"""
Utility
-----
  Supplimentary utility functions for time-related visualization and measurement

Provides:
- Timer
- Progress Bar
"""

import time
import wikipedia

def timer(func):
  """
  Timer
  -----
    Decorator for timing a function
    write '@timer' before the function and it will automaticly time the function
  """
  def wrapper(*args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return end_time - start_time

  return wrapper

def progress_bar(iterable, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', empty = ' '):
  """
  Progress Bar
  -----
    equivilant to tqdm, imagine this as the built in 'range' function 
    that also happpens to print the progress bar for the loop
  -----
  Args
  -----
  - iterable  (Iterable)       : iterable object

  - (Optional) prefix    (Str) : prefix string
  - (Optional) suffix    (Str) : suffix string
  - (Optional) decimals  (Int) : positive number of decimals in percent complete
  - (Optional) length    (Int) : character length of bar
  - (Optional) fill      (Str) : bar fill character
  
  Returns
  -----
    None
  """
  try:
    total = len(iterable)
    # Progress Bar Printing Function
    def printProgressBar (iteration):
      percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
      filledLength = int(length * iteration // total)
      bar = fill * filledLength + empty * (length - filledLength)
      print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
      # Print New Line on Complete
      if iteration == total:
        print()
    # Initial Call
    printProgressBar(0)
    # Update Progress Bar
    for i, item in enumerate(iterable):
      yield item
      printProgressBar(i + 1)
    # Print New Line on Complete
    print()
  except:
    print()
    exit()
