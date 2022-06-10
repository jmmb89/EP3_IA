import os
import time
from ia import train 

current_model = f"models/new_model-{int(time.time())}.keras"

print('\n'+8*'='+ ' EP3 - TREINO '+8*'=')

train(current_model)

print(f"Training complete!\n\nNew file {current_model}")
