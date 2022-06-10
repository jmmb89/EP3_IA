import os, sys
import discord
import requests
from ia import train, load, predict

current_model = "traffic_sign_model"
c_model = None
client = discord.Client()

def download_file(url):
	download = requests.get(url)
	local_file = url.split('/')[-1]
	#os.path.join("/{}/{}")
	path = f"downloads/{local_file}"
	with open(path, 'wb') as f:
		f.write(download.content)	
	return path

@client.event 
async def on_ready():
	print("Connected.")

@client.event
async def on_message(message):
	content = message.content.lower()
	channel = message.channel
	author = message.author.name 
	if(author == "BccBot"):
		return
	else:
		if content == "#placa":
			try:
				image_url = message.attachments[0].url
				print(f"\nDownloading image {image_url.split('/')[-1]}..")
				image = download_file(image_url)
				sign = predict(c_model, image) 
				await channel.send(sign)
			except:
				print("\nFailed loading discord image.")
				await channel.send("Failed loading image..")

print('\n'+8*'='+ ' EP3 - PI '+8*'=')

if len(sys.argv) > 1:
	current_model = os.path.splitext(sys.argv[1])[0]

if os.path.isfile(f"models/{current_model}.keras"):
	print("\nLoading model..")
	c_model = load(f"models/{current_model}.keras")
	print("Model loaded.\n\nConnecting to Discord..")
else:
	print(f"\nModel {current_model} not found. Exiting..")
	os._exit(1)

client.run("OTc4MDc3MDgyOTgwNjEwMDY4.G7Uzw7.rgxOM1z1TaeHbQdT8km_n63xPNpsrstoZeCeRY")
