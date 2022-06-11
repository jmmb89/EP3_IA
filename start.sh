#!/bin/bash

OS=$(uname -o)
USR=$(whoami)

bot() {
	echo -e "\nLoading Discord bot.\n\nUse other model? [N,y]"
	read -p "> " ANS
	if [[ "$ANS" != [Yy]* ]]; then
		python main.py 2> /dev/null
	else
		echo -e "\nWrite the model name:"
		read -p "> " NAME
		python main.py $NAME 2> /dev/null
	fi
}

clear; echo -e "\n======== EP3 - IA =========\n\nWelcome $USR, starting EP3 on $OS.."

[[ "$1" == [Tt]rain ]] &&  python train.py || bot
