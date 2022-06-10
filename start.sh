#!/bin/bash

OS=$(uname -o)
USR=$(whoami)

bot() {
	echo -e "\nLoading Discord bot.\n\nUse other model?"
	read -p "> " ANS
	[[ "$ANS" != [Yy]* ]] && python main.py 2> /dev/null || echo -e "\nWrite the model name:"; read -p "> " NAME; python main.py $NAME 2> /dev/null
}

clear; echo -e "\n======== EP3 - IA =========\n\nWelcome $USR, starting EP3 on $OS.."

[[ "$1" == [Tt]rain ]] &&  python train.py || bot
