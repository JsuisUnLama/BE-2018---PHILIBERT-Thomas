#!/bin/sh

if [ $# -eq 2 ]
	ip = $1
	reverse = $2
	if [ reverse -eq 0 ]
		cat shoot.sh | ssh pi@$ip sh
		rsync -avz --ignore-existing --remove-source-files pi@ip:hell_eyes.jpeg ../images
	elif [ reverse -eq 1 ]
		cat shoot_rev.sh | ssh pi@ip sh
		rsync -avz --ignore-existing --remove-source-files pi@ip:hell_eyes.jpeg ../images
	else
		exit(1)
return('hell_eyes.jpeg')
else
	exit(1)
