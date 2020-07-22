#!/bin/bash

sudo su

apt install unzip
cd ./data
sh figaro.sh && sh lfw.sh
