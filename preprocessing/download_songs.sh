#!/bin/sh
for dir in data/*; do
	python3 getSongInfo.py $dir mongodb://localhost:27017 tracks,audio-features;
	done
