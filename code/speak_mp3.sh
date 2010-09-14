espeak -w mp3/test.wav "This is a test." 
sox mp3/test.wav -c 2 -r 44100 mp3/test-441s.wav
lame -ms -cbr mp3/test-441s.wav mp3/test.mp3

