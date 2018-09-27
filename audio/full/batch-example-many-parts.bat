for %%A in (*.mp3) do (
	ffmpeg -i "%%A" -ar 44100 -ac 1 "..\wav\%%~nA.wav"
    sox --norm=0 "..\wav\%%~nA.wav" -b 16 "..\trim\%%2n%%~nA.wav" trim 140 20 : newfile : trim 160 20 : newfile : trim 180 20 : newfile : trim 200 20 : newfile : trim 220 20 : newfile : trim 240 20 : newfile : trim 260 20 : newfile : trim 280 20 : newfile : trim 300 20 : newfile : trim 320 20 : newfile : trim 340 20 : newfile : trim 360 20 : newfile : trim 380 20 : newfile : trim 400 20 : newfile : trim 420 20 : newfile : trim 440 20 : newfile : trim 460 20 : newfile : trim 480 20
	del "..\wav\%%~nA.wav"
	)
pause