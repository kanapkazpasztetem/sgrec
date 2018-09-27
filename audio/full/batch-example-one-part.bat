for %%A in (*.mp3) do (
	ffmpeg -i "%%A" -ar 44100 -ac 1 "..\wav\%%~nA.wav"
    sox --norm=0 "..\wav\%%~nA.wav" -b 16 "..\trim\%%~nA.wav" trim 60 20 
	del "..\wav\%%~nA.wav"
)
pause