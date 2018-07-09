# TLOU-morse-cricket
Work in progress on decoding a potential easter egg in The Last of Us.  In the beginning of the Museum section, there's a cricket in a corner that chirps what sounds like Morse code, except with no spaces between letters.  The files here help transcode and try to find words in that chirping.

transcribe.Rmd - A simple transcriber in R from audio file (.wav) to Morse code dits and dahs in a text file.

translate.py - Given a length of dits and dahs, tries to impose words from a dictionary.

longest_substring.py - find the longest common substring between two files, to look for overlap.

# Credit
This code was pulled almost entirely from Mark Patterson's work on a Morse translator.  I only modified a few things to work reliably with these particular audio recordings.  The original work is here:

https://www.r-bloggers.com/morse-code-converter/

# Information
The reddit threads on the project are here:

https://www.reddit.com/r/thelastofus/comments/8vl4zx/morse_cricket_decoding_discussion/

https://www.reddit.com/r/thelastofus/comments/5hrhfc/the_last_of_us_morse_code_easter_egg/

https://www.reddit.com/r/morse/comments/5hwpi8/is_this_morse_code/

# Recordings
My own sample recordings of the cricket:

https://youtu.be/WrDUUxLAcck  (transcribed as  7min.txt in the data folder)

https://youtu.be/DM5IY1qt59A  (transcribed as 57min.txt in the data folder)

https://youtu.be/EhunEdlmLdg  (transcribed as 60min.txt in the data folder)

Other people's recordings of the cricket:

https://youtu.be/vdMzYNtFS58
