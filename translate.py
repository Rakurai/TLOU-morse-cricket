#!/bin/bash/python3

# just some experimental stuff to try and find meaningful words in run-together morse.
# searches for common words from a dictionary, outputs a pseudo-visual

D = {
	'e': '.',
	't': '-',
	'a': '.-',
	'o': '---',
	'i': '..',
	'n': '-.',
	's': '...',
	'h': '....',
	'r': '.-.',
	'd': '-..',
	'l': '.-..',
	'c': '-.-.',
	'u': '..-',
	'm': '--',
	'w': '.--',
	'f': '..-.',
	'g': '--.',
	'y': '-.--',
	'p': '.--.',
	'b': '-...',
	'v': '...-',
	'k': '-.-',
	'j': '.---',
	'x': '-..-',
	'q': '--.-',
	'z': '--..',
}


CRYPT = '---..----.--..-.--..---.'#-----....---...----..-.-.-..--.--.---.....---...--..-.--.-.---.----..-.--..----..--..-..-.-.--....--.....----.-.----.-.--..--.....'
#CRYPT = '...---.--..--.--.-.----.--.-..---.-.--.----...-.---..---....---.---....-.--....---...-----.-.-..----------..----.----.--.-..-.---.---.-.--..---.----....----..--..----..-.-.---.-.-..---..--.-----.-----...--..---.-.-.-.-.-..--.-..----..-...-.-.--..----..-.----.-.--....-.--..-.---.-------..--.-.--.-.--..-.--.-..----.----..-.-...---...-.-------.----..----.----.---....--.--.-.--...---...---...----..----.-------.---.-..---.-...---...----.---....-.--..---..-.---.---.....---.-.---..----....--.-..---.----.-....-.--.-------...-.--.-.--....--.-...-.-.-..--.-.-------..-..---..-------.--.---..-------....----.------.-.-.-.-..-.-...---..---....---....-'
CRYPTLEN = len(CRYPT)
#START = min(CRYPT.find("."), CRYPT.find("-"))
PAD = 8
outbuf = ''

def has_code(place, code):
	tail_index = place + len(code)
	if CRYPTLEN >= tail_index and CRYPT[place:tail_index] == code:
		return True
	return False

def recurse(place, outbuf):
	for letter, code in D:
		if has_code(place, code):
			recurse(place + len(code), outbuf + letter)

	print(outbuf)

#recurse(0, '')

def read_dict(fname):
	words = {}
	MIN_LEN = 1

	with open(fname) as f:
		for word in f.readlines():
			word = word.strip()

			if len(word) < MIN_LEN:
				continue

			code = ''

			for c in word:
				code += D[c]

			words[word] = code

	return words

def find_words(words):
	positions = []

	for word, code in words.items():
		index = 0

		while True:
			index = CRYPT.find(code, index)

			if index == -1:
				break

			positions.append(index)
			index += len(code)

		if len(positions) == 0:
			continue

	#	padf = int((len(code) - len(word)) / 2)
	#	padf = 0
		padt = len(code) - len(word)# - padf
		padded_word = word + ' '*padt

		buf = ''
		last_tail = 0
	#	print(positions)

		for pos in positions:
			buf += CRYPT[last_tail:pos]
			buf += padded_word
			last_tail = pos + len(padded_word)

		buf += CRYPT[last_tail:]

	#	print("%-10s %s" % (word, CRYPT))
		print("%-10s %s" % (word, buf))
		positions.clear()

	#	print("%-10s %s%s%s%s%s" % (word, CRYPT[:index], ' '*padf, word, ' '*padt, CRYPT[index+len(code):]))

def get_word_positions(words):
	positions = []
	word_list = []

	for index in range(0, CRYPTLEN):
		word_list = []
		remainder = CRYPT[index:]

		for word, code in words.items():
			if remainder.startswith(code):
				word_list.append(word)

		positions.append(word_list)

	return positions

def generate_phrases_recurse(positions, pos, outbuf):
	if pos >= len(positions) or len(positions[pos]) == 0:
		print(outbuf)
		return

	for word in positions[pos]:
		generate_phrases_recurse(positions, pos + len(word), outbuf + ' ' + word)

import sys
words = read_dict(sys.argv[1])
#find_words(words)
positions = get_word_positions(words)
for i in range(0, CRYPTLEN):
	print("ignoring the first", i, "Morse symbols:")
	generate_phrases_recurse(positions, i, '  '+CRYPT[:i])


