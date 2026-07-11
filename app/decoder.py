# -*- coding: utf-8 -*-
"""
Braille to text decoder using Bharati Braille mapping (OBR - Object to Braille Recognition).
"""
# ---------- Braille bit helpers ----------

def dots_to_bit(dots):
    """Convert braille dot numbers [1..6] to integer bitmask"""
    value = 0
    for d in dots:
        value |= (1 << (d - 1))
    return value


def bits_to_int(bits):
    """Convert [0,1,0,1,0,0] → integer bitmask"""
    value = 0
    for i, b in enumerate(bits):
        if b:
            value |= (1 << i)
    return value

vowel_symbol_map = {
    'अ': '',
    'आ': 'ा',
    'इ': 'ि',
    'ई': 'ी',
    'उ': 'ु',
    'ऊ': 'ू',
    'ऋ': 'ृ',
    'ए': 'े',
    'ऐ': 'ै',
    'ओ': 'ो',
    'औ': 'ौ',
    'अं': 'ं',
    'अः': 'ः',
}

consonant_symbol_map = {
    'क': 'क',
    'ख': 'ख',
    'ग': 'ग',
    'घ': 'घ',
    'ङ': 'ङ',
    'च': 'च',
    'छ': 'छ',
    'ज': 'ज',
    'झ': 'झ',
    'ञ': 'ञ',
    'ट': 'ट',
    'ठ': 'ठ',
    'ड': 'ड',
    'ढ': 'ढ',
    'ण': 'ण',
    'त': 'त',
    'थ': 'थ',
    'द': 'द',
    'ध': 'ध',
    'न': 'न',
    'प': 'प',
    'फ': 'फ',
    'ब': 'ब',
    'भ': 'भ',
    'म': 'म',
    'य': 'य',
    'र': 'र',
    'ल': 'ल',
    'व': 'व',
    'श': 'श',
    'ष': 'ष',
    'स': 'स',
    'ह': 'ह',
    'क्ष': 'क्ष',
    'ज्ञ': 'ज्ञ',
}

half_consonant_symbol_map = {
    'क': 'क्',
    'ख': 'ख्',
    'ग': 'ग्',
    'घ': 'घ्',
    'ङ': 'ङ्',
    'च': 'च्',
    'छ': 'छ्',
    'ज': 'ज्',
    'झ': 'झ्',
    'ञ': 'ञ्',
    'ट': 'ट्',
    'ठ': 'ठ्',
    'ड': 'ड्',
    'ढ': 'ढ',
    'ण': 'ण्',
    'त': 'त्',
    'थ': 'थ्',
    'द': 'द्',
    'ध': 'ध्',
    'न': 'न्',
    'प': 'प्',
    'फ': 'फ्',
    'ब': 'ब्',
    'भ': 'भ्',
    'म': 'म्',
    'य': 'य्',
    'र': 'र्',
    'ल': 'ल्',
    'व': 'व्',
    'श': 'श्',
    'ष': 'ष्',
    'स': 'स्',
    'ह': 'ह्',
    'क्ष': 'क्ष्',
    'ज्ञ': 'ज्ञ्',
}

# ---------- Bharati Braille Map ----------

braille_map_text = {
    # vowels
    dots_to_bit([1]): 'अ',
    dots_to_bit([3, 4, 5]): 'आ',
    dots_to_bit([2, 4]): 'इ',
    dots_to_bit([3, 5]): 'ई',
    dots_to_bit([1, 3, 6]): 'उ',
    dots_to_bit([1, 2, 5, 6]): 'ऊ',
    dots_to_bit([1, 5, 6]): 'ऋ',
    dots_to_bit([1, 5]): 'ए',
    dots_to_bit([3, 4]): 'ऐ',
    dots_to_bit([1, 3, 5]): 'ओ',
    dots_to_bit([2, 4, 6]): 'औ',
    dots_to_bit([1, 6]): 'अं',
    dots_to_bit([6]): 'अः',

    # symbols
    dots_to_bit([4]): ' ्', # comes before consonants
    dots_to_bit([3]): ' ँ', # comes after consonants
    dots_to_bit([5, 6]): ' ं', # comes after consonants
    dots_to_bit([6]): ' ः', # comes after vowels
    dots_to_bit([2, 5, 6]): '।', # comes at end of every sentence
    dots_to_bit([2, 3, 6]): '?', # comes at end of every question
    dots_to_bit([3, 6]): '-', 
    dots_to_bit([2]): ',',

    # consonants
    dots_to_bit([1, 3]): 'क',
    dots_to_bit([4, 6]): 'ख',
    dots_to_bit([1, 2, 4, 5]): 'ग',
    dots_to_bit([1, 2, 6]): 'घ',
    dots_to_bit([3, 4, 6]): 'ङ',
    dots_to_bit([1, 4]): 'च',
    dots_to_bit([1, 6]): 'छ',
    dots_to_bit([2, 4, 5]): 'ज',
    dots_to_bit([3, 5, 6]): 'झ',
    dots_to_bit([2, 5]): 'ञ',
    dots_to_bit([2, 3, 4, 6]): 'ट',
    dots_to_bit([2, 4, 5, 6]): 'ठ',
    dots_to_bit([1, 2, 4, 6]): 'ड',
    dots_to_bit([1, 2, 3, 4, 6]): 'ढ',
    dots_to_bit([3, 4, 5, 6]): 'ण',
    dots_to_bit([2, 3, 4, 5]): 'त',
    dots_to_bit([1, 4, 5, 6]): 'थ',
    dots_to_bit([1, 4, 5]): 'द',
    dots_to_bit([2, 3, 5, 6]): 'ध',
    dots_to_bit([1, 3, 4, 5]): 'न',
    dots_to_bit([1, 2, 3, 4]): 'प',
    dots_to_bit([2, 3, 5]): 'फ',
    dots_to_bit([1, 2]): 'ब',
    dots_to_bit([4, 5]): 'भ',
    dots_to_bit([1, 3, 4]): 'म',
    dots_to_bit([1, 3, 4, 5, 6]): 'य',
    dots_to_bit([1, 2, 3, 5]): 'र',
    dots_to_bit([1, 2, 3]): 'ल',
    dots_to_bit([1, 2, 3, 6]): 'व',
    dots_to_bit([1, 4, 6]): 'श',
    dots_to_bit([1, 2, 3, 4, 6]): 'ष',
    dots_to_bit([2, 3, 4]): 'स',
    dots_to_bit([1, 2, 5]): 'ह',
    dots_to_bit([1, 2, 3, 4, 5]): 'क्ष',
    dots_to_bit([1, 5, 6]): 'ज्ञ',
}

# for numbers:
# 3456 denotes the start of a number string
braille_map_number = {
    dots_to_bit([1]): '१',
    dots_to_bit([1, 2]): '२',
    dots_to_bit([1, 4]): '३',
    dots_to_bit([1, 4, 5]): '४',
    dots_to_bit([1, 5]): '५',
    dots_to_bit([1, 2, 4]): '६',
    dots_to_bit([1, 2, 4, 5]): '७',
    dots_to_bit([1, 2, 5]): '८',
    dots_to_bit([2, 4]): '९',
    dots_to_bit([2, 4, 5]): '०',
}


# ---------- Braille Decoder ----------

def decode_braille_lines(braille_bins, braille_map_text, braille_map_number):
    """
    Decode braille binary representations to text.
    
    Args:
        braille_bins: List[List[str]] - output of braille_grid_detection
        braille_map_text: dict - mapping from bitmask to characters
        braille_map_number: dict - mapping from bitmask to numbers
        
    Returns:
        str: Decoded text with newlines between lines
    """
    decoded_lines = []

    for line in braille_bins:
        tokens = []
        is_number_string = False

        for cell in line:
            bits = [int(b) for b in cell.split(',')]

            # space
            if bits == [0, 0, 0, 0, 0, 0]:
                tokens.append(' ')
                is_number_string = False
                continue

            key = bits_to_int(bits)

            # number mode
            if is_number_string and key in braille_map_number:
                # remove the last letter (ण) if last two letters are a space and the letter ण which is 3,4,5,6 in braille code
                if len(tokens) > 1 and tokens[-1] == 'ण' and tokens[-2] == ' ':
                    tokens.pop()
                tokens.append(braille_map_number[key])
                continue

            # detect number start if last added token is a space
            if len(tokens) > 0 and tokens[-1] == ' ':
                if key == dots_to_bit([3, 4, 5, 6]):
                    tokens.append(braille_map_text[key])
                    is_number_string = True
                    continue

            tokens.append(braille_map_text.get(key, ''))

        decoded_lines.append(tokens)

    return decoded_lines


# ---------- Clean the text ----------
def render_nepali_text(token_lines, half_consonant_symbol_map):
    output_lines = []

    for tokens in token_lines:
        result = []
        i = 0

        while i < len(tokens):
            t = tokens[i]

            # multiple spaces
            if t == ' ':
                while i + 1 < len(tokens) and tokens[i + 1] == ' ':
                    i += 1
                result.append(' ')
                i += 1
                continue

            # SPACE
            if t == ' ':
                result.append(' ')
                i += 1
                continue
            
            # special case: ' ्'
            if t == " ्":
                if i + 1 < len(tokens) and tokens[i + 1] in consonant_symbol_map:
                    result.append(half_consonant_symbol_map[tokens[i + 1]])
                    i += 2
                    continue

            # CONSONANT
            if t in consonant_symbol_map:
                # lookahead for vowel
                if i + 1 < len(tokens) and tokens[i + 1] in vowel_symbol_map:
                    vowel = tokens[i + 1]
                    matra = vowel_symbol_map[vowel]

                    # special case: 'ि'
                    if matra == 'ि':
                        result.append(matra + t)
                    else:
                        result.append(t + matra)

                    i += 2
                else:
                    result.append(t)
                    i += 1

            # VOWEL (standalone)
            elif t in vowel_symbol_map:
                result.append(t)
                i += 1

            else:
                result.append(t)
                i += 1

        output_lines.append(''.join(result))

    # return output_lines
    # return token_lines
    return ''.join(output_lines)