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


# ---------- Bharati Braille Map ----------

braille_map = {
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


# ---------- Braille Decoder ----------

def decode_braille_lines(braille_bins, braille_map):
    """
    Decode braille binary representations to text.
    
    Args:
        braille_bins: List[List[str]] - output of braille_grid_detection
        braille_map: dict - mapping from bitmask to characters
        
    Returns:
        str: Decoded text with newlines between lines
    """
    decoded_lines = []
    
    for line in braille_bins:
        line_text = []
        for cell in line:
            bits = [int(b) for b in cell.split(',')]
            
            # Space
            if bits == [0, 0, 0, 0, 0, 0]:
                line_text.append(' ')
                continue
            
            key = bits_to_int(bits)
            line_text.append(braille_map.get(key, ''))
        
        decoded_lines.append(''.join(line_text).rstrip())
    
    return '\n'.join(decoded_lines)
