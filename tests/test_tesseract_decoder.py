import tesseract_decoder

def test_simple_decode():
    dem = """
error(0.5) D0 L0
error(0.5) D1
detector(0,0,0) D0
detector(0,0,0) D1
"""
    dec = tesseract_decoder.compile_decoder_for_dem(dem)
    assert dec.decode([0]) == 1
    assert dec.decode([]) == 0
