from Levenshtein import ratio

# Ini digunakan untuk memvalidasi hasil prediksi dengan target sebenarnya

def validate_word(predicted, target):
    score = ratio(predicted, target)
    return score >= 0.8, score