import enchant

def english_words_ratio(text):
    """
    Calculates the ratio of English words in the given text.

    Parameters:
    text (str): The input text.

    Returns:
    float: The ratio of English words in the text.
    """
    d = enchant.Dict("en_BG")
    len_text = len(text.split())
    return len([True for word in text.split() if d.check(word)]) / len_text


def unique_words_ratio(text, english_only=False):
    """
    Calculates the ratio of unique words in the given text.

    Parameters:
    - text (str): The input text.
    - english_only (bool): If True, only considers English words.

    Returns:
    - float: The ratio of unique words in the text.
    """
    len_text = len(text.split())
    if len_text <= 0: return 0
    if english_only:
        d = enchant.Dict("en_BG")
        return len([True for word in text.split() if d.check(word)]) / len_text
    return len(set(text.split())) / len_text


def remove_non_english(text):
    """
    Removes non-English words from the given text.

    Args:
        text (str): The input text.

    Returns:
        str: The text with non-English words removed.
    """
    d = enchant.Dict("en_BG")
    res = ""
    for word in text.split(): 
        res += f"{word} " if d.check(word) else ""
    return res