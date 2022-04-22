import re


def remove_pattern(text, regex):
    patterns = re.findall(regex, text)
    for pattern in patterns:
        text = re.sub(pattern, "", text)

    return text


def remove_usernames(text):
    return remove_pattern(text, r"@\w*")


# function to collect hashtags
def hashtag_extract(text):
    return re.findall(r"#(\w+)", text)


def remove_unicode(text):
    # source: https://towardsdatascience.com/a-guide-to-cleaning-text-in-python-943356ac86ca
    # encoding the text to ASCII format
    encoded_text = text.encode(encoding="ascii", errors="ignore")

    # decoding the text
    decoded_text = encoded_text.decode()

    # cleaning the text to remove extra whitespace
    return " ".join([word for word in decoded_text.split()])


def remove_html_entities(text):
    return remove_pattern(text, r"&[a-zA-Z0-9#]+;")


def remove_tags(text):
    return remove_pattern(text, r"<([^>]+)>")
