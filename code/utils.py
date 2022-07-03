from io import StringIO
import tokenize
import re
from bs4 import BeautifulSoup
from markdown import markdown

fails = []

def return_default_value_if_failed(default_value):
    def decorator(func):
        def inner(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                fails.append((func, (args, kwargs), e))
                return default_value
        return inner
    return decorator

def return_unmodified_value_if_failed():
    def decorator(func):
        def inner(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                fails.append((func, (args, kwargs), e))
                return args
        return inner
    return decorator

# Source: https://gist.github.com/lorey/eb15a7f3338f959a78cc3661fbc255fe
# May have some overhead from converting to html back to text, another solution:
# https://stackoverflow.com/questions/761824/python-how-to-convert-markdown-formatted-text-to-text
@return_unmodified_value_if_failed()
def remove_markdown_and_markup(string):
    """ Converts a markdown string to plaintext """
    
    # md -> html -> text since BeautifulSoup can extract text cleanly
    html = markdown(string)
    
    # remove code snippets
    html = re.sub(r'<pre>(.*?)</pre>', ' ', html)
    html = re.sub(r'<code>(.*?)</code >', ' ', html)

    # extract text
    soup = BeautifulSoup(html, "html.parser")
    text = ' '.join(soup.findAll(text=True))
    
    return text

# Source: https://stackoverflow.com/questions/1769332/script-to-remove-python-comments-docstrings
@return_unmodified_value_if_failed()
def remove_comments_and_docstrings(source):
    """
    Returns 'source' minus comments and docstrings.
    """
    io_obj = StringIO(source)
    out = ""
    prev_toktype = tokenize.INDENT
    last_lineno = -1
    last_col = 0
    for tok in tokenize.generate_tokens(io_obj.readline):
        token_type = tok[0]
        token_string = tok[1]
        start_line, start_col = tok[2]
        end_line, end_col = tok[3]
        ltext = tok[4]
        # The following two conditionals preserve indentation.
        # This is necessary because we're not using tokenize.untokenize()
        # (because it spits out code with copious amounts of oddly-placed
        # whitespace).
        if start_line > last_lineno:
            last_col = 0
        if start_col > last_col:
            out += (" " * (start_col - last_col))
        # Remove comments:
        if token_type == tokenize.COMMENT:
            pass
        # This series of conditionals removes docstrings:
        elif token_type == tokenize.STRING:
            if prev_toktype != tokenize.INDENT:
        # This is likely a docstring; double-check we're not inside an operator:
                if prev_toktype != tokenize.NEWLINE:
                    # Note regarding NEWLINE vs NL: The tokenize module
                    # differentiates between newlines that start a new statement
                    # and newlines inside of operators such as parens, brackes,
                    # and curly braces.  Newlines inside of operators are
                    # NEWLINE and newlines that start new code are NL.
                    # Catch whole-module docstrings:
                    if start_col > 0:
                        # Unlabelled indentation means we're inside an operator
                        out += token_string
                    # Note regarding the INDENT token: The tokenize module does
                    # not label indentation inside of an operator (parens,
                    # brackets, and curly braces) as actual indentation.
                    # For example:
                    # def foo():
                    #     "The spaces before this docstring are tokenize.INDENT"
                    #     test = [
                    #         "The spaces before this string do not get a token"
                    #     ]
        else:
            out += token_string
        prev_toktype = token_type
        last_col = end_col
        last_lineno = end_line
    return out

@return_unmodified_value_if_failed()
def remove_special_tokens(document):
    special_tokens = ['<pad>', '</s>', '<unk>', '<s>', '<mask>']
    document_words = document.split()
    cleaned_words  = [word for word in document_words if word.lower() not in special_tokens]
    text = ' '.join(cleaned_words)
    return text

@return_unmodified_value_if_failed()
def clean_markdown(document):
    #document = remove_markdown_and_markup(str(document)) # remove markdown and html
    #document = re.sub(r'https?://\S+|www\.\S+', ' ', str(document)) # remove links
    #document = remove_special_tokens(str(document)) # remove any special tokens
    #document = re.sub(r'[^a-zA-Z0-9]+', ' ', str(document)) # remove all the special characters 
    #document = re.sub(r'[^a-zA-Z0-9#]+', ' ', str(document)) # remove all the special characters except # which indicates heading level
    #document = re.sub(r'\w*\d\w*', ' ', str(document)) # remove words with numbers
    #document = re.sub(r'[0-9]+', ' ', str(document)) # remove numbers from words and on their own
    #document = re.sub(r'^b\s+', ' ', str(document)) # removing prefixed 'b'
    #document = re.sub(r'\b[a-zA-Z]\b', ' ', str(document)) # remove all single characters
    #document = remove_special_tokens(str(document)) # remove any special tokens
    #document = re.sub(r'\s+', ' ', str(document), flags=re.I) # substitute multiple spaces with single space   
    #document = str(document).strip() # remove leading and following spaces
    #document = str(document).lower() # converting to Lowercase
    return document

@return_unmodified_value_if_failed()
def clean_code(cell):
    cell = remove_comments_and_docstrings(str(document)) #remove comments
    cell = str(cell).replace("\\n", "\n") # fix newlines
    cell = str(cell).replace("\n", " ") # remove newlines
    cell = remove_special_tokens(str(cell)) # remove any special tokens
    cell = re.sub(' +', ' ', str(cell)) # remove multiple spaces
    cell = str(cell).strip() # remove leading and following spaces
    cell = str(cell).lower() # converting to Lowercase
    return cell
