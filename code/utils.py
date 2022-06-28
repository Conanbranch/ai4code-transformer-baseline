from io import StringIO
import tokenize
import re

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

@return_unmodified_value_if_failed()
def preprocess_text(document):
    document = re.sub(r"[^a-zA-Z0-9]+", ' ', str(document)) # Remove all the special characters except 
    #document = re.sub(r"[^a-zA-Z0-9#]+", ' ', str(document)) # Remove all the special characters except # which indicates heading level
    #document = re.sub(r"\b[a-zA-Z]\b", ' ', document) # Remove all single characters
    #document = re.sub(r'\s+', ' ', document, flags=re.I) # Substitute multiple spaces with single space   
    #document = re.sub(r'^b\s+', '', document) # Removing prefixed 'b'
    #document = document.lower() # Converting to Lowercase
    #document = document.lstrip()
    return document

@return_unmodified_value_if_failed()
def preprocess_markdown(df):
    return [preprocess_text(message) for message in df.source]

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
def clean_code(cell):
    #cell = str(remove_comments_and_docstrings(cell))
    cell = str(cell).replace("\\n", "\n")
    #cell = str(cell).replace("\n", " ")
    #cell = str(re.sub(' +', ' ', cell))
    #cell = str(cell).lstrip()
    #cell = str(cell).lower()
    return cell
