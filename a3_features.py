import os
import sys
import argparse
import numpy as np
import pandas as pd
# Whatever other imports you need
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
import random

email_address = re.compile(r"(.+)@(.+)")
url = re.compile(r"^https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)$")

names = ["Susan", "Bailey", "Shelley", "Corman", "Craig", "Clint", "Dean", "Tom", "Donohoe", "Kam", "KK", "Keiser", "Kenneth", "Ken", "Lay", "Matthew", "Lenhart", "Larry", "May", "Danny", "Dan", "McCarty", "Stephanie", "Panus", "Dutch", "Quigley", "Eric", "Saibi", "Holden", "Salisbury", "Jim", "Schwieger"]

def remove_author_names(line):
    for name in names:
        line = line.replace(name, "")
    return line

# some headers to skip
def skip_line(line):
    if line.startswith("From:"):
        return True
    if line.startswith("To:"):
        return True
    if line.startswith("Subject:"):
        return True
    if line.startswith("Date:"):
        return True
    if line.startswith("Cc"):
        return True
    if line.startswith("cc"):
        return True
    if line.startswith("Mime-Version"):
        return True
    if line.startswith("Content"):
        return True
    if line.startswith("Bcc"):
        return True
    if line.startswith("X"):
        return True
    if line.startswith("<Embedded"):
        return True
    else:
        return False

# trying to avoid adding signatures or forwarded email text
def end_of_message(line):
    if line.startswith("-----"):
        return True
    if line.startswith(" -----"):
        return True
    if line.startswith("___"):
        return True
    else:
        return False

authortable = {}

# looks up author name in authortable and returns the number representing that name
def author_to_num(author):
    return str(authortable[author])

# used to rended author table of author names and author numbers to write to outputfile
def render_author_table():
    return ' '.join([key + ' ' + str(value) for key,value in authortable.items()])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert directories into table.")
    parser.add_argument("inputdir", type=str, help="The root of the author directories.")
    parser.add_argument("outputfile", type=str, help="The name of the output file containing the table of instances.")
    parser.add_argument("dims", type=int, help="The output feature dimensions.")
    parser.add_argument("--test", "-T", dest="testsize", type=int, default="20", help="The percentage (integer) of instances to label as test.")

    args = parser.parse_args()

    print("Reading {}...".format(args.inputdir))
    # Do what you need to read the documents here.
    emails = []
    authors = []

    for idx, author in enumerate(os.listdir(args.inputdir)):
        # author name and index(as a unique numerical identifier) added to authortable
        authortable[author] = idx
        for filename in os.listdir(args.inputdir + "/" + author):
            with open(args.inputdir + "/" + author + "/" + filename, 'r') as file:
                lines = file.readlines()
                # from a look at a few files, it seems like first 15 lines always are headers
                # so we ignore those from the start
                lines = lines[15:]
                message = "" 
                for line in lines:
                    if(line.isspace()):
                        continue
                    # some headers are longer than 15 lines, so we try to skip those here
                    if(skip_line(line)):
                        continue
                    # trying to avoid text that belongs to signature or forwarded emails
                    if(end_of_message(line)):
                        break
                    else:
                        # remove email addresses
                        line = re.sub(email_address, '', line)
                        # remove url addresses
                        line = re.sub(url, '', line)
                        # remove author names
                        line = remove_author_names(line)
                        # in case we removed all content of this line, just go to next
                        if (line.isspace()):
                            continue
                        # add processed line to message in lower case
                        message = message.lower() + line
                # in case a message is empty, e.g. if there were no original text only forwarded email text
                if(message == ''):
                    continue
                # add the message to emails
                emails.append(message)
                # add the author of the message to authors
                authors.append(author)

    print("Constructing table with {} feature dimensions and {}% test instances...".format(args.dims, args.testsize))
    # Build the table here.
    
    # vectorize the emails
    vec = CountVectorizer()
    vec.fit(emails)
    X = vec.transform(emails)
    
    # reduce dimensions
    X_reduced = SelectKBest(chi2, k=args.dims).fit_transform(X, authors)
    print(X_reduced.shape)

    # randomly split the data into two parts, training and testing.
    # to do this randomly, let's select some rows from the lists at random
    collection = range(0,len(emails)) # all possible indices of emails
    numrandom = int(len(emails) * (args.testsize/100)) # how many random numbers to select
    # sample that many unique numbers from the collection of indices of emails
    random_indices = random.sample(collection, numrandom)
    # now, random_indices contains a list of indices of emails that will belong to the test set

    print("Writing to {}...".format(args.outputfile))
    # Write the table out here.

    content = []
    # first add the table of author names and author numbers to content
    content.append(render_author_table())
    for idx,(author,emailvector) in enumerate(zip(authors,X_reduced)):
        # add 0 to this line if it is test data, 1 is it is training data
        line = '0' if idx in random_indices else '1'
        # flatten the list of vectorized email
        array = ' '.join([str(item) for sublist in emailvector.toarray() for item in sublist])
        # add author number to this line
        line = line + ' ' + author_to_num(author)
        # add vectorized email to this line
        line = line + str(array)
        # add the built line to contents
        content.append(line)

    # each line built will be a row in outputfile
    filecontent = "\n".join(content)
    file = open(args.outputfile, 'w')
    file.write(filecontent)
    file.close()

    print("Done!")