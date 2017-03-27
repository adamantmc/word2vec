import sys
import datetime
import json

def read_datasets(training_set_path, test_set_path, training_set_limit, test_set_limit):
    if training_set_limit != -1:
        training_set = json.load(open(training_set_path))["documents"][0:training_set_limit]
    else:
        training_set = json.load(open(training_set_path))["documents"]

    tlog("Training set read.")

    if test_set_limit !=-1:
        test_set = json.load(open(test_set_path))["documents"][0:test_set_limit]
    else:
        test_set = json.load(open(test_set_path))["documents"]

    tlog("Test set read.")

    return training_set, test_set

def printProgressBar(progress, total):
    prog_bar = ""
    padding = ""
	#int(100*ratio) so we get something like e.g. int(0.733435*100) = 73, halved, to print 50 '=' maximum
    prog = int(int(100*progress / total)*0.5)
    for i in range(prog):
        prog_bar += "="
    for i in range(50-prog):
        padding += " "
    if(int(progress/total) < 1):
        percentage = int(100*progress/total)
        print("\t["+prog_bar+padding+"] " + str(percentage)+"%", end="\r")
        sys.stdout.flush()
    else:
        print("\t["+prog_bar+padding+"] 100%")
        print()


def getTime():
    return str(datetime.datetime.time(datetime.datetime.now()))

def tlog(msg):
    print("["+getTime()+"] "+msg)
