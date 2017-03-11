import sys
import datetime

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
