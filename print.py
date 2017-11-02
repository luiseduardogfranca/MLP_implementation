import sys
import time

# sys.stdout.write("\n\t[%s" %(" " * 14))
# sys.stdout.flush()

percent = 10
cont = 0


char_load = ("#" * cont)
char_blankspace = (" " * (10 - cont))


for i in range(1, 101):

    if(i % percent == 0):
        cont += 1
        char_load = ("#" * cont)
        char_blankspace = (" " * (10 - cont))

    # print(loading)
    loading = ("[%s%s]%d%%" %(char_load, char_blankspace, i))
    sys.stdout.write(loading)
    sys.stdout.flush()

    # print(len(loading))

    if(i != 100):
        sys.stdout.write("\b" * (13 + len(str(i))))

    time.sleep(0.1)
