import pickle
import os
import time
import datetime
# from pushbullet import Pushbullet
import getpass


class command():
    def __init__(self, command_str, status, end_time, elapsed_time):
        self.command_str = command_str
        self.status = status
        self.end_time = end_time
        self.elapsed_time = elapsed_time


command_list = [
###

"python main.py --model_name vgg16 --mode train --hp_bs 8 --hp_lr 1e-3 --hp_ep 1 --hp_sch plateu --num_worker 8 --description 8",
"python main.py --model_name vgg16 --mode train --hp_bs 8 --hp_lr 1e-3 --hp_ep 1 --hp_sch plateu --num_worker 8 --description 12",
"python main.py --model_name vgg16 --mode train --hp_bs 16 --hp_lr 1e-3 --hp_ep 1 --hp_sch plateu --num_worker 8 --description 8",
"python main.py --model_name vgg16 --mode train --hp_bs 16 --hp_lr 1e-3 --hp_ep 1 --hp_sch plateu --num_worker 8 --description 12",
# "python main.py --model_name resnet34 --mode train --hp_bs 128 --hp_lr 1e-3 --hp_ep 1 --hp_sch plateu",
# "python main.py --model_name timm_vit_small_patch16_224 --mode train --hp_bs 10 --hp_lr 1e-3 --hp_ep 1 --hp_sch plateu",

###
]

# initially, generate command objects.
c_list = []
for command_elem in command_list:
    c_i = command(command_elem, " Waiting ", "", "")
    c_list.append(c_i)


# run commands.
for i, c_i in enumerate(c_list):
    # start of command
    print("")
    print("Command: {}/{}".format((i + 1), len(c_list)))
    print("start_time:", str(datetime.datetime.now()))
    print(c_i.command_str)
    print("")

    # logging
    c_i.status = " Running "
    with open("log.pkl", "wb") as f:
        pickle.dump(c_list, f)

    start_time = time.time()
    os.system(c_i.command_str)
    ##
    time.sleep(0.1)
    ##
    end_time = time.time()

    # end of command
    print("")
    print("end_time:", str(datetime.datetime.now()))

    hours, rem = divmod(end_time - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

    # logging
    c_i.status = " Done "
    c_i.end_time = "End time: " + str(datetime.datetime.now())
    c_i.elapsed_time = "Elapsed time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
    with open("log.pkl", "wb") as f:
        pickle.dump(c_list, f)

# # push
# api_key = "o.EePlquoGSmUa2ktvw92G7OjaVka3Uii3"  # new
# # api_key = "o.LcPgEyU7O8z2bF0BAQ63pD7VFljwOFVY"
# pb = Pushbullet(api_key)
# user = getpass.getuser()
# push_title = "[" + user + "] Execution ended. Last command was: " + command_list[-1]
# push_body = str(datetime.datetime.now())
# push = pb.push_note(push_title, push_body)
