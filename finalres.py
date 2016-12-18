'''
from functions import *

def output(pred):
    if pred==0:
        print("SPEED LIMIT-20Km/Hr")
    elif pred==1:
        print("SPEED LIMIT-30Km/Hr")
    elif pred==2:
        print("SPEED LIMIT-50Km/Hr")
    elif pred==3:
        print("SPEED LIMIT-60Km/Hr")
    elif pred==4:
        print("SPEED LIMIT-70Km/Hr")
    elif pred==5:
        print("SPEED LIMIT-80Km/Hr")
    elif pred==6:
        print("END OF SPEED LIMIT ZONE!")
    elif pred==7:
        print("SPEED LIMIT-100Km/Hr")
    elif pred==8:
        print("SPEED LIMIT-120Km/Hr")
    elif pred==9:
        print("NO OVERTAKING!")
    elif pred==10:
        print("NO PASSING FOR VEHICLES OVER 3.5TONS!")
    elif pred==11:
        print("CROSSROADS!")
    elif pred==12:
        print("PRIORITY ROADS!")
    elif pred==13:
        print("DISTANCE TO 'GIVE-WAY' LINE!")
    elif pred==14:
        print("STOP!")
    elif pred==15:
        print("NO VEHICLES!")
    elif pred==16:
        print("TRUCK CROSSING!")
    elif pred==17:
        print("NO ENTRY FOR VEHICULAR TRAFFIC!")
    elif pred==18:
        print("HIDDEN DIP!")
    elif pred==19:
        print("BEND TO LEFT!")
    elif pred==20:
        print("BEND TO RIGHT!")
    elif pred==21:
        print("DOUBLE BEND FIRST TO LEFT!")
    elif pred==22:
        print("UNEVEN ROAD!")
    elif pred==23:
        print("SLIPPERY ROAD!")
    elif pred==24:
        print("ROAD NARROWS ON RIGHT!")
    elif pred==25:
        print("MEN AT WORK!")
    elif pred==26:
        print("TRAFFIC SIGNALS!")
    elif pred==27:
        print("PEDESTRIANS PROHIBITED!")
    elif pred==28:
        print("PEDESTRIAN CROSSING!")
    elif pred==29:
        print("CYCLE ROUTE AHEAD!")
    elif pred==30:
        print("RISK OF ICE!")
    elif pred==31:
        print("WILD ANIMALS!")
    elif pred==32:
        print("NATIONAL SPEED LIMIT APPLIES!")
    elif pred==33:
        print("TURN RIGHT AHEAD!")
    elif pred==34:
        print("TURN LEFT AHEAD!")
    elif pred==35:
        print("AHEAD ONLY!")
    elif pred==36:
        print("JUNCTION AHEAD-RIGHT!")
    elif pred==37:
        print("JUNCTION AHEAD-LEFT!")
    elif pred==38:
        print("KEEP RIGHT!")
    elif pred==39:
        print("KEEP LEFT!")
    elif pred==40:
        print("MINI ROUNDABOUT!")
    elif pred==41:
        print("NO CARS ALLOWED!")
    elif pred==42:
        print("HEAVY VEHICLES NOT ALLOWED!")
    else:
        print("")

'''
from functions1 import *

def output(pred):
    if pred==0:
        print("\033[1;4;91m SPEED LIMIT-20Km/Hr \033[0m")
    elif pred==1:
        print("\033[1;4;91m SPEED LIMIT-30Km/Hr \033[0m")
    elif pred==2:
        print("\033[1;4;91m SPEED LIMIT-50Km/Hr \033[0m")
    elif pred==3:
        print("\033[1;4;91m SPEED LIMIT-60Km/Hr \033[0m")
    elif pred==4:
        print("\033[1;4;91m SPEED LIMIT-70Km/Hr \033[0m")
    elif pred==5:
        print("\033[1;4;91m SPEED LIMIT-80Km/Hr \033[0m")
    elif pred==6:
        print("\033[1;4;91m END OF SPEED LIMIT ZONE! \033[0m")
    elif pred==7:
        print("\033[1;4;91m SPEED LIMIT-100Km/Hr \033[0m")
    elif pred==8:
        print("\033[1;4;91m SPEED LIMIT-120Km/Hr \033[0m")
    elif pred==9:
        print("\033[1;4;91m NO OVERTAKING! \033[0m")
    elif pred==10:
        print("\033[1;4;91m NO PASSING FOR VEHICLES OVER 3.5TONS! \033[0m")
    elif pred==11:
        print("\033[1;4;91m CROSSROADS! \033[0m")
    elif pred==12:
        print("\033[1;4;91m PRIORITY ROADS! \033[0m")
    elif pred==13:
        print("\033[1;4;91m DISTANCE TO 'GIVE-WAY' LINE! \033[0m")
    elif pred==14:
        print("\033[1;4;91m STOP! \033[0m")
    elif pred==15:
        print("\033[1;4;91m NO VEHICLES! \033[0m")
    elif pred==16:
        print("\033[1;4;91m TRUCK CROSSING!  \033[0m")
    elif pred==17:
        print("\033[1;4;91m NO ENTRY FOR VEHICULAR TRAFFIC!  \033[0m")
    elif pred==18:
        print("\033[1;4;91m HIDDEN DIP!  \033[0m")
    elif pred==19:
        print("\033[1;4;91m BEND TO LEFT!  \033[0m")
    elif pred==20:
        print("\033[1;4;91m BEND TO RIGHT!  \033[0m")
    elif pred==21:
        print("\033[1;4;91m DOUBLE BEND FIRST TO LEFT!  \033[0m")
    elif pred==22:
        print("\033[1;4;91m UNEVEN ROAD!  \033[0m")
    elif pred==23:
        print("\033[1;4;91m SLIPPERY ROAD!  \033[0m")
    elif pred==24:
        print("\033[1;4;91m ROAD NARROWS ON RIGHT!  \033[0m")
    elif pred==25:
        print("\033[1;4;91m MEN AT WORK! \033[0m")
    elif pred==26:
        print("\033[1;4;91m TRAFFIC SIGNALS! \033[0m")
    elif pred==27:
        print("\033[1;4;91m PEDESTRIANS PROHIBITED! \033[0m")
    elif pred==28:
        print("\033[1;4;91m PEDESTRIAN CROSSING! \033[0m")
    elif pred==29:
        print("\033[1;4;91m CYCLE ROUTE AHEAD! \033[0m")
    elif pred==30:
        print("\033[1;4;91m RISK OF ICE! \033[0m")
    elif pred==31:
        print("\033[1;4;91m WILD ANIMALS! \033[0m")
    elif pred==32:
        print("\033[1;4;91m NATIONAL SPEED LIMIT APPLIES!  \033[0m")
    elif pred==33:
        print("\033[1;4;91m TURN RIGHT AHEAD!  \033[0m")
    elif pred==34:
        print("\033[1;4;91m TURN LEFT AHEAD!  \033[0m")
    elif pred==35:
        print("\033[1;4;91m AHEAD ONLY!  \033[0m")
    elif pred==36:
        print("\033[1;4;91m JUNCTION AHEAD-RIGHT!  \033[0m")
    elif pred==37:
        print("\033[1;4;91m JUNCTION AHEAD-LEFT!  \033[0m")
    elif pred==38:
        print("\033[1;4;91m KEEP RIGHT! \033[0m")
    elif pred==39:
        print("\033[1;4;91m KEEP LEFT! \033[0m")
    elif pred==40:
        print("\033[1;4;91m MINI ROUNDABOUT! \033[0m")
    elif pred==41:
        print("\033[1;4;91m NO CARS ALLOWED! \033[0m")
    elif pred==42:
        print("\033[1;4;91m HEAVY VEHICLES NOT ALLOWED! \033[0m")
    else:
        print("")

