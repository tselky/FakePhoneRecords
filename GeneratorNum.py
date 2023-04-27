# import module
import random as r
import pandas as pd
import time as t
from numpy.random import choice


#Program to simulate call records based on conditional probability and a hidden weight system.
#Some labels are influenced by the population of others
#A light/medium/heavy weight score system is used in the background to influence the column that flags a call as suspiscious or not.


#All Michigan areacodes
local_codes = ['231', '248', '269', '313', '517', '586', '616', '734', '810', '906', '947', '989']

data = {
     "contact": [], "call_history": [], "sms_history": [],
        "local": [], "answer": [], "duration": [], "cnam": [], "flag": [],
}

col_list = [ "contact", "call_history", "sms_history",
        "local", "answer", "duration", "cnam", "flag"]

#Data appender
def append_data(arr: dict):
    try:
        for key, val in arr.items():
            data[key].append(val)
    except Exception as e:
        print("Append Data failed: ", e)
#Might rename
#Generates a phone number, with an 30% chance to be a michigan number
def generate_call_record():
    sampleList = [r.sample(range(1000000000, 9999999999), 1)[0],
                  str(r.choice(local_codes)) + str(r.sample(range(1000000, 9999999), 1)[0])]
    randomNumberList = choice(
        sampleList, 1, p=[0.7, 0.3])
    number = randomNumberList[0]
    return number
#Uses number to check if its from michigan, then generates conditionally probable stats with a hidden weight score
#['local','contact','cnam','weight',]
def generate_caller_stats(number):
    weight = 0
    l = 0.05
    m = 0.1
    h = 0.15

    #Check if local
    caller = []
    s1 = str(number)
    match s1.startswith(tuple(local_codes)) == True:
        case True:
            local_result = '1'
        case False:
            local_result = '0'
    caller.append(local_result)

    caller.append(local_result)

    #Decide if a contact based on local status
    match local_result == '1':
        case True:
            sampleList = ['1', '0']
            contactList = choice(sampleList, 1, p=[0.6, 0.4])
            weight += l

        case False:
            sampleList = ['1', '0']
            contactList = choice(sampleList, 1, p=[0.3, 0.7])
            weight -= l

    #Calcualtes weight change on contact status
    if contactList[0] == '1':
        weight += h
    elif contactList[0] == '0':
        weight -= h

    caller.append(contactList[0])

    #Decide if CNAM registered based on local status
    match local_result == '1':
        case True:
            sampleList = ['1', '0']
            cnamList = choice(sampleList, 1, p=[0.8, 0.2])
        case False:
            sampleList = ['1', '0']
            cnamList = choice(sampleList, 1, p=[0.5, 0.5])

    if cnamList[0] == '1':
        weight += m
    elif contactList[0] == '0':
        weight -= m

    caller.append(cnamList[0])

    #Adds weight to list for later use
    caller.append(int(weight))

    return caller
#Generates conditionally probable communication history with number based on contact status. Produces hidden weight score as well
#['Call check', 'SMS check', 'Weight']
def generate_history(result):
    weight = 0
    l = 0.1
    m = 0.15
    h = 0.2
    history_result = []

    #call history weighted by contact status
    match result[1] == '1':
        case True:
            sampleList = ['1', '0']
            callHistoryList = choice(sampleList, 1, p=[0.95, 0.05])

        case False:
            sampleList = ['1', '0']
            callHistoryList = choice(sampleList, 1, p=[0.3, 0.7])

    if callHistoryList[0] == '1':
        weight += m
    elif callHistoryList[0] == '0':
        weight -= m

    history_result.append(callHistoryList[0])
    #sms history weighted by call history
    match callHistoryList[0] == '1':
        case True:
            sampleList = ['1', '0']
            smsHistoryList = choice(sampleList, 1, p=[0.8, 0.2])

        case False:
            sampleList = ['1', '0']
            smsHistoryList = choice(sampleList, 1, p=[0.1, 0.9])

    if smsHistoryList[0] == '1':
        weight += h
    elif smsHistoryList[0] == '0':
        weight -= h

    history_result.append(smsHistoryList[0])
    history_result.append(weight)
    return history_result
def generate_call_stats(cstats):
    #["Answer", "Duration", "Weight"]
    weight = 0
    l = 0.05
    m = 0.1
    h = 0.15
    stats = []
    durationList = []
    #Decide if call was answered based on if contact
    match cstats[1] == '1':
        case True:
            sampleList = ['1', '0']
            answerList = choice(sampleList, 1, p=[0.85, 0.15])
        case False:
            sampleList = ['1', '0']
            answerList = choice(sampleList, 1, p=[0.6, 0.4])

    stats.append(answerList[0])
    #Decide on call duration if answered

    if stats[0] == '1':
        match cstats[1] == '1':
            case True:
                sampleList = [r.sample(range(1, 100), 1)[0], r.sample(range(1, 800), 1)[0]]
                durationList = choice(sampleList, 1, p=[0.05, 0.95])
            case False:
                sampleList = [r.sample(range(1, 100), 1)[0], r.sample(range(1, 800), 1)[0]]
                durationList = choice(sampleList, 1, p=[0.7, 0.3])

    elif stats[0] == '0':
        durationList.append(0)

    #Call duration weight score
    if int(durationList[0]) > 70:
        weight+=l
    elif int(durationList[0]) < 70 and int(durationList[0]) > 0:
        weight-=l
    elif durationList[0] == 0:
        pass

    stats.append(str(durationList[0]))
    stats.append(weight)
    return stats
#Uses totaled and normalized weight-scores to pre-flag calls as suspicious
def decide_preflag(weight):
    #Decides if call is deemed suspicious
    flag = []
    sampleList = ['1', '0']
    flagList = choice(sampleList, 1, p=[0.3-weight+0.05, 0.7+weight-0.05])
    flag.append(str(flagList[0]))
    #print(flag)
    return flag
#Produces dataset in csv format of x size.
def generate_dataset(x):
    print("#########################################")
    print("Creating Dataset of "+str(x)+" size. . .")
    start = t.time()
    for i in range(x):
        number = generate_call_record()

        #caller_stats= #['local','contact','cnam','weight',]
        caller_stats = generate_caller_stats(number)

        #call_stats["answer", "duration", "weight"]
        call_stats = generate_call_stats(caller_stats)

        #history = ['call check', 'SMS check', 'Weight']
        history = generate_history(caller_stats)

        local = caller_stats[0]
        contact = caller_stats[1]
        cnam = caller_stats[2]
        answer = call_stats[0]
        duration = call_stats[1]
        if int(duration) >100:
            durationtoggle = 1
        else:
            durationtoggle = 0
        call_history = history[0]
        sms_history = history[1]
        #weight = (int(caller_stats[3])+call_stats[2]+history[2]) *0.2
        #print(weight)
        #flag = weight

        TotalList = [local, contact, cnam, answer, durationtoggle, call_history, sms_history]
        ZeroCount = TotalList.count('0')
        print(ZeroCount)

        if ZeroCount >= 5:
            flag = 1
        elif ZeroCount == 4:
            sampleList = ['1', '0']
            flagList = choice(sampleList, 1, p=[0.6, 0.4])
            flag = str(flagList[0])
        else:
            flag = 0


        #print('Weight: ', weight)

        phone_data = {
             "contact": contact, "call_history": call_history, "sms_history": sms_history,
        "local": local, "answer": answer, "duration": durationtoggle, "cnam": cnam, "flag": flag,
        }

        #print(phone_data)
        append_data(phone_data)

    #create dataframe
    df1 = pd.DataFrame.from_dict(data, orient="index").T
    filename = str(x)+"_phone_records.csv"

    #Creates a dataset file or adds to one and prunes all duplicate rows with the exact same number
    try:
        #print("TRY BLOCK")
        df2 = pd.read_csv(filename)
        merge = pd.concat([df1,df2]).drop_duplicates(subset=["number"], keep='last').reset_index(drop=True)
        merge.to_csv(filename, index=False)
        pd.set_option('display.max_columns', None)

    except:
        #print("EXCEPT BLOCK")
        df1.to_csv(filename, index=False)

    end = t.time()
    timer = str(round((end-start),3))+" sec."
    print("Dataset generated as: " + filename)
    print("Completed in "+timer)

#Produces 3 files of the following set sizes and appropriately named
generate_dataset(100)
generate_dataset(1000)
generate_dataset(10000)



