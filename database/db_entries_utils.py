#db_entries_ultils.py

import os
import datetime
import sqlite3
import time
from datetime import datetime
from configParams import Parameters
from database.classEntries import Entries
from helper.text_decorators import check_similarity_threshold

params = Parameters()

fieldsList = ['platePercent', 'charPercent', 'eDate', 'eTime', 'plateNum']
dbEntries = params.dbEntries


def insertEntries(entry):
    sqlConnect = sqlite3.connect(dbEntries)
    sqlCursor = sqlConnect.cursor()

    sqlCursor.execute(
        "INSERT OR IGNORE INTO entries VALUES (:platePercent, :charPercent, :eDate, :eTime, :plateNum)",
        vars(entry))

    sqlConnect.commit()
    sqlConnect.close()


def dbGetPlateLatestEntry(plateNumber):
    sqlConnect = sqlite3.connect(dbEntries)
    sqlCursor = sqlConnect.cursor()
    FullEntriesSQL = f"""SELECT * FROM entries WHERE plateNum='{plateNumber}' ORDER BY eDate DESC LIMIT 1"""
    FullEntries = sqlCursor.execute(FullEntriesSQL).fetchall()
    if len(FullEntries) != 0:
        FullData = dict(zip([c[0] for c in sqlCursor.description], FullEntries[0]))
        sqlConnect.commit()
        sqlConnect.close()
        # اعتبارسنجی تاریخ
        try:
            datetime.strptime(f"{FullData['eDate']} {FullData['eTime']}", "%Y-%m-%d %H:%M:%S")
            return Entries(**FullData)
        except ValueError:
            print(f"Invalid date in database for plate {plateNumber}: {FullData['eDate']} {FullData['eTime']}")
            return None
    sqlConnect.close()
    return None



def dbGetAllEntries(limit=None, whereLike=''):
    listEntries = []
    sqlConnect = sqlite3.connect(params.dbEntries)
    sqlCursor = sqlConnect.cursor()
    where_clause = f"WHERE plateNum LIKE '%{whereLike}%'" if whereLike else ''
    entriesSQL = f"SELECT plateNum, eTime, eDate, charPercent, platePercent FROM entries {where_clause} ORDER BY eDate DESC, eTime DESC"
    if limit:
        entriesSQL += f" LIMIT {limit}"
    entries = sqlCursor.execute(entriesSQL).fetchall()
    for i in range(len(entries)):
        FullData = dict(zip([c[0] for c in sqlCursor.description], entries[i]))
        FullData['platePic'] = f"temp/{FullData['plateNum']}_{FullData['eTime']}_{FullData['eDate']}.jpg"
        listEntries.append(FullData)
    sqlConnect.commit()
    sqlConnect.close()
    return listEntries


similarityTemp = ''


def db_entries_time(number, charConfAvg, plateConfAvg, croppedPlate, external_service_data: dict = None):
    """
    ثبت یک پلاک در دیتابیس با تاریخ و زمان میلادی
    بررسی مشابهت با آخرین پلاک ثبت شده برای جلوگیری از تکرار سریع
    """
    global similarityTemp
    isSimilar = check_similarity_threshold(similarityTemp, number)
    if not isSimilar:
        similarityTemp = number
        if True:
            timeNow = datetime.now()
            result = dbGetPlateLatestEntry(number)
            if result is not None and number != '':

                strTime = result.getTime()
                strDate = result.getDate()

                if timeDifference(strTime, strDate):
                    display_time = timeNow.strftime("%H:%M:%S")
                    display_date = timeNow.strftime("%Y-%m-%d")

                    plateImgName = 'temp/{}_{}.jpg'.format(number,
                                                           datetime.now().strftime("%H:%M:%S_%Y-%m-%d"))
                    croppedPlate.save(plateImgName, format='jpg')

                    entries = Entries(plateConfAvg, charConfAvg, display_date, display_time, number)

                    insertEntries(entries)
                else:
                    pass
            else:
                if number != '':
                    display_time = time.strftime("%H:%M:%S")
                    display_date = time.strftime("%Y-%m-%d")

                    plateImgName = 'temp/{}_{}.jpg'.format(number, datetime.now().strftime("%H:%M:%S_%Y-%m-%d"))
                    croppedPlate.save(plateImgName, format='jpg')

                    entries = Entries(plateConfAvg, charConfAvg, display_date, display_time, number)

                    insertEntries(entries)



def dbGetEntriesByDateTime(start_time, end_time, plate_filter=''):
    listEntries = []
    sqlConnect = sqlite3.connect(params.dbEntries)
    sqlCursor = sqlConnect.cursor()
    where_clause = f"WHERE datetime(eDate || ' ' || eTime) BETWEEN '{start_time}' AND '{end_time}'"
    if plate_filter:
        where_clause += f" AND plateNum LIKE '%{plate_filter}%'"
    entriesSQL = f"SELECT plateNum, eTime, eDate, charPercent, platePercent FROM entries {where_clause}"
    entries = sqlCursor.execute(entriesSQL).fetchall()
    for i in range(len(entries)):
        FullData = dict(zip([c[0] for c in sqlCursor.description], entries[i]))
        # بدون جایگزینی : با -
        FullData['platePic'] = f"temp/{FullData['plateNum']}_{FullData['eTime']}_{FullData['eDate']}.jpg"
        listEntries.append(FullData)
    sqlConnect.commit()
    sqlConnect.close()
    return listEntries


def timeDifference(strTime, strDate):
    try:
        # بررسی فرمت و اعتبار تاریخ
        datetime.strptime(f"{strDate} {strTime}", "%Y-%m-%d %H:%M:%S")
    except ValueError:
        print(f"Invalid date or time: {strDate} {strTime}")
        return False  # یا مقدار پیش‌فرض دیگه‌ای که منطقی باشه

    start_time = datetime.strptime(f"{strTime} {strDate}", "%H:%M:%S %Y-%m-%d")
    end_time = datetime.strptime(datetime.now().strftime("%H:%M:%S %Y-%m-%d"), "%H:%M:%S %Y-%m-%d")
    delta = end_time - start_time
    sec = delta.total_seconds()
    min = (sec / 60).__ceil__()
    if min > 1:
        return True
    return False


def dbRemoveEntry(plateNum, eTime, eDate):
    """
    Deletes a specific entry from the database based on plateNum, eTime, and eDate.
    Also removes the associated image file from the temp folder.
    """
    sqlConnect = sqlite3.connect(dbEntries)
    sqlCursor = sqlConnect.cursor()
    removeEntrySQL = f"""DELETE FROM entries WHERE plateNum='{plateNum}' AND eTime='{eTime}' AND eDate='{eDate}'"""
    sqlCursor.execute(removeEntrySQL)
    sqlConnect.commit()
    sqlConnect.close()
    
    # حذف فایل عکس مرتبط
    plate_pic_path = f"temp/{plateNum}_{eTime}_{eDate}.jpg"
    if os.path.exists(plate_pic_path):
        os.remove(plate_pic_path)
    
    