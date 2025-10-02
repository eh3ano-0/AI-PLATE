#classEntries.py

from PySide6.QtGui import QColor
from PySide6.QtWidgets import QTableWidgetItem

from helper import jalali
from helper.text_decorators import convert_english_to_persian, split_string_language_specific


class Entries:

    def __init__(self, platePercent, charPercent, eDate, eTime, plateNum):
        self.plateNum = plateNum
        self.eTime = eTime
        self.eDate = eDate

        self.charPercent = charPercent
        self.platePercent = platePercent

    def getTime(self):
        return self.eTime

    def getDate(self, persian=True):
        if persian:
            return jalali.Gregorian(self.eDate).persian_string()
        return self.eDate

    def getPlatePic(self):
        return 'temp/{}_{}_{}.jpg'.format(self.plateNum, self.eTime, self.eDate)

    def getCharPercent(self):
        return "{}%".format(self.charPercent)

    def getPlatePercent(self):
        return "{}%".format(self.platePercent)

    def getPlateNumber(self, display=False):
        return convert_english_to_persian(split_string_language_specific(self.plateNum), display)

